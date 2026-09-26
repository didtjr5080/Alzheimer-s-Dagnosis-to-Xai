"""Orchestrates the merged (OASIS-3+ADNI) live-inference path for the UI:
scan_id -> per-branch probabilities (CLIP / 3D CNN / GBM baseline) -> fixed-
weight ensemble, plus CLIP Grad-ECLIP and 3D CNN Grad-CAM XAI for one
representative view each. A branch with no usable input for a given scan_id
returns `None` and is displayed as "사용 불가"; the ensemble is only computed
when all three branches are available (work order section 4: no
re-weighting of remaining branches).

Model bundles are loaded once per process (`functools.lru_cache`) since
CLIP/3D CNN loading takes several seconds.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .adapters.merged_clip import CLASS_NAMES, load_merged_clip_bundle, predict_slice_batch
from .adapters.merged_cnn3d import load_merged_cnn3d_bundle
from .adapters.merged_cnn3d_gradcam import central_slice_overlays, grad_cam_3d
from .adapters.merged_ensemble import combine_ensemble, ensemble_predicted_class
from .adapters.merged_gbm import load_merged_gbm_bundle
from .adapters.merged_grad_eclip import explain_image as clip_explain_image

REPO_ROOT = Path(__file__).resolve().parents[2]
MERGED_ROOT = REPO_ROOT / "merged_project"


@dataclass(frozen=True)
class ScanMeta:
    scan_id: str
    dataset_source: str
    split: str
    label: str | None


def _read_csv_dicts(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


@lru_cache(maxsize=1)
def _unified_manifest_by_scan() -> dict[str, dict]:
    return {r["scan_id"]: r for r in _read_csv_dicts(MERGED_ROOT / "unified_manifest.csv")}


@lru_cache(maxsize=1)
def _slice_manifest_by_scan() -> dict[str, list[int]]:
    by_scan: dict[str, list[int]] = {}
    for row in _read_csv_dicts(MERGED_ROOT / "unified_slice_manifest.csv"):
        by_scan.setdefault(row["scan_id"], []).append(int(row["slice_index"]))
    return by_scan


def merged_project_available() -> bool:
    return (MERGED_ROOT / "unified_manifest.csv").exists()


def list_test_scan_ids(limit: int | None = 50) -> list[str]:
    ids = sorted(
        scan_id for scan_id, row in _unified_manifest_by_scan().items() if row["split"] == "test"
    )
    return ids[:limit] if limit else ids


def get_scan_meta(scan_id: str) -> ScanMeta | None:
    row = _unified_manifest_by_scan().get(scan_id)
    if row is None:
        return None
    return ScanMeta(scan_id=scan_id, dataset_source=row["dataset_source"], split=row["split"], label=row.get("label") or None)


@lru_cache(maxsize=1)
def _clip_bundle():
    return load_merged_clip_bundle(REPO_ROOT, device=None)


@lru_cache(maxsize=1)
def _gbm_bundle():
    return load_merged_gbm_bundle(REPO_ROOT)


@lru_cache(maxsize=1)
def _cnn3d_bundle():
    return load_merged_cnn3d_bundle(REPO_ROOT, device=None)


def _clip_subject_probs_and_xai(scan_id: str) -> tuple[dict | None, dict]:
    slice_indices = sorted(_slice_manifest_by_scan().get(scan_id, []))
    if not slice_indices:
        return None, {}

    bundle = _clip_bundle()
    slices_dir = MERGED_ROOT / "slices_multi"
    images, valid_indices = [], []
    for idx in slice_indices:
        path = slices_dir / f"{scan_id}_cor{idx:03d}.png"
        if path.exists():
            images.append(Image.open(path))
            valid_indices.append(idx)
    if not images:
        return None, {}

    with torch.no_grad():
        probs = predict_slice_batch(bundle, images).cpu().numpy()
    mean_probs = probs.mean(axis=0)
    clip_probs = {name: float(p) for name, p in zip(CLASS_NAMES, mean_probs)}

    rep_pos = len(valid_indices) // 2
    rep_image = images[rep_pos]
    xai = clip_explain_image(bundle, rep_image)
    return clip_probs, {
        "rep_slice_index": valid_indices[rep_pos],
        "rep_image": rep_image,
        "heatmap_224": xai["heatmap_224"],
        "predicted_class": xai["predicted_class"],
        "target_class": xai["target_class"],
    }


def _gbm_probs(scan_id: str) -> dict | None:
    return _gbm_bundle().predict_by_scan_id(scan_id)


def _cnn3d_probs_and_xai(scan_id: str) -> tuple[dict | None, dict, np.ndarray | None, np.ndarray | None]:
    bundle = _cnn3d_bundle()
    path = bundle.cache_dir / f"{scan_id}.npy"
    if not path.exists():
        return None, {}, None, None
    volume = np.load(path)
    cam_result = grad_cam_3d(bundle, volume)
    overlays = central_slice_overlays(volume, cam_result["cam_volume"])
    return cam_result["probs"], overlays, volume, cam_result["cam_volume"]


@dataclass
class MergedAnalysisResult:
    scan_meta: ScanMeta
    branch_probs: dict[str, dict | None]
    ensemble_probs: dict | None
    ensemble_predicted_class: str | None
    clip_xai: dict
    cnn3d_overlays: dict
    clip_analysis: object | None = None
    cnn3d_analysis: object | None = None
    cnn3d_volume: np.ndarray | None = None
    cnn3d_cam_volume: np.ndarray | None = None


def analyze_merged_scan(scan_id: str, run_region_analysis: bool = True) -> MergedAnalysisResult:
    scan_id = (scan_id or "").strip()
    if not scan_id:
        raise ValueError("scan_id를 입력해 주세요.")
    meta = get_scan_meta(scan_id)
    if meta is None:
        raise ValueError(f"unified_manifest.csv에서 scan_id를 찾을 수 없습니다: {scan_id}")

    clip_probs, clip_xai = _clip_subject_probs_and_xai(scan_id)
    gbm_probs = _gbm_probs(scan_id)
    cnn3d_probs, cnn3d_overlays, cnn3d_volume, cnn3d_cam_volume = _cnn3d_probs_and_xai(scan_id)

    branch_probs = {"CLIP": clip_probs, "3D CNN": cnn3d_probs, "GBM baseline": gbm_probs}

    ensemble_probs = None
    ensemble_pred = None
    if clip_probs is not None and gbm_probs is not None and cnn3d_probs is not None:
        ensemble_probs = combine_ensemble(gbm_probs, cnn3d_probs, clip_probs)
        ensemble_pred = ensemble_predicted_class(ensemble_probs)

    clip_analysis = None
    cnn3d_analysis = None
    if run_region_analysis:
        from .merged_region_analysis import analyze_clip_regions, analyze_cnn3d_regions

        if clip_xai:
            gray = np.array(clip_xai["rep_image"].convert("L"))
            clip_analysis = analyze_clip_regions(
                _clip_bundle(), gray, clip_xai["heatmap_224"], clip_xai["predicted_class"],
            )
        if cnn3d_volume is not None:
            cnn3d_predicted_class = max(cnn3d_probs, key=cnn3d_probs.get)
            cnn3d_analysis = analyze_cnn3d_regions(_cnn3d_bundle(), cnn3d_volume, cnn3d_cam_volume, cnn3d_predicted_class)

    return MergedAnalysisResult(
        scan_meta=meta, branch_probs=branch_probs, ensemble_probs=ensemble_probs,
        clip_analysis=clip_analysis, cnn3d_analysis=cnn3d_analysis,
        cnn3d_volume=cnn3d_volume, cnn3d_cam_volume=cnn3d_cam_volume,
        ensemble_predicted_class=ensemble_pred, clip_xai=clip_xai, cnn3d_overlays=cnn3d_overlays,
    )


def analyze_merged_uploaded_image(image_path: str, run_region_analysis: bool = True) -> MergedAnalysisResult:
    """CLIP-only analysis for a newly uploaded image, not one of
    `merged_project`'s cached scans. 3D CNN needs a full MNI-registered 3D
    volume and GBM baseline needs SynthSeg structural volume ratios -- neither
    can be derived from a single 2D image, so both branches are reported as
    unavailable and the ensemble is not computed (a missing branch is never
    covered by re-weighting the remaining ones, same rule as the scan_id
    path). The image is resized to 224x224 before anything else so the
    region-masking grid stays spatially aligned with the 224x224 CAM."""
    image = Image.open(image_path).convert("L").resize((224, 224))
    bundle = _clip_bundle()
    with torch.no_grad():
        probs_tensor = predict_slice_batch(bundle, [image])
    clip_probs = {name: float(p) for name, p in zip(CLASS_NAMES, probs_tensor[0].cpu().numpy())}
    xai = clip_explain_image(bundle, image)

    clip_xai = {
        "rep_slice_index": None,
        "rep_image": image,
        "heatmap_224": xai["heatmap_224"],
        "predicted_class": xai["predicted_class"],
        "target_class": xai["target_class"],
    }
    branch_probs = {"CLIP": clip_probs, "3D CNN": None, "GBM baseline": None}

    clip_analysis = None
    if run_region_analysis:
        from .merged_region_analysis import analyze_clip_regions

        gray = np.array(image)
        clip_analysis = analyze_clip_regions(bundle, gray, xai["heatmap_224"], xai["predicted_class"])

    meta = ScanMeta(scan_id="(업로드된 이미지)", dataset_source="사용자 업로드", split="N/A", label=None)
    return MergedAnalysisResult(
        scan_meta=meta, branch_probs=branch_probs, ensemble_probs=None, ensemble_predicted_class=None,
        clip_xai=clip_xai, cnn3d_overlays={}, clip_analysis=clip_analysis, cnn3d_analysis=None,
        cnn3d_volume=None, cnn3d_cam_volume=None,
    )
