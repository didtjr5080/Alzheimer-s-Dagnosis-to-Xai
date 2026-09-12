from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from .contracts import HeatmapValidation, SampleIdentity, XAIArtifact, XAIProvenance
from .visualization import heatmap_to_rgb, normalize_heatmap, overlay_heatmap


CLASS_NAMES = ("CN", "MCI", "AD")
METHOD_NAME = "clip-gradient"
MODEL_ID = "legacy_clip_lr"
MODEL_VERSION = "openai/clip-vit-base-patch16+sklearn-logistic-regression"
TARGET_SCORE_TYPE = "LR class logit"
PREPROCESSING_VERSION = "CLIPProcessor.image_processor@handoff"
CLASS_MAPPING_VERSION = "clf.classes_+metadata/model_config.json"
MASK_METHOD = "largest_component_intensity"
MASK_THRESHOLD = 0.08
TOP_ACTIVATION_PERCENT = 5.0
MINIMUM_BRAIN_FRACTION = 0.10
MAXIMUM_BRAIN_FRACTION = 0.70
BORDERLINE_FOREGROUND_BACKGROUND_RATIO = 1.25
XAI_CONFIG = {
    "method_name": METHOD_NAME,
    "mask_method": MASK_METHOD,
    "mask_threshold": MASK_THRESHOLD,
    "top_activation_percent": TOP_ACTIVATION_PERCENT,
    "minimum_brain_fraction": MINIMUM_BRAIN_FRACTION,
    "maximum_brain_fraction": MAXIMUM_BRAIN_FRACTION,
    "borderline_foreground_background_ratio": BORDERLINE_FOREGROUND_BACKGROUND_RATIO,
    "overlay_alpha": 0.45,
    "colormap": "red_blue_internal",
    "interpolation": "bilinear",
    "align_corners": False,
}
INDEX_COLUMNS = [
    "analysis_id",
    "dataset_id",
    "subject_id",
    "session_or_visit_id",
    "scan_id",
    "series_id",
    "input_level",
    "slice_axis",
    "slice_index",
    "source_relative_path",
    "source_sha256",
    "true_label",
    "predicted_class",
    "prediction_status",
    "target_class",
    "target_score_type",
    "model_id",
    "checkpoint_sha256",
    "preprocessing_version",
    "code_commit",
    "xai_method",
    "xai_config_hash",
    "run_id",
    "original_path",
    "raw_heatmap_path",
    "heatmap_png_path",
    "overlay_path",
    "metadata_path",
    "xai_qc_status",
    "created_at",
]


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_digest(payload: dict) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def xai_config_hash() -> str:
    return canonical_json_digest(XAI_CONFIG)


def _safe_token(value: str | None) -> str:
    if not value:
        return "NA"
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-") or "NA"


def parse_slice_filename(filename: str) -> tuple[str, str | None, int | None]:
    name = Path(filename).name
    match = re.match(r"^(?P<scan>.+)_(?P<axis>cor|sag|ax|axi)(?P<index>\d+)\.png$", name, re.IGNORECASE)
    if not match:
        stem = Path(filename).stem
        return stem, None, None
    axis = match.group("axis").lower()
    if axis == "axi":
        axis = "ax"
    return match.group("scan"), axis, int(match.group("index"))


def infer_dataset_id(scan_id: str, filename: str) -> str:
    token = scan_id or filename
    if token.startswith("OAS"):
        return "OASIS3"
    if re.match(r"^\d{3}_S_\d{4}", token):
        return "ADNI"
    return "unknown"


def infer_subject_id(scan_id: str, dataset_id: str) -> str:
    if dataset_id == "OASIS3":
        match = re.match(r"^(OAS\d+)_MR_d\d+", scan_id)
        return match.group(1) if match else scan_id
    if dataset_id == "ADNI":
        match = re.match(r"^(\d{3}_S_\d{4})", scan_id)
        return match.group(1) if match else scan_id
    return scan_id


def relative_to_known_root(path: Path, roots: list[Path]) -> str:
    resolved = path.resolve()
    for root in roots:
        try:
            return resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
    return resolved.as_posix()


def load_sample_manifest(repo_root: Path) -> dict[str, dict]:
    manifest = (
        repo_root
        / "clip_lr_grad_eclip_handoff_v1_20260812_105027"
        / "data"
        / "xai_samples"
        / "xai_sample_manifest.csv"
    )
    if not manifest.exists():
        return {}
    rows = {}
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[Path(row["slice_path_portable"]).name] = row
    return rows


def validate_sample_identity_rows(repo_root: Path) -> dict:
    manifest = (
        repo_root
        / "clip_lr_grad_eclip_handoff_v1_20260812_105027"
        / "data"
        / "xai_samples"
        / "xai_sample_manifest.csv"
    )
    if not manifest.exists():
        return {"status": "UNAVAILABLE_MISSING_FILES", "rows": 0, "duplicates": []}
    keys = {}
    duplicates = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            path = (
                repo_root
                / "clip_lr_grad_eclip_handoff_v1_20260812_105027"
                / "data"
                / "xai_samples"
                / row["slice_path_portable"]
            )
            identity = identity_from_path(path, repo_root)
            key = (
                identity.dataset_id,
                identity.subject_id,
                identity.session_or_visit_id,
                identity.scan_id,
                identity.series_id,
                identity.input_level,
                identity.slice_axis,
                identity.slice_index,
                identity.source_sha256,
            )
            if key in keys:
                duplicates.append({"first": keys[key], "second": row["slice_path_portable"]})
            else:
                keys[key] = row["slice_path_portable"]
    return {"status": "PASS", "rows": len(keys), "duplicates": duplicates}


def migrate_legacy_grad_eclip_samples(repo_root: Path, tolerance: float = 0.001) -> dict:
    manifest_rows = list(load_sample_manifest(repo_root).values())
    samples_dir = repo_root / "clip_lr_grad_eclip_handoff_v1_20260812_105027" / "reports" / "grad_eclip_samples"
    results = []
    if not samples_dir.exists():
        return {"status": "UNAVAILABLE_MISSING_FILES", "results": []}
    for image_path in sorted(samples_dir.glob("*_gradeclip.png")):
        stem = image_path.stem.replace("_gradeclip", "")
        match = re.match(r"(?P<mr_id>.+)_(?P<case>correct|incorrect)$", stem)
        if not match:
            results.append({"legacy_file": image_path.name, "status": "UNRESOLVED", "reason": "filename pattern mismatch"})
            continue
        mr_id = match.group("mr_id")
        case = match.group("case")
        candidates = [row for row in manifest_rows if row["MR_ID"] == mr_id and row["case"] == case]
        status = "UNRESOLVED"
        mapped = None
        reason = "subject/case candidates require displayed probability vector"
        # Known legacy sample images contain subject-only names. The documented OAS30100 image has displayed
        # probabilities matching cor084; this generic rule maps only if one candidate exactly matches that vector.
        documented_probs = {
            "OAS30100_MR_d0158": (0.4191, 0.4529, 0.1280),
        }.get(mr_id)
        if documented_probs is not None:
            matches = []
            for row in candidates:
                vector = (
                    float(row["CN_prob_slice"]),
                    float(row["MCI_prob_slice"]),
                    float(row["AD_prob_slice"]),
                )
                max_delta = max(abs(vector[i] - documented_probs[i]) for i in range(3))
                if max_delta <= tolerance:
                    matches.append((row, max_delta))
            if len(matches) == 1:
                mapped = Path(matches[0][0]["slice_path_portable"]).name
                status = "MAPPED"
                reason = f"unique probability-vector match within tolerance {tolerance}"
            elif len(matches) > 1:
                status = "AMBIGUOUS"
                reason = f"{len(matches)} probability-vector matches"
            else:
                status = "UNRESOLVED"
                reason = "no probability-vector match"
        elif len(candidates) > 1:
            status = "AMBIGUOUS"
            reason = f"{len(candidates)} subject/case candidates and no exact slice metadata"
        elif len(candidates) == 1:
            mapped = Path(candidates[0]["slice_path_portable"]).name
            status = "MAPPED"
            reason = "single subject/case candidate"
        results.append(
            {
                "legacy_file": image_path.name,
                "status": status,
                "mapped_slice": mapped,
                "candidate_count": len(candidates),
                "reason": reason,
            }
        )
    counts = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    return {"status": "PASS", "counts": counts, "results": results}


def write_heatmap_validation_reports(repo_root: Path) -> tuple[Path, Path]:
    out_dir = repo_root / "clip_xai_app" / "artifacts" / "verification"
    out_dir.mkdir(parents=True, exist_ok=True)
    identity = validate_sample_identity_rows(repo_root)
    migration = migrate_legacy_grad_eclip_samples(repo_root)
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sample_identity": identity,
        "legacy_grad_eclip_migration": migration,
    }
    json_path = out_dir / "heatmap_identity_audit.json"
    md_path = out_dir / "heatmap_identity_audit.md"
    _atomic_write_text(json_path, json.dumps(report, indent=2, ensure_ascii=False))
    lines = [
        "# Heatmap Identity Audit",
        "",
        f"- sample identity rows: {identity.get('rows')}",
        f"- duplicate sample identities: {len(identity.get('duplicates', []))}",
        f"- migration counts: `{migration.get('counts', {})}`",
        "",
        "## Legacy Grad-ECLIP Samples",
        "",
        "| legacy_file | status | mapped_slice | candidates | reason |",
        "|---|---|---|---:|---|",
    ]
    for item in migration.get("results", []):
        lines.append(
            f"| {item['legacy_file']} | {item['status']} | {item.get('mapped_slice') or ''} | "
            f"{item.get('candidate_count', 0)} | {item['reason']} |"
        )
    _atomic_write_text(md_path, "\n".join(lines) + "\n")
    return json_path, md_path


def identity_from_path(path: Path, repo_root: Path) -> SampleIdentity:
    scan_id, axis, slice_index = parse_slice_filename(path.name)
    dataset_id = infer_dataset_id(scan_id, path.name)
    subject_id = infer_subject_id(scan_id, dataset_id)
    return SampleIdentity(
        dataset_id=dataset_id,
        subject_id=subject_id,
        session_or_visit_id=None,
        scan_id=scan_id,
        series_id=None,
        source_relative_path=relative_to_known_root(path, [repo_root]),
        source_sha256=file_sha256(path),
        input_level="slice",
        slice_axis=axis,
        slice_index=slice_index,
    )


def _checkpoint_digest(bundle) -> str:
    root = Path(getattr(bundle, "root", ""))
    candidates = [
        root / "models" / "clip_lr_weights.npz",
        root / "models" / "clip_lr_classifier_new_run.joblib",
    ]
    parts = [file_sha256(path) for path in candidates if path.exists()]
    if not parts:
        return "unknown"
    return hashlib.sha256("".join(parts).encode("ascii")).hexdigest()


def _code_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={repo_root.as_posix()}", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return "unknown"
    commit = result.stdout.strip()
    return commit if re.fullmatch(r"[0-9a-fA-F]{40}", commit) else "unknown"


def analysis_id_for(
    identity: SampleIdentity,
    model_id: str,
    checkpoint_sha256: str,
    preprocessing_version: str,
    target_class: str,
    target_score_type: str,
    run_id: str,
    code_commit: str = "unknown",
) -> str:
    payload = {
        "identity": asdict(identity),
        "model_id": model_id,
        "checkpoint_sha256": checkpoint_sha256,
        "code_commit": code_commit,
        "preprocessing_version": preprocessing_version,
        "class_mapping_version": CLASS_MAPPING_VERSION,
        "xai_method": METHOD_NAME,
        "target_class": target_class,
        "target_score_type": target_score_type,
        "xai_config_hash": xai_config_hash(),
        "run_id": run_id,
    }
    return canonical_json_digest(payload)


def artifact_stem(provenance: XAIProvenance) -> str:
    correctness = (
        "unknown"
        if provenance.prediction_correct is None
        else "correct"
        if provenance.prediction_correct
        else "incorrect"
    )
    axis = "slice"
    if provenance.slice_axis and provenance.slice_index >= 0:
        axis = f"{provenance.slice_axis}{provenance.slice_index:03d}"
    elif provenance.slice_index >= 0:
        axis = f"slice{provenance.slice_index:03d}"
    return "_".join(
        [
            _safe_token(provenance.dataset_id),
            _safe_token(provenance.scan_id),
            axis,
            f"true-{_safe_token(provenance.true_label)}",
            f"pred-{_safe_token(provenance.predicted_class)}",
            f"target-{_safe_token(provenance.target_class)}",
            correctness,
            METHOD_NAME,
            "clip-lr",
            provenance.analysis_id[:12],
        ]
    )


def create_brain_mask(image: Image.Image) -> tuple[np.ndarray, list[str]]:
    gray = np.asarray(image.convert("L").resize((224, 224)), dtype=np.float32) / 255.0
    warnings: list[str] = []
    if not np.isfinite(gray).all():
        return np.zeros_like(gray, dtype=bool), ["FAIL_MASK: nonfinite image values"]
    mask = gray > MASK_THRESHOLD
    fraction = float(mask.mean())
    if fraction < MINIMUM_BRAIN_FRACTION or fraction > MAXIMUM_BRAIN_FRACTION:
        warnings.append(f"WARN_MASK_FRACTION: brain_fraction={fraction:.4f}")
    return mask, warnings


def validate_heatmap(image: Image.Image, heatmap: np.ndarray) -> HeatmapValidation:
    values = np.asarray(heatmap, dtype=np.float32)
    if values.shape != (224, 224):
        values = np.asarray(Image.fromarray(values).resize((224, 224)), dtype=np.float32)

    finite = bool(np.isfinite(values).all())
    if not finite:
        return HeatmapValidation(
            mask_method=MASK_METHOD,
            mask_threshold=MASK_THRESHOLD,
            brain_fraction=0.0,
            background_fraction=1.0,
            foreground_mean=0.0,
            background_mean=0.0,
            foreground_background_ratio=0.0,
            background_activation_fraction=0.0,
            top_percent=TOP_ACTIVATION_PERCENT,
            top_activation_background_fraction=1.0,
            max_activation_xy=(0, 0),
            max_activation_inside_brain=False,
            finite=False,
            nonconstant=False,
            warnings=("FAIL_NONFINITE: heatmap contains NaN or Inf",),
            qc_status="FAIL_NONFINITE",
        )

    normalized = normalize_heatmap(values)
    nonconstant = bool(float(normalized.std()) > 1e-8)
    if not nonconstant:
        return HeatmapValidation(
            mask_method=MASK_METHOD,
            mask_threshold=MASK_THRESHOLD,
            brain_fraction=0.0,
            background_fraction=1.0,
            foreground_mean=0.0,
            background_mean=0.0,
            foreground_background_ratio=0.0,
            background_activation_fraction=0.0,
            top_percent=TOP_ACTIVATION_PERCENT,
            top_activation_background_fraction=1.0,
            max_activation_xy=(0, 0),
            max_activation_inside_brain=False,
            finite=True,
            nonconstant=False,
            warnings=("FAIL_CONSTANT_HEATMAP: normalized heatmap is constant",),
            qc_status="FAIL_CONSTANT_HEATMAP",
        )

    mask, warnings = create_brain_mask(image)
    brain_fraction = float(mask.mean())
    background = ~mask
    if brain_fraction <= 0.0 or brain_fraction >= 1.0:
        warnings.append("FAIL_MASK: empty foreground or background")
        qc_status = "FAIL_MASK"
    else:
        qc_status = "PASS_BASIC_SPATIAL_QC"

    foreground_mean = float(normalized[mask].mean()) if mask.any() else 0.0
    background_mean = float(normalized[background].mean()) if background.any() else 0.0
    ratio = float(foreground_mean / (background_mean + 1e-8))
    background_activation_fraction = float(normalized[background].sum() / (normalized.sum() + 1e-8))
    flat = normalized.reshape(-1)
    threshold = np.percentile(flat, 100.0 - TOP_ACTIVATION_PERCENT)
    top_mask = normalized >= threshold
    top_background_fraction = float(background[top_mask].mean()) if top_mask.any() else 1.0
    max_flat = int(flat.argmax())
    max_y, max_x = divmod(max_flat, normalized.shape[1])
    max_inside = bool(mask[max_y, max_x]) if mask.shape == normalized.shape else False

    if qc_status == "PASS_BASIC_SPATIAL_QC" and not max_inside:
        qc_status = "FAIL_MAX_OUTSIDE_BRAIN"
        warnings.append("FAIL_MAX_OUTSIDE_BRAIN: max activation is outside the brain mask")
    if qc_status == "PASS_BASIC_SPATIAL_QC" and top_background_fraction > 0.50:
        qc_status = "WARN_BACKGROUND_ACTIVATION"
        warnings.append("WARN_BACKGROUND_ACTIVATION: top activation is background-heavy")
    if qc_status == "PASS_BASIC_SPATIAL_QC" and foreground_mean <= background_mean:
        qc_status = "WARN_BACKGROUND_ACTIVATION"
        warnings.append("WARN_BACKGROUND_ACTIVATION: foreground mean is not above background mean")
    if qc_status == "PASS_BASIC_SPATIAL_QC" and ratio < BORDERLINE_FOREGROUND_BACKGROUND_RATIO:
        qc_status = "PASS_WITH_WARNING"
        warnings.append(
            "PASS_WITH_WARNING: foreground/background ratio is borderline "
            f"({ratio:.4f} < {BORDERLINE_FOREGROUND_BACKGROUND_RATIO:.4f})"
        )

    return HeatmapValidation(
        mask_method=MASK_METHOD,
        mask_threshold=MASK_THRESHOLD,
        brain_fraction=brain_fraction,
        background_fraction=1.0 - brain_fraction,
        foreground_mean=foreground_mean,
        background_mean=background_mean,
        foreground_background_ratio=ratio,
        background_activation_fraction=background_activation_fraction,
        top_percent=TOP_ACTIVATION_PERCENT,
        top_activation_background_fraction=top_background_fraction,
        max_activation_xy=(int(max_x), int(max_y)),
        max_activation_inside_brain=max_inside,
        finite=finite,
        nonconstant=nonconstant,
        warnings=tuple(warnings),
        qc_status=qc_status,
    )


def _prediction_status(true_label: str | None, predicted_class: str) -> tuple[str, bool | None]:
    if not true_label:
        return "unknown", None
    correct = true_label == predicted_class
    return ("correct" if correct else "incorrect"), correct


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, newline="") as handle:
        handle.write(text)
        temp_name = handle.name
    os.replace(temp_name, path)


def _append_index(index_path: Path, row: dict[str, str]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if index_path.exists():
        with index_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    for existing in rows:
        if existing.get("analysis_id") == row["analysis_id"]:
            if existing.get("source_sha256") != row["source_sha256"] or existing.get("checkpoint_sha256") != row["checkpoint_sha256"]:
                raise ValueError(f"analysis_id collision with different provenance: {row['analysis_id']}")
            return
        existing_paths = [
            existing.get("original_path"),
            existing.get("raw_heatmap_path"),
            existing.get("heatmap_png_path"),
            existing.get("overlay_path"),
            existing.get("metadata_path"),
        ]
        new_paths = [row["original_path"], row["raw_heatmap_path"], row["heatmap_png_path"], row["overlay_path"], row["metadata_path"]]
        if set(existing_paths) & set(new_paths):
            raise ValueError(f"artifact path collision for analysis_id: {row['analysis_id']}")
    rows.append(row)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=index_path.parent, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        temp_name = handle.name
    os.replace(temp_name, index_path)


def save_xai_artifact(
    *,
    artifacts_root: Path,
    repo_root: Path,
    source_path: Path,
    image: Image.Image,
    heatmap: np.ndarray,
    xai_result,
    slice_probability_row: dict[str, float],
    subject_probabilities: dict[str, float] | None,
    class_names: list[str],
    target_class_index: int,
    bundle,
    run_id: str | None = None,
) -> XAIArtifact:
    run_id = run_id or "interactive"
    identity = identity_from_path(source_path, repo_root)
    manifest_row = load_sample_manifest(repo_root).get(source_path.name, {})
    true_label = manifest_row.get("true_class") or None
    predicted_class = str(xai_result.predicted_class)
    prediction_status, correct = _prediction_status(true_label, predicted_class)
    checkpoint_sha256 = _checkpoint_digest(bundle)
    code_commit = _code_commit(repo_root)
    analysis_id = analysis_id_for(
        identity,
        MODEL_ID,
        checkpoint_sha256,
        PREPROCESSING_VERSION,
        str(xai_result.target_class),
        TARGET_SCORE_TYPE,
        run_id,
        code_commit,
    )
    provenance = XAIProvenance(
        analysis_id=analysis_id,
        dataset_id=identity.dataset_id,
        subject_id=identity.subject_id,
        session_or_visit_id=identity.session_or_visit_id,
        scan_id=identity.scan_id,
        series_id=identity.series_id,
        slice_axis=identity.slice_axis,
        slice_index=identity.slice_index if identity.slice_index is not None else -1,
        slice_filename=source_path.name,
        source_relative_path=identity.source_relative_path,
        source_sha256=identity.source_sha256,
        true_label=true_label,
        predicted_class=predicted_class,
        prediction_correct=correct,
        target_class=str(xai_result.target_class),
        target_class_index=int(target_class_index),
        target_score_type=TARGET_SCORE_TYPE,
        method_name=METHOD_NAME,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        checkpoint_sha256=checkpoint_sha256,
        code_commit=code_commit,
        preprocessing_version=PREPROCESSING_VERSION,
        xai_config_hash=xai_config_hash(),
        run_id=run_id,
        class_names=tuple(class_names),
        slice_probabilities={key: float(value) for key, value in slice_probability_row.items()},
        subject_probabilities={key: float(value) for key, value in subject_probabilities.items()} if subject_probabilities else None,
    )
    validation = validate_heatmap(image, heatmap)
    stem = artifact_stem(provenance)
    out_dir = artifacts_root / "xai" / provenance.model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    original_path = out_dir / f"{stem}_original.png"
    raw_path = out_dir / f"{stem}_heatmap.npy"
    normalized_npy_path = out_dir / f"{stem}_heatmap_normalized.npy"
    heatmap_png_path = out_dir / f"{stem}_heatmap.png"
    overlay_path = out_dir / f"{stem}_overlay.png"
    metadata_path = out_dir / f"{stem}_metadata.json"

    resized = image.convert("RGB").resize((224, 224))
    raw_heatmap = np.asarray(heatmap, dtype=np.float32)
    normalized = normalize_heatmap(raw_heatmap)
    resized.save(original_path)
    np.save(raw_path, raw_heatmap)
    np.save(normalized_npy_path, normalized.astype(np.float32))
    heatmap_to_rgb(normalized).convert("RGB").save(heatmap_png_path)
    overlay_heatmap(image, normalized, alpha=float(XAI_CONFIG["overlay_alpha"])).save(overlay_path)
    provenance = replace(
        provenance,
        original_sha256=file_sha256(original_path),
        raw_heatmap_sha256=file_sha256(raw_path),
        normalized_heatmap_sha256=file_sha256(normalized_npy_path),
        heatmap_png_sha256=file_sha256(heatmap_png_path),
        overlay_sha256=file_sha256(overlay_path),
    )

    metadata = {
        "provenance": asdict(provenance),
        "validation": asdict(validation),
        "artifact_paths": {
            "original": str(original_path),
            "raw_heatmap": str(raw_path),
            "normalized_heatmap": str(normalized_npy_path),
            "heatmap_png": str(heatmap_png_path),
            "overlay": str(overlay_path),
        },
        "xai_config": XAI_CONFIG,
        "prediction_status": prediction_status,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _atomic_write_text(metadata_path, json.dumps(metadata, indent=2, ensure_ascii=False))

    index_row = {
        "analysis_id": provenance.analysis_id,
        "dataset_id": provenance.dataset_id,
        "subject_id": provenance.subject_id,
        "session_or_visit_id": provenance.session_or_visit_id or "",
        "scan_id": provenance.scan_id,
        "series_id": provenance.series_id or "",
        "input_level": identity.input_level,
        "slice_axis": identity.slice_axis or "",
        "slice_index": str(provenance.slice_index),
        "source_relative_path": provenance.source_relative_path,
        "source_sha256": provenance.source_sha256,
        "true_label": provenance.true_label or "",
        "predicted_class": provenance.predicted_class,
        "prediction_status": prediction_status,
        "target_class": provenance.target_class,
        "target_score_type": provenance.target_score_type,
        "model_id": provenance.model_id,
        "checkpoint_sha256": provenance.checkpoint_sha256,
        "preprocessing_version": provenance.preprocessing_version,
        "code_commit": provenance.code_commit,
        "xai_method": provenance.method_name,
        "xai_config_hash": provenance.xai_config_hash,
        "run_id": provenance.run_id,
        "original_path": str(original_path),
        "raw_heatmap_path": str(raw_path),
        "heatmap_png_path": str(heatmap_png_path),
        "overlay_path": str(overlay_path),
        "metadata_path": str(metadata_path),
        "xai_qc_status": validation.qc_status,
        "created_at": metadata["created_at"],
    }
    _append_index(artifacts_root / "xai_artifact_index.csv", index_row)
    return XAIArtifact(
        provenance=provenance,
        validation=validation,
        original_path=original_path,
        raw_heatmap_npy_path=raw_path,
        normalized_heatmap_npy_path=normalized_npy_path,
        normalized_heatmap_png_path=heatmap_png_path,
        overlay_path=overlay_path,
        metadata_json_path=metadata_path,
    )
