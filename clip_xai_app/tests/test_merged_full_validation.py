"""Authoritative, full test-set (n=573) reproduction of every validation
criterion in `통합(OASIS-3+ADNI) 모델 라이브 추론 전환 작업지시서.md` sections
1-5, 2, 3-4, and 4-1. This is slow (~7-8 minutes, dominated by the CLIP
branch's 13,752-slice GPU forward pass) and is excluded from routine runs
by pytest.ini's `addopts = -m "not slow"` (same intent as
graph_xai_extension excluding its real-model batch test):

    pytest -q                    # excludes this file (default addopts)

Run explicitly to re-verify the work order's exact pass/fail numbers:

    pytest tests/test_merged_full_validation.py -v -m slow
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "clip_xai_app"
sys.path.insert(0, str(APP_ROOT))

MERGED_ROOT = REPO_ROOT / "merged_project"
_merged_available = (MERGED_ROOT / "unified_manifest.csv").exists()

pytestmark = [
    pytest.mark.skipif(not _merged_available, reason="merged_project/ not available locally"),
    pytest.mark.slow,
]

CLASS_NAMES = ["CN", "MCI", "AD"]


def _read_csv_dicts(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _macro_f1(y_true, y_pred, classes) -> float:
    f1s = []
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def _confusion_matrix(y_true, y_pred, classes) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm


@pytest.fixture(scope="module")
def unified_manifest() -> dict[str, dict]:
    return {r["scan_id"]: r for r in _read_csv_dicts(MERGED_ROOT / "unified_manifest.csv")}


@pytest.fixture(scope="module")
def test_scan_ids(unified_manifest) -> list[str]:
    ids = sorted(sid for sid, row in unified_manifest.items() if row["split"] == "test")
    assert len(ids) == 573
    return ids


def test_clip_branch_reproduces_test_set_exactly(unified_manifest, test_scan_ids):
    from src.adapters.merged_clip import load_merged_clip_bundle, predict_slice_batch

    bundle = load_merged_clip_bundle(REPO_ROOT, device=None)
    manifest = _read_csv_dicts(MERGED_ROOT / "unified_slice_manifest.csv")
    test_rows = [r for r in manifest if r["split"] == "test"]
    assert len(test_rows) == 13752

    by_scan = defaultdict(list)
    for r in test_rows:
        by_scan[r["scan_id"]].append(int(r["slice_index"]))

    slices_dir = MERGED_ROOT / "slices_multi"
    flat = [(scan_id, idx) for scan_id in test_scan_ids for idx in sorted(by_scan[scan_id])]

    slice_probs: dict[tuple[str, int], np.ndarray] = {}
    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(flat), batch_size):
            batch = flat[i:i + batch_size]
            images = [Image.open(slices_dir / f"{scan_id}_cor{idx:03d}.png") for scan_id, idx in batch]
            probs = predict_slice_batch(bundle, images).cpu().numpy()
            for (scan_id, idx), p in zip(batch, probs):
                slice_probs[(scan_id, idx)] = p

    subject_probs = {
        scan_id: np.mean([slice_probs[(scan_id, idx)] for idx in sorted(by_scan[scan_id])], axis=0)
        for scan_id in test_scan_ids
    }

    ref_by_scan = {r["scan_id"]: r for r in _read_csv_dicts(MERGED_ROOT / "clip_test_probs.csv")}
    max_abs_err = 0.0
    y_true, y_pred = [], []
    for scan_id in test_scan_ids:
        pred_p = subject_probs[scan_id]
        ref = ref_by_scan[scan_id]
        ref_p = np.array([float(ref["clip_CN"]), float(ref["clip_MCI"]), float(ref["clip_AD"])])
        max_abs_err = max(max_abs_err, float(np.abs(pred_p - ref_p).max()))
        y_true.append(ref["label"])
        y_pred.append(CLASS_NAMES[int(np.argmax(pred_p))])

    assert max_abs_err < 1e-3
    assert sum(1 for t, p in zip(y_true, y_pred) if t == p) >= 0  # sanity, real check below

    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    f1 = _macro_f1(y_true, y_pred, CLASS_NAMES)
    assert round(acc, 3) == pytest.approx(0.609, abs=0.001)
    assert round(f1, 3) == pytest.approx(0.576, abs=0.001)

    cm = _confusion_matrix(y_true, y_pred, CLASS_NAMES)
    assert cm.tolist() == [[212, 47, 7], [86, 81, 25], [18, 41, 56]]


def test_gbm_baseline_branch_reproduces_test_set_exactly(unified_manifest, test_scan_ids):
    from src.adapters.merged_gbm import load_merged_gbm_bundle

    bundle = load_merged_gbm_bundle(REPO_ROOT)
    final_ref = {r["scan_id"]: r for r in _read_csv_dicts(MERGED_ROOT / "final_ensemble_results.csv")}

    y_true, y_pred = [], []
    max_ref_err = 0.0
    for scan_id in test_scan_ids:
        probs = bundle.predict_by_scan_id(scan_id)
        assert probs is not None
        pred_class = max(CLASS_NAMES, key=lambda c: probs[c])
        y_true.append(unified_manifest[scan_id]["label"])
        y_pred.append(pred_class)
        ref = final_ref[scan_id]
        max_ref_err = max(max_ref_err, max(abs(probs[c] - float(ref[f"gbm_{c}"])) for c in CLASS_NAMES))

    assert max_ref_err < 1e-6

    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    f1 = _macro_f1(y_true, y_pred, CLASS_NAMES)
    assert round(acc, 3) == pytest.approx(0.632, abs=0.001)
    assert round(f1, 3) == pytest.approx(0.604, abs=0.001)

    cm = _confusion_matrix(y_true, y_pred, CLASS_NAMES)
    assert cm.tolist() == [[207, 43, 16], [65, 80, 47], [10, 30, 75]]


def test_cnn3d_branch_reproduces_test_set_exactly(unified_manifest, test_scan_ids):
    from src.adapters.merged_cnn3d import load_merged_cnn3d_bundle

    bundle = load_merged_cnn3d_bundle(REPO_ROOT, device=None)
    ref_rows = {r["scan_id"]: r for r in _read_csv_dicts(MERGED_ROOT / "3dcnn_test_probs.csv")}

    y_true, y_pred = [], []
    max_abs_err = 0.0
    for scan_id in test_scan_ids:
        probs = bundle.predict_by_scan_id(scan_id)
        assert probs is not None
        pred_class = max(CLASS_NAMES, key=lambda c: probs[c])
        y_true.append(unified_manifest[scan_id]["label"])
        y_pred.append(pred_class)
        ref = ref_rows[scan_id]
        ref_probs = {"CN": float(ref["prob_CN"]), "MCI": float(ref["prob_MCI"]), "AD": float(ref["prob_AD"])}
        max_abs_err = max(max_abs_err, max(abs(probs[c] - ref_probs[c]) for c in CLASS_NAMES))

    assert max_abs_err < 1e-4

    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    f1 = _macro_f1(y_true, y_pred, CLASS_NAMES)
    assert round(acc, 3) == pytest.approx(0.649, abs=0.001)
    assert round(f1, 3) == pytest.approx(0.622, abs=0.001)

    cm = _confusion_matrix(y_true, y_pred, CLASS_NAMES)
    assert cm.tolist() == [[224, 39, 3], [86, 72, 34], [6, 33, 76]]


def test_ensemble_reproduces_final_ensemble_results_exactly(unified_manifest, test_scan_ids):
    from src.adapters.merged_ensemble import combine_ensemble, ensemble_predicted_class

    final_ref = {r["scan_id"]: r for r in _read_csv_dicts(MERGED_ROOT / "final_ensemble_results.csv")}
    clip_ref = {r["scan_id"]: r for r in _read_csv_dicts(MERGED_ROOT / "clip_test_probs.csv")}
    cnn_ref = {r["scan_id"]: r for r in _read_csv_dicts(MERGED_ROOT / "3dcnn_test_probs.csv")}

    pred_matches = 0
    y_true, y_pred = [], []
    for scan_id in test_scan_ids:
        clip_p = {c: float(clip_ref[scan_id][f"clip_{c}"]) for c in CLASS_NAMES}
        cnn_p = {c: float(cnn_ref[scan_id][f"prob_{c}"]) for c in CLASS_NAMES}
        gbm_p = {c: float(final_ref[scan_id][f"gbm_{c}"]) for c in CLASS_NAMES}

        ensemble = combine_ensemble(gbm_p, cnn_p, clip_p)
        pred_class = ensemble_predicted_class(ensemble)
        if pred_class == CLASS_NAMES[int(final_ref[scan_id]["pred"])]:
            pred_matches += 1

        y_true.append(unified_manifest[scan_id]["label"])
        y_pred.append(pred_class)

    assert pred_matches == 573

    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    f1 = _macro_f1(y_true, y_pred, CLASS_NAMES)
    assert round(acc, 3) == pytest.approx(0.656, abs=0.001)
    assert round(f1, 3) == pytest.approx(0.634, abs=0.001)

    cm = _confusion_matrix(y_true, y_pred, CLASS_NAMES)
    assert cm.tolist() == [[219, 43, 4], [79, 77, 36], [9, 26, 80]]
