from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "clip_xai_app"
sys.path.insert(0, str(APP_ROOT))


def test_merged_artifact_audit_contract():
    from src.merged_artifacts import audit_merged_project

    audit = audit_merged_project(REPO_ROOT)
    assert audit["file_count"] == 95425
    assert audit["extension_counts"][".png"] == 91593
    assert audit["extension_counts"][".npy"] == 3816
    assert audit["contains_adni"] is True
    assert audit["contains_oasis"] is True
    assert audit["csv_contracts"]["unified_slice_manifest.csv"]["rows"] == 91584
    assert audit["overlap_checks"]["slice_scans_missing_from_manifest"] == 0
    assert audit["dynamic_model_contracts"]["ensemble"].startswith("BLOCKED_MODEL_CONTRACT")


def test_reference_csv_adapters_return_common_prediction_contract():
    from src.adapters.merged_reference import build_default_reference_adapters

    adapters = build_default_reference_adapters(REPO_ROOT)
    sample = "OAS30564_MR_d0000"
    by_id = {adapter.model_id: adapter for adapter in adapters}

    assert set(by_id) == {"merged_clip_reference", "cnn3d_reference", "gbm_reference"}
    for adapter in adapters:
        assert adapter.health().status == "AVAILABLE"
        result = adapter.predict_by_scan_id(sample)
        result.validate(tolerance=1e-3)
        assert result.level == "subject"
        assert result.class_names == ["CN", "MCI", "AD"]
