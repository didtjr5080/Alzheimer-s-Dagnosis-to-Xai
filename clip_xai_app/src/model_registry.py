from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .contracts import AdapterHealth


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_model_registry() -> list[AdapterHealth]:
    root = repo_root()
    handoff = root / "clip_lr_grad_eclip_handoff_v1_20260812_105027"
    merged = root / "merged_project"
    has_handoff = all(
        path.exists()
        for path in [
            handoff / "code" / "inference_clip_lr.py",
            handoff / "code" / "grad_eclip.py",
            handoff / "models" / "clip_model",
            handoff / "models" / "clip_processor",
            handoff / "models" / "clip_lr_classifier_new_run.joblib",
        ]
    )
    has_merged_csv = all(
        (merged / name).exists()
        for name in [
            "clip_test_probs.csv",
            "3dcnn_test_probs.csv",
            "gbm_hierarchical_test_probs.csv",
            "final_ensemble_results.csv",
        ]
    )
    return [
        AdapterHealth(
            model_id="legacy_clip_lr",
            status="AVAILABLE" if has_handoff else "UNAVAILABLE_MISSING_FILES",
            execution_mode="live_png_inference",
            reason=None if has_handoff else "Required CLIP+LR handoff files are missing.",
            available_inputs=["coronal_png_slice", "multi_slice_subject"],
        ),
        AdapterHealth(
            model_id="merged_clip_reference",
            status="AVAILABLE" if has_merged_csv else "UNAVAILABLE_MISSING_FILES",
            execution_mode="reference_csv_by_scan_id",
            reason="Dynamic checkpoint inference is blocked; reference CSV predictions are exposed for verified scan_ids.",
            available_inputs=["scan_id"],
        ),
        AdapterHealth(
            model_id="merged_clip_checkpoint",
            status="BLOCKED_MODEL_CONTRACT",
            execution_mode="blocked_live_inference",
            reason="clip_best.pt stores class_embeds only; backbone/preprocessing linkage is not present.",
            available_inputs=[],
        ),
        AdapterHealth(
            model_id="cnn3d_reference",
            status="AVAILABLE" if has_merged_csv else "UNAVAILABLE_MISSING_FILES",
            execution_mode="reference_csv_by_scan_id",
            reason="3D CNN live inference is blocked; verified CSV predictions are exposed for scan_ids.",
            available_inputs=["scan_id"],
        ),
        AdapterHealth(
            model_id="cnn3d_checkpoint",
            status="BLOCKED_MODEL_CONTRACT",
            execution_mode="blocked_live_inference",
            reason="Checkpoint state_dict is readable, but model class and preprocessing contract are absent.",
            available_inputs=[],
        ),
        AdapterHealth(
            model_id="gbm_reference",
            status="AVAILABLE" if has_merged_csv else "UNAVAILABLE_MISSING_FILES",
            execution_mode="reference_csv_by_scan_id",
            reason="Live GBM inference requires caller-supplied validated feature schema.",
            available_inputs=["scan_id"],
        ),
        AdapterHealth(
            model_id="ensemble_reference",
            status="BLOCKED_MODEL_CONTRACT" if has_merged_csv else "UNAVAILABLE_MISSING_FILES",
            execution_mode="reference_csv_by_scan_id",
            reason="final_ensemble_results.csv has final pred but no final ensemble probability columns; combining rule is not fully recoverable.",
            available_inputs=["scan_id"],
        ),
    ]


def registry_rows() -> list[dict]:
    return [asdict(item) for item in get_model_registry()]
