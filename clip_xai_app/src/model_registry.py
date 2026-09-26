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
    has_merged_clip_ckpt = (merged / "checkpoints" / "clip_best.pt").exists() and has_handoff
    has_merged_cnn3d_ckpt = (merged / "checkpoints" / "3dcnn_best.pt").exists()
    has_gbm_baseline = (merged / "models" / "gbm_baseline.txt").exists() and (
        merged / "unified_final_features_gbm.csv"
    ).exists()
    has_merged_live = has_merged_clip_ckpt and has_merged_cnn3d_ckpt and has_gbm_baseline

    # Test-set reproduction validated in this session against every criterion
    # in `통합(OASIS-3+ADNI) 모델 라이브 추론 전환 작업지시서.md` sections 1-5/2/3-4/4-1
    # (n=573; see WorkOrder/통합모델_라이브추론_보고서_*.md for full evidence).
    return [
        AdapterHealth(
            model_id="legacy_clip_lr",
            status="AVAILABLE" if has_handoff else "UNAVAILABLE_MISSING_FILES",
            execution_mode="live_png_inference",
            reason=None if has_handoff else "Required CLIP+LR handoff files are missing.",
            available_inputs=["coronal_png_slice", "multi_slice_subject"],
        ),
        AdapterHealth(
            model_id="merged_clip",
            status="AVAILABLE" if has_merged_clip_ckpt else "UNAVAILABLE_MISSING_FILES",
            execution_mode="live_inference_by_scan_id",
            reason=(
                "Live inference: frozen CLIP backbone + class_embeds cosine-similarity head "
                "(clip_best.pt). Test-set reproduction validated: Acc=60.9%, Macro-F1=0.576, n=573."
            ),
            available_inputs=["scan_id"],
        ),
        AdapterHealth(
            model_id="merged_cnn3d",
            status="AVAILABLE" if has_merged_cnn3d_ckpt else "UNAVAILABLE_MISSING_FILES",
            execution_mode="live_inference_by_scan_id",
            reason=(
                "Live inference: Simple3DCNN_Stable loaded from 3dcnn_best.pt (strict state_dict "
                "match). Test-set reproduction validated: Acc=64.9%, Macro-F1=0.622, n=573."
            ),
            available_inputs=["scan_id"],
        ),
        AdapterHealth(
            model_id="merged_gbm_baseline",
            status="AVAILABLE" if has_gbm_baseline else "UNAVAILABLE_MISSING_FILES",
            execution_mode="live_inference_by_scan_id",
            reason=(
                "Live inference: LightGBM Booster (gbm_baseline.txt) over unified_final_features_gbm.csv "
                "(is_adni derived from dataset_source). Test-set reproduction validated: "
                "Acc=63.2%, Macro-F1=0.604, n=573."
            ),
            available_inputs=["scan_id"],
        ),
        AdapterHealth(
            model_id="merged_ensemble",
            status="AVAILABLE" if has_merged_live else "UNAVAILABLE_MISSING_FILES",
            execution_mode="live_inference_by_scan_id",
            reason=(
                "Fixed-weight combination (GBM 0.35 + 3D CNN 0.45 + CLIP 0.20) of the three live "
                "branches above; a missing branch makes the ensemble UNAVAILABLE for that scan_id "
                "(no re-weighting of remaining branches). Test-set reproduction validated: "
                "Acc=65.6%, Macro-F1=0.634, n=573, 573/573 predicted-class matches vs "
                "final_ensemble_results.csv."
            ),
            available_inputs=["scan_id"],
        ),
        AdapterHealth(
            model_id="gbm_hierarchical_reference",
            status="AVAILABLE" if has_merged_csv else "UNAVAILABLE_MISSING_FILES",
            execution_mode="reference_csv_by_scan_id",
            reason=(
                "Not used by merged_ensemble (which uses gbm_baseline). Kept for reference lookup "
                "of the hierarchical GBM's own precomputed test predictions."
            ),
            available_inputs=["scan_id"],
        ),
    ]


def registry_rows() -> list[dict]:
    return [asdict(item) for item in get_model_registry()]
