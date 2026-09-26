from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "clip_xai_app"
sys.path.insert(0, str(APP_ROOT))


def test_ensemble_is_available_with_the_recovered_fixed_weight_rule():
    """Superseded by `통합(OASIS-3+ADNI) 모델 라이브 추론 전환 작업지시서.md`: the
    combining rule was not "unrecoverable" -- it is a fixed weighted sum
    (GBM 0.35 + 3D CNN 0.45 + CLIP 0.20) using the *baseline* GBM branch
    (not the hierarchical one), which reproduces `final_ensemble_results.csv`
    exactly (validated 573/573 predicted-class matches in this session, see
    WorkOrder/통합모델_라이브추론_보고서_*.md). The registry entry moved from
    `ensemble_reference` (BLOCKED) to `merged_ensemble` (AVAILABLE)."""
    from src.model_registry import get_model_registry

    registry = {item.model_id: item for item in get_model_registry()}
    ensemble = registry["merged_ensemble"]
    assert ensemble.status == "AVAILABLE"
    assert "0.35" in ensemble.reason and "0.45" in ensemble.reason and "0.20" in ensemble.reason


def test_ensemble_combine_reproduces_final_ensemble_results_weights():
    from src.adapters.merged_ensemble import ENSEMBLE_WEIGHTS, combine_ensemble

    assert ENSEMBLE_WEIGHTS == {"gbm": 0.35, "cnn": 0.45, "clip": 0.20}
    result = combine_ensemble(
        gbm_probs={"CN": 1.0, "MCI": 0.0, "AD": 0.0},
        cnn_probs={"CN": 0.0, "MCI": 1.0, "AD": 0.0},
        clip_probs={"CN": 0.0, "MCI": 0.0, "AD": 1.0},
    )
    assert result == {"CN": 0.35, "MCI": 0.45, "AD": 0.20}
