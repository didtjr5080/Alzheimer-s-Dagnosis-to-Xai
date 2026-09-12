from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "clip_xai_app"
sys.path.insert(0, str(APP_ROOT))


def test_ensemble_is_blocked_without_final_probabilities_or_rule():
    from src.model_registry import get_model_registry

    registry = {item.model_id: item for item in get_model_registry()}
    ensemble = registry["ensemble_reference"]
    assert ensemble.status == "BLOCKED_MODEL_CONTRACT"
    assert "no final ensemble probability" in (ensemble.reason or "")
