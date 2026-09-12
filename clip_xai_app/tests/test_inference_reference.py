from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "clip_xai_app"
SAMPLE = (
    REPO_ROOT
    / "clip_lr_grad_eclip_handoff_v1_20260812_105027"
    / "data"
    / "xai_samples"
    / "images"
    / "OAS30009_MR_d2457_cor094.png"
)
sys.path.insert(0, str(APP_ROOT))


def test_reference_slice_prediction_matches_documented_sample():
    from src.inference import analyze_subject

    image = Image.open(SAMPLE).convert("RGB")
    _, slice_predictions, subject_prediction, _ = analyze_subject([image], device="cpu")

    expected = {"CN": 0.9180, "MCI": 0.0619, "AD": 0.0200}
    actual = dict(zip(slice_predictions.class_names, slice_predictions.probs[0]))
    for class_name, expected_value in expected.items():
        assert abs(float(actual[class_name]) - expected_value) < 0.005
    assert subject_prediction.predicted_class == "CN"
    assert subject_prediction.single_slice_warning
