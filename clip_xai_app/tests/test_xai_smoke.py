from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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


def test_clip_gradient_xai_smoke():
    from src.inference import analyze_subject
    from src.xai import generate_representative_xai

    image = Image.open(SAMPLE).convert("RGB")
    bundle, _, subject_prediction, _ = analyze_subject([image], device="cpu")
    xai = generate_representative_xai([image], [0], subject_prediction.predicted_class_idx, bundle)[0]
    heatmap = np.asarray(xai.heatmap_224)

    assert heatmap.shape == (224, 224)
    assert np.isfinite(heatmap).all()
    assert 0.0 <= float(heatmap.min()) <= float(heatmap.max()) <= 1.0
    assert float(heatmap.std()) > 0.0
    assert xai.target_class == "CN"
