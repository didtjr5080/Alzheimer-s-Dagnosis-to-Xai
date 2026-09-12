from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "clip_xai_app"
HANDOFF_ROOT = REPO_ROOT / "clip_lr_grad_eclip_handoff_v1_20260812_105027"
sys.path.insert(0, str(APP_ROOT))


def test_legacy_clip_lr_model_contract():
    from src.model_loader import load_pipeline

    bundle = load_pipeline(device="cpu")
    vision = bundle.predictor.clip_model.config.vision_config
    clf = bundle.predictor.clf

    assert bundle.root == HANDOFF_ROOT
    assert bundle.class_names == ["CN", "MCI", "AD"]
    assert vision.image_size == 224
    assert vision.patch_size == 16
    assert vision.image_size // vision.patch_size == 14
    assert bundle.predictor.clip_model.config.projection_dim == 512
    assert clf.coef_.shape == (3, 512)
    assert clf.intercept_.shape == (3,)
    assert clf.classes_.tolist() == [0, 1, 2]
