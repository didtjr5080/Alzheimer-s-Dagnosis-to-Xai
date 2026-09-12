from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "clip_xai_app"
sys.path.insert(0, str(APP_ROOT))


def _brain_image() -> Image.Image:
    arr = np.zeros((224, 224), dtype=np.uint8)
    arr[56:168, 56:168] = 180
    return Image.fromarray(arr, mode="L").convert("RGB")


def test_foreground_heatmap_passes_basic_spatial_qc():
    from src.xai_artifacts import validate_heatmap

    heatmap = np.zeros((224, 224), dtype=np.float32)
    heatmap[80:144, 80:144] = 1.0
    result = validate_heatmap(_brain_image(), heatmap)

    assert result.qc_status == "PASS_BASIC_SPATIAL_QC"
    assert result.finite
    assert result.nonconstant
    assert result.max_activation_inside_brain
    assert result.foreground_mean > result.background_mean


def test_background_max_fails_spatial_qc():
    from src.xai_artifacts import validate_heatmap

    heatmap = np.zeros((224, 224), dtype=np.float32)
    heatmap[5:20, 5:20] = 1.0
    result = validate_heatmap(_brain_image(), heatmap)

    assert result.qc_status == "FAIL_MAX_OUTSIDE_BRAIN"
    assert not result.max_activation_inside_brain


def test_borderline_foreground_background_ratio_is_warning_not_pass():
    from src.xai_artifacts import validate_heatmap

    heatmap = np.full((224, 224), 0.80, dtype=np.float32)
    heatmap[56:168, 56:168] = 0.86
    heatmap[100, 100] = 1.0
    heatmap[0, 0] = 0.0
    result = validate_heatmap(_brain_image(), heatmap)

    assert result.max_activation_inside_brain
    assert 1.0 < result.foreground_background_ratio < 1.25
    assert result.qc_status == "PASS_WITH_WARNING"


def test_constant_and_nonfinite_heatmaps_fail():
    from src.xai_artifacts import validate_heatmap

    constant = validate_heatmap(_brain_image(), np.ones((224, 224), dtype=np.float32))
    nonfinite = np.zeros((224, 224), dtype=np.float32)
    nonfinite[0, 0] = np.nan
    nonfinite_result = validate_heatmap(_brain_image(), nonfinite)

    assert constant.qc_status == "FAIL_CONSTANT_HEATMAP"
    assert nonfinite_result.qc_status == "FAIL_NONFINITE"
