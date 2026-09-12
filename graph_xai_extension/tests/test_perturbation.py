from __future__ import annotations

import numpy as np
import pytest

from graph_xai.perturbation import run_region_perturbation
from graph_xai.region_grid import split_into_regions


class DictPredictor:
    """Mock predictor: probability of the original class is proportional to how much
    of the 'signal' region (rows 0:3, cols 0:3) is intact in the image."""

    def __init__(self, class_names):
        self.class_names = class_names

    def __call__(self, images: list[np.ndarray]) -> np.ndarray:
        out = []
        for image in images:
            signal = float(image[0:3, 0:3].mean())
            p_class0 = min(max(signal / 10.0, 0.0), 1.0)
            remainder = (1.0 - p_class0) / (len(self.class_names) - 1)
            row = [p_class0] + [remainder] * (len(self.class_names) - 1)
            out.append(row)
        return np.array(out, dtype=np.float32)


def test_probability_drop_uses_original_class_not_argmax():
    image = np.zeros((9, 9), dtype=np.float32)
    image[0:3, 0:3] = 10.0  # top_left carries all the "signal"
    cam = image.copy()
    class_names = ["AD", "CN"]
    predictor = DictPredictor(class_names)

    results = run_region_perturbation(
        image=image,
        cam=cam,
        predict_fn=predictor,
        class_names=class_names,
        original_class_idx=0,
        masking_method="zero",
        grid_size=3,
    )
    assert len(results) == 9
    by_name = {r.region_name: r for r in results}
    top_left = by_name["top_left"]
    assert top_left.original_predicted_class == "AD"
    assert top_left.original_class_probability == pytest.approx(1.0)
    assert top_left.masked_original_class_probability == pytest.approx(0.0, abs=1e-5)
    assert top_left.probability_drop == pytest.approx(1.0, abs=1e-5)
    assert top_left.absolute_probability_change == pytest.approx(1.0, abs=1e-5)

    other = by_name["bottom_right"]
    assert other.probability_drop == pytest.approx(0.0, abs=1e-5)


def test_probability_drop_formula():
    image = np.full((9, 9), 5.0, dtype=np.float32)
    cam = image.copy()
    class_names = ["AD", "CN"]
    predictor = DictPredictor(class_names)
    results = run_region_perturbation(
        image=image,
        cam=cam,
        predict_fn=predictor,
        class_names=class_names,
        original_class_idx=0,
        masking_method="mean",
    )
    for r in results:
        assert r.probability_drop == pytest.approx(
            r.original_class_probability - r.masked_original_class_probability, abs=1e-6
        )
        assert r.absolute_probability_change == pytest.approx(abs(r.probability_drop), abs=1e-6)


def test_uses_provided_original_class_idx_not_argmax_of_masked():
    image = np.zeros((9, 9), dtype=np.float32)
    cam = image.copy()
    class_names = ["AD", "CN", "MCI"]
    predictor = DictPredictor(class_names)
    results = run_region_perturbation(
        image=image, cam=cam, predict_fn=predictor, class_names=class_names,
        original_class_idx=1, masking_method="zero",
    )
    for r in results:
        assert r.original_predicted_class == "CN"


def test_region_results_carry_cam_stats():
    image = np.random.rand(12, 12).astype(np.float32)
    cam = np.random.rand(12, 12).astype(np.float32)
    class_names = ["AD", "CN"]
    predictor = DictPredictor(class_names)
    results = run_region_perturbation(
        image=image, cam=cam, predict_fn=predictor, class_names=class_names,
        original_class_idx=0,
    )
    regions = split_into_regions(cam, image=image)
    region_by_name = {r.name: r for r in regions}
    for result in results:
        region = region_by_name[result.region_name]
        assert result.cam_mean == pytest.approx(region.cam_mean)
        assert result.cam_ratio == pytest.approx(region.cam_ratio)


def test_invalid_class_idx_raises():
    image = np.zeros((9, 9), dtype=np.float32)
    class_names = ["AD", "CN"]
    predictor = DictPredictor(class_names)
    with pytest.raises((ValueError, IndexError)):
        run_region_perturbation(
            image=image, cam=image, predict_fn=predictor, class_names=class_names,
            original_class_idx=5,
        )
