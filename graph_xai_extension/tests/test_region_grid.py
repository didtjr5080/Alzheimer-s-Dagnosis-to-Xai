from __future__ import annotations

import numpy as np
import pytest

from graph_xai.region_grid import split_into_regions
from graph_xai.schemas import REGION_NAMES_3X3


def _region_mask_grid(regions, shape):
    """Rebuild a label grid marking which region owns each pixel."""
    owner = np.full(shape, -1, dtype=int)
    for i, region in enumerate(regions):
        owner[region.row_start:region.row_end, region.col_start:region.col_end] = i
    return owner


@pytest.mark.parametrize("shape", [(30, 30), (33, 33), (28, 31), (10, 10), (100, 97)])
def test_every_pixel_covered_exactly_once(shape):
    cam = np.random.rand(*shape).astype(np.float32)
    regions = split_into_regions(cam)
    assert len(regions) == 9
    owner = _region_mask_grid(regions, shape)
    assert (owner >= 0).all(), "some pixels not assigned to any region"
    total_area = sum(r.area_pixels for r in regions)
    assert total_area == shape[0] * shape[1]


def test_region_names_match_expected_layout():
    cam = np.zeros((9, 9), dtype=np.float32)
    regions = split_into_regions(cam)
    names = [r.name for r in regions]
    assert names == list(REGION_NAMES_3X3)
    for region in regions:
        assert 0 <= region.row <= 2
        assert 0 <= region.col <= 2


def test_cam_stats_are_correct_for_uniform_regions():
    cam = np.zeros((9, 9), dtype=np.float32)
    cam[0:3, 0:3] = 2.0  # top_left region only
    regions = split_into_regions(cam)
    top_left = regions[0]
    assert top_left.name == "top_left"
    assert top_left.cam_mean == pytest.approx(2.0)
    assert top_left.cam_max == pytest.approx(2.0)
    assert top_left.cam_sum == pytest.approx(18.0)
    total = float(cam.sum())
    assert top_left.cam_ratio == pytest.approx(18.0 / total)


def test_handles_zero_sum_cam_without_division_error():
    cam = np.zeros((9, 9), dtype=np.float32)
    regions = split_into_regions(cam)
    assert len(regions) == 9
    for region in regions:
        assert region.cam_ratio == 0.0
        assert np.isfinite(region.cam_mean)


def test_handles_constant_cam():
    cam = np.full((12, 12), 5.0, dtype=np.float32)
    regions = split_into_regions(cam)
    for region in regions:
        assert region.cam_mean == pytest.approx(5.0)


def test_rejects_nan_and_inf():
    cam = np.zeros((9, 9), dtype=np.float32)
    cam[0, 0] = np.nan
    with pytest.raises(ValueError):
        split_into_regions(cam)

    cam2 = np.zeros((9, 9), dtype=np.float32)
    cam2[0, 0] = np.inf
    with pytest.raises(ValueError):
        split_into_regions(cam2)


def test_rejects_empty_array():
    with pytest.raises(ValueError):
        split_into_regions(np.zeros((0, 0), dtype=np.float32))


def test_resizes_mismatched_cam_to_image_shape():
    cam = np.random.rand(14, 14).astype(np.float32)
    image = np.zeros((224, 224), dtype=np.float32)
    regions = split_into_regions(cam, image=image, interpolation="bilinear")
    assert sum(r.area_pixels for r in regions) == 224 * 224


def test_does_not_mutate_input_arrays():
    cam = np.random.rand(30, 30).astype(np.float32)
    image = np.random.rand(30, 30).astype(np.float32)
    cam_copy = cam.copy()
    image_copy = image.copy()
    split_into_regions(cam, image=image)
    np.testing.assert_array_equal(cam, cam_copy)
    np.testing.assert_array_equal(image, image_copy)


def test_image_mean_brightness_is_computed_per_region():
    cam = np.zeros((9, 9), dtype=np.float32)
    image = np.zeros((9, 9), dtype=np.float32)
    image[0:3, 0:3] = 10.0
    regions = split_into_regions(cam, image=image)
    assert regions[0].image_mean_brightness == pytest.approx(10.0)
    assert regions[1].image_mean_brightness == pytest.approx(0.0)


def test_negative_cam_values_do_not_crash():
    cam = np.full((9, 9), -3.0, dtype=np.float32)
    regions = split_into_regions(cam)
    assert len(regions) == 9
    assert regions[0].cam_mean == pytest.approx(-3.0)
