from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from graph_xai.cam_visuals import (
    CAM_NOT_LESION_CAPTION,
    render_cam_heatmap,
    render_grid_cam_overlay,
    render_mri_cam_overlay,
    render_region_masking_before_after,
    render_top_cam_regions_highlight,
)
from graph_xai.region_grid import split_into_regions
from graph_xai.visualization import overlay_grid_on_image


def _image(size=32):
    rng = np.random.default_rng(0)
    return (rng.random((size, size, 3)) * 255).astype(np.uint8)


def test_cam_heatmap_is_resized_to_requested_image_size():
    cam = np.random.default_rng(1).random((14, 14)).astype(np.float32)
    result = render_cam_heatmap(cam, image_size_hw=(64, 64))
    assert result["image"].size == (64, 64)  # PIL size is (W, H)
    assert result["interpolation"] == "bilinear"
    assert result["colormap"] == "jet"
    assert len(result["cam_value_range"]) == 2
    assert result["caption"] == CAM_NOT_LESION_CAPTION


def test_mri_cam_overlay_does_not_mutate_inputs():
    image = _image()
    image_copy = image.copy()
    cam = np.random.default_rng(2).random((14, 14)).astype(np.float32)
    cam_copy = cam.copy()
    result = render_mri_cam_overlay(image, cam, alpha=0.3)
    np.testing.assert_array_equal(image, image_copy)
    np.testing.assert_array_equal(cam, cam_copy)
    assert result["alpha"] == 0.3
    assert result["image"].size == (image.shape[1], image.shape[0])


def test_grid_cam_overlay_differs_from_plain_grid_overlay_and_plain_cam_overlay():
    image = _image()
    cam = np.random.default_rng(3).random((14, 14)).astype(np.float32)
    regions = split_into_regions(cam, image=image)

    grid_only = overlay_grid_on_image(Image.fromarray(image), regions)
    cam_only = render_mri_cam_overlay(image, cam)["image"]
    grid_and_cam = render_grid_cam_overlay(image, cam, regions)["image"]

    assert list(grid_and_cam.getdata()) != list(grid_only.getdata())
    assert list(grid_and_cam.getdata()) != list(cam_only.getdata())


def test_grid_cam_overlay_actually_draws_grid_lines():
    image = np.zeros((30, 30, 3), dtype=np.uint8)
    cam = np.zeros((30, 30), dtype=np.float32)
    regions = split_into_regions(cam, image=image)
    result = render_grid_cam_overlay(image, cam, regions, grid_color=(255, 255, 0))
    pixels = np.array(result["image"])
    # a boundary column between region 0 and region 1 should contain the grid color somewhere
    boundary_col = regions[0].col_end - 1
    assert any(tuple(pixels[y, boundary_col]) == (255, 255, 0) for y in range(pixels.shape[0]))


def test_top_cam_regions_highlight_selects_highest_cam_ratio_region():
    image = np.zeros((9, 9, 3), dtype=np.uint8)
    cam = np.zeros((9, 9), dtype=np.float32)
    cam[0:3, 0:3] = 5.0  # top_left region has all the CAM signal
    regions = split_into_regions(cam, image=image)
    result = render_top_cam_regions_highlight(image, regions, top_k=1)
    assert result["region_names"] == ["top_left"]
    assert result["caption"] == CAM_NOT_LESION_CAPTION


def test_region_masking_before_after_matches_apply_region_mask():
    from graph_xai.masking import apply_region_mask

    image = _image()
    cam = np.zeros((32, 32), dtype=np.float32)
    regions = split_into_regions(cam, image=image)
    region = regions[0]

    result = render_region_masking_before_after(image, region, method="zero")
    expected_after = apply_region_mask(
        image, row_start=region.row_start, row_end=region.row_end,
        col_start=region.col_start, col_end=region.col_end, method="zero",
    )
    np.testing.assert_array_equal(np.array(result["after"]), expected_after)
    np.testing.assert_array_equal(np.array(result["before"]), image)
    assert result["region_name"] == region.name
    assert result["masking_method"] == "zero"


def test_cam_alignment_bright_top_left_stays_top_left_after_resize():
    cam = np.zeros((2, 2), dtype=np.float32)
    cam[0, 0] = 1.0  # bright top-left cell only
    result = render_cam_heatmap(cam, image_size_hw=(20, 20), interpolation="nearest")
    array = np.array(result["image"].convert("L"))
    top_left_mean = array[0:10, 0:10].mean()
    bottom_right_mean = array[10:20, 10:20].mean()
    assert top_left_mean > bottom_right_mean


def test_cam_heatmap_rejects_nan_and_inf():
    cam = np.zeros((10, 10), dtype=np.float32)
    cam[0, 0] = np.nan
    with pytest.raises(ValueError):
        render_cam_heatmap(cam, image_size_hw=(10, 10))
