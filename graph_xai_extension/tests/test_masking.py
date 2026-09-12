from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from graph_xai.masking import apply_region_mask


def _bbox():
    return dict(row_start=0, row_end=3, col_start=0, col_end=3)


@pytest.mark.parametrize("method", ["zero", "mean", "blur"])
def test_masking_methods_run_on_grayscale_numpy(method):
    image = np.random.rand(9, 9).astype(np.float32)
    original = image.copy()
    masked = apply_region_mask(image, **_bbox(), method=method)
    assert masked.shape == image.shape
    np.testing.assert_array_equal(image, original), "original must not be mutated"


@pytest.mark.parametrize("method", ["zero", "mean", "blur"])
def test_masking_methods_run_on_rgb_numpy(method):
    image = (np.random.rand(9, 9, 3) * 255).astype(np.uint8)
    original = image.copy()
    masked = apply_region_mask(image, **_bbox(), method=method)
    assert masked.shape == image.shape
    assert masked.dtype == image.dtype
    np.testing.assert_array_equal(image, original)


def test_zero_masking_sets_region_to_zero():
    image = np.full((9, 9), 200, dtype=np.uint8)
    masked = apply_region_mask(image, **_bbox(), method="zero")
    assert (masked[0:3, 0:3] == 0).all()
    assert (masked[3:, 3:] == 200).all()


def test_mean_masking_sets_region_to_image_mean():
    image = np.zeros((9, 9), dtype=np.float32)
    image[6:9, 6:9] = 90.0  # only bottom-right nonzero, mean over whole image
    expected_mean = float(image.mean())
    masked = apply_region_mask(image, **_bbox(), method="mean")
    assert np.allclose(masked[0:3, 0:3], expected_mean)
    assert masked[6, 6] == pytest.approx(90.0)


def test_blur_masking_changes_region_but_not_outside():
    rng = np.random.default_rng(0)
    image = rng.random((12, 12)).astype(np.float32)
    masked = apply_region_mask(image, row_start=3, row_end=9, col_start=3, col_end=9, method="blur")
    outside_before = image.copy()
    outside_before[3:9, 3:9] = 0
    outside_after = masked.copy()
    outside_after[3:9, 3:9] = 0
    np.testing.assert_array_equal(outside_before, outside_after)
    assert not np.array_equal(masked[3:9, 3:9], image[3:9, 3:9])


def test_default_method_is_mean():
    image = np.random.rand(9, 9).astype(np.float32)
    masked_default = apply_region_mask(image, **_bbox())
    masked_mean = apply_region_mask(image, **_bbox(), method="mean")
    np.testing.assert_array_almost_equal(masked_default, masked_mean)


def test_pil_image_input_and_immutability():
    image = Image.fromarray((np.random.rand(9, 9, 3) * 255).astype(np.uint8), mode="RGB")
    fingerprint = np.array(image).copy()
    masked = apply_region_mask(image, **_bbox(), method="zero")
    assert isinstance(masked, Image.Image)
    np.testing.assert_array_equal(np.array(image), fingerprint)
    assert (np.array(masked)[0:3, 0:3] == 0).all()


def test_torch_tensor_chw_input_and_immutability():
    tensor = torch.rand(3, 9, 9)
    fingerprint = tensor.clone()
    masked = apply_region_mask(tensor, **_bbox(), method="zero")
    assert isinstance(masked, torch.Tensor)
    assert masked.shape == tensor.shape
    torch.testing.assert_close(tensor, fingerprint)
    assert torch.all(masked[:, 0:3, 0:3] == 0)


def test_invalid_bbox_raises():
    image = np.random.rand(9, 9).astype(np.float32)
    with pytest.raises(ValueError):
        apply_region_mask(image, row_start=5, row_end=2, col_start=0, col_end=3, method="zero")
    with pytest.raises(ValueError):
        apply_region_mask(image, row_start=0, row_end=3, col_start=0, col_end=100, method="zero")


def test_unknown_method_raises():
    image = np.random.rand(9, 9).astype(np.float32)
    with pytest.raises(ValueError):
        apply_region_mask(image, **_bbox(), method="not_a_real_method")


def test_masking_records_method_and_params():
    from graph_xai.masking import MaskingRecord

    image = np.random.rand(9, 9).astype(np.float32)
    _masked, record = apply_region_mask(image, **_bbox(), method="blur", blur_sigma=2.0, return_record=True)
    assert isinstance(record, MaskingRecord)
    assert record.method == "blur"
    assert record.params.get("blur_sigma") == 2.0
