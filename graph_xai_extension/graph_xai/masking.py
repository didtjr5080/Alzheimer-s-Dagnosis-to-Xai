"""Region masking: zero / mean / blur perturbations for grayscale/RGB numpy,
PIL, and torch (CHW) images. Never mutates the input; preserves dtype and
value range; does not re-apply any model preprocessing/normalization.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

from .schemas import MaskingRecord

__all__ = ["apply_region_mask", "MaskingRecord"]

_METHODS = ("zero", "mean", "blur")
_DEFAULT_BLUR_SIGMA = 3.0


def _validate_bbox(row_start: int, row_end: int, col_start: int, col_end: int, height: int, width: int) -> None:
    if row_start < 0 or col_start < 0:
        raise ValueError("bbox start indices must be non-negative")
    if row_end <= row_start or col_end <= col_start:
        raise ValueError("bbox end must be strictly greater than bbox start")
    if row_end > height or col_end > width:
        raise ValueError(f"bbox [{row_start}:{row_end}, {col_start}:{col_end}] exceeds image bounds {(height, width)}")


def _core_mask(arr: np.ndarray, row_start: int, row_end: int, col_start: int, col_end: int,
               method: str, params: dict) -> np.ndarray:
    arr = arr.astype(np.float64, copy=True)
    if method == "zero":
        arr[row_start:row_end, col_start:col_end, ...] = 0.0
    elif method == "mean":
        fill_value = float(arr.mean())
        arr[row_start:row_end, col_start:col_end, ...] = fill_value
    elif method == "blur":
        sigma = float(params.get("blur_sigma", _DEFAULT_BLUR_SIGMA))
        region = arr[row_start:row_end, col_start:col_end, ...]
        if region.ndim == 3:
            blurred = np.stack(
                [gaussian_filter(region[..., c], sigma=sigma) for c in range(region.shape[-1])], axis=-1
            )
        else:
            blurred = gaussian_filter(region, sigma=sigma)
        arr[row_start:row_end, col_start:col_end, ...] = blurred
    else:
        raise ValueError(f"Unknown masking method: {method!r}. Expected one of {_METHODS}")
    return arr


def _restore_dtype(arr_float: np.ndarray, original_dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        return np.clip(np.round(arr_float), info.min, info.max).astype(original_dtype)
    return arr_float.astype(original_dtype)


def apply_region_mask(
    image,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    method: str = "mean",
    return_record: bool = False,
    **params,
):
    if method not in _METHODS:
        raise ValueError(f"Unknown masking method: {method!r}. Expected one of {_METHODS}")

    if isinstance(image, Image.Image):
        arr = np.array(image)
        height, width = arr.shape[0], arr.shape[1]
        _validate_bbox(row_start, row_end, col_start, col_end, height, width)
        masked_float = _core_mask(arr, row_start, row_end, col_start, col_end, method, params)
        masked = _restore_dtype(masked_float, arr.dtype)
        result = Image.fromarray(masked, mode=image.mode)

    elif torch.is_tensor(image):
        arr = image.detach().cpu().numpy()
        if arr.ndim == 3:
            hwc = np.transpose(arr, (1, 2, 0))
        elif arr.ndim == 2:
            hwc = arr
        else:
            raise ValueError(f"Unsupported tensor ndim={arr.ndim}; expected (C,H,W) or (H,W)")
        height, width = hwc.shape[0], hwc.shape[1]
        _validate_bbox(row_start, row_end, col_start, col_end, height, width)
        masked_hwc_float = _core_mask(hwc, row_start, row_end, col_start, col_end, method, params)
        masked_hwc = _restore_dtype(masked_hwc_float, arr.dtype)
        masked_arr = np.transpose(masked_hwc, (2, 0, 1)) if arr.ndim == 3 else masked_hwc
        result = torch.from_numpy(np.ascontiguousarray(masked_arr)).to(dtype=image.dtype, device=image.device)

    else:
        arr = np.asarray(image)
        if arr.ndim not in (2, 3):
            raise ValueError(f"Unsupported image ndim={arr.ndim}; expected (H,W) or (H,W,C)")
        height, width = arr.shape[0], arr.shape[1]
        _validate_bbox(row_start, row_end, col_start, col_end, height, width)
        masked_float = _core_mask(arr, row_start, row_end, col_start, col_end, method, params)
        result = _restore_dtype(masked_float, arr.dtype)

    if return_record:
        return result, MaskingRecord(method=method, params=dict(params))
    return result
