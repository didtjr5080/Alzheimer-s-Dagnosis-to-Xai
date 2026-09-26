"""zero/mean/blur masking, generic over N-dimensional numpy arrays (2D
grayscale/RGB images or 3D volumes). A bbox is a tuple of (start, end)
index pairs, one per spatial axis, applied via `array[slices]`. Blurring is
applied to the WHOLE array first (never mixing an unblurred image with a
locally-blurred patch, which would create a hard discontinuity at the
region border), then the blurred *region* is spliced into a copy of the
original -- everywhere outside the region is untouched, byte-for-byte."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

MaskingMethod = str  # "zero" | "mean" | "blur"


def _bbox_to_slices(bbox: tuple[tuple[int, int], ...]) -> tuple[slice, ...]:
    return tuple(slice(start, end) for start, end in bbox)


def apply_region_mask(
    array: np.ndarray, bbox: tuple[tuple[int, int], ...], method: MaskingMethod = "mean",
    blur_sigma: float = 4.0,
) -> np.ndarray:
    """Returns a NEW array (never mutates `array`) with the region at `bbox`
    masked. `array` may be (H, W), (H, W, C), or (D, H, W) -- `bbox` must
    have one (start, end) pair per leading spatial axis; trailing channel
    axes (if any) are masked in full for each masked spatial location."""
    if method not in ("zero", "mean", "blur"):
        raise ValueError(f"unknown masking method: {method!r}")

    out = array.copy()
    region_slices = _bbox_to_slices(bbox)

    if method == "zero":
        out[region_slices] = 0
    elif method == "mean":
        mean_value = array.mean()
        out[region_slices] = mean_value
    else:  # blur
        # Trailing axes beyond `bbox` (e.g. an RGB channel axis) are never
        # blurred -- sigma=0 on those axes leaves them untouched.
        n_spatial = len(bbox)
        sigma_per_axis = tuple([blur_sigma] * n_spatial + [0] * (array.ndim - n_spatial))
        blurred_whole = gaussian_filter(array.astype(np.float64), sigma=sigma_per_axis)
        out = out.astype(array.dtype, copy=True)
        out[region_slices] = blurred_whole[region_slices].astype(array.dtype)

    return out
