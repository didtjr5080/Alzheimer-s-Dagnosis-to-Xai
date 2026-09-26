"""3x3 2D grid split for the merged CLIP branch's 224x224 CAM (`heatmap_224`
from `merged_grad_eclip.explain_image`). Same region names and exact-pixel-
coverage boundary rule as `graph_xai_extension/graph_xai/region_grid.py`
(`round(i*size/grid)`, no gaps/overlaps), reimplemented independently so
this package never imports `graph_xai_extension`."""
from __future__ import annotations

import numpy as np

from .schemas import RegionInfo

REGION_NAMES_3X3 = (
    "top_left", "top_center", "top_right",
    "middle_left", "middle_center", "middle_right",
    "bottom_left", "bottom_center", "bottom_right",
)


def _boundaries(size: int, grid: int = 3) -> list[int]:
    return [round(i * size / grid) for i in range(grid + 1)]


def split_into_regions_2d(cam: np.ndarray, grid: int = 3) -> list[RegionInfo]:
    """`cam` must be 2D (H, W), already aligned to the image it will be
    masked on (no resizing here -- callers pass an already-224x224 CAM)."""
    if cam.ndim != 2:
        raise ValueError(f"cam must be 2D, got shape {cam.shape}")
    if not np.isfinite(cam).all():
        raise ValueError("cam contains NaN or Inf values")

    h, w = cam.shape
    row_bounds = _boundaries(h, grid)
    col_bounds = _boundaries(w, grid)
    total = float(cam.sum())

    regions = []
    for r in range(grid):
        for c in range(grid):
            r0, r1 = row_bounds[r], row_bounds[r + 1]
            c0, c1 = col_bounds[c], col_bounds[c + 1]
            patch = cam[r0:r1, c0:c1]
            cam_sum = float(patch.sum())
            regions.append(RegionInfo(
                name=REGION_NAMES_3X3[r * grid + c],
                bbox=((r0, r1), (c0, c1)),
                cam_mean=float(patch.mean()) if patch.size else 0.0,
                cam_sum=cam_sum,
                cam_ratio=(cam_sum / total) if total > 0 else 0.0,
            ))
    return regions
