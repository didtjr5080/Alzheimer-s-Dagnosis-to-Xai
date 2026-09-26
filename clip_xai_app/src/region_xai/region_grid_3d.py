"""3x3x3 (27-region) grid split for the merged 3D CNN branch's cache volume
+ Grad-CAM volume. Region names are grid-index based (`block_{d}_{h}_{w}`,
each index in {0,1,2} = low/mid/high third along that array axis) -- NOT
anatomical labels, consistent with this project's "좌우 미검증" /
axis-mapping caveats (see `merged_cnn3d_gradcam.py`)."""
from __future__ import annotations

import numpy as np

from .schemas import RegionInfo


def _boundaries(size: int, grid: int = 3) -> list[int]:
    return [round(i * size / grid) for i in range(grid + 1)]


def region_name_3d(d: int, h: int, w: int) -> str:
    return f"block_{d}_{h}_{w}"


def split_into_regions_3d(cam_volume: np.ndarray, grid: int = 3) -> list[RegionInfo]:
    if cam_volume.ndim != 3:
        raise ValueError(f"cam_volume must be 3D, got shape {cam_volume.shape}")
    if not np.isfinite(cam_volume).all():
        raise ValueError("cam_volume contains NaN or Inf values")

    d, h, w = cam_volume.shape
    d_bounds, h_bounds, w_bounds = _boundaries(d, grid), _boundaries(h, grid), _boundaries(w, grid)
    total = float(cam_volume.sum())

    regions = []
    for di in range(grid):
        for hi in range(grid):
            for wi in range(grid):
                d0, d1 = d_bounds[di], d_bounds[di + 1]
                h0, h1 = h_bounds[hi], h_bounds[hi + 1]
                w0, w1 = w_bounds[wi], w_bounds[wi + 1]
                patch = cam_volume[d0:d1, h0:h1, w0:w1]
                cam_sum = float(patch.sum())
                regions.append(RegionInfo(
                    name=region_name_3d(di, hi, wi),
                    bbox=((d0, d1), (h0, h1), (w0, w1)),
                    cam_mean=float(patch.mean()) if patch.size else 0.0,
                    cam_sum=cam_sum,
                    cam_ratio=(cam_sum / total) if total > 0 else 0.0,
                ))
    return regions
