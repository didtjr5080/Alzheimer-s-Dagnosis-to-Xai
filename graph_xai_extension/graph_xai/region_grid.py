"""Divide a CAM/image into a grid of purely spatial regions (no anatomical labels).

Region boundaries are computed with `round(i * size / grid)` cut points so that
every pixel belongs to exactly one region even when the image/CAM size is not
evenly divisible by the grid size. No pixel is skipped or double-counted.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .schemas import REGION_NAMES_3X3, RegionInfo


def _boundaries(size: int, grid: int) -> list[int]:
    edges = [round(i * size / grid) for i in range(grid + 1)]
    edges[0] = 0
    edges[-1] = size
    return edges


def _resize_2d(array: np.ndarray, target_shape: tuple[int, int], interpolation: str) -> np.ndarray:
    if array.shape == tuple(target_shape):
        return array.astype(np.float32, copy=True)
    tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    kwargs = {}
    if interpolation in ("bilinear", "bicubic"):
        kwargs["align_corners"] = False
    resized = F.interpolate(tensor, size=tuple(int(s) for s in target_shape), mode=interpolation, **kwargs)
    return resized[0, 0].numpy()


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        return image.mean(axis=-1)
    raise ValueError(f"Unsupported image ndim={image.ndim}; expected 2 (H,W) or 3 (H,W,C)")


def split_into_regions(
    cam: np.ndarray,
    image: np.ndarray | None = None,
    grid: int = 3,
    interpolation: str = "bilinear",
) -> list[RegionInfo]:
    cam = np.asarray(cam)
    if cam.size == 0 or any(dim == 0 for dim in cam.shape):
        raise ValueError("cam array is empty")
    if cam.ndim != 2:
        raise ValueError(f"cam must be a 2D array, got shape {cam.shape}")
    cam = cam.astype(np.float32, copy=True)
    if not np.isfinite(cam).all():
        raise ValueError("cam contains NaN or Inf values")

    image_gray = None
    target_shape = cam.shape
    if image is not None:
        image_arr = np.asarray(image, dtype=np.float32)
        image_gray = _to_grayscale(image_arr)
        target_shape = image_gray.shape

    cam_resized = _resize_2d(cam, target_shape, interpolation)
    total_sum = float(cam_resized.sum())

    height, width = target_shape
    row_edges = _boundaries(height, grid)
    col_edges = _boundaries(width, grid)

    if grid == 3:
        names = list(REGION_NAMES_3X3)
    else:
        names = [f"row{r}_col{c}" for r in range(grid) for c in range(grid)]

    regions: list[RegionInfo] = []
    idx = 0
    for row in range(grid):
        row_start, row_end = row_edges[row], row_edges[row + 1]
        for col in range(grid):
            col_start, col_end = col_edges[col], col_edges[col + 1]
            block = cam_resized[row_start:row_end, col_start:col_end]
            block_sum = float(block.sum())
            image_mean_brightness = float("nan")
            if image_gray is not None:
                image_block = image_gray[row_start:row_end, col_start:col_end]
                image_mean_brightness = float(image_block.mean())
            regions.append(RegionInfo(
                name=names[idx],
                row=row,
                col=col,
                row_start=int(row_start),
                row_end=int(row_end),
                col_start=int(col_start),
                col_end=int(col_end),
                cam_mean=float(block.mean()),
                cam_max=float(block.max()),
                cam_sum=block_sum,
                cam_ratio=(block_sum / total_sum) if total_sum != 0 else 0.0,
                image_mean_brightness=image_mean_brightness,
                area_pixels=int(block.size),
            ))
            idx += 1
    return regions
