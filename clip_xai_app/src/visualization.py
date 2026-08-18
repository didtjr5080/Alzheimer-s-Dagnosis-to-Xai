from __future__ import annotations

import numpy as np
from PIL import Image


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if not np.isfinite(heatmap).all():
        raise ValueError("Heatmap contains NaN or Inf.")
    min_value = float(heatmap.min())
    max_value = float(heatmap.max())
    if max_value <= min_value:
        return np.zeros_like(heatmap, dtype=np.float32)
    return (heatmap - min_value) / (max_value - min_value)


def heatmap_to_rgb(heatmap: np.ndarray) -> Image.Image:
    normalized = normalize_heatmap(heatmap)
    red = (normalized * 255).astype(np.uint8)
    green = np.zeros_like(red)
    blue = ((1.0 - normalized) * 64).astype(np.uint8)
    alpha = np.full_like(red, 180, dtype=np.uint8)
    rgba = np.stack([red, green, blue, alpha], axis=-1)
    return Image.fromarray(rgba, mode="RGBA")


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = image.convert("RGBA").resize((224, 224))
    heat = heatmap_to_rgb(heatmap)
    blended = Image.blend(base, heat, alpha=alpha)
    return blended.convert("RGB")
