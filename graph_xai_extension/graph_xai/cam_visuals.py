"""CAM/MRI visualization primitives: a CAM-only heatmap, an MRI+CAM overlay,
a grid-boundary+CAM overlay (visually distinct from a plain 3x3 grid-only
overlay), a top-CAM-region highlight, and before/after masking snapshots for
a chosen region.

Every image here is produced at the SAME size as the input MRI (the CAM is
resized/aligned to it, recording which interpolation method was used), and
none of the input arrays/images are mutated. Every renderer records its CAM
value range, colormap, and alpha (where applicable) so callers can display or
log them, and every renderer's caption should be paired with
`CAM_NOT_LESION_CAPTION` wherever it is shown.
"""
from __future__ import annotations

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .masking import apply_region_mask
from .region_grid import _resize_2d

CAM_NOT_LESION_CAPTION = (
    "CAM은 모델 출력에 대한 근사적 민감도 지도이며, 실제 병변 위치를 의미하지 않습니다."
)


def _to_rgb_image(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def _normalize_cam(cam: np.ndarray) -> tuple[np.ndarray, float, float]:
    cam = np.asarray(cam, dtype=np.float32)
    if not np.isfinite(cam).all():
        raise ValueError("cam contains NaN or Inf values")
    cam_min, cam_max = float(cam.min()), float(cam.max())
    if cam_max <= cam_min:
        return np.zeros_like(cam), cam_min, cam_max
    return (cam - cam_min) / (cam_max - cam_min), cam_min, cam_max


def render_cam_heatmap(cam, image_size_hw: tuple[int, int], colormap: str = "jet", interpolation: str = "bilinear") -> dict:
    """CAM-only heatmap resized/aligned to `image_size_hw` (height, width)."""
    resized = _resize_2d(np.asarray(cam, dtype=np.float32), image_size_hw, interpolation)
    normalized, cam_min, cam_max = _normalize_cam(resized)
    cmap = matplotlib.colormaps[colormap]
    rgba = (cmap(normalized) * 255).astype(np.uint8)
    return {
        "image": Image.fromarray(rgba, mode="RGBA"),
        "cam_value_range": (cam_min, cam_max),
        "colormap": colormap,
        "interpolation": interpolation,
        "caption": CAM_NOT_LESION_CAPTION,
    }


def _apply_alpha(rgba_image: Image.Image, alpha: float) -> Image.Image:
    array = np.array(rgba_image)
    array[..., 3] = (array[..., 3].astype(np.float32) * alpha).astype(np.uint8)
    return Image.fromarray(array, mode="RGBA")


def render_mri_cam_overlay(image, cam, alpha: float = 0.45, colormap: str = "jet", interpolation: str = "bilinear") -> dict:
    """MRI + CAM overlay, no grid lines. Distinct from `render_grid_cam_overlay`."""
    base = _to_rgb_image(image)
    heat = render_cam_heatmap(cam, (base.height, base.width), colormap=colormap, interpolation=interpolation)
    blended = Image.alpha_composite(base.convert("RGBA"), _apply_alpha(heat["image"], alpha)).convert("RGB")
    heat["image"] = blended
    heat["alpha"] = alpha
    return heat


def render_grid_cam_overlay(
    image, cam, regions, alpha: float = 0.45, colormap: str = "jet",
    interpolation: str = "bilinear", grid_color: tuple = (255, 255, 0),
) -> dict:
    """3x3 grid boundaries drawn ON TOP of an MRI+CAM overlay -- visually
    distinct from a plain grid-only overlay (no CAM) or a plain CAM overlay
    (no grid)."""
    overlay = render_mri_cam_overlay(image, cam, alpha=alpha, colormap=colormap, interpolation=interpolation)
    img = overlay["image"].copy()
    draw = ImageDraw.Draw(img)
    for region in regions:
        draw.rectangle(
            [region.col_start, region.row_start, region.col_end - 1, region.row_end - 1],
            outline=grid_color, width=2,
        )
    overlay["image"] = img
    overlay["grid_color"] = grid_color
    return overlay


def render_top_cam_regions_highlight(image, regions, top_k: int = 3, highlight_color: tuple = (255, 0, 0)) -> dict:
    """Highlights the `top_k` regions ranked by `cam_ratio` with a colored
    border and rank label."""
    img = _to_rgb_image(image).copy()
    # Scale the label up with the image so "#1"/"#2"/"#3" stay legible on a
    # small MRI slice instead of rendering at PIL's tiny default font size.
    font_size = max(14, img.height // 12)
    try:
        font = ImageFont.truetype("arialbd.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(img)
    top_regions = sorted(regions, key=lambda r: r.cam_ratio, reverse=True)[:top_k]
    for rank, region in enumerate(top_regions, start=1):
        draw.rectangle(
            [region.col_start, region.row_start, region.col_end - 1, region.row_end - 1],
            outline=highlight_color, width=3,
        )
        # White outline behind the colored text so the rank number stays
        # readable regardless of what's underneath it.
        label_xy = (region.col_start + 4, region.row_start + 2)
        draw.text(label_xy, f"#{rank}", fill=highlight_color, font=font, stroke_width=2, stroke_fill=(255, 255, 255))
    return {"image": img, "top_k": top_k, "region_names": [r.name for r in top_regions], "caption": CAM_NOT_LESION_CAPTION}


def render_region_masking_before_after(image, region, method: str, **mask_params) -> dict:
    """Before/after snapshot of masking one specific region, for showing the
    top supporting or top suppressing region's actual masked appearance."""
    masked = apply_region_mask(
        image, row_start=region.row_start, row_end=region.row_end,
        col_start=region.col_start, col_end=region.col_end, method=method, **mask_params,
    )
    return {
        "before": _to_rgb_image(image),
        "after": _to_rgb_image(masked),
        "region_name": region.name,
        "masking_method": method,
    }
