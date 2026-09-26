"""Grad-ECLIP-style XAI for the merged (OASIS-3+ADNI) CLIP branch.

Reuses the exact hook location and "attention 결합 제거" (gradient-only,
no attention-weight combination) approach from the OASIS-3-only handoff's
`grad_eclip.py` (work order section 1-4: "기존 hook 위치와 attention 결합
제거 방식은 그대로 쓴다"). The only thing that changes is the backpropagated
target: instead of the LR class logit `W_c f(x) + b_c`, this uses the merged
branch's own class-embedding cosine-similarity logit,

    z_c = logit_scale * (f(x)/||f(x)||) . (e_c/||e_c||)

`torch.no_grad()` is never used in this module, per section 1-4.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .merged_clip import CLASS_NAMES, HEATMAP_METHOD_NAME, MergedClipBundle, _images_to_pixel_values


def _forward_with_hook(bundle: MergedClipBundle, pixel_values: torch.Tensor):
    holder: dict[str, torch.Tensor] = {}

    def capture_hook(module, inp, out):
        out.retain_grad()
        holder["tokens"] = out

    last_attn_module = bundle.clip_model.vision_model.encoder.layers[-1].self_attn
    handle = last_attn_module.v_proj.register_forward_hook(capture_hook)
    vision_outputs = bundle.clip_model.vision_model(pixel_values=pixel_values)
    handle.remove()

    pooled_output = vision_outputs.pooler_output
    image_embeds = bundle.clip_model.visual_projection(pooled_output)
    return image_embeds, holder["tokens"]


def explain_image(bundle: MergedClipBundle, image: Image.Image, target_class_name: str | None = None) -> dict:
    """Single-image Grad-ECLIP-style explanation for the merged CLIP branch.
    Returns a dict with `heatmap_224`, `predicted_class`, `target_class`,
    and `probs` -- same shape as the legacy handoff's `explain_image()`."""
    pixel_values = _images_to_pixel_values([image], bundle.clip_processor).to(bundle.device)
    image_embeds, last_layer_tokens = _forward_with_hook(bundle, pixel_values)

    image_embeds_norm = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    logits = bundle.logit_scale * (image_embeds_norm @ bundle.class_embeds.T)
    probs = torch.softmax(logits, dim=-1)[0]

    pred_local_idx = int(logits.argmax(dim=-1).item())
    predicted_class = CLASS_NAMES[pred_local_idx]
    target_local_idx = pred_local_idx if target_class_name is None else CLASS_NAMES.index(target_class_name)

    target_logit = logits[0, target_local_idx]
    bundle.clip_model.zero_grad()
    target_logit.backward()

    grad = last_layer_tokens.grad
    if grad is None:
        raise RuntimeError("gradient가 계산되지 않았습니다 (last_layer_tokens.grad is None)")

    tokens = last_layer_tokens.detach()[0]
    grads = grad.detach()[0]

    patch_tokens = tokens[1:]
    patch_grads = grads[1:]

    channel_weights = patch_grads.mean(dim=0)
    cam = (patch_tokens * channel_weights).sum(dim=-1)
    cam = F.relu(cam)
    if cam.max() > 0:
        cam = cam / cam.max()

    vision_cfg = bundle.clip_model.config.vision_config
    grid_size = vision_cfg.image_size // vision_cfg.patch_size
    image_size = vision_cfg.image_size

    heatmap_grid = cam.reshape(grid_size, grid_size).cpu().numpy()
    heatmap_tensor = torch.tensor(heatmap_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    heatmap_224 = F.interpolate(
        heatmap_tensor, size=(image_size, image_size), mode="bilinear", align_corners=False
    )[0, 0].numpy()

    return {
        "heatmap_224": heatmap_224,
        "predicted_class": predicted_class,
        "target_class": CLASS_NAMES[target_local_idx],
        "probs": dict(zip(CLASS_NAMES, probs.detach().cpu().tolist())),
        "method_name": HEATMAP_METHOD_NAME,
    }


# A background pixel in these coronal-plane MRI PNGs is near 0 (black); a
# small positive threshold (out of 255) absorbs near-zero noise without
# masking real low-signal brain tissue. Same masking rationale as the 3D
# branch's BACKGROUND_THRESHOLD in merged_cnn3d_gradcam.py (work order
# section A-4: "2D CLIP 히트맵에서... 3D와 같이 배경 마스크를 적용").
BACKGROUND_THRESHOLD_255 = 10


def render_masked_overlay(image: Image.Image, heatmap_224: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """MRI slice + CAM overlay with the heat layer masked to fully
    transparent outside the brain (background pixels), so the heatmap is
    never painted over the black background/corners. The heatmap values
    themselves are not smoothed or thresholded -- only the overlay's alpha
    is masked based on the underlying image brightness."""
    base = image.convert("L").resize((224, 224))
    gray = np.array(base)
    base_rgba = base.convert("RGBA")

    heat_clipped = np.clip(heatmap_224, 0.0, 1.0)
    red = (heat_clipped * 255).astype(np.uint8)
    zeros = np.zeros_like(red)
    heat_alpha = np.full_like(red, 180, dtype=np.uint8)
    background_mask = gray <= BACKGROUND_THRESHOLD_255
    heat_alpha = np.where(background_mask, 0, heat_alpha).astype(np.uint8)
    heat_alpha = (heat_alpha.astype(np.float32) * alpha).astype(np.uint8)
    heat = Image.fromarray(np.stack([red, zeros, zeros, heat_alpha], axis=-1), mode="RGBA")

    return Image.alpha_composite(base_rgba, heat).convert("RGB")


def render_cam_only(heatmap_224: np.ndarray, colormap: str = "jet") -> Image.Image:
    """CAM-only heatmap (no MRI underneath), matplotlib colormap."""
    import matplotlib
    normalized = np.clip(heatmap_224, 0.0, 1.0)
    cmap = matplotlib.colormaps[colormap]
    rgba = (cmap(normalized) * 255).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA").convert("RGB")


def render_grid_overlay(image: Image.Image, heatmap_224: np.ndarray, regions, alpha: float = 0.45) -> Image.Image:
    """MRI+CAM overlay (background-masked) with 3x3 grid boundary lines
    drawn on top -- visually distinct from a plain masked overlay."""
    from PIL import ImageDraw

    base = render_masked_overlay(image, heatmap_224, alpha=alpha).convert("RGB")
    draw = ImageDraw.Draw(base)
    for region in regions:
        (r0, r1), (c0, c1) = region.bbox
        draw.rectangle([c0, r0, c1 - 1, r1 - 1], outline=(255, 255, 0), width=2)
    return base


def render_top_regions_highlight(image: Image.Image, regions, top_k: int = 3) -> Image.Image:
    """Highlights the `top_k` regions ranked by `cam_ratio` with a colored
    border and rank label, at 224x224 resolution."""
    from PIL import ImageDraw, ImageFont

    base = image.convert("RGB").resize((224, 224))
    draw = ImageDraw.Draw(base)
    font_size = max(14, base.height // 12)
    try:
        font = ImageFont.truetype("arialbd.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    top_regions = sorted(regions, key=lambda r: r.cam_ratio, reverse=True)[:top_k]
    for rank, region in enumerate(top_regions, start=1):
        (r0, r1), (c0, c1) = region.bbox
        draw.rectangle([c0, r0, c1 - 1, r1 - 1], outline=(255, 0, 0), width=3)
        draw.text((c0 + 4, r0 + 2), f"#{rank}", fill=(255, 0, 0), font=font, stroke_width=2, stroke_fill=(255, 255, 255))
    return base


def background_heat_ratio_2d(image: Image.Image, heatmap_224: np.ndarray) -> dict:
    """Fraction of total CAM "heat" (sum of heatmap values) inside vs
    outside the brain (grayscale > BACKGROUND_THRESHOLD_255), for reporting
    how much of the unmasked heatmap would have painted outside the
    anatomy (work order section A-4)."""
    gray = np.array(image.convert("L").resize((224, 224)))
    in_brain = gray > BACKGROUND_THRESHOLD_255
    total = float(np.clip(heatmap_224, 0.0, 1.0).sum())
    if total <= 0:
        return {"in_brain_fraction": float("nan"), "background_fraction": float("nan")}
    in_brain_heat = float(np.clip(heatmap_224, 0.0, 1.0)[in_brain].sum())
    return {
        "in_brain_fraction": in_brain_heat / total,
        "background_fraction": 1.0 - (in_brain_heat / total),
    }
