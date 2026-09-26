"""3D Grad-CAM for the merged (OASIS-3+ADNI) 3D CNN branch (work order
`통합(OASIS-3+ADNI) 모델 라이브 추론 전환 작업지시서.md` section 3-3, axis
mapping corrected by `통합 모델 XAI 후속 수정 작업지시서.md` section A).

Axis mapping (verified against `slices_multi/` reference PNGs and gross
brain anatomy on 3 test scans, work order section A / A-1):

    axis 0 (size 98)  -> coronal   : vol[i, :, :]
    axis 1 (size 116) -> axial     : np.rot90(vol[:, j, :])  (matches slices_multi)
    axis 2 (size 94)  -> sagittal  : vol[:, :, k]

`slices_multi/{scan_id}_cor{idx:03d}.png` files are axial slices despite the
"cor" in their filename (that name reflects the original preprocessing
code's own naming, not the anatomical plane).

Left/right laterality of the coronal and axial panels has NOT been
independently verified in this work (좌우 미검증) -- only the
axial/coronal/sagittal plane identity was confirmed.

The activation/gradient hook location is configurable (`target_layer`,
default `"block4"`, spatial resolution (6, 7, 5) for a (98, 116, 94) input;
`"block3"` gives (12, 14, 11)) so the two resolutions can be compared.
Regardless of `target_layer`, upsampling to the cache volume's own
resolution always uses
`F.interpolate(cam[None, None], size=volume.shape, mode="trilinear", align_corners=False)`,
and the CAM is normalized exactly once, over the whole 3D volume (never
per 2D slice).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .merged_cnn3d import CLASS_NAMES, MergedCnn3dBundle

HEATMAP_METHOD_NAME_TEMPLATE = "통합 3D CNN {layer} 활성화 기반 Grad-CAM"
LATERALITY_NOT_VERIFIED_NOTE = "좌우(L/R) 방향은 별도로 검증하지 않았습니다 (좌우 미검증)."

# A background voxel in the cache volume is exactly 0.0 (whole-volume
# min-max normalization during caching); a small positive threshold absorbs
# any near-zero noise without masking real low-signal brain tissue.
BACKGROUND_THRESHOLD = 0.02


def grad_cam_3d(
    bundle: MergedCnn3dBundle, volume: np.ndarray, target_class_name: str | None = None,
    target_layer: str = "block4",
) -> dict:
    """`volume` is the raw cached (98, 116, 94) float16/float32 array. Returns
    a dict with `cam_volume` (same shape as `volume`, in [0, 1]),
    `predicted_class`, `target_class`, `probs`, and `target_layer`."""
    if target_layer not in ("block3", "block4"):
        raise ValueError(f"target_layer must be 'block3' or 'block4', got {target_layer!r}")

    model = bundle.model
    tensor = torch.from_numpy(volume.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(bundle.device)

    activation_holder: dict[str, torch.Tensor] = {}

    def fwd_hook(module, inp, out):
        out.retain_grad()
        activation_holder["act"] = out

    hook_module = getattr(model, target_layer)
    handle = hook_module.register_forward_hook(fwd_hook)
    logits = model(tensor)
    handle.remove()

    probs = logits.softmax(dim=-1)[0]
    pred_idx = int(logits.argmax(dim=-1).item())
    target_idx = pred_idx if target_class_name is None else CLASS_NAMES.index(target_class_name)

    target_logit = logits[0, target_idx]
    model.zero_grad()
    target_logit.backward()

    activation = activation_holder["act"]
    grad = activation.grad
    if grad is None:
        raise RuntimeError(f"gradient가 계산되지 않았습니다 ({target_layer} activation.grad is None)")

    act = activation.detach()[0]  # (C, d, h, w)
    grad = grad.detach()[0]       # (C, d, h, w)
    weights = grad.mean(dim=(1, 2, 3))  # (C,) global-average-pooled gradient, per Grad-CAM
    cam = (act * weights[:, None, None, None]).sum(dim=0)
    cam = F.relu(cam)
    if cam.max() > 0:
        # Normalized exactly once, over the whole 3D CAM volume -- never
        # renormalized per 2D slice after extraction.
        cam = cam / cam.max()

    cam_5d = cam.unsqueeze(0).unsqueeze(0)
    cam_up = F.interpolate(cam_5d, size=volume.shape, mode="trilinear", align_corners=False)[0, 0]
    cam_volume = cam_up.cpu().numpy()

    return {
        "cam_volume": cam_volume,
        "predicted_class": CLASS_NAMES[pred_idx],
        "target_class": CLASS_NAMES[target_idx],
        "probs": dict(zip(CLASS_NAMES, probs.detach().cpu().tolist())),
        "method_name": HEATMAP_METHOD_NAME_TEMPLATE.format(layer=target_layer),
        "target_layer": target_layer,
    }


def render_plain_slice(volume_slice: np.ndarray) -> Image.Image:
    """Plain grayscale render of one volume slice, no CAM overlay -- used
    for before/after masked-region comparisons."""
    gray = (np.clip(volume_slice, 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(gray, mode="L").convert("RGB")


def central_slice_overlays(volume: np.ndarray, cam_volume: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Central coronal/axial/sagittal (volume_slice, cam_slice) pairs, at the
    cache volume's own resolution (no resampling to original scan
    resolution). Axis mapping per this module's docstring; the axial panel
    is rotated 90 degrees to match `slices_multi/` orientation."""
    d, h, w = volume.shape
    return {
        "coronal": (volume[d // 2, :, :], cam_volume[d // 2, :, :]),
        "axial": (np.rot90(volume[:, h // 2, :]), np.rot90(cam_volume[:, h // 2, :])),
        "sagittal": (volume[:, :, w // 2], cam_volume[:, :, w // 2]),
    }


def render_overlay_image(volume_slice: np.ndarray, cam_slice: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Grayscale volume slice (already in [0, 1], the cache volume's own
    whole-volume min-max normalization) blended with a red-scale CAM overlay
    (also already normalized once, over the whole 3D CAM volume). Kept at
    the cache volume's native slice resolution, no resize.

    Voxels at or below `BACKGROUND_THRESHOLD` in the *volume* (i.e. outside
    the brain) are masked to fully transparent in the heat layer, so the
    CAM is never painted outside the anatomy -- this masks the *overlay
    presentation* only; the underlying CAM values themselves are not
    smoothed, thresholded, or otherwise altered.
    """
    gray = (np.clip(volume_slice, 0.0, 1.0) * 255).astype(np.uint8)
    base = Image.fromarray(gray, mode="L").convert("RGBA")

    cam_clipped = np.clip(cam_slice, 0.0, 1.0)
    red = (cam_clipped * 255).astype(np.uint8)
    zeros = np.zeros_like(red)
    heat_alpha = np.full_like(red, 180, dtype=np.uint8)
    background_mask = volume_slice <= BACKGROUND_THRESHOLD
    heat_alpha = np.where(background_mask, 0, heat_alpha).astype(np.uint8)
    heat = Image.fromarray(np.stack([red, zeros, zeros, heat_alpha], axis=-1), mode="RGBA")

    return Image.alpha_composite(base, _scale_alpha(heat, alpha)).convert("RGB")


def _scale_alpha(rgba_image: Image.Image, alpha: float) -> Image.Image:
    array = np.array(rgba_image)
    array[..., 3] = (array[..., 3].astype(np.float32) * alpha).astype(np.uint8)
    return Image.fromarray(array, mode="RGBA")


def background_heat_ratio(volume_slice: np.ndarray, cam_slice: np.ndarray) -> dict:
    """Fraction of total CAM "heat" (sum of cam values) that falls inside vs
    outside the brain (`volume_slice > BACKGROUND_THRESHOLD`), for reporting
    how much of the unmasked heatmap would have painted outside the
    anatomy (work order section A-4)."""
    in_brain = volume_slice > BACKGROUND_THRESHOLD
    total = float(cam_slice.sum())
    if total <= 0:
        return {"in_brain_fraction": float("nan"), "background_fraction": float("nan")}
    in_brain_heat = float(cam_slice[in_brain].sum())
    return {
        "in_brain_fraction": in_brain_heat / total,
        "background_fraction": 1.0 - (in_brain_heat / total),
    }
