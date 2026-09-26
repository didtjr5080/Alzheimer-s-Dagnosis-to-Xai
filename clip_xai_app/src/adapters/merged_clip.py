"""Live-inference adapter for the merged (OASIS-3+ADNI) CLIP branch.

Reuses the same frozen `openai/clip-vit-base-patch16` backbone/processor
already shipped in the OASIS-3-only handoff package (same commit,
`57c216476eefef5ab752ec549e440a49ae4ae5f3`, confirmed against
`clip_lr_grad_eclip_handoff_v1_20260812_105027/metadata/model_config.json`),
but replaces the classification head: instead of a Logistic Regression on
raw (non-L2-normalized) embeddings, this branch L2-normalizes the CLIP image
embedding and scores it against three L2-normalized per-class embedding
vectors (`class_embeds`, shape (3, 512)) stored in
`merged_project/checkpoints/clip_best.pt`, scaled by CLIP's own learned
`logit_scale.exp()` -- i.e. the same cosine-similarity-times-temperature
formula CLIP uses for its own image/text contrastive logits, per work order
`통합(OASIS-3+ADNI) 모델 라이브 추론 전환 작업지시서.md` section 1-1 (mirrors
notebook cell 42, `CLIPSimilarityClassifierFixed2`).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

CLASS_NAMES = ["CN", "MCI", "AD"]
HEATMAP_METHOD_NAME = "통합 CLIP class-embedding 유사도 기반 gradient heatmap"


@dataclass
class MergedClipBundle:
    clip_model: CLIPModel
    clip_processor: CLIPProcessor
    class_embeds: torch.Tensor  # (3, 512), L2-normalized, on `device`
    logit_scale: float
    epoch: int
    device: torch.device


def load_merged_clip_bundle(repo_root: Path, device: str | None = None) -> MergedClipBundle:
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    handoff_root = repo_root / "clip_lr_grad_eclip_handoff_v1_20260812_105027"
    clip_model_dir = handoff_root / "models" / "clip_model"
    clip_processor_dir = handoff_root / "models" / "clip_processor"
    ckpt_path = repo_root / "merged_project" / "checkpoints" / "clip_best.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing merged CLIP checkpoint: {ckpt_path}")

    clip_model = CLIPModel.from_pretrained(str(clip_model_dir)).to(device_obj)
    clip_model.eval()
    # Parameters are left trainable (`requires_grad=True`, the HF default) --
    # never optimized, but required for Grad-ECLIP's backward-to-activation
    # pass in merged_grad_eclip.py to build a gradient graph at all.
    clip_processor = CLIPProcessor.from_pretrained(str(clip_processor_dir))

    ckpt = torch.load(ckpt_path, map_location=device_obj, weights_only=True)
    class_embeds_raw = ckpt["class_embeds"].to(device_obj).float()
    if tuple(class_embeds_raw.shape) != (3, 512):
        raise ValueError(f"unexpected class_embeds shape: {tuple(class_embeds_raw.shape)} (expected (3, 512))")
    class_embeds = class_embeds_raw / class_embeds_raw.norm(dim=-1, keepdim=True)
    logit_scale = clip_model.logit_scale.exp().item()
    epoch = int(ckpt["epoch"])

    return MergedClipBundle(
        clip_model=clip_model, clip_processor=clip_processor, class_embeds=class_embeds,
        logit_scale=logit_scale, epoch=epoch, device=device_obj,
    )


def _images_to_pixel_values(images: list[Image.Image], processor: CLIPProcessor) -> torch.Tensor:
    rgb_images = [img.convert("RGB") for img in images]
    return processor.image_processor(images=rgb_images, return_tensors="pt")["pixel_values"]


def _image_embeds_l2(bundle: MergedClipBundle, pixel_values: torch.Tensor) -> torch.Tensor:
    image_embeds = bundle.clip_model.get_image_features(pixel_values=pixel_values)
    return image_embeds / image_embeds.norm(dim=-1, keepdim=True)


def predict_slice_batch(bundle: MergedClipBundle, images: list[Image.Image]) -> torch.Tensor:
    """(n, 3) softmax probabilities for a batch of slice images, with
    gradients tracked (needed for the Grad-ECLIP XAI path, work order
    section 1-4). Bulk validation should wrap calls in `torch.no_grad()`
    itself for speed -- the math is identical either way."""
    pixel_values = _images_to_pixel_values(images, bundle.clip_processor).to(bundle.device)
    image_embeds_norm = _image_embeds_l2(bundle, pixel_values)
    logits = bundle.logit_scale * (image_embeds_norm @ bundle.class_embeds.T)
    return logits.softmax(dim=-1)


def xai_target_logit(bundle: MergedClipBundle, pixel_values: torch.Tensor, target_class_idx: int) -> torch.Tensor:
    """z_c = logit_scale * (f(x)/||f(x)||) . (e_c/||e_c||) for a single class,
    WITHOUT torch.no_grad(), for Grad-ECLIP backprop (work order section 1-4)."""
    image_embeds_norm = _image_embeds_l2(bundle, pixel_values)
    class_embed = bundle.class_embeds[target_class_idx]
    return bundle.logit_scale * (image_embeds_norm @ class_embed)


def predict_array(bundle: MergedClipBundle, image_array: np.ndarray) -> np.ndarray:
    """(3,) CN/MCI/AD probabilities for a single uint8 numpy array, either
    grayscale (H, W) or RGB (H, W, 3) (e.g. a region-perturbation-masked
    slice). No gradients tracked."""
    clipped = np.clip(image_array, 0, 255).astype(np.uint8)
    mode = "L" if clipped.ndim == 2 else "RGB"
    pil_image = Image.fromarray(clipped, mode=mode)
    with torch.no_grad():
        probs = predict_slice_batch(bundle, [pil_image])
    return probs[0].cpu().numpy()
