from __future__ import annotations

from PIL import Image


def generate_representative_xai(images: list[Image.Image], indices: list[int], target_class_idx: int, bundle):
    from inference_clip_lr import generate_xai

    return [
        generate_xai(images[index], target_class_idx=target_class_idx, bundle=bundle)
        for index in indices
    ]
