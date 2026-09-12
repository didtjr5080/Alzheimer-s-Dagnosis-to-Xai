"""Region perturbation analysis: mask each of the 9 grid regions and measure
how the ORIGINAL predicted class's probability changes. The top-1 class of the
masked image is never substituted for the original class.
"""
from __future__ import annotations

import numpy as np

from .masking import apply_region_mask
from .region_grid import split_into_regions
from .schemas import RegionPerturbationResult


def run_region_perturbation(
    image,
    cam: np.ndarray,
    predict_fn,
    class_names: list[str],
    original_class_idx: int,
    masking_method: str = "mean",
    grid_size: int = 3,
    interpolation: str = "bilinear",
    **mask_params,
) -> list[RegionPerturbationResult]:
    if not 0 <= int(original_class_idx) < len(class_names):
        raise ValueError(f"original_class_idx={original_class_idx} out of range for {len(class_names)} classes")

    original_probs = np.asarray(predict_fn([image])[0], dtype=np.float64)
    original_predicted_class = class_names[int(original_probs.argmax())]
    original_class_probability = float(original_probs[original_class_idx])

    regions = split_into_regions(cam, image=np.asarray(image, dtype=np.float32) if not hasattr(image, "mode") else image,
                                  grid=grid_size, interpolation=interpolation)

    masked_images = [
        apply_region_mask(
            image,
            row_start=region.row_start, row_end=region.row_end,
            col_start=region.col_start, col_end=region.col_end,
            method=masking_method, **mask_params,
        )
        for region in regions
    ]

    masked_probs_batch = np.asarray(predict_fn(masked_images), dtype=np.float64)

    results = []
    for region, masked_probs in zip(regions, masked_probs_batch):
        masked_predicted_class = class_names[int(masked_probs.argmax())]
        masked_original_class_probability = float(masked_probs[original_class_idx])
        probability_drop = original_class_probability - masked_original_class_probability
        results.append(RegionPerturbationResult(
            region_name=region.name,
            original_predicted_class=original_predicted_class,
            original_class_probability=original_class_probability,
            masked_predicted_class=masked_predicted_class,
            masked_original_class_probability=masked_original_class_probability,
            probability_drop=probability_drop,
            absolute_probability_change=abs(probability_drop),
            masking_method=masking_method,
            cam_mean=region.cam_mean,
            cam_ratio=region.cam_ratio,
        ))
    return results
