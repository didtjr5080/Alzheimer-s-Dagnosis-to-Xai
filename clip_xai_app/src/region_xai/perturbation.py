"""Generic region-perturbation runner: mask each region, rerun `predict_fn`,
and compare the probability of the ORIGINAL predicted class before/after
(never re-argmax the masked result to decide "did it change" -- the drop is
always measured against the class the model originally picked). Works for
both the 2D CLIP branch and the 3D CNN branch; only `predict_fn` and the
region list differ between them.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .masking import MaskingMethod, apply_region_mask
from .schemas import RegionInfo, RegionPerturbationResult


def run_region_perturbation(
    array: np.ndarray,
    regions: Sequence[RegionInfo],
    predict_fn: Callable[[np.ndarray], np.ndarray],
    class_names: list[str],
    original_class_idx: int,
    masking_method: MaskingMethod = "mean",
    **mask_params,
) -> list[RegionPerturbationResult]:
    original_probs = np.asarray(predict_fn(array), dtype=np.float64)
    original_class_probability = float(original_probs[original_class_idx])
    original_predicted_class = class_names[original_class_idx]

    results = []
    for region in regions:
        masked = apply_region_mask(array, region.bbox, method=masking_method, **mask_params)
        masked_probs = np.asarray(predict_fn(masked), dtype=np.float64)
        masked_original_class_probability = float(masked_probs[original_class_idx])
        masked_predicted_class = class_names[int(np.argmax(masked_probs))]
        drop = original_class_probability - masked_original_class_probability
        results.append(RegionPerturbationResult(
            region_name=region.name,
            original_predicted_class=original_predicted_class,
            original_class_probability=original_class_probability,
            masked_predicted_class=masked_predicted_class,
            masked_original_class_probability=masked_original_class_probability,
            probability_drop=drop,
            absolute_probability_change=abs(drop),
            masking_method=masking_method,
            cam_mean=region.cam_mean,
            cam_ratio=region.cam_ratio,
        ))
    return results


ALL_MASKING_METHODS: tuple[MaskingMethod, ...] = ("zero", "mean", "blur")


def run_all_masking_methods(
    array: np.ndarray, regions: Sequence[RegionInfo], predict_fn, class_names: list[str],
    original_class_idx: int, methods: tuple[MaskingMethod, ...] = ALL_MASKING_METHODS,
) -> dict[str, list[RegionPerturbationResult]]:
    return {
        method: run_region_perturbation(array, regions, predict_fn, class_names, original_class_idx, method)
        for method in methods
    }
