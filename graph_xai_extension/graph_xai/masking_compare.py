"""Run all three masking methods (`zero`, `mean`, `blur`) on the same image
and assemble the region-by-region comparison table. `mean` alone must never
be treated as the final explanation -- see work order section 8.
"""
from __future__ import annotations

from .perturbation import run_region_perturbation

ALL_MASKING_METHODS = ("zero", "mean", "blur")


def run_all_masking_methods(
    image, cam, predict_fn, class_names, original_class_idx,
    grid_size: int = 3, methods=ALL_MASKING_METHODS,
) -> dict:
    """Returns {masking_method: [9 RegionPerturbationResult, ...]}."""
    return {
        method: run_region_perturbation(
            image=image, cam=cam, predict_fn=predict_fn, class_names=class_names,
            original_class_idx=original_class_idx, masking_method=method, grid_size=grid_size,
        )
        for method in methods
    }


def build_masking_comparison_table(results_by_method: dict) -> list[dict]:
    """One row per region, with each masking method's probability_drop and
    absolute_probability_change as separate columns, for a side-by-side view."""
    methods = list(results_by_method.keys())
    if not methods:
        return []
    region_names = [r.region_name for r in results_by_method[methods[0]]]

    rows = []
    for name in region_names:
        row = {"region_name": name}
        for method in methods:
            match = next((r for r in results_by_method[method] if r.region_name == name), None)
            row[f"{method}_probability_drop"] = match.probability_drop if match else None
            row[f"{method}_absolute_change"] = match.absolute_probability_change if match else None
            row[f"{method}_masked_predicted_class"] = match.masked_predicted_class if match else None
        rows.append(row)
    return rows
