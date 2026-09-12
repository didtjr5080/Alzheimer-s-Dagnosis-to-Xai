"""CAM-vs-perturbation agreement for a single (image, masking_method) result
set. Always exploratory: only 9 regions per image, so every correlation here
is a descriptive statistic, not a validated measurement. Disagreements (a
region with a high CAM ratio but a negative probability_drop) are preserved
and surfaced, never dropped or sign-flipped to make CAM and perturbation look
more consistent than they are.
"""
from __future__ import annotations

from scipy.stats import spearmanr

from .ranking import rank_by_absolute_sensitivity, rank_supporting_regions
from .schemas import CamAgreementReport

SAMPLE_SIZE_NOTE = "n=9 regions per image; exploratory only, not a validated statistic."


def _safe_spearman(a, b):
    if len(a) < 2 or len(set(a)) == 1 or len(set(b)) == 1:
        return None
    return float(spearmanr(a, b).correlation)


def _jaccard(set_a, set_b):
    union = set_a | set_b
    return (len(set_a & set_b) / len(union)) if union else 1.0


def compute_cam_perturbation_agreement(region_results, masking_method: str | None = None) -> CamAgreementReport:
    masking_method = masking_method or (region_results[0].masking_method if region_results else "unknown")

    cam_ratios = [r.cam_ratio for r in region_results]
    drops = [r.probability_drop for r in region_results]
    abs_changes = [r.absolute_probability_change for r in region_results]

    spearman_drop = _safe_spearman(cam_ratios, drops)
    spearman_abs = _safe_spearman(cam_ratios, abs_changes)

    cam_top3 = {r.region_name for r in sorted(region_results, key=lambda r: r.cam_ratio, reverse=True)[:3]}
    support_top3 = {r.region_name for r in rank_supporting_regions(region_results)[:3]}
    absolute_top3 = {r.region_name for r in rank_by_absolute_sensitivity(region_results)[:3]}

    cam_top3_vs_support_overlap = len(cam_top3 & support_top3)
    cam_top3_vs_absolute_jaccard = _jaccard(cam_top3, absolute_top3)

    high_cam_negative_drop = sorted(
        r.region_name for r in region_results
        if r.region_name in cam_top3 and r.probability_drop < 0
    )

    return CamAgreementReport(
        masking_method=masking_method,
        spearman_cam_ratio_vs_drop=spearman_drop,
        spearman_cam_ratio_vs_absolute=spearman_abs,
        cam_top3_vs_support_top3_overlap=cam_top3_vs_support_overlap,
        cam_top3_vs_absolute_top3_jaccard=cam_top3_vs_absolute_jaccard,
        high_cam_negative_drop_regions=high_cam_negative_drop,
        sample_size_note=SAMPLE_SIZE_NOTE,
    )
