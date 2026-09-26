"""CAM-vs-perturbation agreement: does the CAM's own spatial signal line up
with the measured probability drops? Always exploratory -- n=9 regions for
CLIP, n=27 for the 3D CNN, both far too small for a validated statistic."""
from __future__ import annotations

from scipy.stats import spearmanr

from .ranking import rank_by_absolute_sensitivity, rank_supporting_regions
from .schemas import CamAgreementReport, RegionPerturbationResult

SAMPLE_SIZE_NOTE = "표본이 구역 개수뿐이므로 모든 값은 탐색적 지표입니다."


def compute_cam_perturbation_agreement(
    results: list[RegionPerturbationResult], masking_method: str, top_k: int = 3,
) -> CamAgreementReport:
    cam_ratios = [r.cam_ratio for r in results]
    drops = [r.probability_drop for r in results]
    abs_drops = [abs(d) for d in drops]

    def safe_spearman(a, b):
        if len(set(a)) < 2 or len(set(b)) < 2:
            return None
        rho, _ = spearmanr(a, b)
        return None if rho != rho else float(rho)

    rho_drop = safe_spearman(cam_ratios, drops)
    rho_abs = safe_spearman(cam_ratios, abs_drops)

    cam_top_k = {r.region_name for r in sorted(results, key=lambda r: r.cam_ratio, reverse=True)[:top_k]}
    support_top_k = {ranked.region_name for ranked in rank_supporting_regions(results)[:top_k]}
    absolute_top_k = {ranked.region_name for ranked in rank_by_absolute_sensitivity(results)[:top_k]}

    overlap_support = len(cam_top_k & support_top_k)
    union_abs = cam_top_k | absolute_top_k
    jaccard_abs = (len(cam_top_k & absolute_top_k) / len(union_abs)) if union_abs else 0.0

    median_cam = sorted(cam_ratios)[len(cam_ratios) // 2] if cam_ratios else 0.0
    high_cam_negative = [
        r.region_name for r in results if r.cam_ratio >= median_cam and r.probability_drop < 0
    ]

    return CamAgreementReport(
        masking_method=masking_method,
        spearman_cam_ratio_vs_drop=rho_drop,
        spearman_cam_ratio_vs_absolute=rho_abs,
        top3_overlap_support=overlap_support,
        top3_jaccard_absolute=jaccard_abs,
        high_cam_negative_drop_regions=high_cam_negative,
        n_regions=len(results),
    )
