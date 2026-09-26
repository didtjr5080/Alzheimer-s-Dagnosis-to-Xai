"""Cross-masking-method stability: how much do the zero/mean/blur results
agree with each other? Same threshold values as
`graph_xai_extension/graph_xai/stability.py` for consistency across the
project's reports -- these are a project engineering threshold, not a
medically validated one."""
from __future__ import annotations

from scipy.stats import spearmanr

from .schemas import RegionPerturbationResult, StabilityReport

DEFAULT_STABILITY_THRESHOLDS = {
    "high_min_mean_spearman": 0.70,
    "high_min_sign_agreement": 0.80,
    "moderate_min_mean_spearman": 0.40,
    "moderate_min_sign_agreement": 0.50,
}

STABILITY_METHODOLOGY_NOTE = (
    "이 안정성 판정은 프로젝트 운영 기준이며 의학적으로 검증된 기준이 아닙니다. "
    "'mean' 결과 하나만으로 최종 설명을 확정하지 않았습니다."
)


def compare_masking_methods(
    results_by_method: dict[str, list[RegionPerturbationResult]], thresholds: dict | None = None,
) -> StabilityReport:
    thresholds = thresholds or DEFAULT_STABILITY_THRESHOLDS
    methods = list(results_by_method.keys())
    region_names = [r.region_name for r in next(iter(results_by_method.values()))]

    drop_by_region = {
        name: {m: next(r.probability_drop for r in results_by_method[m] if r.region_name == name) for m in methods}
        for name in region_names
    }

    # Mean pairwise Spearman correlation on |probability_drop| (absolute
    # sensitivity ranking) across every pair of masking methods.
    pairwise_spearman = []
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            a = [abs(drop_by_region[name][methods[i]]) for name in region_names]
            b = [abs(drop_by_region[name][methods[j]]) for name in region_names]
            rho, _ = spearmanr(a, b)
            pairwise_spearman.append(0.0 if rho != rho else float(rho))  # NaN-safe
    mean_spearman = sum(pairwise_spearman) / len(pairwise_spearman) if pairwise_spearman else float("nan")

    def sign(value: float) -> int:
        return 0 if value == 0 else (1 if value > 0 else -1)

    sign_flipped = []
    agreement_count = 0
    for name in region_names:
        signs = {sign(drop_by_region[name][m]) for m in methods}
        if len(signs) == 1:
            agreement_count += 1
        else:
            sign_flipped.append(name)
    sign_agreement_rate = agreement_count / len(region_names) if region_names else float("nan")

    most_unstable = None
    if region_names:
        ranges = {
            name: max(drop_by_region[name].values()) - min(drop_by_region[name].values())
            for name in region_names
        }
        most_unstable = max(ranges, key=ranges.get)

    if mean_spearman >= thresholds["high_min_mean_spearman"] and sign_agreement_rate >= thresholds["high_min_sign_agreement"]:
        verdict = "높음"
    elif mean_spearman >= thresholds["moderate_min_mean_spearman"] and sign_agreement_rate >= thresholds["moderate_min_sign_agreement"]:
        verdict = "보통"
    else:
        verdict = "낮음"

    return StabilityReport(
        masking_methods=methods,
        mean_spearman=mean_spearman,
        sign_agreement_rate=sign_agreement_rate,
        sign_flipped_regions=sign_flipped,
        most_unstable_region=most_unstable,
        stability_verdict=verdict,
        stability_thresholds=thresholds,
    )
