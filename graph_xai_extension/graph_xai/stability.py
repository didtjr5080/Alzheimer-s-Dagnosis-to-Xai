"""Cross-masking-method stability of the region explanation: how much do the
`zero` / `mean` / `blur` masking results agree with each other? A single
masking method's result (e.g. `mean` alone) must never be presented as "the"
explanation -- see work order section 8/9.

Thresholds are configuration, not a medically validated cutoff. They only
describe how consistent this project's own numbers are with each other.
"""
from __future__ import annotations

from itertools import combinations

from scipy.stats import kendalltau, spearmanr

from .ranking import rank_by_absolute_sensitivity, rank_supporting_regions
from .schemas import StabilityReport

DEFAULT_STABILITY_THRESHOLDS = {
    "high_min_mean_spearman": 0.70,
    "high_min_sign_agreement": 0.80,
    "moderate_min_mean_spearman": 0.40,
    # Not in the work order's literal example, which only gates "moderate" on
    # mean Spearman. Added because Spearman correlation on |probability_drop|
    # is blind to sign: a result set where every region's drop flipped sign
    # between masking methods (worst case for interpretability) can still
    # score a perfect absolute-value correlation. Requiring some minimum sign
    # agreement even for "moderate" avoids calling that case merely "보통".
    "moderate_min_sign_agreement": 0.50,
}

STABILITY_METHODOLOGY_NOTE = (
    "This stability rating is a project engineering threshold on internal numerical agreement "
    "between masking methods. It is not a medically validated criterion."
)


def _align_by_region(results_a, results_b, value_attr):
    by_name_b = {r.region_name: r for r in results_b}
    names, values_a, values_b = [], [], []
    for r_a in results_a:
        r_b = by_name_b.get(r_a.region_name)
        if r_b is None:
            continue
        names.append(r_a.region_name)
        values_a.append(getattr(r_a, value_attr))
        values_b.append(getattr(r_b, value_attr))
    return names, values_a, values_b


def _safe_spearman(a, b):
    if len(a) < 2 or len(set(a)) == 1 or len(set(b)) == 1:
        return float("nan")
    return float(spearmanr(a, b).correlation)


def _safe_kendall(a, b):
    if len(a) < 2 or len(set(a)) == 1 or len(set(b)) == 1:
        return float("nan")
    return float(kendalltau(a, b).correlation)


def _top_k_names(ranked_regions, k):
    return {r.region_name for r in ranked_regions[:k]}


def _jaccard(set_a, set_b):
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def compare_masking_methods(results_by_method: dict, thresholds: dict | None = None) -> StabilityReport:
    """`results_by_method`: {masking_method_name: [9 RegionPerturbationResult, ...]}."""
    thresholds = {**DEFAULT_STABILITY_THRESHOLDS, **(thresholds or {})}
    methods = list(results_by_method.keys())
    if len(methods) < 2:
        raise ValueError("Need at least 2 masking methods to compare stability")

    pairwise_spearman_support: dict = {}
    pairwise_spearman_absolute: dict = {}
    pairwise_kendall_support: dict = {}
    pairwise_kendall_absolute: dict = {}
    top1_overlap_support: dict = {}
    top3_jaccard_support: dict = {}
    top1_overlap_absolute: dict = {}
    top3_jaccard_absolute: dict = {}

    for method_a, method_b in combinations(methods, 2):
        pair_key = f"{method_a}_vs_{method_b}"
        results_a, results_b = results_by_method[method_a], results_by_method[method_b]

        _, drop_a, drop_b = _align_by_region(results_a, results_b, "probability_drop")
        pairwise_spearman_support[pair_key] = _safe_spearman(drop_a, drop_b)
        pairwise_kendall_support[pair_key] = _safe_kendall(drop_a, drop_b)

        _, abs_a, abs_b = _align_by_region(results_a, results_b, "absolute_probability_change")
        pairwise_spearman_absolute[pair_key] = _safe_spearman(abs_a, abs_b)
        pairwise_kendall_absolute[pair_key] = _safe_kendall(abs_a, abs_b)

        support_a = rank_supporting_regions(results_a)
        support_b = rank_supporting_regions(results_b)
        top1_a = support_a[0].region_name if support_a else None
        top1_b = support_b[0].region_name if support_b else None
        top1_overlap_support[pair_key] = (top1_a is not None and top1_a == top1_b)
        top3_jaccard_support[pair_key] = _jaccard(_top_k_names(support_a, 3), _top_k_names(support_b, 3))

        absolute_a = rank_by_absolute_sensitivity(results_a)
        absolute_b = rank_by_absolute_sensitivity(results_b)
        top1_overlap_absolute[pair_key] = (absolute_a[0].region_name == absolute_b[0].region_name)
        top3_jaccard_absolute[pair_key] = _jaccard(_top_k_names(absolute_a, 3), _top_k_names(absolute_b, 3))

    all_region_names = {r.region_name for results in results_by_method.values() for r in results}
    sign_matches, sign_total = 0, 0
    flipped_regions = []
    drop_range_by_region: dict = {}
    for region_name in all_region_names:
        drops = []
        for method in methods:
            match = next((r for r in results_by_method[method] if r.region_name == region_name), None)
            if match is not None:
                drops.append(match.probability_drop)
        if len(drops) < 2:
            continue
        drop_range_by_region[region_name] = max(drops) - min(drops)
        for d_a, d_b in combinations(drops, 2):
            sign_total += 1
            same_sign = (d_a > 0 and d_b > 0) or (d_a < 0 and d_b < 0) or (d_a == 0 and d_b == 0)
            if same_sign:
                sign_matches += 1
            else:
                flipped_regions.append(region_name)

    sign_agreement_rate = (sign_matches / sign_total) if sign_total else float("nan")
    flipped_regions = sorted(set(flipped_regions))
    most_unstable_region = max(drop_range_by_region, key=drop_range_by_region.get) if drop_range_by_region else None

    all_spearman_values = [v for v in pairwise_spearman_absolute.values() if v == v]  # drop NaN
    mean_spearman = (sum(all_spearman_values) / len(all_spearman_values)) if all_spearman_values else float("nan")

    spearman_ok = mean_spearman == mean_spearman  # not NaN
    sign_ok = sign_agreement_rate == sign_agreement_rate  # not NaN

    if (spearman_ok and mean_spearman >= thresholds["high_min_mean_spearman"]
            and sign_ok and sign_agreement_rate >= thresholds["high_min_sign_agreement"]):
        verdict = "높음"
    elif (spearman_ok and mean_spearman >= thresholds["moderate_min_mean_spearman"]
            and sign_ok and sign_agreement_rate >= thresholds["moderate_min_sign_agreement"]):
        verdict = "보통"
    else:
        verdict = "낮음"

    return StabilityReport(
        masking_methods=methods,
        pairwise_spearman_support=pairwise_spearman_support,
        pairwise_spearman_absolute=pairwise_spearman_absolute,
        pairwise_kendall_support=pairwise_kendall_support,
        pairwise_kendall_absolute=pairwise_kendall_absolute,
        mean_spearman=mean_spearman,
        top1_overlap_support=top1_overlap_support,
        top3_jaccard_support=top3_jaccard_support,
        top1_overlap_absolute=top1_overlap_absolute,
        top3_jaccard_absolute=top3_jaccard_absolute,
        sign_agreement_rate=sign_agreement_rate,
        sign_flipped_regions=flipped_regions,
        most_unstable_region=most_unstable_region,
        stability_verdict=verdict,
        stability_thresholds=thresholds,
    )
