from __future__ import annotations

import math

import pytest

from graph_xai.schemas import REGION_NAMES_3X3, RegionPerturbationResult
from graph_xai.stability import compare_masking_methods


def _make_results(drops, method):
    return [
        RegionPerturbationResult(
            region_name=name,
            original_predicted_class="AD",
            original_class_probability=0.7,
            masked_predicted_class="AD",
            masked_original_class_probability=0.7 - drop,
            probability_drop=drop,
            absolute_probability_change=abs(drop),
            masking_method=method,
            cam_mean=0.1,
            cam_ratio=1.0 / 9,
        )
        for name, drop in zip(REGION_NAMES_3X3, drops)
    ]


def test_identical_results_across_methods_are_perfectly_stable():
    drops = [0.5, 0.4, 0.3, 0.2, 0.1, -0.1, -0.2, -0.3, -0.4]
    results_by_method = {
        "zero": _make_results(drops, "zero"),
        "mean": _make_results(drops, "mean"),
        "blur": _make_results(drops, "blur"),
    }
    report = compare_masking_methods(results_by_method)
    assert report.sign_agreement_rate == pytest.approx(1.0)
    assert report.mean_spearman == pytest.approx(1.0)
    assert report.stability_verdict == "높음"
    assert report.sign_flipped_regions == []


def test_fully_sign_flipped_results_are_rated_unstable_even_if_magnitudes_match():
    # abs(drop) is identical either way, so magnitude-only correlation would
    # look "perfect" here -- the verdict must not be fooled by that; full
    # sign disagreement has to pull the verdict down to 낮음.
    drops_a = [0.5, 0.4, 0.3, 0.2, 0.1, -0.1, -0.2, -0.3, -0.4]
    drops_b = [-d for d in drops_a]
    results_by_method = {
        "zero": _make_results(drops_a, "zero"),
        "mean": _make_results(drops_b, "mean"),
    }
    report = compare_masking_methods(results_by_method)
    assert report.sign_agreement_rate == pytest.approx(0.0)
    assert report.mean_spearman == pytest.approx(1.0)  # magnitudes alone DO correlate perfectly
    assert report.stability_verdict == "낮음"  # but the verdict still isn't fooled
    assert set(report.sign_flipped_regions) == set(REGION_NAMES_3X3)


def test_most_unstable_region_has_largest_range_across_methods():
    drops_a = [0.1] * 9
    drops_b = [0.1] * 9
    drops_b[3] = -0.9  # region index 3 = middle_left, flips hard
    results_by_method = {"zero": _make_results(drops_a, "zero"), "mean": _make_results(drops_b, "mean")}
    report = compare_masking_methods(results_by_method)
    assert report.most_unstable_region == REGION_NAMES_3X3[3]


def test_requires_at_least_two_methods():
    with pytest.raises(ValueError):
        compare_masking_methods({"zero": _make_results([0.1] * 9, "zero")})


def test_constant_values_do_not_crash_spearman():
    drops = [0.2] * 9
    results_by_method = {"zero": _make_results(drops, "zero"), "mean": _make_results(drops, "mean")}
    report = compare_masking_methods(results_by_method)
    # spearman is undefined for constant arrays -> NaN, must not raise
    for value in report.pairwise_spearman_absolute.values():
        assert value != value or isinstance(value, float)  # NaN or a float, never a crash


def test_custom_thresholds_are_respected():
    drops = [0.5, 0.4, 0.3, 0.2, 0.1, -0.1, -0.2, -0.3, -0.4]
    results_by_method = {"zero": _make_results(drops, "zero"), "mean": _make_results(drops, "mean")}
    report = compare_masking_methods(results_by_method, thresholds={"high_min_mean_spearman": 1.5})
    assert report.stability_verdict != "높음"  # impossible threshold forces it down
    assert report.stability_thresholds["high_min_mean_spearman"] == 1.5


def test_verdict_is_not_asserted_as_medically_validated():
    from graph_xai.stability import STABILITY_METHODOLOGY_NOTE
    assert "not" in STABILITY_METHODOLOGY_NOTE.lower() or "아니" in STABILITY_METHODOLOGY_NOTE
