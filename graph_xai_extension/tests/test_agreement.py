from __future__ import annotations

import pytest

from graph_xai.agreement import compute_cam_perturbation_agreement
from graph_xai.schemas import REGION_NAMES_3X3, RegionPerturbationResult


def _make_results(cam_ratios, drops):
    return [
        RegionPerturbationResult(
            region_name=name,
            original_predicted_class="AD",
            original_class_probability=0.7,
            masked_predicted_class="AD",
            masked_original_class_probability=0.7 - drop,
            probability_drop=drop,
            absolute_probability_change=abs(drop),
            masking_method="mean",
            cam_mean=cam_ratio,
            cam_ratio=cam_ratio,
        )
        for name, cam_ratio, drop in zip(REGION_NAMES_3X3, cam_ratios, drops)
    ]


def test_perfectly_correlated_cam_and_drop_gives_high_positive_spearman():
    cam_ratios = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    drops = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    results = _make_results(cam_ratios, drops)
    report = compute_cam_perturbation_agreement(results, masking_method="mean")
    assert report.spearman_cam_ratio_vs_drop == pytest.approx(1.0)
    assert report.spearman_cam_ratio_vs_absolute == pytest.approx(1.0)
    assert report.cam_top3_vs_support_top3_overlap == 3
    assert report.high_cam_negative_drop_regions == []


def test_high_cam_but_negative_drop_is_preserved_not_deleted_or_flipped():
    cam_ratios = [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # top_left has highest CAM
    drops = [-0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # but masking it INCREASED probability
    results = _make_results(cam_ratios, drops)
    report = compute_cam_perturbation_agreement(results, masking_method="mean")
    assert "top_left" in report.high_cam_negative_drop_regions
    # the underlying result for top_left must be untouched (still negative, not flipped to positive)
    top_left_result = next(r for r in results if r.region_name == "top_left")
    assert top_left_result.probability_drop == -0.5


def test_sample_size_note_present_and_exploratory():
    results = _make_results([0.1] * 9, [0.1] * 9)
    report = compute_cam_perturbation_agreement(results)
    assert "n=9" in report.sample_size_note
    assert "exploratory" in report.sample_size_note.lower()


def test_constant_cam_ratio_returns_none_correlation_not_a_crash():
    results = _make_results([0.111] * 9, [0.1, 0.2, -0.1, 0.3, -0.2, 0.05, -0.05, 0.15, -0.15])
    report = compute_cam_perturbation_agreement(results)
    assert report.spearman_cam_ratio_vs_drop is None


def test_masking_method_defaults_to_the_results_own_method():
    results = _make_results([0.1] * 9, [0.1] * 9)
    report = compute_cam_perturbation_agreement(results)
    assert report.masking_method == "mean"
