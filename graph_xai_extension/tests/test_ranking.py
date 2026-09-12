from __future__ import annotations

import pytest

from graph_xai.ranking import rank_by_absolute_sensitivity, rank_supporting_regions, rank_suppressing_regions
from graph_xai.schemas import RegionPerturbationResult


def _result(name, drop):
    return RegionPerturbationResult(
        region_name=name,
        original_predicted_class="AD",
        original_class_probability=0.7,
        masked_predicted_class="AD",
        masked_original_class_probability=0.7 - drop,
        probability_drop=drop,
        absolute_probability_change=abs(drop),
        masking_method="mean",
        cam_mean=0.1,
        cam_ratio=0.1,
    )


def test_supporting_regions_are_only_positive_drop_descending():
    results = [_result("a", 0.3), _result("b", -0.2), _result("c", 0.5), _result("d", 0.0)]
    ranking = rank_supporting_regions(results)
    assert [r.region_name for r in ranking] == ["c", "a"]
    assert all(r.value > 0 for r in ranking)
    assert [r.rank for r in ranking] == [1, 2]


def test_suppressing_regions_are_only_negative_drop_most_suppressing_first():
    results = [_result("a", 0.3), _result("b", -0.2), _result("c", -0.6), _result("d", 0.0)]
    ranking = rank_suppressing_regions(results)
    assert [r.region_name for r in ranking] == ["c", "b"]
    assert all(r.value < 0 for r in ranking)


def test_zero_drop_region_excluded_from_support_and_suppress_but_present_in_absolute():
    results = [_result("a", 0.0), _result("b", 0.4)]
    assert "a" not in [r.region_name for r in rank_supporting_regions(results)]
    assert "a" not in [r.region_name for r in rank_suppressing_regions(results)]
    absolute = rank_by_absolute_sensitivity(results)
    assert "a" in [r.region_name for r in absolute]
    # zero is the smallest magnitude, so it ranks last among these two
    assert absolute[-1].region_name == "a"
    assert absolute[-1].value == 0.0


def test_absolute_sensitivity_is_direction_agnostic():
    results = [_result("a", 0.4), _result("b", -0.4)]
    absolute = rank_by_absolute_sensitivity(results)
    assert absolute[0].rank == absolute[1].rank == 1  # tied magnitude
    assert {r.region_name for r in absolute} == {"a", "b"}


def test_tied_values_share_competition_rank_and_next_rank_skips():
    results = [_result("a", 0.5), _result("b", 0.5), _result("c", 0.2)]
    ranking = rank_supporting_regions(results)
    ranks = {r.region_name: r.rank for r in ranking}
    assert ranks["a"] == 1
    assert ranks["b"] == 1
    assert ranks["c"] == 3  # not 2 -- two items tied for 1st


def test_all_positive_or_all_negative_edge_cases_do_not_crash():
    all_positive = [_result("a", 0.2), _result("b", 0.4)]
    assert rank_suppressing_regions(all_positive) == []
    assert len(rank_supporting_regions(all_positive)) == 2

    all_negative = [_result("a", -0.2), _result("b", -0.4)]
    assert rank_supporting_regions(all_negative) == []
    assert len(rank_suppressing_regions(all_negative)) == 2


def test_ranked_region_carries_back_reference_to_original_result():
    results = [_result("a", 0.3)]
    ranking = rank_supporting_regions(results)
    assert ranking[0].result is results[0]
