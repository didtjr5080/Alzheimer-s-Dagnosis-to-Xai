"""Three separated rankings -- supporting (probability_drop > 0), suppressing
(probability_drop < 0), and absolute sensitivity (|probability_drop|,
direction-agnostic) -- using competition ranking (ties share a rank; the
next rank skips accordingly, e.g. 1,1,3). Never merged into one "importance"
list."""
from __future__ import annotations

from .schemas import RankedRegion, RegionPerturbationResult


def _competition_rank(sorted_results: list[RegionPerturbationResult], value_fn) -> list[RankedRegion]:
    ranked = []
    previous_value = None
    previous_rank = 0
    for index, result in enumerate(sorted_results, start=1):
        value = value_fn(result)
        rank = previous_rank if value == previous_value else index
        ranked.append(RankedRegion(rank=rank, region_name=result.region_name, value=value, result=result))
        previous_value, previous_rank = value, rank
    return ranked


def rank_supporting_regions(results: list[RegionPerturbationResult]) -> list[RankedRegion]:
    supporting = [r for r in results if r.probability_drop > 0]
    supporting.sort(key=lambda r: r.probability_drop, reverse=True)
    return _competition_rank(supporting, lambda r: r.probability_drop)


def rank_suppressing_regions(results: list[RegionPerturbationResult]) -> list[RankedRegion]:
    suppressing = [r for r in results if r.probability_drop < 0]
    suppressing.sort(key=lambda r: r.probability_drop)
    return _competition_rank(suppressing, lambda r: r.probability_drop)


def rank_by_absolute_sensitivity(results: list[RegionPerturbationResult]) -> list[RankedRegion]:
    ordered = sorted(results, key=lambda r: abs(r.probability_drop), reverse=True)
    return _competition_rank(ordered, lambda r: abs(r.probability_drop))
