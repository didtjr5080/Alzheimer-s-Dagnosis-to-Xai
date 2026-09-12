"""Three separated region rankings, replacing a single ambiguous ranking that
mixed positive and negative `probability_drop` values under one "importance"
label. A positive drop (masking decreased the original class's probability)
and a negative drop (masking increased it) have opposite meanings and must
never be shown as the same kind of "supporting" evidence.
"""
from __future__ import annotations

from .schemas import RankedRegion


def _competition_rank(sorted_results, value_fn) -> list[RankedRegion]:
    """Standard competition ranking: equal values share a rank, and the next
    distinct value's rank accounts for how many items tied ahead of it
    (e.g. 1, 1, 3 -- not 1, 1, 2)."""
    ranked = []
    prev_value = None
    prev_rank = 0
    for position, result in enumerate(sorted_results, start=1):
        value = value_fn(result)
        rank = prev_rank if prev_value is not None and value == prev_value else position
        ranked.append(RankedRegion(rank=rank, region_name=result.region_name, value=value, result=result))
        prev_value = value
        prev_rank = rank
    return ranked


def rank_supporting_regions(results) -> list[RankedRegion]:
    """Regions where masking DECREASED the original predicted class's
    probability (probability_drop > 0). This is evidence that appears to have
    supported the model's original prediction -- not proof of causation, and
    not a claim about pathology or anatomy. Zero-drop regions are excluded
    (they belong only in the absolute-sensitivity ranking)."""
    supporting = [r for r in results if r.probability_drop > 0]
    ordered = sorted(supporting, key=lambda r: r.probability_drop, reverse=True)
    return _competition_rank(ordered, lambda r: r.probability_drop)


def rank_suppressing_regions(results) -> list[RankedRegion]:
    """Regions where masking INCREASED the original predicted class's
    probability (probability_drop < 0). This is evidence that appears to have
    worked against the model's confidence in its original prediction. Ranked
    most-suppressing first (largest probability increase). Zero-drop regions
    are excluded."""
    suppressing = [r for r in results if r.probability_drop < 0]
    ordered = sorted(suppressing, key=lambda r: r.probability_drop)
    return _competition_rank(ordered, lambda r: r.probability_drop)


def rank_by_absolute_sensitivity(results) -> list[RankedRegion]:
    """All regions (including zero-drop ones) ranked by |probability_drop|,
    direction-agnostic: how much the model's output moved, regardless of
    which way."""
    ordered = sorted(results, key=lambda r: r.absolute_probability_change, reverse=True)
    return _competition_rank(ordered, lambda r: r.absolute_probability_change)
