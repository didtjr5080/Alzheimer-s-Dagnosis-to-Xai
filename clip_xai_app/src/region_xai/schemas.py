"""Shared dataclasses, usable for both the 2D (CLIP, 3x3=9 regions) and 3D
(3D CNN, 3x3x3=27 regions) branches."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RegionInfo:
    """Pure grid-cell geometry + CAM stats. `bbox` is a tuple of (start, end)
    pairs, one per spatial axis (2 for CLIP, 3 for 3D CNN). No anatomical
    meaning is implied by `name` -- these are array-index grid cells."""

    name: str
    bbox: tuple[tuple[int, int], ...]
    cam_mean: float
    cam_sum: float
    cam_ratio: float


@dataclass(frozen=True)
class RegionPerturbationResult:
    region_name: str
    original_predicted_class: str
    original_class_probability: float
    masked_predicted_class: str
    masked_original_class_probability: float
    probability_drop: float
    absolute_probability_change: float
    masking_method: str
    cam_mean: float
    cam_ratio: float


@dataclass(frozen=True)
class RankedRegion:
    """One row of a support/suppress/sensitivity ranking. `rank` uses
    competition ranking (ties share a rank)."""

    rank: int
    region_name: str
    value: float
    result: RegionPerturbationResult


@dataclass(frozen=True)
class StabilityReport:
    masking_methods: list
    mean_spearman: float
    sign_agreement_rate: float
    sign_flipped_regions: list
    most_unstable_region: str | None
    stability_verdict: str
    stability_thresholds: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CamAgreementReport:
    masking_method: str
    spearman_cam_ratio_vs_drop: float | None
    spearman_cam_ratio_vs_absolute: float | None
    top3_overlap_support: int
    top3_jaccard_absolute: float
    high_cam_negative_drop_regions: list
    n_regions: int
