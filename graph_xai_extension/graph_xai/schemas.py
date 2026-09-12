"""Shared dataclasses for the Graph XAI extension."""
from __future__ import annotations

from dataclasses import dataclass, field


REGION_NAMES_3X3 = (
    "top_left", "top_center", "top_right",
    "middle_left", "middle_center", "middle_right",
    "bottom_left", "bottom_center", "bottom_right",
)


@dataclass(frozen=True)
class RegionInfo:
    """Pure spatial + CAM statistics for one grid region. No anatomical meaning."""

    name: str
    row: int
    col: int
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    cam_mean: float
    cam_max: float
    cam_sum: float
    cam_ratio: float
    image_mean_brightness: float
    area_pixels: int


@dataclass(frozen=True)
class MaskingRecord:
    method: str
    params: dict = field(default_factory=dict)


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
    """One row of a support/suppress/sensitivity ranking table. `rank` uses
    competition ranking (ties share a rank; the next rank skips accordingly)."""

    rank: int
    region_name: str
    value: float
    result: RegionPerturbationResult


@dataclass(frozen=True)
class ClassMappingReport:
    """Evidence that class order was read from the model, never assumed."""

    classes_idx: list
    class_names_ordered: list
    n_classes: int
    is_binary: bool
    probability_array_order_matches_class_names: bool
    probability_sum_ok: bool
    probability_sum_value: float
    unexpected_classes: list
    ok: bool
    detail: str


@dataclass(frozen=True)
class StabilityReport:
    """Cross-masking-method agreement for one image's region rankings."""

    masking_methods: list
    pairwise_spearman_support: dict
    pairwise_spearman_absolute: dict
    pairwise_kendall_support: dict
    pairwise_kendall_absolute: dict
    mean_spearman: float
    top1_overlap_support: dict
    top3_jaccard_support: dict
    top1_overlap_absolute: dict
    top3_jaccard_absolute: dict
    sign_agreement_rate: float
    sign_flipped_regions: list
    most_unstable_region: str | None
    stability_verdict: str
    stability_thresholds: dict


@dataclass(frozen=True)
class CamAgreementReport:
    """CAM-vs-perturbation agreement for one (image, masking_method) pair.
    n=9 regions only -- always exploratory, never a validated statistic."""

    masking_method: str
    spearman_cam_ratio_vs_drop: float | None
    spearman_cam_ratio_vs_absolute: float | None
    cam_top3_vs_support_top3_overlap: int
    cam_top3_vs_absolute_top3_jaccard: float
    high_cam_negative_drop_regions: list
    sample_size_note: str = "n=9 regions per image; exploratory only, not a validated statistic."


@dataclass(frozen=True)
class BatchSampleResult:
    path: str  # basename only, never a full local path
    subject_id: str
    predicted_class: str
    original_class_probability: float
    probs: dict
    region_results: tuple
    supporting_top_region: str | None
    suppressing_top_region: str | None
    true_label: str | None = None


@dataclass(frozen=True)
class BatchSummary:
    total_images: int
    total_subjects: int
    masking_method: str
    class_counts_images: dict
    class_counts_subjects: dict
    predicted_class_distribution_images: dict
    predicted_class_distribution_subjects: dict
    supporting_region_frequency_by_class: dict
    suppressing_region_frequency_by_class: dict
    mean_probability_drop_by_region: dict
    cam_perturbation_mean_agreement: dict
    classification_metrics: dict | None
    failed_samples: list
    samples: tuple
    dedup_note: str = (
        "Class distributions and any classification metrics are computed once per subject "
        "(subject-averaged probabilities), not once per slice, so repeated slices of the same "
        "subject are not double-counted as independent samples."
    )


@dataclass(frozen=True)
class GraphXAIRunResult:
    region_info: tuple
    perturbation_results: tuple
    class_names: list
    original_predicted_class: str
    original_class_probability: float
    masking_method: str
    grid_size: int
    diagonal_edges: bool
    mock_mode: bool
    device: str
    elapsed_seconds: float
