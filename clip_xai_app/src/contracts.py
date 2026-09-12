from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


AdapterStatus = Literal[
    "AVAILABLE",
    "UNAVAILABLE_MISSING_FILES",
    "BLOCKED_MODEL_CONTRACT",
    "BLOCKED_PREPROCESSING_CONTRACT",
    "FAILED_VALIDATION",
]

PredictionLevel = Literal["slice", "subject"]


@dataclass(frozen=True)
class PredictionResult:
    model_id: str
    model_version: str | None
    level: PredictionLevel
    sample_id: str
    class_names: list[str]
    probabilities: list[float]
    predicted_class: str
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def validate(self, tolerance: float = 1e-4) -> None:
        if len(self.class_names) != len(self.probabilities):
            raise ValueError("class_names and probabilities must have the same length")
        if self.predicted_class not in self.class_names:
            raise ValueError(f"predicted_class is not in class_names: {self.predicted_class}")
        if not all(isinstance(value, (int, float)) for value in self.probabilities):
            raise ValueError("probabilities must be numeric")
        if not all(value == value and value not in (float("inf"), float("-inf")) for value in self.probabilities):
            raise ValueError("probabilities must be finite")
        total = float(sum(self.probabilities))
        if abs(total - 1.0) > tolerance:
            raise ValueError(f"probabilities must sum to 1 within {tolerance}: {total}")


@dataclass(frozen=True)
class XAIResult:
    model_id: str
    target_class: str
    target_score_type: str
    method_name: str
    source_level: PredictionLevel
    heatmap_2d: Any | None = None
    heatmap_3d: Any | None = None
    feature_contributions: list[dict[str, Any]] = field(default_factory=list)
    model_contributions: list[dict[str, Any]] = field(default_factory=list)
    representative_inputs: list[str] = field(default_factory=list)
    validation_metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.target_class:
            raise ValueError("target_class is required")
        if not self.target_score_type:
            raise ValueError("target_score_type is required")
        if self.heatmap_2d is None and self.heatmap_3d is None and not self.feature_contributions and not self.model_contributions:
            raise ValueError("XAIResult must contain heatmap, feature contributions, or model contributions")


@dataclass(frozen=True)
class AdapterHealth:
    model_id: str
    status: AdapterStatus
    execution_mode: str
    reason: str | None = None
    available_inputs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SampleIdentity:
    dataset_id: str
    subject_id: str
    session_or_visit_id: str | None
    scan_id: str
    series_id: str | None
    source_relative_path: str
    source_sha256: str
    input_level: str
    slice_axis: str | None
    slice_index: int | None


@dataclass(frozen=True)
class XAIProvenance:
    analysis_id: str
    dataset_id: str
    subject_id: str
    session_or_visit_id: str | None
    scan_id: str
    series_id: str | None
    slice_axis: str | None
    slice_index: int
    slice_filename: str
    source_relative_path: str
    source_sha256: str
    true_label: str | None
    predicted_class: str
    prediction_correct: bool | None
    target_class: str
    target_class_index: int
    target_score_type: str
    method_name: str
    model_id: str
    model_version: str | None
    checkpoint_sha256: str
    code_commit: str
    preprocessing_version: str
    xai_config_hash: str
    run_id: str
    class_names: tuple[str, ...]
    slice_probabilities: dict[str, float]
    subject_probabilities: dict[str, float] | None
    original_sha256: str | None = None
    raw_heatmap_sha256: str | None = None
    normalized_heatmap_sha256: str | None = None
    heatmap_png_sha256: str | None = None
    overlay_sha256: str | None = None


@dataclass(frozen=True)
class HeatmapValidation:
    mask_method: str
    mask_threshold: float | None
    brain_fraction: float
    background_fraction: float
    foreground_mean: float
    background_mean: float
    foreground_background_ratio: float
    background_activation_fraction: float
    top_percent: float
    top_activation_background_fraction: float
    max_activation_xy: tuple[int, int]
    max_activation_inside_brain: bool
    finite: bool
    nonconstant: bool
    warnings: tuple[str, ...]
    qc_status: str


@dataclass(frozen=True)
class XAIArtifact:
    provenance: XAIProvenance
    validation: HeatmapValidation
    original_path: Path
    raw_heatmap_npy_path: Path
    normalized_heatmap_npy_path: Path
    normalized_heatmap_png_path: Path
    overlay_path: Path
    metadata_json_path: Path
