from __future__ import annotations

import csv
from pathlib import Path

from ..contracts import AdapterHealth, PredictionResult


CLASS_NAMES = ["CN", "MCI", "AD"]


class MergedReferenceAdapter:
    def __init__(
        self,
        model_id: str,
        csv_path: Path,
        probability_columns: list[str],
        label_column: str | None,
        prediction_column: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.csv_path = csv_path
        self.probability_columns = probability_columns
        self.label_column = label_column
        self.prediction_column = prediction_column

    def health(self) -> AdapterHealth:
        if not self.csv_path.exists():
            return AdapterHealth(
                model_id=self.model_id,
                status="UNAVAILABLE_MISSING_FILES",
                execution_mode="reference_csv_by_scan_id",
                reason=f"Missing CSV: {self.csv_path}",
                available_inputs=[],
            )
        header = self._header()
        missing = [name for name in ["scan_id", *self.probability_columns] if name not in header]
        if missing:
            return AdapterHealth(
                model_id=self.model_id,
                status="FAILED_VALIDATION",
                execution_mode="reference_csv_by_scan_id",
                reason=f"Missing required columns: {missing}",
                available_inputs=[],
            )
        return AdapterHealth(
            model_id=self.model_id,
            status="AVAILABLE",
            execution_mode="reference_csv_by_scan_id",
            reason="Verified reference CSV predictions.",
            available_inputs=["scan_id"],
        )

    def _header(self) -> list[str]:
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            return next(reader)

    def _find_row(self, scan_id: str) -> dict[str, str]:
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("scan_id") == scan_id:
                    return row
        raise KeyError(f"scan_id not found in {self.csv_path.name}: {scan_id}")

    def predict_by_scan_id(self, scan_id: str) -> PredictionResult:
        health = self.health()
        if health.status != "AVAILABLE":
            raise RuntimeError(health.reason or f"{self.model_id} is not available")
        row = self._find_row(scan_id)
        probabilities = [float(row[column]) for column in self.probability_columns]
        pred_index = max(range(len(probabilities)), key=lambda index: probabilities[index])
        predicted_class = CLASS_NAMES[pred_index]
        if self.prediction_column and row.get(self.prediction_column, "") != "":
            predicted_class = CLASS_NAMES[int(row[self.prediction_column])]
        result = PredictionResult(
            model_id=self.model_id,
            model_version=self.csv_path.name,
            level="subject",
            sample_id=scan_id,
            class_names=CLASS_NAMES,
            probabilities=probabilities,
            predicted_class=predicted_class,
            warnings=[] if self.prediction_column is None else ["Prediction class is replayed from reference CSV."],
            provenance={"csv_path": str(self.csv_path), "label": row.get(self.label_column or "", None)},
        )
        result.validate(tolerance=1e-3)
        return result


def build_default_reference_adapters(repo_root: Path) -> list[MergedReferenceAdapter]:
    root = repo_root / "merged_project"
    return [
        MergedReferenceAdapter("merged_clip_reference", root / "clip_test_probs.csv", ["clip_CN", "clip_MCI", "clip_AD"], "label"),
        MergedReferenceAdapter("cnn3d_reference", root / "3dcnn_test_probs.csv", ["prob_CN", "prob_MCI", "prob_AD"], "true_label"),
        MergedReferenceAdapter("gbm_reference", root / "gbm_hierarchical_test_probs.csv", ["gbm_CN", "gbm_MCI", "gbm_AD"], "true_label"),
    ]
