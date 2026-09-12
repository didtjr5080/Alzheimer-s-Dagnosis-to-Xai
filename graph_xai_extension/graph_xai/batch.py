"""Multi-sample batch evaluation. Accepts a list of file paths, a directory,
or a CSV manifest (columns: `image_path`, optional `true_label`) -- whichever
is safest against the caller's actual input, per work order section 11.

Subject-level de-duplication: a subject id is inferred from filenames that
follow the existing project's `OASxxxxx_MR_dxxxx_corNNN.png` convention (see
`clip_xai_app/src/report.py::infer_subject_id`, reimplemented here
independently rather than imported, since this module must not depend on
`clip_xai_app`). Class distributions and classification metrics are computed
once per subject (using that subject's mean probability vector across its
slices), not once per slice, so multiple slices of one subject are never
double-counted as independent samples.
"""
from __future__ import annotations

import csv
import re
import statistics
from pathlib import Path

import numpy as np
from PIL import Image

from .agreement import compute_cam_perturbation_agreement
from .cam_adapter import get_original_prediction_and_cam
from .perturbation import run_region_perturbation
from .ranking import rank_suppressing_regions, rank_supporting_regions
from .schemas import BatchSampleResult, BatchSummary

_SUBJECT_ID_PATTERN = re.compile(r"^(OAS\d+_MR_d\d+)_cor\d+\.(png|jpg|jpeg)$", re.IGNORECASE)


def infer_subject_id(filename: str) -> str | None:
    match = _SUBJECT_ID_PATTERN.match(Path(filename).name)
    return match.group(1) if match else None


def discover_batch_inputs(source) -> list[dict]:
    """Normalizes any of the 3 supported input forms into
    `[{"image_path": str, "true_label": str | None}, ...]`."""
    if isinstance(source, (list, tuple)):
        return [{"image_path": str(p), "true_label": None} for p in source]

    if not isinstance(source, (str, Path)):
        raise ValueError(
            f"Unsupported batch source: {source!r}. Expected a list of file paths, a directory, "
            "or a .csv manifest with an 'image_path' column (optionally 'true_label')."
        )
    path = Path(source)
    if path.is_dir():
        images = sorted(p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        return [{"image_path": str(p), "true_label": None} for p in images]

    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if rows and "image_path" not in rows[0]:
            raise ValueError("CSV manifest must have an 'image_path' column")
        return [{"image_path": row["image_path"], "true_label": row.get("true_label") or None} for row in rows]

    raise ValueError(
        f"Unsupported batch source: {source!r}. Expected a list of file paths, a directory, "
        "or a .csv manifest with an 'image_path' column (optionally 'true_label')."
    )


def _class_counts(class_names: list[str], predicted_classes: list[str]) -> dict:
    return {name: predicted_classes.count(name) for name in class_names}


def run_batch_evaluation(source, model, masking_method: str = "mean", grid_size: int = 3) -> BatchSummary:
    entries = discover_batch_inputs(source)
    class_names = list(model.class_names)

    samples: list[BatchSampleResult] = []
    failed_samples: list[dict] = []

    for entry in entries:
        image_path = entry["image_path"]
        try:
            image = Image.open(image_path).convert("RGB")
            idx, predicted_class, probs, cam = get_original_prediction_and_cam(model, image)
            region_results = run_region_perturbation(
                image=np.array(image), cam=cam, predict_fn=model.predict_proba,
                class_names=class_names, original_class_idx=idx,
                masking_method=masking_method, grid_size=grid_size,
            )
            support = rank_supporting_regions(region_results)
            suppress = rank_suppressing_regions(region_results)
            subject_id = infer_subject_id(image_path) or Path(image_path).stem

            samples.append(BatchSampleResult(
                path=Path(image_path).name,
                subject_id=subject_id,
                predicted_class=predicted_class,
                original_class_probability=float(probs[idx]),
                probs={name: float(p) for name, p in zip(class_names, probs)},
                region_results=tuple(region_results),
                supporting_top_region=support[0].region_name if support else None,
                suppressing_top_region=suppress[0].region_name if suppress else None,
                true_label=entry.get("true_label"),
            ))
        except Exception as exc:  # noqa: BLE001 -- batch runs must not abort on one bad sample
            failed_samples.append({"path": Path(str(image_path)).name, "reason": str(exc)})

    return _summarize(samples, failed_samples, class_names, masking_method)


def _summarize(samples: list[BatchSampleResult], failed_samples: list[dict], class_names: list[str], masking_method: str) -> BatchSummary:
    predicted_classes_images = [s.predicted_class for s in samples]
    class_counts_images = _class_counts(class_names, predicted_classes_images)
    predicted_class_distribution_images = (
        {name: count / len(samples) for name, count in class_counts_images.items()} if samples else {}
    )

    by_subject: dict[str, list[BatchSampleResult]] = {}
    for sample in samples:
        by_subject.setdefault(sample.subject_id, []).append(sample)

    subject_predicted_classes = []
    subject_true_labels: dict[str, str | None] = {}
    for subject_id, subject_samples in by_subject.items():
        mean_probs = np.mean([[s.probs[name] for name in class_names] for s in subject_samples], axis=0)
        subject_predicted_classes.append(class_names[int(mean_probs.argmax())])
        true_labels_seen = {s.true_label for s in subject_samples if s.true_label}
        subject_true_labels[subject_id] = next(iter(true_labels_seen)) if true_labels_seen else None

    class_counts_subjects = _class_counts(class_names, subject_predicted_classes)
    predicted_class_distribution_subjects = (
        {name: count / len(by_subject) for name, count in class_counts_subjects.items()} if by_subject else {}
    )

    supporting_freq: dict[str, dict[str, int]] = {name: {} for name in class_names}
    suppressing_freq: dict[str, dict[str, int]] = {name: {} for name in class_names}
    for sample in samples:
        if sample.supporting_top_region:
            bucket = supporting_freq.setdefault(sample.predicted_class, {})
            bucket[sample.supporting_top_region] = bucket.get(sample.supporting_top_region, 0) + 1
        if sample.suppressing_top_region:
            bucket = suppressing_freq.setdefault(sample.predicted_class, {})
            bucket[sample.suppressing_top_region] = bucket.get(sample.suppressing_top_region, 0) + 1

    drops_by_region: dict[str, list[float]] = {}
    for sample in samples:
        for region in sample.region_results:
            drops_by_region.setdefault(region.region_name, []).append(region.probability_drop)
    mean_probability_drop_by_region = {
        name: {
            "mean": statistics.fmean(values),
            "stdev": (statistics.pstdev(values) if len(values) > 1 else 0.0),
            "n": len(values),
        }
        for name, values in drops_by_region.items()
    }

    agreement_reports = [compute_cam_perturbation_agreement(sample.region_results, masking_method) for sample in samples]
    spearman_drop_values = [a.spearman_cam_ratio_vs_drop for a in agreement_reports if a.spearman_cam_ratio_vs_drop is not None]
    spearman_abs_values = [a.spearman_cam_ratio_vs_absolute for a in agreement_reports if a.spearman_cam_ratio_vs_absolute is not None]
    cam_perturbation_mean_agreement = {
        "mean_spearman_cam_ratio_vs_drop": (statistics.fmean(spearman_drop_values) if spearman_drop_values else None),
        "mean_spearman_cam_ratio_vs_absolute": (statistics.fmean(spearman_abs_values) if spearman_abs_values else None),
        "n_images_with_defined_correlation": len(spearman_drop_values),
        "sample_size_note": "Each image contributes n=9 regions; this average is itself only over the images that had a defined (non-constant-CAM) correlation.",
    }

    classification_metrics = _classification_metrics(class_names, subject_predicted_classes, subject_true_labels, by_subject)

    return BatchSummary(
        total_images=len(samples),
        total_subjects=len(by_subject),
        masking_method=masking_method,
        class_counts_images=class_counts_images,
        class_counts_subjects=class_counts_subjects,
        predicted_class_distribution_images=predicted_class_distribution_images,
        predicted_class_distribution_subjects=predicted_class_distribution_subjects,
        supporting_region_frequency_by_class=supporting_freq,
        suppressing_region_frequency_by_class=suppressing_freq,
        mean_probability_drop_by_region=mean_probability_drop_by_region,
        cam_perturbation_mean_agreement=cam_perturbation_mean_agreement,
        classification_metrics=classification_metrics,
        failed_samples=failed_samples,
        samples=tuple(samples),
    )


def _classification_metrics(class_names, subject_predicted_classes, subject_true_labels, by_subject) -> dict | None:
    subject_ids = list(by_subject.keys())
    y_true = [subject_true_labels[sid] for sid in subject_ids]
    if not subject_ids or any(label is None for label in y_true):
        return None  # no ground truth available -- do not compute classification performance

    y_pred = [
        class_names[int(np.mean(
            [[s.probs[name] for name in class_names] for s in by_subject[sid]], axis=0
        ).argmax())]
        for sid in subject_ids
    ]

    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

    return {
        "n_subjects_with_ground_truth": len(subject_ids),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=class_names, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=class_names).tolist(),
        "confusion_matrix_labels": list(class_names),
    }
