from __future__ import annotations

from PIL import Image

from .model_loader import load_pipeline


def analyze_subject(images: list[Image.Image], device: str | None = None):
    bundle = load_pipeline(device=device)

    from inference_clip_lr import aggregate_subject, predict_slices, select_representative_slices

    slice_predictions = predict_slices(images, bundle)
    subject_prediction = aggregate_subject(slice_predictions)
    representative_indices = select_representative_slices(slice_predictions)
    return bundle, slice_predictions, subject_prediction, representative_indices
