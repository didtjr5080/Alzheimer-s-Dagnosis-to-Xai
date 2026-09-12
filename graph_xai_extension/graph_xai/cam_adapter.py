"""CAM-generation glue: always explains the ORIGINAL predicted class, never an
arbitrary or user-picked target, so downstream perturbation numbers line up
with "how much did the model's own decision rely on this region".
"""
from __future__ import annotations

import numpy as np

from .model_adapter import GraphXAIModel


def get_original_prediction_and_cam(model: GraphXAIModel, image) -> tuple[int, str, np.ndarray, np.ndarray]:
    """Runs one forward pass to get the prediction, then one CAM pass targeted
    at that same predicted class.

    Returns (predicted_class_idx, predicted_class_name, class_probabilities, cam_heatmap).
    """
    predicted_idx, predicted_class, probs = model.predict_original(image)
    cam = model.compute_cam(image, target_class_idx=predicted_idx)
    return predicted_idx, predicted_class, probs, cam
