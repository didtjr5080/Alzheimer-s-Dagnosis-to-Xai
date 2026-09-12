"""Deterministic, explicit mock classifier for tests ONLY. Never used to make
real-analysis claims -- see mock_mode=True below and the JSON export flag.
"""
from __future__ import annotations

import numpy as np


class MockGraphXAIModel:
    mock_mode = True
    model_identifier = "mock::deterministic-brightness-classifier"
    device = "cpu"

    def __init__(self, class_names=("AD", "CN")):
        self.class_names = list(class_names)

    @staticmethod
    def _brightness(image) -> float:
        arr = np.asarray(image, dtype=np.float64)
        if arr.ndim == 3:
            arr = arr.mean(axis=-1)
        return float(arr.mean())

    def predict_proba(self, images) -> np.ndarray:
        rows = []
        n = len(self.class_names)
        for image in images:
            brightness = self._brightness(image)
            logits = np.array([brightness - i * 15.0 for i in range(n)], dtype=np.float64)
            exp = np.exp(logits - logits.max())
            rows.append(exp / exp.sum())
        return np.stack(rows)

    def predict_original(self, image):
        probs = self.predict_proba([image])[0]
        idx = int(probs.argmax())
        return idx, self.class_names[idx], probs

    def compute_cam(self, image, target_class_idx: int) -> np.ndarray:
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=-1)
        cam = arr.copy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam
