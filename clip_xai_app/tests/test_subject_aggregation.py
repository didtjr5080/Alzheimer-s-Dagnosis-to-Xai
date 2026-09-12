from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
HANDOFF_CODE = REPO_ROOT / "clip_lr_grad_eclip_handoff_v1_20260812_105027" / "code"
sys.path.insert(0, str(HANDOFF_CODE))


def test_subject_aggregation_is_order_invariant_mean_probability():
    module = importlib.import_module("inference_clip_lr")
    batch = module.SlicePredictionBatch(
        logits=np.zeros((3, 3), dtype=float),
        probs=np.array(
            [
                [0.9, 0.05, 0.05],
                [0.8, 0.1, 0.1],
                [0.9886, 0.1083, -0.0969],
            ],
            dtype=float,
        ),
        embeddings=np.zeros((3, 512), dtype=float),
        predicted_class_indices=np.array([0, 0, 0]),
        predicted_classes=["CN", "CN", "CN"],
        confidences=np.array([0.9, 0.8, 0.9886]),
        class_names=["CN", "MCI", "AD"],
        classes_idx=np.array([0, 1, 2]),
    )

    result = module.aggregate_subject(batch)
    reversed_result = module.aggregate_subject(
        module.SlicePredictionBatch(
            logits=batch.logits[::-1],
            probs=batch.probs[::-1],
            embeddings=batch.embeddings[::-1],
            predicted_class_indices=batch.predicted_class_indices[::-1],
            predicted_classes=list(reversed(batch.predicted_classes)),
            confidences=batch.confidences[::-1],
            class_names=batch.class_names,
            classes_idx=batch.classes_idx,
        )
    )

    assert result.predicted_class == "CN"
    assert abs(result.probs["CN"] - 0.8962) < 1e-4
    assert abs(result.probs["MCI"] - 0.0861) < 1e-4
    assert abs(result.probs["AD"] - 0.0177) < 1e-4
    for class_name in result.probs:
        assert abs(result.probs[class_name] - reversed_result.probs[class_name]) < 1e-12
