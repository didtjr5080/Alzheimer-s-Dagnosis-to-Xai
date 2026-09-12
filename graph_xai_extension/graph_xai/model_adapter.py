"""High-level model access used by the Graph XAI extension.

Wraps ``legacy_adapter.LegacyPipelineHandle`` so the rest of this package only
ever talks to a small ``predict_proba(images) -> (n, num_classes)`` /
``compute_cam(image, target_class_idx) -> (H, W)`` surface. If the real model
files are missing, ``check_model_available`` reports why and real analysis
must be disabled by the caller -- this module never fabricates a classifier.
"""
from __future__ import annotations

import numpy as np

from .class_mapping import ClassMappingError, validate_class_mapping, validate_probabilities
from .legacy_adapter import ModelUnavailableError, load_real_pipeline, resolve_handoff_root, validate_handoff_root

__all__ = [
    "GraphXAIModel", "ModelUnavailableError", "ClassMappingError",
    "check_model_available", "MODEL_MISSING_MESSAGE",
]

MODEL_MISSING_MESSAGE = (
    "분류 모델 파일을 찾을 수 없어 실제 Graph XAI 분석을 실행할 수 없습니다. "
    "학습된 분류기 경로를 지정해 주세요. 테스트용 mock 모델 결과는 실제 의료영상 분석 결과가 아닙니다."
)


def check_model_available(classifier_path: str | None = None) -> tuple[bool, str, list[str]]:
    """Returns (is_available, resolved_root_str, missing_paths)."""
    try:
        root = resolve_handoff_root(classifier_path)
    except ModelUnavailableError as exc:
        return False, "", [str(exc)]
    missing = validate_handoff_root(root)
    return (len(missing) == 0), str(root), missing


class GraphXAIModel:
    """Real-model wrapper. Raises ModelUnavailableError at construction time if
    the underlying handoff package cannot be found -- callers must catch this
    and disable real analysis rather than substituting a mock silently."""

    def __init__(self, classifier_path: str | None = None, device: str | None = None, allow_binary: bool = False):
        self._handle = load_real_pipeline(classifier_path, device=device)
        self.mock_mode = False
        # Never trust CN=0/MCI=1/AD=2 -- validate the class names this specific
        # loaded model reports, and refuse to proceed on anything unexpected
        # (see ClassMappingError). Raises before any analysis can run.
        self.class_mapping_report = validate_class_mapping(self._handle.class_names, allow_binary=allow_binary)

    @property
    def class_names(self) -> list[str]:
        return self._handle.class_names

    @property
    def device(self) -> str:
        return str(self._handle.device)

    @property
    def model_identifier(self) -> str:
        return f"clip_lr::{self._handle.root.name}"

    def predict_proba(self, images) -> np.ndarray:
        return np.asarray(self._handle.predict_proba(images), dtype=np.float64)

    def predict_original(self, image) -> tuple[int, str, np.ndarray]:
        probs = self.predict_proba([image])[0]
        validate_probabilities(probs, self.class_names)
        idx = int(probs.argmax())
        return idx, self.class_names[idx], probs

    def compute_cam(self, image, target_class_idx: int) -> np.ndarray:
        return np.asarray(self._handle.generate_cam(image, target_class_idx), dtype=np.float32)
