from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class UploadedSlice:
    name: str
    image: Image.Image


@dataclass
class AnalysisResult:
    mr_id: str
    class_names: list[str]
    slice_probs: np.ndarray
    subject_probs: dict[str, float]
    predicted_class: str
    confidence: float
    representative_indices: list[int]
    report_path: Path | None = None
    warning: str | None = None
