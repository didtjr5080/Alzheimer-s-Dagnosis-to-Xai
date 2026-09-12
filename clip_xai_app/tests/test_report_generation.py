from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from pypdf import PdfReader


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "clip_xai_app"
HANDOFF_CODE = REPO_ROOT / "clip_lr_grad_eclip_handoff_v1_20260812_105027" / "code"
SAMPLE_ROOT = REPO_ROOT / "clip_lr_grad_eclip_handoff_v1_20260812_105027" / "data" / "xai_samples"
sys.path.insert(0, str(APP_ROOT))
sys.path.insert(0, str(HANDOFF_CODE))


def _batch(probs):
    module = importlib.import_module("inference_clip_lr")
    probs = np.asarray(probs, dtype=float)
    return module.SlicePredictionBatch(
        logits=np.zeros_like(probs),
        probs=probs,
        embeddings=np.zeros((probs.shape[0], 512), dtype=float),
        predicted_class_indices=probs.argmax(axis=1),
        predicted_classes=[["CN", "MCI", "AD"][int(i)] for i in probs.argmax(axis=1)],
        confidences=probs.max(axis=1),
        class_names=["CN", "MCI", "AD"],
        classes_idx=np.array([0, 1, 2]),
    )


def test_pdf_report_uses_research_and_model_output_language(tmp_path):
    from inference_clip_lr import aggregate_subject
    from src.report import create_basic_pdf_report

    batch = _batch([[0.9180, 0.0619, 0.0201]])
    subject = aggregate_subject(batch)
    report = create_basic_pdf_report(
        tmp_path,
        "OAS30009_MR_d2457",
        subject,
        batch,
        [0],
        ["OAS30009_MR_d2457_cor094.png"],
        [],
    )
    text = "\n".join(page.extract_text() or "" for page in PdfReader(str(report)).pages)

    assert "Research" in text
    assert "모델 출력 확률" in text
    assert "Confidence" not in text
    assert "진단 결과" not in text
    assert "병변 위치" not in text
