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


def _saved_artifact(tmp_path, filename, probs, pred="CN", target="CN"):
    from src.xai_artifacts import save_xai_artifact

    path = SAMPLE_ROOT / "images" / filename
    image = Image.open(path).convert("RGB")
    heatmap = np.zeros((224, 224), dtype=np.float32)
    heatmap[70:150, 70:150] = 1.0
    fake_xai = SimpleNamespace(predicted_class=pred, target_class=target)
    fake_bundle = SimpleNamespace(root=REPO_ROOT / "clip_lr_grad_eclip_handoff_v1_20260812_105027")
    return save_xai_artifact(
        artifacts_root=tmp_path,
        repo_root=REPO_ROOT,
        source_path=path,
        image=image,
        heatmap=heatmap,
        xai_result=fake_xai,
        slice_probability_row=probs,
        subject_probabilities=probs,
        class_names=["CN", "MCI", "AD"],
        target_class_index=["CN", "MCI", "AD"].index(target),
        bundle=fake_bundle,
        run_id="report-test",
    )


def _xai_item(artifact):
    return {
        "title": artifact.provenance.slice_filename,
        "original": Image.open(artifact.original_path),
        "heatmap": Image.open(artifact.normalized_heatmap_png_path),
        "overlay": Image.open(artifact.overlay_path),
        "artifact": artifact,
    }


def test_pdf_report_marks_mci_to_cn_misclassification_and_xai_target(tmp_path):
    from inference_clip_lr import aggregate_subject
    from src.report import create_basic_pdf_report

    filename = "OAS30051_MR_d1286_cor084.png"
    probs = {"CN": 0.6740367151534636, "MCI": 0.22393300809046773, "AD": 0.10203027675606859}
    artifact = _saved_artifact(tmp_path, filename, probs, pred="CN", target="CN")
    batch = _batch([[probs["CN"], probs["MCI"], probs["AD"]]])
    subject = aggregate_subject(batch)
    report = create_basic_pdf_report(
        tmp_path,
        "OAS30051_MR_d1286",
        subject,
        batch,
        [0],
        [filename],
        [_xai_item(artifact)],
    )
    text = "\n".join(page.extract_text() or "" for page in PdfReader(str(report)).pages)

    assert "오분류" in text
    assert "true" in text
    assert "pred" in text
    assert "XAI target" in text
    assert "CN prediction logit" in text or "CN 예측 logit" in text
    assert "전경 마스크가 없는" not in text
    assert "mask_method" in text
    assert "foreground_" in text
    assert "background_" in text


def test_pdf_generation_aborts_on_manifest_probability_mismatch(tmp_path):
    from inference_clip_lr import aggregate_subject
    from src.report import create_basic_pdf_report

    filename = "OAS30051_MR_d1286_cor084.png"
    manifest_probs = {"CN": 0.6740367151534636, "MCI": 0.22393300809046773, "AD": 0.10203027675606859}
    artifact = _saved_artifact(tmp_path, filename, manifest_probs, pred="CN", target="CN")
    batch = _batch([[0.50, 0.30, 0.20]])
    subject = aggregate_subject(batch)

    with pytest.raises(ValueError, match="probability mismatch|predicted_class mismatch"):
        create_basic_pdf_report(
            tmp_path,
            "OAS30051_MR_d1286",
            subject,
            batch,
            [0],
            [filename],
            [_xai_item(artifact)],
        )
