from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from graph_xai.batch import discover_batch_inputs, infer_subject_id, run_batch_evaluation
from graph_xai.model_adapter import GraphXAIModel, check_model_available

from _mock_model import MockGraphXAIModel


def _make_png(path, brightness):
    array = np.full((20, 20, 3), brightness, dtype=np.uint8)
    Image.fromarray(array).save(path)


def test_infer_subject_id_recognizes_the_project_slice_naming_convention():
    assert infer_subject_id("OAS30217_MR_d0077_cor090.png") == "OAS30217_MR_d0077"
    assert infer_subject_id("some_random_file.png") is None


def test_discover_batch_inputs_from_list():
    entries = discover_batch_inputs(["a.png", "b.png"])
    assert entries == [{"image_path": "a.png", "true_label": None}, {"image_path": "b.png", "true_label": None}]


def test_discover_batch_inputs_from_directory(tmp_path):
    _make_png(tmp_path / "OAS1_MR_d1_cor001.png", 50)
    _make_png(tmp_path / "OAS1_MR_d1_cor002.png", 60)
    (tmp_path / "not_an_image.txt").write_text("x")
    entries = discover_batch_inputs(tmp_path)
    assert len(entries) == 2
    assert all(e["image_path"].endswith(".png") for e in entries)


def test_discover_batch_inputs_from_csv_manifest(tmp_path):
    csv_path = tmp_path / "manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "true_label"])
        writer.writeheader()
        writer.writerow({"image_path": "OAS1_MR_d1_cor001.png", "true_label": "CN"})
        writer.writerow({"image_path": "OAS1_MR_d1_cor002.png", "true_label": ""})
    entries = discover_batch_inputs(csv_path)
    assert entries[0]["true_label"] == "CN"
    assert entries[1]["true_label"] is None


def test_discover_batch_inputs_rejects_unsupported_source():
    with pytest.raises(ValueError):
        discover_batch_inputs(12345)


def test_batch_evaluation_reports_totals_and_class_distributions(tmp_path):
    # two "subjects", two slices each -- subject dedup must collapse to 2, not 4
    _make_png(tmp_path / "OAS30001_MR_d1_cor001.png", 220)
    _make_png(tmp_path / "OAS30001_MR_d1_cor002.png", 210)
    _make_png(tmp_path / "OAS30002_MR_d1_cor001.png", 10)
    _make_png(tmp_path / "OAS30002_MR_d1_cor002.png", 20)

    model = MockGraphXAIModel(class_names=["AD", "CN"])
    summary = run_batch_evaluation(tmp_path, model, masking_method="mean")

    assert summary.total_images == 4
    assert summary.total_subjects == 2
    assert sum(summary.class_counts_subjects.values()) == 2
    assert sum(summary.class_counts_images.values()) == 4
    assert summary.failed_samples == []


def test_batch_evaluation_no_ground_truth_skips_classification_metrics(tmp_path):
    _make_png(tmp_path / "OASA_MR_d1_cor001.png", 220)
    model = MockGraphXAIModel(class_names=["AD", "CN"])
    summary = run_batch_evaluation(tmp_path, model)
    assert summary.classification_metrics is None


def test_batch_evaluation_with_ground_truth_computes_classification_metrics(tmp_path):
    _make_png(tmp_path / "OASA_MR_d1_cor001.png", 220)  # bright -> mock predicts class index 0
    _make_png(tmp_path / "OASB_MR_d1_cor001.png", 10)   # dark -> mock predicts a later class

    csv_path = tmp_path / "manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "true_label"])
        writer.writeheader()
        writer.writerow({"image_path": str(tmp_path / "OASA_MR_d1_cor001.png"), "true_label": "AD"})
        writer.writerow({"image_path": str(tmp_path / "OASB_MR_d1_cor001.png"), "true_label": "AD"})

    model = MockGraphXAIModel(class_names=["AD", "CN"])
    summary = run_batch_evaluation(csv_path, model)
    assert summary.classification_metrics is not None
    assert summary.classification_metrics["n_subjects_with_ground_truth"] == 2
    assert 0.0 <= summary.classification_metrics["accuracy"] <= 1.0
    assert "confusion_matrix" in summary.classification_metrics


def test_batch_evaluation_records_failed_samples_without_aborting(tmp_path):
    good_path = tmp_path / "OASA_MR_d1_cor001.png"
    _make_png(good_path, 100)
    bad_path = tmp_path / "does_not_exist.png"

    model = MockGraphXAIModel(class_names=["AD", "CN"])
    summary = run_batch_evaluation([str(good_path), str(bad_path)], model)
    assert summary.total_images == 1
    assert len(summary.failed_samples) == 1
    assert summary.failed_samples[0]["path"] == "does_not_exist.png"


def test_batch_summary_never_stores_full_local_paths():
    model = MockGraphXAIModel(class_names=["AD", "CN"])
    summary = run_batch_evaluation([], model)
    assert summary.total_images == 0
    assert summary.samples == ()


REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_SAMPLES_DIR = REPO_ROOT / "clip_lr_grad_eclip_handoff_v1_20260812_105027" / "data" / "xai_samples" / "images"
_real_available, _real_root, _real_missing = check_model_available(None)


@pytest.mark.skipif(
    not (_real_available and REAL_SAMPLES_DIR.exists()),
    reason=f"Real classifier/samples not available locally (missing={_real_missing})",
)
def test_real_model_batch_evaluation_deduplicates_six_known_subjects():
    model = GraphXAIModel(classifier_path=None, device="cpu")
    summary = run_batch_evaluation(REAL_SAMPLES_DIR, model, masking_method="mean")

    assert summary.total_images == 152
    assert summary.total_subjects == 6  # not 152 -- subject-level dedup must collapse repeated slices
    assert summary.failed_samples == []
    assert sum(summary.class_counts_subjects.values()) == 6
    assert set(summary.class_counts_images.keys()) == {"CN", "MCI", "AD"}
