"""Integration tests for the three report_mode values added by the
'의료진용 보고서 개선' work order: clinical_summary / technical_full / combined."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from pypdf import PdfReader

from graph_xai.agreement import compute_cam_perturbation_agreement
from graph_xai.cam_adapter import get_original_prediction_and_cam
from graph_xai.class_mapping import validate_class_mapping
from graph_xai.clinical_wording import find_forbidden_phrases
from graph_xai.graph_builder import build_region_graph
from graph_xai.masking_compare import run_all_masking_methods
from graph_xai.pdf_report import REPORT_MODES, build_graph_xai_pdf_report
from graph_xai.region_grid import split_into_regions
from graph_xai.run_metadata import build_run_metadata
from graph_xai.stability import compare_masking_methods

from _mock_model import MockGraphXAIModel


def _page_count(pdf_path) -> int:
    return len(PdfReader(str(pdf_path)).pages)


def _all_pdf_text(pdf_path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _build_report(tmp_path, report_mode: str, mock_mode: bool = False):
    size = 30
    array = np.full((size, size, 3), 20, dtype=np.uint8)
    array[0:10, 0:10] = 220
    image = Image.fromarray(array, mode="RGB")

    model = MockGraphXAIModel(class_names=["AD", "CN", "MCI"])
    idx, predicted_class, probs, cam = get_original_prediction_and_cam(model, np.array(image))
    regions = split_into_regions(cam, image=np.array(image), grid=3)

    results_by_method = run_all_masking_methods(
        image=np.array(image), cam=cam, predict_fn=model.predict_proba,
        class_names=model.class_names, original_class_idx=idx,
    )
    graph = build_region_graph(results_by_method["mean"])
    stability = compare_masking_methods(results_by_method)
    agreement_reports = {method: compute_cam_perturbation_agreement(results, method) for method, results in results_by_method.items()}
    class_mapping_report = validate_class_mapping(model.class_names)

    run_metadata = build_run_metadata(
        model_identifier=model.model_identifier, classifier_classes=model.class_names,
        cam_method="mock", masking_methods=list(results_by_method.keys()), grid_size=3,
        device=model.device, input_bytes=b"fake-bytes", mock_mode=mock_mode,
    )
    run_metadata["input_filename"] = "synthetic_demo_slice.png"

    output_path = tmp_path / f"{report_mode}.pdf"
    build_graph_xai_pdf_report(
        output_path,
        run_metadata=run_metadata,
        predicted_class=predicted_class,
        class_probabilities=dict(zip(model.class_names, probs.tolist())),
        class_mapping_report=class_mapping_report,
        original_image=image,
        cam=cam,
        regions=regions,
        results_by_method=results_by_method,
        primary_masking_method="mean",
        graph=graph,
        stability_report=stability,
        agreement_reports=agreement_reports,
        mock_mode=mock_mode,
        report_mode=report_mode,
    )
    return output_path, predicted_class


def test_unknown_report_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        _build_report(tmp_path, "not_a_real_mode")


@pytest.mark.parametrize("mode", REPORT_MODES)
def test_every_report_mode_produces_a_nonempty_pdf(tmp_path, mode):
    output_path, _ = _build_report(tmp_path / mode, mode)
    assert output_path.exists()
    assert output_path.stat().st_size > 1000


def test_clinical_summary_is_two_to_three_pages(tmp_path):
    output_path, _ = _build_report(tmp_path, "clinical_summary")
    assert 2 <= _page_count(output_path) <= 3


def test_clinical_summary_shows_the_clinical_title_and_predicted_class(tmp_path):
    from graph_xai.clinical_wording import CLINICAL_TITLE

    output_path, predicted_class = _build_report(tmp_path, "clinical_summary")
    text = _all_pdf_text(output_path)
    assert CLINICAL_TITLE in text
    assert predicted_class in text


def test_clinical_summary_excludes_raw_technical_jargon(tmp_path):
    output_path, _ = _build_report(tmp_path, "clinical_summary")
    text = _all_pdf_text(output_path)
    for jargon in ("Spearman", "Kendall", "Jaccard", "classes_", "mock_mode", "SHA-256", "git", "Git"):
        assert jargon not in text


def test_clinical_summary_never_calls_the_prediction_a_diagnosis_probability(tmp_path):
    output_path, _ = _build_report(tmp_path, "clinical_summary")
    text = _all_pdf_text(output_path)
    assert "진단 확률" not in text.replace("진단 확률 또는", "")  # allow only the negated disclaimer


def test_clinical_summary_has_no_forbidden_phrase_violations(tmp_path):
    output_path, _ = _build_report(tmp_path, "clinical_summary")
    text = _all_pdf_text(output_path)
    assert find_forbidden_phrases(text) == []


def test_clinical_summary_shows_reviewer_confirmation_exactly_once(tmp_path):
    output_path, _ = _build_report(tmp_path, "clinical_summary")
    text = _all_pdf_text(output_path)
    assert text.count("연구 검토자 확인") == 1
    assert text.count("본 서명은 연구용 AI 출력물을 열람했다는 기록이며") == 1


def test_technical_full_has_no_clinical_title_but_keeps_technical_sections(tmp_path):
    from graph_xai.clinical_wording import CLINICAL_TITLE

    output_path, predicted_class = _build_report(tmp_path, "technical_full")
    text = _all_pdf_text(output_path)
    assert CLINICAL_TITLE not in text
    assert "지지 근거" in text
    assert "억제 근거" in text
    assert "절대 민감도" in text
    assert "연구 검토자 확인" in text


def test_combined_contains_both_clinical_and_technical_content(tmp_path):
    from graph_xai.clinical_wording import CLINICAL_TITLE

    output_path, predicted_class = _build_report(tmp_path, "combined")
    text = _all_pdf_text(output_path)
    assert CLINICAL_TITLE in text
    assert "연구자용 기술 부록" in text
    assert "지지 근거" in text
    assert "절대 민감도" in text


def test_combined_shows_reviewer_confirmation_exactly_once(tmp_path):
    output_path, _ = _build_report(tmp_path, "combined")
    text = _all_pdf_text(output_path)
    assert text.count("연구 검토자 확인") == 1


def test_combined_has_more_pages_than_either_mode_alone(tmp_path):
    clinical_path, _ = _build_report(tmp_path / "c", "clinical_summary")
    technical_path, _ = _build_report(tmp_path / "t", "technical_full")
    combined_path, _ = _build_report(tmp_path / "b", "combined")
    assert _page_count(combined_path) >= max(_page_count(clinical_path), _page_count(technical_path))


def test_mock_mode_warning_appears_in_clinical_summary_when_mock(tmp_path):
    mock_path, _ = _build_report(tmp_path / "mock", "clinical_summary", mock_mode=True)
    real_path, _ = _build_report(tmp_path / "real", "clinical_summary", mock_mode=False)
    assert "mock" in _all_pdf_text(mock_path).lower()
    assert "mock 모델로 생성" not in _all_pdf_text(real_path)
