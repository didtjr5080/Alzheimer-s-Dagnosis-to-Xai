from __future__ import annotations

import numpy as np
from PIL import Image
from pypdf import PdfReader

from graph_xai.agreement import compute_cam_perturbation_agreement
from graph_xai.cam_adapter import get_original_prediction_and_cam
from graph_xai.class_mapping import validate_class_mapping
from graph_xai.graph_builder import build_region_graph
from graph_xai.masking_compare import run_all_masking_methods
from graph_xai.pdf_report import build_graph_xai_pdf_report
from graph_xai.region_grid import split_into_regions
from graph_xai.run_metadata import build_run_metadata
from graph_xai.stability import compare_masking_methods

from _mock_model import MockGraphXAIModel


def _all_pdf_text(pdf_path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _build_report(tmp_path, mock_mode: bool):
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

    output_path = tmp_path / "graph_xai_report.pdf"
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
    )
    return output_path, predicted_class


def test_pdf_report_is_created_and_nonempty(tmp_path):
    output_path, _ = _build_report(tmp_path, mock_mode=True)
    assert output_path.exists()
    assert output_path.stat().st_size > 1000


def test_pdf_contains_prediction_ranking_sections_and_signature(tmp_path):
    output_path, predicted_class = _build_report(tmp_path, mock_mode=True)
    text = _all_pdf_text(output_path)

    assert predicted_class in text
    assert "모델 예측" in text
    assert "지지 근거" in text
    assert "억제 근거" in text
    assert "절대 민감도" in text
    assert "연구 검토자 확인" in text
    assert "서명" in text
    assert "진단" in text  # part of the sign-off disclaimer
    assert "Graph XAI" in text
    assert "세 마스킹 방식 비교" in text
    assert "CAM-Perturbation 일치도" in text


def test_pdf_never_calls_the_prediction_a_judgment():
    # "판단" (judgment) must not appear anywhere in the wording constants
    from graph_xai import pdf_report
    for name in ("RESEARCH_USE_WARNING", "SENSITIVITY_EXPLANATION_SENTENCE", "GRAPH_XAI_UI_WARNING",
                 "REVIEWER_SIGNATURE_DISCLAIMER", "DEFAULT_REVIEWER_SECTION_TITLE"):
        assert "판단" not in getattr(pdf_report, name)


def test_pdf_never_uses_banned_correlation_based_explanation_phrase():
    from graph_xai import pdf_report
    assert "상관관계 기반 설명" not in pdf_report.SENSITIVITY_EXPLANATION_SENTENCE


def test_pdf_shows_mock_mode_warning_only_when_mock(tmp_path):
    mock_path, _ = _build_report(tmp_path / "mock", mock_mode=True)
    real_path, _ = _build_report(tmp_path / "real", mock_mode=False)

    mock_text = _all_pdf_text(mock_path)
    real_text = _all_pdf_text(real_path)

    assert "mock" in mock_text.lower()
    assert "mock_mode: true" not in real_text.lower() and "mock 모델로 생성" not in real_text


def test_pdf_region_table_lists_all_nine_regions(tmp_path):
    output_path, _ = _build_report(tmp_path, mock_mode=False)
    text = _all_pdf_text(output_path)
    for region_name in (
        "top_left", "top_center", "top_right",
        "middle_left", "middle_center", "middle_right",
        "bottom_left", "bottom_center", "bottom_right",
    ):
        assert region_name in text


def test_pdf_includes_class_mapping_evidence(tmp_path):
    output_path, _ = _build_report(tmp_path, mock_mode=False)
    text = _all_pdf_text(output_path)
    assert "클래스 매핑" in text
    assert "CN=0/MCI=1/AD=2를 임의로 가정하지 않았습니다" in text or "임의로 가정" in text
