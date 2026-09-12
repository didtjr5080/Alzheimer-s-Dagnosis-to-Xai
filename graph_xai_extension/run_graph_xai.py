"""Standalone Gradio app for the Graph XAI extension.

Does NOT modify or import clip_xai_app/app.py. Run directly:

    python graph_xai_extension/run_graph_xai.py

If the real classifier files are not found, the "5. 분석 실행" button stays
disabled and the model-status box explains why -- this app never silently
substitutes a mock model for a real analysis result.

Every analysis run now runs all three masking methods (zero/mean/blur) so the
report always shows their agreement/disagreement rather than presenting one
method's numbers (typically the UI-selected one, shown as "primary") as the
whole explanation. See docs/IMPLEMENTATION_REPORT.md for the rationale.
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from uuid import uuid4

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image

from graph_xai.agreement import compute_cam_perturbation_agreement
from graph_xai.batch import run_batch_evaluation
from graph_xai.cam_adapter import get_original_prediction_and_cam
from graph_xai.class_mapping import ClassMappingError
from graph_xai.clinical_wording import STABILITY_LOW_WARNING, build_clinical_summary_sentence
from graph_xai.exporter import MEDICAL_LIMITATION_NOTICE, build_export_payload, export_csv, export_html, export_json
from graph_xai.graph_builder import build_region_graph
from graph_xai.masking import apply_region_mask
from graph_xai.masking_compare import build_masking_comparison_table, run_all_masking_methods
from graph_xai.model_adapter import MODEL_MISSING_MESSAGE, GraphXAIModel, check_model_available
from graph_xai.pdf_report import build_graph_xai_pdf_report
from graph_xai.ranking import rank_by_absolute_sensitivity, rank_supporting_regions, rank_suppressing_regions
from graph_xai.region_grid import split_into_regions
from graph_xai.run_metadata import build_run_metadata
from graph_xai.schemas import REGION_NAMES_3X3
from graph_xai.stability import compare_masking_methods
from graph_xai.visualization import overlay_grid_on_image, plot_probability_bar_chart, plot_region_graph

EXTENSION_ROOT = Path(__file__).resolve().parent

UI_WARNING = (
    "이 분석은 모델이 특정 영상 구역을 얼마나 참고했는지 평가하는 설명가능 AI 결과입니다. "
    "표시된 구역은 실제 병변 위치, 알츠하이머병의 원인 또는 임상 진단을 의미하지 않습니다. "
    "3×3 구역은 해부학적 뇌 영역이 아닙니다."
)

REPORT_TYPE_CHOICES = ("통합", "의료진용", "기술 상세")


def _resolve_report_mode(report_type_choice: str, include_appendix: bool) -> str:
    """Maps the Korean UI choice (work order section 15) onto the pdf_report
    module's report_mode values."""
    if report_type_choice == "기술 상세":
        return "technical_full"
    if report_type_choice == "의료진용":
        return "combined" if include_appendix else "clinical_summary"
    return "combined"


def _resolve_upload_path(file_obj) -> str:
    return file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", file_obj)


def _output_dir() -> Path:
    return Path(os.environ.get("GRAPH_XAI_OUTPUT_DIR") or (EXTENSION_ROOT / "outputs"))


def check_model_status(classifier_path: str):
    classifier_path = (classifier_path or "").strip() or None
    available, root, missing = check_model_available(classifier_path)
    if available:
        return f"모델 사용 가능: {root}", gr.update(interactive=True)
    detail = "\n".join(missing) if missing else ""
    return f"{MODEL_MISSING_MESSAGE}\n(확인한 경로: {root or classifier_path})\n{detail}", gr.update(interactive=False)


def _load_model(classifier_path: str, device: str | None) -> GraphXAIModel:
    try:
        return GraphXAIModel(classifier_path=classifier_path, device=device)
    except ClassMappingError as exc:
        raise gr.Error(f"클래스 매핑 검증에 실패하여 분석을 중단합니다: {exc}")


def run_analysis(
    image_file, classifier_path: str, cam_array_file, masking_method: str,
    report_type_choice: str = "통합", include_appendix: bool = True,
):
    if image_file is None:
        raise gr.Error("MRI 이미지를 업로드해 주세요.")

    classifier_path = (classifier_path or "").strip() or os.environ.get("GRAPH_XAI_CLASSIFIER_PATH") or None
    available, _root, missing = check_model_available(classifier_path)
    if not available:
        raise gr.Error(MODEL_MISSING_MESSAGE + " (" + "; ".join(missing) + ")")

    device = os.environ.get("GRAPH_XAI_DEVICE") or None
    model = _load_model(classifier_path, device)

    image_path = _resolve_upload_path(image_file)
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    image_bytes = Path(image_path).read_bytes()

    if cam_array_file is not None:
        cam = np.load(_resolve_upload_path(cam_array_file))
        cam_method = "user_provided_cam_array"
        idx, predicted_class, probs = model.predict_original(image)
    else:
        idx, predicted_class, probs, cam = get_original_prediction_and_cam(model, image)
        cam_method = "grad_eclip_lr_logit_applied"

    grid_size = int(os.environ.get("GRAPH_XAI_GRID_SIZE", "3"))
    regions = split_into_regions(cam, image=image_np, grid=grid_size)

    results_by_method = run_all_masking_methods(
        image=image_np, cam=cam, predict_fn=model.predict_proba,
        class_names=model.class_names, original_class_idx=idx, grid_size=grid_size,
    )
    primary_results = results_by_method[masking_method]
    graph = build_region_graph(primary_results, diagonal_edges=False)

    supporting = rank_supporting_regions(primary_results)
    suppressing = rank_suppressing_regions(primary_results)
    absolute_sensitivity = rank_by_absolute_sensitivity(primary_results)
    stability = compare_masking_methods(results_by_method)
    agreement_reports = {method: compute_cam_perturbation_agreement(r, method) for method, r in results_by_method.items()}

    overlay_img = overlay_grid_on_image(image, regions)
    bar_fig = plot_probability_bar_chart(primary_results)
    graph_fig = plot_region_graph(graph)
    table_df = pd.DataFrame([dataclasses.asdict(r) for r in primary_results])

    masked_previews = {}
    for region in regions:
        masked_previews[region.name] = apply_region_mask(
            image_np, row_start=region.row_start, row_end=region.row_end,
            col_start=region.col_start, col_end=region.col_end, method=masking_method,
        )

    output_dir = _output_dir()
    class_mapping_report = model.class_mapping_report
    run_metadata = build_run_metadata(
        model_identifier=model.model_identifier, classifier_classes=model.class_names,
        cam_method=cam_method, masking_methods=list(results_by_method.keys()), grid_size=grid_size,
        device=model.device, input_bytes=image_bytes, mock_mode=False,
    )
    run_metadata["input_filename"] = Path(image_path).name

    payload = build_export_payload(
        input_filename=Path(image_path).name,
        timestamp=run_metadata["timestamp"],
        model_identifier=model.model_identifier,
        predicted_class=predicted_class,
        original_probability=float(probs[idx]),
        cam_method=cam_method,
        masking_method=masking_method,
        grid_size=grid_size,
        region_results=primary_results,
        mock_mode=False,
        warning=MEDICAL_LIMITATION_NOTICE,
        run_metadata=run_metadata,
        class_mapping_report=class_mapping_report,
        supporting_ranking=supporting,
        suppressing_ranking=suppressing,
        absolute_ranking=absolute_sensitivity,
        masking_comparison_table=build_masking_comparison_table(results_by_method),
        stability_report=stability,
        agreement_report=agreement_reports[masking_method],
    )
    run_id = run_metadata["run_id"][:10]
    csv_path = export_csv(primary_results, output_dir / f"{run_id}_regions.csv")
    json_path = export_json(payload, output_dir / f"{run_id}_report.json")
    html_path = export_html(payload, graph_fig=graph_fig, bar_fig=bar_fig, output_path=output_dir / f"{run_id}_report.html")

    class_probabilities = {name: float(p) for name, p in zip(model.class_names, probs)}
    report_mode = _resolve_report_mode(report_type_choice, include_appendix)
    pdf_path = build_graph_xai_pdf_report(
        output_dir / f"{run_id}_report.pdf",
        run_metadata=run_metadata,
        predicted_class=predicted_class,
        class_probabilities=class_probabilities,
        class_mapping_report=class_mapping_report,
        original_image=image,
        cam=cam,
        regions=regions,
        results_by_method=results_by_method,
        primary_masking_method=masking_method,
        graph=graph,
        stability_report=stability,
        agreement_reports=agreement_reports,
        mock_mode=False,
        report_mode=report_mode,
    )

    status = (
        f"predicted_class={predicted_class} | original_probability={probs[idx]:.4f} | device={model.device} | "
        f"primary_masking={masking_method} | grid={grid_size}x{grid_size} | "
        f"stability={stability.stability_verdict} | sign_agreement={stability.sign_agreement_rate:.0%} | "
        f"report_mode={report_mode}"
    )

    clinical_preview = ""
    if absolute_sensitivity:
        top = absolute_sensitivity[0].result
        clinical_preview = build_clinical_summary_sentence(
            predicted_class=predicted_class, predicted_probability=float(probs[idx]),
            primary_masking_method=masking_method, most_sensitive_region=top.region_name,
            probability_before=top.original_class_probability,
            probability_after=top.masked_original_class_probability,
            stability_verdict=stability.stability_verdict,
        )
    stability_warning = STABILITY_LOW_WARNING if stability.stability_verdict == "낮음" else ""

    return (
        overlay_img, table_df, bar_fig, graph_fig, status,
        str(csv_path), str(json_path), str(html_path), str(pdf_path),
        masked_previews, gr.update(choices=list(masked_previews.keys()), value=None),
        clinical_preview, stability_warning,
    )


def show_masked_region(region_name, masked_previews):
    if not masked_previews or region_name not in masked_previews:
        return None
    return masked_previews[region_name]


def _batch_summary_to_json_safe(summary) -> dict:
    return {
        "total_images": summary.total_images,
        "total_subjects": summary.total_subjects,
        "masking_method": summary.masking_method,
        "class_counts_images": summary.class_counts_images,
        "class_counts_subjects": summary.class_counts_subjects,
        "predicted_class_distribution_images": summary.predicted_class_distribution_images,
        "predicted_class_distribution_subjects": summary.predicted_class_distribution_subjects,
        "supporting_region_frequency_by_class": summary.supporting_region_frequency_by_class,
        "suppressing_region_frequency_by_class": summary.suppressing_region_frequency_by_class,
        "mean_probability_drop_by_region": summary.mean_probability_drop_by_region,
        "cam_perturbation_mean_agreement": summary.cam_perturbation_mean_agreement,
        "classification_metrics": summary.classification_metrics,
        "failed_samples": summary.failed_samples,
        "dedup_note": summary.dedup_note,
        "samples": [
            {
                "path": s.path, "subject_id": s.subject_id, "predicted_class": s.predicted_class,
                "original_class_probability": s.original_class_probability, "true_label": s.true_label,
            }
            for s in summary.samples
        ],
    }


def run_batch_ui(source_dir: str, classifier_path: str, masking_method: str):
    source_dir = (source_dir or "").strip()
    if not source_dir:
        raise gr.Error("배치 평가할 디렉터리 또는 CSV manifest 경로를 입력해 주세요.")

    classifier_path = (classifier_path or "").strip() or os.environ.get("GRAPH_XAI_CLASSIFIER_PATH") or None
    available, _root, missing = check_model_available(classifier_path)
    if not available:
        raise gr.Error(MODEL_MISSING_MESSAGE + " (" + "; ".join(missing) + ")")

    device = os.environ.get("GRAPH_XAI_DEVICE") or None
    model = _load_model(classifier_path, device)

    summary = run_batch_evaluation(source_dir, model, masking_method=masking_method)

    output_dir = _output_dir()
    run_id = uuid4().hex[:10]
    json_path = output_dir / f"{run_id}_batch_summary.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(_batch_summary_to_json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8")

    status = (
        f"total_images={summary.total_images} | total_subjects={summary.total_subjects} | "
        f"failed={len(summary.failed_samples)} | class_counts_subjects={summary.class_counts_subjects}"
    )
    samples_df = pd.DataFrame([
        {
            "path": s.path, "subject_id": s.subject_id, "predicted_class": s.predicted_class,
            "original_class_probability": s.original_class_probability, "true_label": s.true_label,
        }
        for s in summary.samples
    ])
    return status, samples_df, str(json_path)


def build_app() -> gr.Blocks:
    default_classifier_path = os.environ.get("GRAPH_XAI_CLASSIFIER_PATH", "")
    initial_available, initial_root, initial_missing = check_model_available(default_classifier_path or None)
    initial_status = f"모델 사용 가능: {initial_root}" if initial_available else (
        f"{MODEL_MISSING_MESSAGE}\n(확인한 경로: {initial_root})"
    )

    with gr.Blocks(title="Graph XAI Extension (Standalone, Research Use Only)") as demo:
        gr.Markdown("# Graph XAI Extension — 3x3 Region Perturbation + Graph XAI (Standalone)")
        gr.Markdown(f"**{UI_WARNING}**")

        with gr.Tabs():
            with gr.Tab("단일 이미지 분석"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.File(label="1. MRI 입력 (coronal PNG/JPG)", file_types=[".png", ".jpg", ".jpeg"], type="filepath")
                        classifier_path_input = gr.Textbox(label="2. Classifier 경로 (GRAPH_XAI_CLASSIFIER_PATH)", value=default_classifier_path)
                        check_button = gr.Button("모델 확인")
                        model_status = gr.Textbox(label="모델 로딩 상태", value=initial_status, interactive=False, lines=3)
                        cam_array_input = gr.File(label="3. (선택) CAM 배열(.npy) 직접 입력 — 비우면 Grad-ECLIP 자동 생성", file_types=[".npy"], type="filepath")
                        masking_method_input = gr.Dropdown(choices=["mean", "zero", "blur"], value="mean", label="4. 주 마스킹 방법 (세 방식 모두 실행 후 비교표 제공)")
                        report_type_input = gr.Radio(choices=list(REPORT_TYPE_CHOICES), value="통합", label="보고서 유형")
                        language_input = gr.Radio(choices=["한국어"], value="한국어", label="표시 언어", interactive=False)
                        include_appendix_input = gr.Checkbox(value=True, label="기술 부록 포함 (의료진용 선택 시에만 적용; 통합/기술 상세는 항상 포함)")
                        analyze_button = gr.Button("5. 분석 실행", variant="primary", interactive=initial_available)
                    with gr.Column(scale=2):
                        status_box = gr.Textbox(label="상태 (예측 + 마스킹 방식 간 안정성 요약)", lines=2, interactive=False)
                        overlay_image = gr.Image(label="6. 원본 이미지 + 3x3 오버레이 (해부학적 영역 아님)")

                region_table = gr.Dataframe(label="7. 주 마스킹 방법 기준 구역별 결과 (지지/억제/절대 순위는 PDF·JSON 참고)")
                bar_plot = gr.Plot(label="8. 확률 변화 막대그래프 (+ 지지 / - 억제)")
                graph_plot = gr.Plot(label="9. Graph XAI (엣지 = 공간 인접성일 뿐, 신경/구조/기능적 연결 아님)")

                with gr.Row():
                    region_dropdown = gr.Dropdown(choices=list(REGION_NAMES_3X3), label="10. 마스킹 영상을 확인할 구역")
                    masked_region_image = gr.Image(label="선택 구역의 마스킹된 영상")
                masked_previews_state = gr.State({})

                with gr.Row():
                    csv_file = gr.File(label="11. CSV 다운로드")
                    json_file = gr.File(label="JSON 다운로드")
                    html_file = gr.File(label="HTML 다운로드")
                    pdf_file = gr.File(label="PDF 보고서 다운로드 (예측 + 민감도 분석 + 연구 검토자 확인란)")

                clinical_preview_box = gr.Textbox(label="12. 의료진용 미리보기 (자동 요약 문장)", lines=4, interactive=False)
                stability_warning_box = gr.Textbox(label="설명 일관성 경고 (안정성 '낮음'일 때만 표시)", lines=2, interactive=False)

                gr.Markdown(f"**의료적 한계:** {MEDICAL_LIMITATION_NOTICE}")

                check_button.click(check_model_status, inputs=[classifier_path_input], outputs=[model_status, analyze_button])
                analyze_button.click(
                    run_analysis,
                    inputs=[
                        image_input, classifier_path_input, cam_array_input, masking_method_input,
                        report_type_input, include_appendix_input,
                    ],
                    outputs=[
                        overlay_image, region_table, bar_plot, graph_plot, status_box,
                        csv_file, json_file, html_file, pdf_file, masked_previews_state, region_dropdown,
                        clinical_preview_box, stability_warning_box,
                    ],
                )
                region_dropdown.change(show_masked_region, inputs=[region_dropdown, masked_previews_state], outputs=[masked_region_image])

            with gr.Tab("배치 평가"):
                gr.Markdown(
                    "여러 MRI 슬라이스를 한 번에 평가합니다. 디렉터리 경로 또는 "
                    "`image_path`(선택: `true_label`) 컬럼을 가진 CSV manifest 경로를 입력하세요. "
                    "동일 피험자의 여러 슬라이스는 피험자 단위로 집계되어 중복 표본으로 계산되지 않습니다."
                )
                with gr.Row():
                    batch_source_input = gr.Textbox(label="디렉터리 또는 CSV manifest 경로")
                    batch_classifier_path_input = gr.Textbox(label="Classifier 경로 (비우면 기본값)", value=default_classifier_path)
                    batch_masking_method_input = gr.Dropdown(choices=["mean", "zero", "blur"], value="mean", label="마스킹 방법")
                batch_run_button = gr.Button("배치 평가 실행", variant="primary")
                batch_status_box = gr.Textbox(label="배치 요약", interactive=False)
                batch_samples_table = gr.Dataframe(label="샘플별 결과")
                batch_json_file = gr.File(label="배치 요약 JSON 다운로드")

                batch_run_button.click(
                    run_batch_ui,
                    inputs=[batch_source_input, batch_classifier_path_input, batch_masking_method_input],
                    outputs=[batch_status_box, batch_samples_table, batch_json_file],
                )

    return demo


def main() -> None:
    build_app().launch(server_name="127.0.0.1", server_port=7861, show_error=True)


if __name__ == "__main__":
    main()
