from __future__ import annotations

import logging
import subprocess
import traceback
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import gradio as gr
import pandas as pd
import torch
from PIL import Image

from src.adapters.merged_cnn3d_gradcam import LATERALITY_NOT_VERIFIED_NOTE, render_overlay_image
from src.adapters.merged_grad_eclip import render_masked_overlay
from src.config import AppConfig
from src.inference import analyze_subject
from src.merged_inference import (
    analyze_merged_scan, analyze_merged_uploaded_image, list_test_scan_ids, merged_project_available,
)
from src.merged_pdf_report import (
    MERGED_FULL_XAI_LIMITATION_TEXT, build_merged_full_pdf_report, build_presentation_image_bundle,
)
from src.model_registry import registry_rows
from src.report import create_basic_pdf_report
from src.visualization import heatmap_to_rgb, overlay_heatmap
from src.warnings import RESEARCH_USE_WARNING, SINGLE_SLICE_WARNING, XAI_LIMITATION_TEXT
from src.xai import generate_representative_xai
from src.xai_artifacts import save_xai_artifact

REPO_ROOT_FOR_GIT = Path(__file__).resolve().parent.parent


def _git_commit_identifier() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT_FOR_GIT, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT_FOR_GIT, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return f"{commit}-dirty" if dirty else commit
    except Exception:
        return "N/A"

MERGED_TRAIN_VAL_WARNING = (
    "이 scan은 모델 학습(train) 또는 검증(val) 데이터에 포함되어 있어, 여기 표시된 결과는 "
    "일반화 성능 평가 근거가 아닙니다."
)


LOGGER = logging.getLogger("clip_xai_app")


def configure_logging(config: AppConfig) -> None:
    log_dir = config.artifacts_root / "verification"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / "ui_errors.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _load_uploaded_images(files) -> tuple[list[Image.Image], list[str], list[Path]]:
    if not files:
        raise ValueError("Upload one or more coronal MRI PNG slices.")

    file_paths = []
    for item in files:
        path = getattr(item, "name", item)
        file_paths.append(Path(path))
    file_paths = sorted(file_paths, key=lambda path: path.name)

    images = [Image.open(path).convert("RGB") for path in file_paths]
    return images, [path.name for path in file_paths], file_paths


def _format_subject_markdown(mr_id: str, subject_prediction) -> str:
    warning = ""
    if subject_prediction.single_slice_warning:
        warning = f"\n\n**주의:** {SINGLE_SLICE_WARNING}"
    return (
        f"### Subject Result\n"
        f"- MR_ID: `{mr_id}`\n"
        f"- Final prediction: **{subject_prediction.predicted_class}**\n"
        f"- Model output probability: **{subject_prediction.confidence:.4f}**\n"
        f"- Slices used: `{subject_prediction.n_slices_used}`"
        f"{warning}"
    )


def _artifact_markdown(artifacts) -> str:
    if not artifacts:
        return ""
    lines = ["\n\n### XAI Artifact QC"]
    for artifact in artifacts:
        p = artifact.provenance
        v = artifact.validation
        status = "unknown" if p.prediction_correct is None else "correct" if p.prediction_correct else "incorrect"
        lines.append(
            f"- `{p.slice_filename}` | true={p.true_label or 'NA'} | pred={p.predicted_class} | "
            f"target={p.target_class} | {status} | QC={v.qc_status} | analysis_id={p.analysis_id[:12]}"
        )
    return "\n".join(lines)


def _probability_table(subject_prediction) -> pd.DataFrame:
    return pd.DataFrame(
        [{"class": name, "subject_probability": prob} for name, prob in subject_prediction.probs.items()]
    )


def _slice_table(slice_names: list[str], slice_predictions) -> pd.DataFrame:
    rows = []
    for index, name in enumerate(slice_names):
        row = {
            "slice_index": index,
            "file": name,
            "predicted_class": slice_predictions.predicted_classes[index],
            "model_output_probability": float(slice_predictions.confidences[index]),
        }
        for class_index, class_name in enumerate(slice_predictions.class_names):
            row[f"{class_name}_prob"] = float(slice_predictions.probs[index, class_index])
        rows.append(row)
    return pd.DataFrame(rows)


def analyze(files, mr_id: str):
    config = AppConfig.from_env()
    configure_logging(config)

    try:
        images, slice_names, source_paths = _load_uploaded_images(files)
        resolved_mr_id = (mr_id or "").strip() or f"session_{uuid4().hex[:8]}"
        run_id = f"ui-{uuid4().hex[:12]}"

        bundle, slice_predictions, subject_prediction, representative_indices = analyze_subject(
            images, device="cpu"
        )
        xai_results = generate_representative_xai(
            images,
            representative_indices,
            target_class_idx=subject_prediction.predicted_class_idx,
            bundle=bundle,
        )

        gallery_items = []
        report_images = []
        xai_artifacts = []
        for index, xai_result in zip(representative_indices, xai_results):
            original = images[index].convert("RGB").resize((224, 224))
            heatmap = heatmap_to_rgb(xai_result.heatmap_224).convert("RGB")
            overlay = overlay_heatmap(images[index], xai_result.heatmap_224)
            slice_probs = {
                class_name: float(slice_predictions.probs[index, class_index])
                for class_index, class_name in enumerate(slice_predictions.class_names)
            }
            artifact = save_xai_artifact(
                artifacts_root=config.artifacts_root,
                repo_root=config.app_root.parent,
                source_path=source_paths[index],
                image=images[index],
                heatmap=xai_result.heatmap_224,
                xai_result=xai_result,
                slice_probability_row=slice_probs,
                subject_probabilities={name: float(value) for name, value in subject_prediction.probs.items()},
                class_names=list(slice_predictions.class_names),
                target_class_index=subject_prediction.predicted_class_idx,
                bundle=bundle,
                run_id=run_id,
            )
            xai_artifacts.append(artifact)
            truth = artifact.provenance.true_label or "NA"
            status = (
                "unknown"
                if artifact.provenance.prediction_correct is None
                else "correct"
                if artifact.provenance.prediction_correct
                else "incorrect"
            )
            slice_label = (
                f"cor{artifact.provenance.slice_index:03d}"
                if artifact.provenance.slice_index >= 0
                else f"slice {index}"
            )
            label = (
                f"{slice_label} | true={truth} | pred={artifact.provenance.predicted_class} | "
                f"target={artifact.provenance.target_class} | {status}"
            )
            gallery_items.extend([
                (original, f"{label} | original"),
                (heatmap, f"{label} | heatmap"),
                (overlay, f"{label} | overlay"),
            ])
            report_images.append({
                "title": f"대표 슬라이스 {index}: {slice_names[index]}",
                "original": original,
                "heatmap": heatmap,
                "overlay": overlay,
                "artifact": artifact,
            })

        report_path = create_basic_pdf_report(
            config.artifacts_root,
            resolved_mr_id,
            subject_prediction,
            slice_predictions,
            representative_indices,
            slice_names=slice_names,
            representative_images=report_images,
        )

        return (
            _format_subject_markdown(resolved_mr_id, subject_prediction) + _artifact_markdown(xai_artifacts),
            _probability_table(subject_prediction),
            _slice_table(slice_names, slice_predictions),
            gallery_items,
            str(report_path),
            "Analysis completed.",
        )
    except Exception as exc:
        error_id = uuid4().hex[:8]
        LOGGER.error("UI analysis failed [%s]\n%s", error_id, traceback.format_exc())
        message = (
            f"Analysis failed. Error ID: `{error_id}`. "
            "Details were written to the local verification log."
        )
        return message, pd.DataFrame(), pd.DataFrame(), [], None, str(exc)


def _merged_branch_table(branch_probs: dict) -> pd.DataFrame:
    rows = []
    for branch, probs in branch_probs.items():
        if probs is None:
            rows.append({"branch": branch, "CN": None, "MCI": None, "AD": None, "predicted_class": "사용 불가"})
            continue
        rows.append({
            "branch": branch, "CN": probs["CN"], "MCI": probs["MCI"], "AD": probs["AD"],
            "predicted_class": max(probs, key=probs.get),
        })
    return pd.DataFrame(rows)


def analyze_merged_ui(scan_id: str, image_file=None):
    config = AppConfig.from_env()
    configure_logging(config)

    try:
        image_path = getattr(image_file, "name", image_file)
        if image_path:
            result = analyze_merged_uploaded_image(image_path)
        else:
            result = analyze_merged_scan(scan_id)
        meta = result.scan_meta

        gallery_items = []
        if result.clip_xai:
            rep_image = result.clip_xai["rep_image"]
            heatmap = heatmap_to_rgb(result.clip_xai["heatmap_224"]).convert("RGB")
            overlay = render_masked_overlay(rep_image, result.clip_xai["heatmap_224"])
            clip_target = result.clip_xai.get("target_class", "N/A")
            rep_slice_index = result.clip_xai.get("rep_slice_index")
            slice_label = (
                f"CLIP axial 슬라이스 (파일명 cor{rep_slice_index:03d})" if rep_slice_index is not None
                else "CLIP 입력 이미지 (업로드됨)"
            )
            gallery_items.extend([
                (rep_image.convert("RGB"), f"{slice_label} | original"),
                (heatmap, f"{slice_label} | heatmap"),
                (overlay, f"{slice_label} | overlay, target={clip_target}"),
            ])

        for view_name, (volume_slice, cam_slice) in result.cnn3d_overlays.items():
            overlay_img = render_overlay_image(volume_slice, cam_slice)
            laterality_note = "" if view_name == "sagittal" else ", 좌우 미검증"
            gallery_items.append((overlay_img, f"3D CNN Grad-CAM | {view_name}{laterality_note}"))

        status_lines = [
            f"scan_id={meta.scan_id} | dataset_source={meta.dataset_source} | split={meta.split} | "
            f"true_label={meta.label or 'N/A'}",
        ]
        if result.ensemble_predicted_class:
            status_lines.append(f"앙상블 예측: {result.ensemble_predicted_class}")
        else:
            status_lines.append("앙상블 예측: 하나 이상의 브랜치가 사용 불가하여 계산되지 않음")
        if meta.split in ("train", "val"):
            status_lines.append(f"경고: {MERGED_TRAIN_VAL_WARNING}")

        run_metadata = {
            "run_id": uuid4().hex[:12], "timestamp": datetime.now().isoformat(),
            "git_commit_or_worktree_identifier": _git_commit_identifier(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
        report_path = build_merged_full_pdf_report(
            config.artifacts_root,
            scan_id=meta.scan_id, dataset_source=meta.dataset_source, split=meta.split,
            true_label=meta.label, branch_probs=result.branch_probs, ensemble_probs=result.ensemble_probs,
            clip_original_image=result.clip_xai["rep_image"], clip_heatmap_224=result.clip_xai["heatmap_224"],
            clip_analysis=result.clip_analysis, cnn3d_volume=result.cnn3d_volume,
            cnn3d_cam_volume=result.cnn3d_cam_volume, cnn3d_analysis=result.cnn3d_analysis,
            run_metadata=run_metadata,
        )
        images_zip_path = build_presentation_image_bundle(
            config.artifacts_root,
            identifier=run_metadata["run_id"],
            clip_original_image=result.clip_xai["rep_image"], clip_heatmap_224=result.clip_xai["heatmap_224"],
            clip_analysis=result.clip_analysis, cnn3d_volume=result.cnn3d_volume,
            cnn3d_cam_volume=result.cnn3d_cam_volume,
        )

        return (
            "\n\n".join(status_lines),
            _merged_branch_table(result.branch_probs),
            gallery_items,
            str(report_path),
            str(images_zip_path),
            "Analysis completed.",
        )
    except Exception as exc:
        error_id = uuid4().hex[:8]
        LOGGER.error("Merged UI analysis failed [%s]\n%s", error_id, traceback.format_exc())
        message = f"Analysis failed. Error ID: `{error_id}`. {exc}"
        return message, pd.DataFrame(), [], None, None, str(exc)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Alzheimer MRI XAI Decision Support Prototype") as demo:
        gr.Markdown("# Alzheimer MRI XAI Decision Support Prototype")
        gr.Markdown(f"**Research-use warning:** {RESEARCH_USE_WARNING}")
        gr.Dataframe(
            value=pd.DataFrame(registry_rows()),
            label="Model availability",
            interactive=False,
        )

        with gr.Tabs():
            with gr.Tab("통합 모델 (OASIS-3+ADNI)"):
                if merged_project_available():
                    gr.Markdown(
                        "CLIP(class-embedding 유사도) + 3D CNN + GBM baseline 앙상블. "
                        "저장소에 캐시된 scan_id를 조회하거나, 이미지를 직접 업로드해 분석할 수 있습니다. "
                        "**이미지를 업로드하면 CLIP 분석만 수행되며(3D CNN/GBM/앙상블은 MNI 정합 3D 볼륨/구조적 부피비가 "
                        "필요해 사용 불가), scan_id 입력란은 무시됩니다.**"
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            scan_id_input = gr.Dropdown(
                                choices=list_test_scan_ids(limit=50),
                                label="scan_id (test split 예시 50개 표시, 직접 입력도 가능)",
                                allow_custom_value=True,
                            )
                            merged_image_input = gr.File(
                                label="또는 이미지 업로드 (선택 사항, 업로드 시 CLIP 전용 분석)",
                                file_types=["image"],
                            )
                            merged_analyze_button = gr.Button("Analyze (통합 모델)", variant="primary")
                        with gr.Column(scale=2):
                            merged_status_markdown = gr.Markdown("scan_id를 입력하고 분석을 실행하세요.")
                            merged_status_box = gr.Textbox(label="Status", interactive=False)
                    merged_branch_table = gr.Dataframe(label="브랜치별 예측 확률 (CLIP / 3D CNN / GBM baseline)", interactive=False)
                    merged_gallery = gr.Gallery(
                        label="XAI: CLIP Grad-ECLIP (대표 슬라이스) + 3D CNN Grad-CAM (axial/coronal/sagittal)",
                        columns=3, object_fit="contain", height="auto",
                    )
                    gr.Markdown(f"**XAI limitation:** {MERGED_FULL_XAI_LIMITATION_TEXT} {LATERALITY_NOT_VERIFIED_NOTE}")
                    merged_pdf_output = gr.File(label="PDF report download")
                    merged_images_zip_output = gr.File(
                        label="발표용 이미지 모음 (zip): 원본 MRI / CLIP CAM 지도 / 3D CNN CAM 중첩 / Graph XAI(CLIP)"
                    )

                    merged_analyze_button.click(
                        analyze_merged_ui,
                        inputs=[scan_id_input, merged_image_input],
                        outputs=[
                            merged_status_markdown, merged_branch_table, merged_gallery,
                            merged_pdf_output, merged_images_zip_output, merged_status_box,
                        ],
                    )
                else:
                    gr.Markdown(
                        "**통합 모델(merged_project) 데이터를 찾을 수 없습니다.** "
                        "`merged_project/`가 저장소 루트에 있어야 이 탭을 사용할 수 있습니다."
                    )

            with gr.Tab("OASIS-3 단독 (legacy)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        file_input = gr.File(
                            label="Coronal MRI PNG slices",
                            file_count="multiple",
                            file_types=[".png"],
                            type="filepath",
                        )
                        mr_id_input = gr.Textbox(label="MR_ID (optional)", placeholder="Auto session ID if empty")
                        analyze_button = gr.Button("Analyze", variant="primary")
                    with gr.Column(scale=2):
                        result_markdown = gr.Markdown("Upload PNG slices and run analysis.")
                        subject_table = gr.Dataframe(label="Subject probabilities", interactive=False)
                        status_box = gr.Textbox(label="Status", interactive=False)

                slice_table = gr.Dataframe(label="Slice-level probabilities", interactive=False)
                gallery = gr.Gallery(
                    label="Representative slices: original / heatmap / overlay",
                    columns=3,
                    object_fit="contain",
                    height="auto",
                )
                gr.Markdown(f"**XAI limitation:** {XAI_LIMITATION_TEXT}")
                pdf_output = gr.File(label="PDF report download")

                analyze_button.click(
                    analyze,
                    inputs=[file_input, mr_id_input],
                    outputs=[result_markdown, subject_table, slice_table, gallery, pdf_output, status_box],
                )

    return demo


def main() -> None:
    config = AppConfig.from_env()
    config.validate_handoff()
    configure_logging(config)
    build_app().launch(server_name="127.0.0.1", server_port=7860, show_error=False)


if __name__ == "__main__":
    main()
