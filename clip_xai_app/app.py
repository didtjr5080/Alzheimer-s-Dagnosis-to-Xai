from __future__ import annotations

import logging
import traceback
from pathlib import Path
from uuid import uuid4

import gradio as gr
import pandas as pd
from PIL import Image

from src.config import AppConfig
from src.inference import analyze_subject
from src.report import create_basic_pdf_report
from src.visualization import heatmap_to_rgb, overlay_heatmap
from src.warnings import RESEARCH_USE_WARNING, SINGLE_SLICE_WARNING, XAI_LIMITATION_TEXT
from src.xai import generate_representative_xai


LOGGER = logging.getLogger("clip_xai_app")


def configure_logging(config: AppConfig) -> None:
    log_dir = config.artifacts_root / "verification"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / "ui_errors.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _load_uploaded_images(files) -> tuple[list[Image.Image], list[str]]:
    if not files:
        raise ValueError("Upload one or more coronal MRI PNG slices.")

    file_paths = []
    for item in files:
        path = getattr(item, "name", item)
        file_paths.append(Path(path))
    file_paths = sorted(file_paths, key=lambda path: path.name)

    images = [Image.open(path).convert("RGB") for path in file_paths]
    return images, [path.name for path in file_paths]


def _format_subject_markdown(mr_id: str, subject_prediction) -> str:
    warning = ""
    if subject_prediction.single_slice_warning:
        warning = f"\n\n**주의:** {SINGLE_SLICE_WARNING}"
    return (
        f"### Subject Result\n"
        f"- MR_ID: `{mr_id}`\n"
        f"- Final prediction: **{subject_prediction.predicted_class}**\n"
        f"- Confidence: **{subject_prediction.confidence:.4f}**\n"
        f"- Slices used: `{subject_prediction.n_slices_used}`"
        f"{warning}"
    )


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
            "confidence": float(slice_predictions.confidences[index]),
        }
        for class_index, class_name in enumerate(slice_predictions.class_names):
            row[f"{class_name}_prob"] = float(slice_predictions.probs[index, class_index])
        rows.append(row)
    return pd.DataFrame(rows)


def analyze(files, mr_id: str):
    config = AppConfig.from_env()
    configure_logging(config)

    try:
        images, slice_names = _load_uploaded_images(files)
        resolved_mr_id = (mr_id or "").strip() or f"session_{uuid4().hex[:8]}"

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
        for index, xai_result in zip(representative_indices, xai_results):
            original = images[index].convert("RGB").resize((224, 224))
            heatmap = heatmap_to_rgb(xai_result.heatmap_224).convert("RGB")
            overlay = overlay_heatmap(images[index], xai_result.heatmap_224)
            label = f"slice {index} | {slice_names[index]}"
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
            _format_subject_markdown(resolved_mr_id, subject_prediction),
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


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Alzheimer MRI XAI Decision Support Prototype") as demo:
        gr.Markdown("# Alzheimer MRI XAI Decision Support Prototype")
        gr.Markdown(f"**Research-use warning:** {RESEARCH_USE_WARNING}")

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
