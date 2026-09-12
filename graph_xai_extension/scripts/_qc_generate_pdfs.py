"""One-off QC helper (not part of the test suite): generates one PDF per
report_mode using the real model + a real MCI sample, for the PyMuPDF visual
QC pass. Run manually with `python`."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from graph_xai.agreement import compute_cam_perturbation_agreement
from graph_xai.cam_adapter import get_original_prediction_and_cam
from graph_xai.class_mapping import validate_class_mapping
from graph_xai.graph_builder import build_region_graph
from graph_xai.masking_compare import run_all_masking_methods
from graph_xai.model_adapter import GraphXAIModel
from graph_xai.pdf_report import build_graph_xai_pdf_report
from graph_xai.region_grid import split_into_regions
from graph_xai.run_metadata import build_run_metadata
from graph_xai.stability import compare_masking_methods

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE = REPO_ROOT / "clip_lr_grad_eclip_handoff_v1_20260812_105027" / "data" / "xai_samples" / "images" / "OAS30217_MR_d0077_cor090.png"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "qc_medical_report"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    model = GraphXAIModel(classifier_path=None, device="cpu")
    image = Image.open(SAMPLE).convert("RGB")
    idx, predicted_class, probs, cam = get_original_prediction_and_cam(model, image)
    regions = split_into_regions(cam, image=__import__("numpy").array(image), grid=3)

    results_by_method = run_all_masking_methods(
        image=__import__("numpy").array(image), cam=cam, predict_fn=model.predict_proba,
        class_names=model.class_names, original_class_idx=idx,
    )
    graph = build_region_graph(results_by_method["mean"])
    stability = compare_masking_methods(results_by_method)
    agreement_reports = {m: compute_cam_perturbation_agreement(r, m) for m, r in results_by_method.items()}
    class_mapping_report = validate_class_mapping(model.class_names)

    run_metadata = build_run_metadata(
        model_identifier=model.model_identifier, classifier_classes=model.class_names,
        cam_method="grad_eclip_lr_logit_applied", masking_methods=list(results_by_method.keys()), grid_size=3,
        device=model.device, input_bytes=SAMPLE.read_bytes(), mock_mode=False,
    )
    run_metadata["input_filename"] = SAMPLE.name
    class_probabilities = {name: float(p) for name, p in zip(model.class_names, probs)}

    print(f"predicted_class={predicted_class} probs={class_probabilities} stability={stability.stability_verdict}")

    for mode in ("clinical_summary", "technical_full", "combined"):
        out_path = OUT_DIR / f"{mode}.pdf"
        build_graph_xai_pdf_report(
            out_path,
            run_metadata=run_metadata, predicted_class=predicted_class, class_probabilities=class_probabilities,
            class_mapping_report=class_mapping_report, original_image=image, cam=cam, regions=regions,
            results_by_method=results_by_method, primary_masking_method="mean", graph=graph,
            stability_report=stability, agreement_reports=agreement_reports, mock_mode=False,
            report_mode=mode,
        )
        print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
