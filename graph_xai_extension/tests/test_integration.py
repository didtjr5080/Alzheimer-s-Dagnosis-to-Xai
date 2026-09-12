from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from graph_xai.cam_adapter import get_original_prediction_and_cam
from graph_xai.exporter import build_export_payload, export_csv, export_html, export_json
from graph_xai.graph_builder import build_region_graph
from graph_xai.model_adapter import GraphXAIModel, ModelUnavailableError, check_model_available
from graph_xai.perturbation import run_region_perturbation
from graph_xai.visualization import plot_probability_bar_chart, plot_region_graph

from _mock_model import MockGraphXAIModel


def _synthetic_bright_top_left_image(size=30) -> np.ndarray:
    image = np.full((size, size, 3), 20, dtype=np.uint8)
    third = size // 3
    image[0:third, 0:third] = 220
    return image


def test_full_pipeline_with_synthetic_image_and_explicit_mock_classifier(tmp_path):
    image = _synthetic_bright_top_left_image()
    model = MockGraphXAIModel(class_names=["AD", "CN"])

    idx, predicted_class, probs, cam = get_original_prediction_and_cam(model, image)
    assert predicted_class in model.class_names
    assert probs.shape == (2,)

    results = run_region_perturbation(
        image=image, cam=cam, predict_fn=model.predict_proba,
        class_names=model.class_names, original_class_idx=idx,
        masking_method="mean", grid_size=3,
    )
    assert len(results) == 9

    graph = build_region_graph(results, diagonal_edges=False)
    assert graph.number_of_nodes() == 9

    payload = build_export_payload(
        input_filename="synthetic_test_slice.png",
        timestamp="2026-09-12T00:00:00+09:00",
        model_identifier=model.model_identifier,
        predicted_class=predicted_class,
        original_probability=float(probs[idx]),
        cam_method="mock",
        masking_method="mean",
        grid_size=3,
        region_results=results,
        mock_mode=True,
    )
    assert payload["mock_mode"] is True
    assert len(payload["region_results"]) == 9

    csv_path = export_csv(results, tmp_path / "regions.csv")
    json_path = export_json(payload, tmp_path / "report.json")
    graph_fig = plot_region_graph(graph)
    bar_fig = plot_probability_bar_chart(results)
    html_path = export_html(payload, graph_fig=graph_fig, bar_fig=bar_fig, output_path=tmp_path / "report.html")

    assert csv_path.exists() and csv_path.stat().st_size > 0
    assert json_path.exists()
    reloaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert reloaded["mock_mode"] is True
    assert "patient" not in json.dumps(reloaded).lower()
    assert html_path.exists() and "Graph XAI Report" in html_path.read_text(encoding="utf-8")


def test_top_left_region_shows_largest_probability_drop_for_bright_signal():
    image = _synthetic_bright_top_left_image()
    model = MockGraphXAIModel(class_names=["AD", "CN"])
    idx, _, _, cam = get_original_prediction_and_cam(model, image)
    results = run_region_perturbation(
        image=image, cam=cam, predict_fn=model.predict_proba,
        class_names=model.class_names, original_class_idx=idx, masking_method="zero",
    )
    by_name = {r.region_name: r.absolute_probability_change for r in results}
    assert by_name["top_left"] >= max(by_name.values()) - 1e-9


def test_model_missing_path_disables_real_analysis_with_clear_error():
    available, root, missing = check_model_available("Z:/definitely/not/a/real/path.joblib")
    assert available is False
    assert missing

    with pytest.raises(ModelUnavailableError):
        GraphXAIModel(classifier_path="Z:/definitely/not/a/real/path.joblib")


REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_HANDOFF_ROOT = REPO_ROOT / "clip_lr_grad_eclip_handoff_v1_20260812_105027"
REAL_SAMPLE_IMAGE = REAL_HANDOFF_ROOT / "data" / "xai_samples" / "images" / "OAS30009_MR_d2457_cor094.png"

_real_available, _real_root, _real_missing = check_model_available(None)


@pytest.mark.skipif(
    not (_real_available and REAL_SAMPLE_IMAGE.exists()),
    reason=f"Real classifier/sample not available locally (missing={_real_missing})",
)
def test_real_model_smoke_end_to_end_on_cpu():
    model = GraphXAIModel(classifier_path=None, device="cpu")
    assert model.mock_mode is False
    assert set(model.class_names) == {"CN", "MCI", "AD"}

    image = Image.open(REAL_SAMPLE_IMAGE).convert("RGB")
    idx, predicted_class, probs, cam = get_original_prediction_and_cam(model, image)
    assert predicted_class in model.class_names
    assert np.isfinite(cam).all()

    results = run_region_perturbation(
        image=image, cam=cam, predict_fn=model.predict_proba,
        class_names=model.class_names, original_class_idx=idx,
        masking_method="mean", grid_size=3,
    )
    assert len(results) == 9
    for r in results:
        assert np.isfinite(r.probability_drop)
        assert r.original_predicted_class == predicted_class


def test_standalone_ui_module_imports_and_builds_blocks_without_launching():
    if "run_graph_xai" in sys.modules:
        del sys.modules["run_graph_xai"]
    module = importlib.import_module("run_graph_xai")
    demo = module.build_app()
    assert demo is not None
    assert type(demo).__name__ == "Blocks"
