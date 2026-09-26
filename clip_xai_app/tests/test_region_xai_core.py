from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "clip_xai_app"
sys.path.insert(0, str(APP_ROOT))

from src.region_xai.agreement import compute_cam_perturbation_agreement  # noqa: E402
from src.region_xai.graph_builder import build_region_graph_2d, build_region_graph_3d  # noqa: E402
from src.region_xai.masking import apply_region_mask  # noqa: E402
from src.region_xai.perturbation import run_all_masking_methods, run_region_perturbation  # noqa: E402
from src.region_xai.ranking import (  # noqa: E402
    rank_by_absolute_sensitivity, rank_supporting_regions, rank_suppressing_regions,
)
from src.region_xai.region_grid_2d import REGION_NAMES_3X3, split_into_regions_2d  # noqa: E402
from src.region_xai.region_grid_3d import split_into_regions_3d  # noqa: E402
from src.region_xai.stability import compare_masking_methods  # noqa: E402

CLASS_NAMES = ["CN", "MCI", "AD"]


def _mock_predict_2d(image: np.ndarray) -> np.ndarray:
    """Deterministic: brightness of the top-left quadrant drives MCI score."""
    brightness = image[:10, :10].mean() / 255.0
    mci = 0.1 + 0.8 * brightness
    remainder = 1.0 - mci
    return np.array([remainder * 0.5, mci, remainder * 0.5])


def _mock_predict_3d(volume: np.ndarray) -> np.ndarray:
    brightness = volume[:5, :5, :5].mean()
    mci = 0.1 + 0.8 * min(1.0, brightness)
    remainder = 1.0 - mci
    return np.array([remainder * 0.5, mci, remainder * 0.5])


def test_masking_zero_mean_blur_never_mutate_input():
    array = np.random.RandomState(0).rand(30, 30, 3) * 255
    original = array.copy()
    for method in ("zero", "mean", "blur"):
        out = apply_region_mask(array, ((0, 10), (0, 10)), method=method)
        assert out.shape == array.shape
        assert np.array_equal(array, original)  # never mutated
    zeroed = apply_region_mask(array, ((0, 10), (0, 10)), method="zero")
    assert (zeroed[:10, :10] == 0).all()
    assert np.array_equal(zeroed[10:], array[10:])


def test_2d_region_grid_covers_all_pixels_without_overlap():
    cam = np.random.RandomState(1).rand(224, 224)
    regions = split_into_regions_2d(cam, grid=3)
    assert len(regions) == 9
    assert {r.name for r in regions} == set(REGION_NAMES_3X3)
    covered = np.zeros_like(cam, dtype=bool)
    for r in regions:
        (r0, r1), (c0, c1) = r.bbox
        assert not covered[r0:r1, c0:c1].any()  # no overlap
        covered[r0:r1, c0:c1] = True
    assert covered.all()  # full coverage


def test_3d_region_grid_produces_27_nonoverlapping_regions():
    cam = np.random.RandomState(2).rand(98, 116, 94)
    regions = split_into_regions_3d(cam, grid=3)
    assert len(regions) == 27
    covered = np.zeros_like(cam, dtype=bool)
    for r in regions:
        (d0, d1), (h0, h1), (w0, w1) = r.bbox
        assert not covered[d0:d1, h0:h1, w0:w1].any()
        covered[d0:d1, h0:h1, w0:w1] = True
    assert covered.all()


def test_perturbation_and_ranking_separate_support_and_suppress():
    image = np.zeros((30, 30, 3))
    image[:10, :10] = 255  # top_left is bright -> masking it should DROP MCI
    cam = np.random.RandomState(3).rand(30, 30)
    regions = split_into_regions_2d(cam, grid=3)
    idx = CLASS_NAMES.index("MCI")

    results = run_region_perturbation(image, regions, _mock_predict_2d, CLASS_NAMES, idx, masking_method="mean")
    assert len(results) == 9
    top_left = next(r for r in results if r.region_name == "top_left")
    assert top_left.probability_drop > 0  # masking away brightness lowers MCI -> supporting

    supporting = rank_supporting_regions(results)
    suppressing = rank_suppressing_regions(results)
    absolute_sensitivity = rank_by_absolute_sensitivity(results)
    assert all(r.result.probability_drop > 0 for r in supporting)
    assert all(r.result.probability_drop < 0 for r in suppressing)
    assert len(absolute_sensitivity) == 9
    assert absolute_sensitivity[0].region_name == "top_left"  # largest |drop|


def test_stability_across_masking_methods():
    image = np.random.RandomState(4).rand(30, 30, 3) * 255
    cam = np.random.RandomState(5).rand(30, 30)
    regions = split_into_regions_2d(cam, grid=3)
    idx = CLASS_NAMES.index("MCI")
    results_by_method = run_all_masking_methods(image, regions, _mock_predict_2d, CLASS_NAMES, idx)
    assert set(results_by_method.keys()) == {"zero", "mean", "blur"}
    report = compare_masking_methods(results_by_method)
    assert report.stability_verdict in ("높음", "보통", "낮음")
    assert 0.0 <= report.sign_agreement_rate <= 1.0


def test_cam_perturbation_agreement_is_exploratory_and_preserves_contradictions():
    image = np.random.RandomState(6).rand(30, 30, 3) * 255
    cam = np.random.RandomState(7).rand(30, 30)
    regions = split_into_regions_2d(cam, grid=3)
    idx = CLASS_NAMES.index("MCI")
    results = run_region_perturbation(image, regions, _mock_predict_2d, CLASS_NAMES, idx)
    report = compute_cam_perturbation_agreement(results, "mean")
    assert report.n_regions == 9
    assert isinstance(report.high_cam_negative_drop_regions, list)  # never deleted, just listed


def test_graph_builder_2d_has_9_nodes_and_rook_adjacency():
    image = np.random.RandomState(8).rand(30, 30, 3) * 255
    cam = np.random.RandomState(9).rand(30, 30)
    regions = split_into_regions_2d(cam, grid=3)
    idx = CLASS_NAMES.index("MCI")
    results = run_region_perturbation(image, regions, _mock_predict_2d, CLASS_NAMES, idx)
    graph = build_region_graph_2d(results)
    assert graph.number_of_nodes() == 9
    assert graph.has_edge("middle_center", "top_center")
    assert not graph.has_edge("top_left", "bottom_right")


def test_graph_builder_3d_has_27_nodes_and_face_adjacency():
    volume = np.random.RandomState(10).rand(98, 116, 94)
    cam = np.random.RandomState(11).rand(98, 116, 94)
    regions = split_into_regions_3d(cam, grid=3)
    idx = CLASS_NAMES.index("MCI")
    results = run_region_perturbation(volume, regions, _mock_predict_3d, CLASS_NAMES, idx)
    graph = build_region_graph_3d(results)
    assert graph.number_of_nodes() == 27
    assert graph.has_edge("block_1_1_1", "block_1_1_0")
    assert not graph.has_edge("block_0_0_0", "block_2_2_2")
