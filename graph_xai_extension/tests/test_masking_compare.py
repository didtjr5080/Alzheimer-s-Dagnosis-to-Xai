from __future__ import annotations

import numpy as np

from graph_xai.masking_compare import ALL_MASKING_METHODS, build_masking_comparison_table, run_all_masking_methods
from graph_xai.schemas import REGION_NAMES_3X3

from _mock_model import MockGraphXAIModel


def test_runs_all_three_masking_methods_on_same_image():
    image = np.random.default_rng(0).random((30, 30, 3)).astype(np.float32) * 255
    model = MockGraphXAIModel(class_names=["AD", "CN"])
    cam = model.compute_cam(image, target_class_idx=0)

    results_by_method = run_all_masking_methods(
        image=image, cam=cam, predict_fn=model.predict_proba,
        class_names=model.class_names, original_class_idx=0,
    )
    assert set(results_by_method.keys()) == set(ALL_MASKING_METHODS)
    for method, results in results_by_method.items():
        assert len(results) == 9
        assert all(r.masking_method == method for r in results)


def test_results_across_methods_share_the_same_schema():
    image = np.random.default_rng(1).random((30, 30, 3)).astype(np.float32) * 255
    model = MockGraphXAIModel(class_names=["AD", "CN"])
    cam = model.compute_cam(image, target_class_idx=0)
    results_by_method = run_all_masking_methods(
        image=image, cam=cam, predict_fn=model.predict_proba,
        class_names=model.class_names, original_class_idx=0,
    )
    field_sets = {method: {f for r in results for f in r.__dataclass_fields__} for method, results in results_by_method.items()}
    assert len(set(map(frozenset, field_sets.values()))) == 1  # identical schema across methods


def test_comparison_table_has_one_row_per_region_with_all_method_columns():
    image = np.random.default_rng(2).random((30, 30, 3)).astype(np.float32) * 255
    model = MockGraphXAIModel(class_names=["AD", "CN"])
    cam = model.compute_cam(image, target_class_idx=0)
    results_by_method = run_all_masking_methods(
        image=image, cam=cam, predict_fn=model.predict_proba,
        class_names=model.class_names, original_class_idx=0,
    )
    table = build_masking_comparison_table(results_by_method)
    assert len(table) == 9
    assert {row["region_name"] for row in table} == set(REGION_NAMES_3X3)
    for row in table:
        for method in ALL_MASKING_METHODS:
            assert f"{method}_probability_drop" in row
            assert f"{method}_absolute_change" in row


def test_custom_subset_of_methods():
    image = np.random.default_rng(3).random((30, 30, 3)).astype(np.float32) * 255
    model = MockGraphXAIModel(class_names=["AD", "CN"])
    cam = model.compute_cam(image, target_class_idx=0)
    results_by_method = run_all_masking_methods(
        image=image, cam=cam, predict_fn=model.predict_proba,
        class_names=model.class_names, original_class_idx=0, methods=("zero", "mean"),
    )
    assert set(results_by_method.keys()) == {"zero", "mean"}
