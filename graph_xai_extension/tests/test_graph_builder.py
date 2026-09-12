from __future__ import annotations

import pytest

from graph_xai.schemas import RegionPerturbationResult
from graph_xai.graph_builder import build_region_graph


def _make_results():
    from graph_xai.schemas import REGION_NAMES_3X3

    results = []
    for i, name in enumerate(REGION_NAMES_3X3):
        results.append(RegionPerturbationResult(
            region_name=name,
            original_predicted_class="AD",
            original_class_probability=0.8,
            masked_predicted_class="AD",
            masked_original_class_probability=0.8 - i * 0.05,
            probability_drop=i * 0.05,
            absolute_probability_change=i * 0.05,
            masking_method="mean",
            cam_mean=float(i),
            cam_ratio=i / 45.0,
        ))
    return results


def test_graph_has_nine_nodes():
    graph = build_region_graph(_make_results())
    assert graph.number_of_nodes() == 9


def test_default_edges_are_4_neighbor_grid_only():
    graph = build_region_graph(_make_results(), diagonal_edges=False)
    # a 3x3 grid-graph (rook adjacency) has exactly 12 edges
    assert graph.number_of_edges() == 12
    assert not graph.has_edge("top_left", "middle_center")


def test_diagonal_edges_add_king_move_adjacency():
    graph = build_region_graph(_make_results(), diagonal_edges=True)
    # king-move adjacency on a 3x3 grid has 20 edges
    assert graph.number_of_edges() == 20
    assert graph.has_edge("top_left", "middle_center")


def test_node_attributes_present():
    graph = build_region_graph(_make_results())
    node = graph.nodes["top_left"]
    assert node["row"] == 0
    assert node["col"] == 0
    assert node["cam_mean"] == pytest.approx(0.0)
    assert node["probability_drop"] == pytest.approx(0.0)
    assert node["absolute_probability_change"] == pytest.approx(0.0)
    assert "cam_ratio" in node


def test_edges_are_spatial_adjacency_only_flagged():
    graph = build_region_graph(_make_results())
    for _, _, data in graph.edges(data=True):
        assert data.get("edge_type") == "spatial_adjacency"
        assert data.get("not_a_neural_connection") is True
