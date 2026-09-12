"""Build a NetworkX graph over the 9 spatial grid regions.

Edges represent spatial (row/col) adjacency ONLY. They are not neural,
structural, or functional connections of any kind, biological or otherwise.
"""
from __future__ import annotations

import networkx as nx

from .schemas import REGION_NAMES_3X3


def _grid_position(name: str) -> tuple[int, int]:
    idx = list(REGION_NAMES_3X3).index(name)
    return idx // 3, idx % 3


def build_region_graph(results, diagonal_edges: bool = False) -> nx.Graph:
    graph = nx.Graph()
    positions: dict[str, tuple[int, int]] = {}

    for result in results:
        row, col = _grid_position(result.region_name)
        positions[result.region_name] = (row, col)
        graph.add_node(
            result.region_name,
            row=row,
            col=col,
            cam_mean=result.cam_mean,
            cam_ratio=result.cam_ratio,
            probability_drop=result.probability_drop,
            absolute_probability_change=result.absolute_probability_change,
            masking_method=result.masking_method,
            original_class_probability=result.original_class_probability,
            masked_original_class_probability=result.masked_original_class_probability,
        )

    names = list(positions.keys())
    for i, name_a in enumerate(names):
        row_a, col_a = positions[name_a]
        for name_b in names[i + 1:]:
            row_b, col_b = positions[name_b]
            d_row, d_col = abs(row_a - row_b), abs(col_a - col_b)
            is_rook_adjacent = (d_row + d_col == 1)
            is_diagonal_adjacent = (d_row == 1 and d_col == 1)
            if is_rook_adjacent or (diagonal_edges and is_diagonal_adjacent):
                graph.add_edge(
                    name_a, name_b,
                    edge_type="spatial_adjacency",
                    not_a_neural_connection=True,
                )
    return graph


def rank_regions_by_importance(results, key: str = "absolute_probability_change") -> list:
    return sorted(results, key=lambda r: getattr(r, key), reverse=True)
