"""Region-adjacency graph (NetworkX) for both the 2D (3x3=9 node) and 3D
(3x3x3=27 node) grids. Edges encode spatial adjacency ONLY -- never a real
neural, structural, or functional connection of any kind."""
from __future__ import annotations

import networkx as nx

from .region_grid_2d import REGION_NAMES_3X3
from .schemas import RegionPerturbationResult

_POSITION_2D = {name: (index // 3, index % 3) for index, name in enumerate(REGION_NAMES_3X3)}


def _parse_3d_position(region_name: str) -> tuple[int, int, int]:
    _, d, h, w = region_name.split("_")
    return int(d), int(h), int(w)


def _add_nodes(graph: nx.Graph, results: list[RegionPerturbationResult], positions: dict) -> None:
    for result in results:
        pos = positions[result.region_name]
        graph.add_node(
            result.region_name, position=pos,
            cam_mean=result.cam_mean, cam_ratio=result.cam_ratio,
            probability_drop=result.probability_drop,
            absolute_probability_change=result.absolute_probability_change,
            masking_method=result.masking_method,
        )


def build_region_graph_2d(results: list[RegionPerturbationResult], diagonal_edges: bool = False) -> nx.Graph:
    graph = nx.Graph()
    _add_nodes(graph, results, _POSITION_2D)
    names = list(graph.nodes())
    for i, a in enumerate(names):
        ra, ca = graph.nodes[a]["position"]
        for b in names[i + 1:]:
            rb, cb = graph.nodes[b]["position"]
            dr, dc = abs(ra - rb), abs(ca - cb)
            is_rook = (dr + dc == 1)
            is_diagonal = (dr == 1 and dc == 1)
            if is_rook or (diagonal_edges and is_diagonal):
                graph.add_edge(a, b)
    return graph


def build_region_graph_3d(results: list[RegionPerturbationResult]) -> nx.Graph:
    """Face-adjacency (6-connectivity) in the 3x3x3 grid."""
    positions = {r.region_name: _parse_3d_position(r.region_name) for r in results}
    graph = nx.Graph()
    _add_nodes(graph, results, positions)
    names = list(graph.nodes())
    for i, a in enumerate(names):
        pa = graph.nodes[a]["position"]
        for b in names[i + 1:]:
            pb = graph.nodes[b]["position"]
            manhattan = sum(abs(x - y) for x, y in zip(pa, pb))
            if manhattan == 1:
                graph.add_edge(a, b)
    return graph
