"""Static (matplotlib/Agg) renderers for the bar chart and region-adjacency
network, for embedding in the PDF report. Both 2D (CLIP, 9 nodes) and 3D
(3D CNN, 27 nodes, rendered as 3 small-multiple depth-layer panels since a
single 3D-projected layout of 27 nodes is not legible at report size).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Patch

from .graph_builder import _parse_3d_position
from .schemas import RegionPerturbationResult

_KOREAN_FONT_PATH = Path("C:/Windows/Fonts/malgun.ttf")
if _KOREAN_FONT_PATH.exists():
    font_manager.fontManager.addfont(str(_KOREAN_FONT_PATH))
    KOREAN_FONT_PROPERTIES = font_manager.FontProperties(fname=str(_KOREAN_FONT_PATH))
else:
    KOREAN_FONT_PROPERTIES = font_manager.FontProperties()

NOT_A_NEURAL_CONNECTION_NOTE = (
    "연결선은 구역이 공간적으로 이웃해 있다는 표시일 뿐이며, 실제 신경 연결을 의미하지 않습니다."
)


def render_probability_bar_png(region_results: list[RegionPerturbationResult], output_path, title: str = "") -> None:
    names = [r.region_name for r in region_results]
    drops = [r.probability_drop for r in region_results]
    colors = ["crimson" if d > 0 else ("steelblue" if d < 0 else "lightgray") for d in drops]

    fig, ax = plt.subplots(figsize=(max(7.2, len(names) * 0.28), 3.6))
    ax.bar(range(len(names)), drops, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60 if len(names) > 12 else 35, fontsize=6 if len(names) > 12 else 7, ha="right")
    ax.set_ylabel("probability_drop", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    if title:
        ax.set_title(title, fontsize=9, fontproperties=KOREAN_FONT_PROPERTIES)
    legend_handles = [
        Patch(facecolor="crimson", label="+ 지지 (마스킹 시 확률 감소)"),
        Patch(facecolor="steelblue", label="- 억제 (마스킹 시 확률 증가)"),
    ]
    ax.legend(handles=legend_handles, fontsize=6.5, loc="upper right", prop=KOREAN_FONT_PROPERTIES)
    fig.tight_layout(pad=1.2)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_region_graph_2d_png(graph, output_path, diagonal_edges: bool = False) -> None:
    positions = {name: (data["position"][1], -data["position"][0]) for name, data in graph.nodes(data=True)}
    node_names = list(graph.nodes())
    cam_ratio = np.array([graph.nodes[n]["cam_ratio"] for n in node_names], dtype=np.float64)
    prob_drop = np.array([graph.nodes[n]["probability_drop"] for n in node_names], dtype=np.float64)
    max_ratio = float(cam_ratio.max()) if cam_ratio.size and cam_ratio.max() > 0 else 1.0
    sizes = 600 + 2400 * (cam_ratio / max_ratio)

    fig, ax = plt.subplots(figsize=(5.5, 5.8))
    for u, v in graph.edges():
        x0, y0 = positions[u]
        x1, y1 = positions[v]
        ax.plot([x0, x1], [y0, y1], color="0.7", linewidth=1.2, zorder=1)

    max_abs_drop = float(np.abs(prob_drop).max()) if prob_drop.size and np.abs(prob_drop).max() > 0 else 1.0
    scatter = ax.scatter(
        [positions[n][0] for n in node_names], [positions[n][1] for n in node_names],
        s=sizes, c=prob_drop, cmap="PuOr_r", vmin=-max_abs_drop, vmax=max_abs_drop,
        edgecolors="black", linewidths=1.0, zorder=2,
    )
    for n, size in zip(node_names, sizes):
        x, y = positions[n]
        sign = "+" if graph.nodes[n]["probability_drop"] > 0 else ("-" if graph.nodes[n]["probability_drop"] < 0 else "0")
        ax.annotate(sign, (x, y), ha="center", va="center", fontsize=11, fontweight="bold", zorder=3)
        label_offset = -(10 + (size ** 0.5) / 2)
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(0, label_offset), ha="center", fontsize=6.5)

    ax.margins(0.3)
    fig.colorbar(scatter, ax=ax, shrink=0.8)
    ax.set_title("영상 구역별 예측 변화 지도", fontsize=12, fontproperties=KOREAN_FONT_PROPERTIES)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(0.5, 0.02, NOT_A_NEURAL_CONNECTION_NOTE, ha="center", fontsize=6.5, color="#8a1f11", wrap=True, fontproperties=KOREAN_FONT_PROPERTIES)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_region_graph_3d_layers_png(graph, output_path) -> None:
    """3 small-multiple panels, one per depth-layer (axis-0 third), each a
    3x3 grid -- a full 3D node-link layout of 27 nodes is not legible at
    report size, so this renders the same node/edge data as three readable
    2D facets instead."""
    node_names = list(graph.nodes())
    cam_ratio = np.array([graph.nodes[n]["cam_ratio"] for n in node_names], dtype=np.float64)
    prob_drop = np.array([graph.nodes[n]["probability_drop"] for n in node_names], dtype=np.float64)
    max_ratio = float(cam_ratio.max()) if cam_ratio.size and cam_ratio.max() > 0 else 1.0
    max_abs_drop = float(np.abs(prob_drop).max()) if prob_drop.size and np.abs(prob_drop).max() > 0 else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    scatter = None
    for layer in range(3):
        ax = axes[layer]
        layer_nodes = [n for n in node_names if _parse_3d_position(n)[0] == layer]
        for u, v in graph.edges():
            pu, pv = _parse_3d_position(u), _parse_3d_position(v)
            if pu[0] == layer and pv[0] == layer:
                ax.plot([pu[2], pv[2]], [-pu[1], -pv[1]], color="0.7", linewidth=1.0, zorder=1)
        for n in layer_nodes:
            _, h, w = _parse_3d_position(n)
            size = 300 + 1400 * (graph.nodes[n]["cam_ratio"] / max_ratio)
            drop = graph.nodes[n]["probability_drop"]
            scatter = ax.scatter(
                [w], [-h], s=[size], c=[drop], cmap="PuOr_r", vmin=-max_abs_drop, vmax=max_abs_drop,
                edgecolors="black", linewidths=0.8, zorder=2,
            )
            sign = "+" if drop > 0 else ("-" if drop < 0 else "0")
            ax.annotate(sign, (w, -h), ha="center", va="center", fontsize=8, fontweight="bold", zorder=3)
        ax.set_title(f"axis0 layer {layer}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.margins(0.35)
        for spine in ax.spines.values():
            spine.set_visible(False)
    if scatter is not None:
        fig.colorbar(scatter, ax=axes, shrink=0.7, label="probability_drop")
    fig.suptitle("3D CNN 구역별 예측 변화 지도 (axis0 3개 층으로 분할 표시)", fontsize=11, fontproperties=KOREAN_FONT_PROPERTIES)
    fig.text(0.5, 0.02, NOT_A_NEURAL_CONNECTION_NOTE, ha="center", fontsize=6.5, color="#8a1f11", fontproperties=KOREAN_FONT_PROPERTIES)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
