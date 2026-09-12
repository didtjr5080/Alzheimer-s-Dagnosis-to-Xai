"""Plotly visualizations for the Graph XAI extension: the region graph, a
grid overlay on the source image, and a probability-drop bar chart.

Everywhere edges are drawn, the accompanying text makes clear that they encode
spatial adjacency only -- not a real neural, structural, or functional
connection of any kind.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageDraw

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

NOT_A_NEURAL_CONNECTION_NOTE = (
    "Edges show spatial adjacency between 3x3 grid regions only. "
    "They are NOT neural, structural, or functional brain connections."
)

_KOREAN_FONT_PATH = Path("C:/Windows/Fonts/malgun.ttf")
if _KOREAN_FONT_PATH.exists():
    font_manager.fontManager.addfont(str(_KOREAN_FONT_PATH))
    KOREAN_FONT_PROPERTIES = font_manager.FontProperties(fname=str(_KOREAN_FONT_PATH))
else:
    # Falls back to matplotlib's default (DejaVu Sans, no Hangul glyphs) --
    # Korean text will render as missing-glyph boxes on machines without
    # Malgun Gothic, same limitation the PDF's own Korean font handling has.
    KOREAN_FONT_PROPERTIES = font_manager.FontProperties()


def plot_region_graph(graph, diagonal_edges: bool = False) -> go.Figure:
    positions = {name: (data["col"], -data["row"]) for name, data in graph.nodes(data=True)}

    edge_x, edge_y = [], []
    for u, v in graph.edges():
        x0, y0 = positions[u]
        x1, y1 = positions[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1.5, color="rgba(120,120,120,0.6)"),
        hoverinfo="none", name="spatial adjacency (not a neural connection)",
    )

    node_names = list(graph.nodes())
    node_x = [positions[n][0] for n in node_names]
    node_y = [positions[n][1] for n in node_names]
    cam_ratio = np.array([graph.nodes[n]["cam_ratio"] for n in node_names], dtype=np.float64)
    prob_drop = np.array([graph.nodes[n]["probability_drop"] for n in node_names], dtype=np.float64)

    max_ratio = float(cam_ratio.max()) if cam_ratio.size and cam_ratio.max() > 0 else 1.0
    sizes = 24 + 40 * (cam_ratio / max_ratio)

    hover_text = [
        f"<b>{n}</b><br>"
        f"cam_mean={graph.nodes[n]['cam_mean']:.4f}<br>"
        f"cam_ratio={graph.nodes[n]['cam_ratio']:.4f}<br>"
        f"probability_drop={graph.nodes[n]['probability_drop']:.4f}<br>"
        f"absolute_change={graph.nodes[n]['absolute_probability_change']:.4f}<br>"
        f"masking={graph.nodes[n]['masking_method']}"
        for n in node_names
    ]

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=node_names, textposition="bottom center",
        hovertext=hover_text, hoverinfo="text",
        marker=dict(
            size=sizes, color=prob_drop, colorscale="RdBu_r",
            cmid=0.0, colorbar=dict(title="probability drop"),
            line=dict(width=1, color="black"),
        ),
        name="regions",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=f"Graph XAI: 3x3 region adjacency ({'king-move' if diagonal_edges else 'rook'} edges)",
        showlegend=False,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=60, b=80),
        annotations=[dict(
            text=NOT_A_NEURAL_CONNECTION_NOTE, showarrow=False,
            xref="paper", yref="paper", x=0.5, y=-0.12, font=dict(size=11, color="gray"),
        )],
    )
    return fig


def overlay_grid_on_image(image: Image.Image, regions) -> Image.Image:
    base = image.convert("RGB").copy()
    draw = ImageDraw.Draw(base)
    for region in regions:
        draw.rectangle(
            [region.col_start, region.row_start, region.col_end - 1, region.row_end - 1],
            outline=(255, 255, 0), width=2,
        )
    return base


def render_region_graph_png(graph, output_path, diagonal_edges: bool = False) -> None:
    """Static (matplotlib) rendering of the region graph, for embedding in the
    PDF report. No `kaleido` dependency needed for a plain PNG snapshot."""
    positions = {name: (data["col"], -data["row"]) for name, data in graph.nodes(data=True)}
    node_names = list(graph.nodes())
    cam_ratio = np.array([graph.nodes[n]["cam_ratio"] for n in node_names], dtype=np.float64)
    prob_drop = np.array([graph.nodes[n]["probability_drop"] for n in node_names], dtype=np.float64)
    max_ratio = float(cam_ratio.max()) if cam_ratio.size and cam_ratio.max() > 0 else 1.0
    sizes = 600 + 2400 * (cam_ratio / max_ratio)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for u, v in graph.edges():
        x0, y0 = positions[u]
        x1, y1 = positions[v]
        ax.plot([x0, x1], [y0, y1], color="0.7", linewidth=1.2, zorder=1)

    max_abs_drop = float(np.abs(prob_drop).max()) if prob_drop.size and np.abs(prob_drop).max() > 0 else 1.0
    scatter = ax.scatter(
        [positions[n][0] for n in node_names], [positions[n][1] for n in node_names],
        s=sizes, c=prob_drop, cmap="RdBu_r", vmin=-max_abs_drop, vmax=max_abs_drop,
        edgecolors="black", linewidths=1.0, zorder=2,
    )
    for n in node_names:
        x, y = positions[n]
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=7)

    # Without this, marker edges/labels on the outermost nodes (e.g. top_left)
    # get clipped right at the axes boundary since matplotlib's default data
    # limits hug the scatter points exactly.
    ax.margins(0.2)

    fig.colorbar(scatter, ax=ax, label="probability_drop (+ = supporting, - = suppressing)", shrink=0.8)
    ax.set_title(f"Graph XAI: 3x3 region adjacency ({'king-move' if diagonal_edges else 'rook'} edges)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(
        0.5, 0.04,
        "node size = CAM ratio | red (+) = masking DECREASED the predicted class's probability (supporting evidence) | "
        "blue (-) = masking INCREASED it (suppressing evidence)",
        ha="center", fontsize=6.5, color="dimgray", wrap=True,
    )
    fig.text(0.5, 0.005, NOT_A_NEURAL_CONNECTION_NOTE, ha="center", fontsize=6, color="gray", wrap=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_clinical_region_graph_png(graph, output_path) -> None:
    """Medical-staff-facing rendering of the region graph (work order section
    12): plain Korean title, a colorblind-safe diverging palette (PuOr,
    avoiding the red=disease/blue=normal association of RdBu), a "+"/"-"
    label on every node so meaning never depends on color alone, and an
    explicit "not a disease/normal color" legend note."""
    from .clinical_wording import (
        GRAPH_XAI_CLINICAL_DESCRIPTION,
        GRAPH_XAI_CLINICAL_LEGEND_NOTE,
        GRAPH_XAI_CLINICAL_TITLE,
    )

    positions = {name: (data["col"], -data["row"]) for name, data in graph.nodes(data=True)}
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
        # Offset scales with the marker's on-page radius so the region-name
        # label clears large nodes instead of overlapping their circle.
        label_offset = -(10 + (size ** 0.5) / 2)
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(0, label_offset), ha="center", fontsize=7)

    ax.margins(0.3)
    # The colorbar's own numeric scale is enough here; the +/- meaning is
    # already spelled out in the legend note below, so no separate rotated
    # colorbar label is needed (a rotated Korean label there was previously
    # hard to read at this figure size).
    fig.colorbar(scatter, ax=ax, shrink=0.8)
    ax.set_title(GRAPH_XAI_CLINICAL_TITLE, fontsize=12, fontproperties=KOREAN_FONT_PROPERTIES)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(0.5, 0.075, GRAPH_XAI_CLINICAL_DESCRIPTION, ha="center", fontsize=6.8, color="dimgray", wrap=True, fontproperties=KOREAN_FONT_PROPERTIES)
    fig.text(0.5, 0.01, GRAPH_XAI_CLINICAL_LEGEND_NOTE, ha="center", fontsize=6.8, color="#8a1f11", wrap=True, fontproperties=KOREAN_FONT_PROPERTIES)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_probability_bar_png(region_results, output_path) -> None:
    from matplotlib.patches import Patch

    names = [r.region_name for r in region_results]
    drops = [r.probability_drop for r in region_results]
    colors = ["crimson" if d > 0 else ("steelblue" if d < 0 else "lightgray") for d in drops]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(names, drops, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("probability_drop = original_class_probability - masked_original_class_probability", fontsize=8.5)
    # kept short deliberately: a longer label here previously got clipped by
    # tight_layout when combined with rotated x tick labels; the legend below
    # already spells out what + and - mean.
    ax.set_ylabel("probability_drop", fontsize=8)
    ax.tick_params(axis="x", labelrotation=35, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    legend_handles = [
        Patch(facecolor="crimson", label="+ supporting (masking decreased predicted-class probability)"),
        Patch(facecolor="steelblue", label="- suppressing (masking increased predicted-class probability)"),
    ]
    ax.legend(handles=legend_handles, fontsize=6.5, loc="upper right")
    fig.tight_layout(pad=1.4)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_probability_bar_chart(region_results) -> go.Figure:
    names = [r.region_name for r in region_results]
    drops = [r.probability_drop for r in region_results]
    colors = ["crimson" if d >= 0 else "steelblue" for d in drops]
    fig = go.Figure(go.Bar(x=names, y=drops, marker_color=colors))
    fig.update_layout(
        title="Probability drop of the original predicted class after masking each region",
        xaxis_title="region", yaxis_title="probability_drop (original - masked)",
        margin=dict(l=40, r=20, t=60, b=80),
    )
    return fig
