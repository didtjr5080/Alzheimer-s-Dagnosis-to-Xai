"""CSV / JSON / HTML export for a Graph XAI run.

No patient identifiers or full local filesystem paths are written -- only the
uploaded file's basename, and only the fields listed in the work order.
"""
from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path

MEDICAL_LIMITATION_NOTICE = (
    "This is an explainable-AI (XAI) result showing how much the model's score relied on each "
    "image region. Highlighted regions are NOT evidence of an actual lesion location, the cause "
    "of Alzheimer's disease, or a clinical diagnosis. The 3x3 regions are arbitrary spatial grid "
    "cells, not anatomical brain regions. CAM is an approximate explanation; results vary by "
    "masking method. This output is for research use only and is not for clinical diagnosis."
)


def region_results_to_rows(region_results) -> list[dict]:
    return [dataclasses.asdict(r) for r in region_results]


def _ranked_rows(ranking) -> list[dict]:
    if not ranking:
        return []
    return [
        {"rank": r.rank, "region_name": r.region_name, "value": r.value, **dataclasses.asdict(r.result)}
        for r in ranking
    ]


def build_export_payload(
    input_filename: str,
    timestamp: str,
    model_identifier: str,
    predicted_class: str,
    original_probability: float,
    cam_method: str,
    masking_method: str,
    grid_size: int,
    region_results,
    mock_mode: bool,
    warning: str = MEDICAL_LIMITATION_NOTICE,
    run_metadata: dict | None = None,
    class_mapping_report=None,
    supporting_ranking=None,
    suppressing_ranking=None,
    absolute_ranking=None,
    masking_comparison_table: list[dict] | None = None,
    stability_report=None,
    agreement_report=None,
) -> dict:
    """Builds the JSON export payload. All parameters after `warning` are
    optional and additive -- omitting them reproduces exactly the original
    (pre-improvement) single-masking-method payload shape, so existing
    callers and regression tests are unaffected."""
    payload = {
        "input_filename": Path(input_filename).name,
        "timestamp": timestamp,
        "model_identifier": model_identifier,
        "predicted_class": predicted_class,
        "original_probability": float(original_probability),
        "cam_method": cam_method,
        "masking_method": masking_method,
        "grid_size": grid_size,
        "region_results": region_results_to_rows(region_results),
        "warning": warning,
        "mock_mode": bool(mock_mode),
    }
    if run_metadata is not None:
        payload["run_metadata"] = run_metadata
    if class_mapping_report is not None:
        payload["class_mapping_report"] = dataclasses.asdict(class_mapping_report)
    if supporting_ranking is not None:
        payload["supporting_ranking"] = _ranked_rows(supporting_ranking)
    if suppressing_ranking is not None:
        payload["suppressing_ranking"] = _ranked_rows(suppressing_ranking)
    if absolute_ranking is not None:
        payload["absolute_sensitivity_ranking"] = _ranked_rows(absolute_ranking)
    if masking_comparison_table is not None:
        payload["masking_comparison_table"] = masking_comparison_table
    if stability_report is not None:
        payload["stability_report"] = dataclasses.asdict(stability_report)
    if agreement_report is not None:
        payload["cam_perturbation_agreement"] = dataclasses.asdict(agreement_report)
    return payload


def export_json(payload: dict, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def export_csv(region_results, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = region_results_to_rows(region_results)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path


def export_html(payload: dict, graph_fig=None, bar_fig=None, output_path: str | Path = "report.html") -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_html = "".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in row.values()) + "</tr>"
        for row in payload["region_results"]
    )
    headers_html = "".join(f"<th>{key}</th>" for key in (payload["region_results"][0].keys() if payload["region_results"] else []))

    graph_div = graph_fig.to_html(full_html=False, include_plotlyjs="cdn") if graph_fig is not None else ""
    bar_div = bar_fig.to_html(full_html=False, include_plotlyjs=False) if bar_fig is not None else ""

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Graph XAI Report</title></head>
<body>
<h1>Graph XAI Report</h1>
<p><b>Warning:</b> {payload['warning']}</p>
<p>input_filename: {payload['input_filename']} | timestamp: {payload['timestamp']} |
model_identifier: {payload['model_identifier']} | mock_mode: {payload['mock_mode']}</p>
<p>predicted_class: {payload['predicted_class']} | original_probability: {payload['original_probability']:.4f} |
cam_method: {payload['cam_method']} | masking_method: {payload['masking_method']} | grid_size: {payload['grid_size']}</p>
{graph_div}
{bar_div}
<table border="1" cellspacing="0" cellpadding="4">
<tr>{headers_html}</tr>
{rows_html}
</table>
</body></html>"""
    output_path.write_text(html, encoding="utf-8")
    return output_path
