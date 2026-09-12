"""Tests for run_graph_xai._resolve_report_mode: maps the Korean UI radio
choice (work order section 15: 보고서 유형 선택) onto pdf_report's report_mode
values without launching the Gradio server."""
from __future__ import annotations

import importlib
import sys


def _resolve_report_mode():
    if "run_graph_xai" in sys.modules:
        del sys.modules["run_graph_xai"]
    module = importlib.import_module("run_graph_xai")
    return module._resolve_report_mode


def test_unified_choice_always_maps_to_combined():
    resolve = _resolve_report_mode()
    assert resolve("통합", True) == "combined"
    assert resolve("통합", False) == "combined"


def test_technical_choice_always_maps_to_technical_full():
    resolve = _resolve_report_mode()
    assert resolve("기술 상세", True) == "technical_full"
    assert resolve("기술 상세", False) == "technical_full"


def test_medical_choice_depends_on_appendix_toggle():
    resolve = _resolve_report_mode()
    assert resolve("의료진용", True) == "combined"
    assert resolve("의료진용", False) == "clinical_summary"
