"""Real-model smoke test for the report_mode PDF pipeline end-to-end through
run_graph_xai.run_analysis (work order section 16: '실제 모델 결과를 이용한 보고서
smoke test'). Skips gracefully when the real classifier/sample are not
available locally, same convention as test_integration.py."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from pypdf import PdfReader

from graph_xai.model_adapter import check_model_available

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_HANDOFF_ROOT = REPO_ROOT / "clip_lr_grad_eclip_handoff_v1_20260812_105027"
REAL_SAMPLE_IMAGE = REAL_HANDOFF_ROOT / "data" / "xai_samples" / "images" / "OAS30009_MR_d2457_cor094.png"

_real_available, _real_root, _real_missing = check_model_available(None)


@pytest.mark.skipif(
    not (_real_available and REAL_SAMPLE_IMAGE.exists()),
    reason=f"Real classifier/sample not available locally (missing={_real_missing})",
)
def test_real_model_run_analysis_produces_a_valid_pdf_for_each_report_type(tmp_path, monkeypatch):
    if "run_graph_xai" in sys.modules:
        del sys.modules["run_graph_xai"]
    module = importlib.import_module("run_graph_xai")
    monkeypatch.setenv("GRAPH_XAI_OUTPUT_DIR", str(tmp_path))

    for report_type_choice, include_appendix in (("의료진용", False), ("기술 상세", True), ("통합", True)):
        result = module.run_analysis(
            str(REAL_SAMPLE_IMAGE), None, None, "mean",
            report_type_choice=report_type_choice, include_appendix=include_appendix,
        )
        pdf_path = Path(result[8])
        assert pdf_path.exists()
        reader = PdfReader(str(pdf_path))
        assert len(reader.pages) >= 1
        clinical_preview = result[11]
        if report_type_choice != "기술 상세":
            assert clinical_preview  # a plain-language summary sentence was generated
