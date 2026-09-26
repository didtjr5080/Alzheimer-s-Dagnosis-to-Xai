"""Focused unit tests for the small, pure building-block functions in
`src/report.py` that back PDF report generation (subject-id parsing,
uncertainty classification, probability-sum validation, and the
`_merged_*` table helpers used by `merged_pdf_report.py`). The higher-level
end-to-end PDF assembly (`create_basic_pdf_report`,
`build_merged_full_pdf_report`) is covered separately by
test_report_generation.py / test_xai_report_preflight.py /
test_merged_live_adapters.py; this file targets the helpers those tests
only exercise indirectly."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "clip_xai_app"
sys.path.insert(0, str(APP_ROOT))

from src.report import (  # noqa: E402
    ExplanationConfig,
    _artifact_slice_token,
    _assert_probabilities_sum_to_one,
    _merged_branch_prob_table,
    _merged_cell,
    _merged_kv_table,
    _prediction_status_text,
    _prob_sum,
    _short_digest,
    _styles,
    _table_style,
    classify_output_separation,
    infer_subject_id,
    validate_same_mr_id,
)


def test_infer_subject_id_matches_expected_oasis_filename_pattern():
    assert infer_subject_id("OAS30009_MR_d2457_cor094.png") == "OAS30009_MR_d2457"
    assert infer_subject_id("/some/dir/OAS30009_MR_d2457_cor005.png") == "OAS30009_MR_d2457"


def test_infer_subject_id_returns_none_for_non_matching_filename():
    assert infer_subject_id("not_a_slice.png") is None
    assert infer_subject_id("OAS30009_MR_d2457.png") is None  # missing _corNNN


def test_validate_same_mr_id_accepts_a_single_consistent_subject():
    names = ["OAS30009_MR_d2457_cor094.png", "OAS30009_MR_d2457_cor096.png"]
    assert validate_same_mr_id(names) == "OAS30009_MR_d2457"


def test_validate_same_mr_id_returns_none_when_no_filename_matches():
    assert validate_same_mr_id(["slice_0.png", "slice_1.png"]) is None


def test_validate_same_mr_id_rejects_mixed_subjects():
    names = ["OAS30009_MR_d2457_cor094.png", "OAS30010_MR_d0001_cor094.png"]
    with pytest.raises(ValueError, match="Mixed MR_ID"):
        validate_same_mr_id(names)


def test_classify_output_separation_four_verdict_bands():
    config = ExplanationConfig()
    assert classify_output_separation(0.40, 0.35, config) == "높은 불확실성"  # top1 < 0.50
    assert classify_output_separation(0.55, 0.50, config) == "경계 영역"  # margin 0.05 < 0.10
    assert classify_output_separation(0.60, 0.40, config) == "중간 수준 분리"  # margin 0.20 in [0.10, 0.25)
    assert classify_output_separation(0.90, 0.05, config) == "상대적으로 큰 클래스 분리"  # margin 0.85


def test_prob_sum_and_assert_probabilities_sum_to_one():
    valid = {"CN": 0.5, "MCI": 0.3, "AD": 0.2}
    assert _prob_sum(valid) == pytest.approx(1.0)
    _assert_probabilities_sum_to_one("test", valid)  # must not raise

    invalid = {"CN": 0.5, "MCI": 0.3, "AD": 0.1}
    with pytest.raises(ValueError, match="must sum to 1"):
        _assert_probabilities_sum_to_one("test", invalid)


def test_short_digest_passes_through_short_strings_and_truncates_long_ones():
    assert _short_digest(None) == "NA"
    assert _short_digest("short_value") == "short_value"
    long_value = "a" * 40
    digest = _short_digest(long_value)
    assert digest == f"{long_value[:12]}<br/>{long_value[-12:]}"


def test_prediction_status_text_unknown_correct_incorrect():
    assert _prediction_status_text(None, "CN") == "unknown"
    assert _prediction_status_text("CN", "CN") == "correct"
    assert _prediction_status_text("AD", "CN") == "incorrect / 오분류"


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("OAS30009_MR_d2457_cor094.png", ("cor", 94)),
        ("scan_sag012.png", ("sag", 12)),
        ("scan_axi003.png", ("ax", 3)),
        ("no_axis_token.png", (None, None)),
        ("two_tokens_cor001_sag002.png", (None, None)),  # ambiguous: must match exactly once
    ],
)
def test_artifact_slice_token_parses_axis_and_index(filename, expected):
    assert _artifact_slice_token(Path(filename)) == expected


def test_merged_cell_wraps_text_in_a_paragraph_for_column_wrapping():
    from reportlab.platypus import Paragraph

    styles, _, _ = _styles()
    cell = _merged_cell("일부러 긴 텍스트라도 컬럼 폭에 맞춰 줄바꿈되어야 합니다", styles)
    assert isinstance(cell, Paragraph)


def test_merged_branch_prob_table_marks_missing_branches_unavailable():
    styles, font_name, bold_name = _styles()
    table_style = _table_style(font_name, bold_name)
    branch_probs = {
        "CLIP": {"CN": 0.7, "MCI": 0.2, "AD": 0.1},
        "3D CNN": None,
    }
    table = _merged_branch_prob_table(branch_probs, table_style, styles)
    # header row + one row per branch
    assert len(table._cellvalues) == 3
    unavailable_row_text = table._cellvalues[2][-1].getPlainText()
    assert unavailable_row_text == "사용 불가"


def test_merged_kv_table_builds_one_row_per_pair():
    styles, font_name, bold_name = _styles()
    table_style = _table_style(font_name, bold_name)
    rows = [("scan_id", "OAS30009_MR_d2457"), ("split", "test")]
    table = _merged_kv_table(rows, [55, 109], table_style, styles)
    assert len(table._cellvalues) == 2
    assert table._cellvalues[0][0].getPlainText() == "scan_id"
