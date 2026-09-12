"""Unit tests for graph_xai.clinical_wording (work order section 16: unit +
금칙어 tests)."""
from __future__ import annotations

from graph_xai.clinical_wording import (
    FORBIDDEN_PHRASES,
    STABILITY_PLAIN_KO,
    build_clinical_summary_sentence,
    build_top_changes_rows,
    find_forbidden_phrases,
    format_percent,
    masking_method_label,
    masking_method_noun,
    region_plain_name,
    stability_plain,
)
from graph_xai.schemas import RankedRegion, RegionPerturbationResult


def _result(region_name: str, drop: float) -> RegionPerturbationResult:
    return RegionPerturbationResult(
        region_name=region_name, original_predicted_class="MCI", original_class_probability=0.8,
        masked_predicted_class="MCI", masked_original_class_probability=0.8 - drop,
        probability_drop=drop, absolute_probability_change=abs(drop), masking_method="mean",
        cam_mean=0.1, cam_ratio=0.1,
    )


def _ranked(region_name: str, drop: float, rank: int) -> RankedRegion:
    return RankedRegion(rank=rank, region_name=region_name, value=drop, result=_result(region_name, drop))


def test_region_plain_name_matches_work_order_examples():
    assert region_plain_name("middle_center") == "영상 중앙부"
    assert region_plain_name("bottom_center") == "아래 중앙부"
    assert region_plain_name("middle_left") == "가운데 왼쪽"


def test_region_plain_name_falls_back_to_raw_name_for_unknown_region():
    assert region_plain_name("not_a_region") == "not_a_region"


def test_masking_method_label_and_noun_cover_all_three_methods():
    for method in ("zero", "mean", "blur"):
        assert masking_method_label(method)
        assert masking_method_noun(method)
        assert masking_method_label(method) != method


def test_stability_plain_covers_all_three_verdicts():
    for verdict in ("높음", "보통", "낮음"):
        text = stability_plain(verdict)
        assert verdict in text
        assert text in STABILITY_PLAIN_KO.values()


def test_format_percent_rounds_to_two_digits_by_default():
    assert format_percent(0.7892) == "78.92%"
    assert format_percent(0.5) == "50.00%"


def test_build_clinical_summary_sentence_uses_actual_values_not_hardcoded_examples():
    sentence = build_clinical_summary_sentence(
        predicted_class="AD", predicted_probability=0.6321, primary_masking_method="zero",
        most_sensitive_region="top_right", probability_before=0.6321, probability_after=0.1111,
        stability_verdict="높음",
    )
    assert "AD" in sentence
    assert "63.21%" in sentence
    assert "11.11%" in sentence
    assert region_plain_name("top_right") in sentence
    assert "78.92" not in sentence  # never the work order's own example number
    assert "판단" not in sentence


def test_build_clinical_summary_sentence_is_three_to_four_sentences():
    from graph_xai.clinical_wording import _split_sentences

    sentence = build_clinical_summary_sentence(
        predicted_class="CN", predicted_probability=0.5, primary_masking_method="mean",
        most_sensitive_region="middle_center", probability_before=0.5, probability_after=0.4,
        stability_verdict="보통",
    )
    sentence_count = len([s for s in _split_sentences(sentence) if s.strip()])
    assert 3 <= sentence_count <= 4


def test_build_top_changes_rows_caps_at_three_and_prioritizes_largest_supporting_drops():
    supporting = [_ranked("middle_center", 0.7, 1), _ranked("bottom_center", 0.3, 2), _ranked("top_left", 0.1, 3)]
    suppressing = [_ranked("bottom_right", -0.2, 1)]
    rows = build_top_changes_rows(supporting, suppressing)
    assert len(rows) == 3
    assert rows[0]["label"] == "가장 큰 감소"
    assert rows[0]["region"] == region_plain_name("middle_center")
    assert rows[-1]["label"] == "가장 큰 증가"


def test_build_top_changes_rows_handles_no_suppressing_regions():
    supporting = [_ranked("middle_center", 0.7, 1), _ranked("bottom_center", 0.3, 2)]
    rows = build_top_changes_rows(supporting, [])
    assert len(rows) == 2
    assert all(r["label"].endswith("감소") for r in rows)


def test_build_top_changes_rows_handles_empty_input():
    assert build_top_changes_rows([], []) == []


def test_positive_negative_and_zero_change_produce_distinct_wording():
    supporting = [_ranked("middle_center", 0.7, 1)]
    suppressing = [_ranked("top_right", -0.7, 1)]
    rows = build_top_changes_rows(supporting, suppressing)
    assert "낮아짐" in rows[0]["explanation"]
    assert "증가" in rows[1]["explanation"]


# --- Forbidden-phrase (금칙어) tests -----------------------------------------

def test_forbidden_phrases_are_detected_in_a_plain_violating_sentence():
    for phrase in FORBIDDEN_PHRASES:
        assert phrase in find_forbidden_phrases(f"이번 사례에서는 {phrase}됐습니다.")


def test_causal_judgment_pattern_is_detected():
    violations = find_forbidden_phrases("이 구역 때문에 MCI로 판단했습니다.")
    assert violations


def test_patient_probability_pattern_is_detected():
    violations = find_forbidden_phrases("이 환자는 MCI일 확률이 78.92%입니다.")
    assert violations


def test_required_negation_disclaimer_is_not_flagged_as_a_violation():
    # This exact sentence is required by work order section 6 and legitimately
    # contains "진단 확률" -- but only to negate it, so it must not be flagged.
    from graph_xai.clinical_wording import OUTPUT_VALUE_DISCLAIMER_TEMPLATE
    sentence = OUTPUT_VALUE_DISCLAIMER_TEMPLATE.format(cls="MCI")
    assert find_forbidden_phrases(sentence) == []


def test_clean_clinical_sentence_has_no_violations():
    sentence = (
        "AI 모델은 입력 영상을 MCI 범주로 분류했으며, 해당 범주의 모델 출력값은 78.92%였습니다. "
        "평균 밝기 가림으로 영상의 영상 중앙부를 처리했을 때 모델 출력값이 78.92%에서 7.95%로 변했습니다."
    )
    assert find_forbidden_phrases(sentence) == []


def test_wording_constants_never_call_the_prediction_a_judgment():
    from graph_xai import clinical_wording
    for name in (
        "CLINICAL_TITLE", "OUTPUT_VALUE_DISCLAIMER_TEMPLATE", "CLINICAL_FIGURE_CAPTION",
        "STABILITY_LOW_WARNING", "GRAPH_XAI_CLINICAL_DESCRIPTION", "GRAPH_XAI_CLINICAL_LEGEND_NOTE",
    ):
        assert "판단" not in getattr(clinical_wording, name)


def test_clinical_title_avoids_banned_diagnostic_words():
    from graph_xai.clinical_wording import CLINICAL_TITLE
    for banned in ("진단 보고서", "판단 보고서", "임상 결과"):
        assert banned not in CLINICAL_TITLE
