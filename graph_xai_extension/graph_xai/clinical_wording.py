"""Plain-language Korean text and terminology guardrails for the
medical-staff-facing report body (work order: Graph_XAI_의료진용_보고서_개선).

Every sentence built here comes from real run values passed in by the
caller -- nothing is a hardcoded example number. Technical statistics
(Spearman/Kendall/Jaccard/SHA-256/git commit/mock_mode/classes_) are never
produced by this module; they stay in the technical appendix, built
separately in `pdf_report.py`'s existing section functions.

`find_forbidden_phrases` is deliberately sentence-scoped: a sentence that
negates a phrase (e.g. "...진단 확률...을 의미하지 않습니다.") is a required
disclaimer, not a violation, so any sentence containing a negation marker is
skipped. This mirrors the work order's own instruction to separate allowed
disclaimer sentences from genuinely misleading ones rather than doing a
context-free substring ban.
"""
from __future__ import annotations

import re

REGION_PLAIN_KO = {
    "top_left": "영상 위쪽 왼쪽",
    "top_center": "영상 위쪽 중앙부",
    "top_right": "영상 위쪽 오른쪽",
    "middle_left": "가운데 왼쪽",
    "middle_center": "영상 중앙부",
    "middle_right": "가운데 오른쪽",
    "bottom_left": "아래쪽 왼쪽",
    "bottom_center": "아래 중앙부",
    "bottom_right": "아래쪽 오른쪽",
}

# Verb-phrase form, for standalone display (e.g. table cells: "적용한 처리 방식").
MASKING_METHOD_LABEL_KO = {
    "zero": "검은색으로 가림",
    "mean": "평균 밝기로 가림",
    "blur": "흐리게 처리",
}

# Noun-phrase form, for embedding mid-sentence ahead of a "...으로" particle.
MASKING_METHOD_NOUN_KO = {
    "zero": "검은색 가림",
    "mean": "평균 밝기 가림",
    "blur": "흐림 처리",
}

STABILITY_PLAIN_KO = {
    "높음": "높음 - 가림 방식을 바꿔도 주요 결과가 대체로 비슷함",
    "보통": "보통 - 일부 위치의 결과가 달라짐",
    "낮음": "낮음 - 가림 방식에 따라 여러 위치의 결과 방향이나 순위가 달라짐",
}

STABILITY_LOW_WARNING = (
    "가림 방식에 따라 일부 구역의 결과가 달라졌습니다. "
    "따라서 특정 위치를 일관된 설명 위치로 단정하기 어렵습니다."
)

OUTPUT_VALUE_DISCLAIMER_TEMPLATE = (
    "이 수치는 실제 {cls} 가능성, 진단 확률 또는 모델 정확도를 의미하지 않습니다."
)

CLINICAL_FIGURE_CAPTION = (
    "색이 강하게 표시된 부분은 모델 계산에 상대적으로 많이 반영된 위치입니다. "
    "색상은 병변의 존재, 질환의 심각도 또는 조직 손상을 의미하지 않습니다."
)

CLINICAL_MASKED_IMAGE_CAPTION_TEMPLATE = (
    "처리 전 모델 출력값 {before} / 처리 후 모델 출력값 {after} / 변화량 {delta} / 적용한 처리 방식: {method}"
)

CLINICAL_TITLE = "AI MRI 분석 요약 - 연구용"

GRAPH_XAI_CLINICAL_TITLE = "영상 구역별 예측 변화 지도"
GRAPH_XAI_CLINICAL_DESCRIPTION = (
    "각 원은 영상의 한 구역을 나타냅니다. 원의 크기는 설명 지도에서의 상대 반응 정도를 나타냅니다. "
    "색상은 해당 구역을 처리한 뒤 모델 출력값이 감소했는지 증가했는지를 나타냅니다. "
    "연결선은 영상에서 서로 이웃한 구역임을 표시할 뿐 실제 신경 연결을 의미하지 않습니다."
)
GRAPH_XAI_CLINICAL_LEGEND_NOTE = (
    "색상은 질환/정상을 나타내는 색이 아닙니다 (빨강=질환, 파랑=정상이 아님). "
    "기호 +는 처리 후 출력값 감소, -는 처리 후 출력값 증가를 의미합니다."
)

RESEARCH_ONLY_HEADER_NOTE = "연구용(Research Use Only)"


def region_plain_name(region_name: str) -> str:
    return REGION_PLAIN_KO.get(region_name, region_name)


def masking_method_label(method: str) -> str:
    return MASKING_METHOD_LABEL_KO.get(method, method)


def masking_method_noun(method: str) -> str:
    return MASKING_METHOD_NOUN_KO.get(method, method)


def stability_plain(verdict: str) -> str:
    return STABILITY_PLAIN_KO.get(verdict, verdict)


def format_percent(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def build_clinical_summary_sentence(
    *,
    predicted_class: str,
    predicted_probability: float,
    primary_masking_method: str,
    most_sensitive_region: str,
    probability_before: float,
    probability_after: float,
    stability_verdict: str,
) -> str:
    """3-4 sentence auto-summary per work order section 6. All values are
    plugged in from the actual run; nothing here is a fixed example."""
    return (
        f"AI 모델은 입력 영상을 {predicted_class} 범주로 분류했으며, "
        f"해당 범주의 모델 출력값은 {format_percent(predicted_probability)}였습니다. "
        f"{masking_method_noun(primary_masking_method)}으로 영상의 {region_plain_name(most_sensitive_region)}를 "
        f"처리했을 때 모델 출력값이 {format_percent(probability_before)}에서 "
        f"{format_percent(probability_after)}로 변했습니다. "
        f"마스킹 방식 간 설명 일관성은 {stability_plain(stability_verdict).split(' - ')[0]}(으)로 평가됐습니다."
    )


def _change_magnitude_word(value: float, max_abs_value: float) -> str:
    ratio = abs(value) / max_abs_value if max_abs_value else 0.0
    if ratio >= 0.5:
        return "크게"
    if ratio >= 0.15:
        return "다소"
    return "소폭"


def build_top_changes_rows(supporting, suppressing, max_rows: int = 3) -> list[dict]:
    """Up to `max_rows` plain-language "주요 변화 요약" rows (work order
    section 8): the largest supporting drops first, then the largest
    suppressing increase, filling no more than `max_rows` total. Every value
    is read from the ranked results passed in -- never hardcoded."""
    all_values = [abs(r.result.probability_drop) for r in list(supporting) + list(suppressing)]
    max_abs = max(all_values) if all_values else 0.0

    support_labels = ["가장 큰 감소", "두 번째 감소", "세 번째 감소"]
    rows: list[dict] = []
    for i, ranked in enumerate(supporting):
        if len(rows) >= max_rows or (suppressing and len(rows) >= max_rows - 1):
            break
        r = ranked.result
        explanation = (
            "이 위치를 처리했을 때 출력값이 크게 낮아짐" if i == 0
            else "이 위치의 변화에도 모델이 민감하게 반응함"
        )
        rows.append({
            "label": support_labels[i] if i < len(support_labels) else f"{i + 1}번째 감소",
            "region": region_plain_name(r.region_name),
            "before": r.original_class_probability,
            "after": r.masked_original_class_probability,
            "explanation": explanation,
        })
    if suppressing and len(rows) < max_rows:
        r = suppressing[0].result
        magnitude = _change_magnitude_word(r.probability_drop, max_abs)
        rows.append({
            "label": "가장 큰 증가",
            "region": region_plain_name(r.region_name),
            "before": r.original_class_probability,
            "after": r.masked_original_class_probability,
            "explanation": f"처리 후 출력값이 {magnitude} 증가함",
        })
    return rows[:max_rows]


FORBIDDEN_PHRASES = (
    "병변이 검출",
    "병변 검출",
    "진단 확률",
    "질환 원인",
    "질환의 원인",
    "질환에 중요한",
    "원인이 되는 영역",
    "신경 연결이 확인",
    "신경 연결 확인",
    "진단 정확도",
    "정상 또는 비정상",
)

_NEGATION_MARKERS = ("의미하지 않습니다", "아닙니다", "않습니다")


def _split_sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+|\n+", text)


def find_forbidden_phrases(text: str) -> list[str]:
    """Returns the list of forbidden-phrase violations found in `text`,
    scanned sentence by sentence so that a required negation disclaimer
    (e.g. "...진단 확률...을 의미하지 않습니다.") is never flagged."""
    violations: list[str] = []
    for sentence in _split_sentences(text):
        if any(marker in sentence for marker in _NEGATION_MARKERS):
            continue
        for phrase in FORBIDDEN_PHRASES:
            if phrase in sentence:
                violations.append(phrase)
        if "환자는" in sentence and ("확률" in sentence or "가능성" in sentence):
            violations.append('"환자는 ... 확률/가능성" 패턴')
        if "때문에" in sentence and any(w in sentence for w in ("판단했습니다", "판단합니다", "판단됩니다")):
            violations.append('"때문에 ... 판단" 패턴')
    return violations
