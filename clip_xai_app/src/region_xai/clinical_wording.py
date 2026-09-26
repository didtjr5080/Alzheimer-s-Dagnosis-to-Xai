"""Plain-language Korean text for the merged-model clinical report, mirroring
`graph_xai_extension`'s wording rules (never call the prediction a "판단",
never claim a region caused the prediction, supporting vs suppressing are
never merged). Independently reimplemented -- does not import
`graph_xai_extension`.
"""
from __future__ import annotations

REGION_PLAIN_KO_2D = {
    "top_left": "영상 위쪽 왼쪽", "top_center": "영상 위쪽 중앙부", "top_right": "영상 위쪽 오른쪽",
    "middle_left": "가운데 왼쪽", "middle_center": "영상 중앙부", "middle_right": "가운데 오른쪽",
    "bottom_left": "아래쪽 왼쪽", "bottom_center": "아래 중앙부", "bottom_right": "아래쪽 오른쪽",
}

MASKING_METHOD_LABEL_KO = {"zero": "검은색으로 가림", "mean": "평균 밝기로 가림", "blur": "흐리게 처리"}
MASKING_METHOD_NOUN_KO = {"zero": "검은색 가림", "mean": "평균 밝기 가림", "blur": "흐림 처리"}

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

REGION_3D_DISCLAIMER = (
    "3D 구역명(block_d_h_w)은 배열 좌표일 뿐이며, 해부학적 방향(좌/우/전/후)은 검증되지 않았습니다."
)


def region_plain_name_2d(region_name: str) -> str:
    return REGION_PLAIN_KO_2D.get(region_name, region_name)


def region_plain_name_3d(region_name: str) -> str:
    return f"3D 구역 {region_name}"


def masking_method_label(method: str) -> str:
    return MASKING_METHOD_LABEL_KO.get(method, method)


def masking_method_noun(method: str) -> str:
    return MASKING_METHOD_NOUN_KO.get(method, method)


def stability_plain(verdict: str) -> str:
    return STABILITY_PLAIN_KO.get(verdict, verdict)


def format_percent(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def build_branch_summary_sentence(
    *, branch_label: str, predicted_class: str, predicted_probability: float, primary_masking_method: str,
    most_sensitive_region_plain: str, probability_before: float, probability_after: float, stability_verdict: str,
) -> str:
    return (
        f"{branch_label}은 입력을 {predicted_class} 범주로 분류했으며, 해당 범주의 모델 출력값은 "
        f"{format_percent(predicted_probability)}였습니다. "
        f"{masking_method_noun(primary_masking_method)}으로 {most_sensitive_region_plain}를 처리했을 때 "
        f"모델 출력값이 {format_percent(probability_before)}에서 {format_percent(probability_after)}로 변했습니다. "
        f"마스킹 방식 간 설명 일관성은 {stability_plain(stability_verdict).split(' - ')[0]}(으)로 평가됐습니다."
    )


def build_top_changes_rows(supporting, suppressing, region_plain_fn, max_rows: int = 3) -> list[dict]:
    all_values = [abs(r.result.probability_drop) for r in list(supporting) + list(suppressing)]
    max_abs = max(all_values) if all_values else 0.0

    support_labels = ["가장 큰 감소", "두 번째 감소", "세 번째 감소"]
    rows: list[dict] = []
    for i, ranked in enumerate(supporting):
        if len(rows) >= max_rows or (suppressing and len(rows) >= max_rows - 1):
            break
        r = ranked.result
        explanation = "이 위치를 처리했을 때 출력값이 크게 낮아짐" if i == 0 else "이 위치의 변화에도 모델이 민감하게 반응함"
        rows.append({
            "label": support_labels[i] if i < len(support_labels) else f"{i + 1}번째 감소",
            "region": region_plain_fn(r.region_name),
            "before": r.original_class_probability, "after": r.masked_original_class_probability,
            "explanation": explanation,
        })
    if suppressing and len(rows) < max_rows:
        r = suppressing[0].result
        ratio = abs(r.probability_drop) / max_abs if max_abs else 0.0
        magnitude = "크게" if ratio >= 0.5 else ("다소" if ratio >= 0.15 else "소폭")
        rows.append({
            "label": "가장 큰 증가", "region": region_plain_fn(r.region_name),
            "before": r.original_class_probability, "after": r.masked_original_class_probability,
            "explanation": f"처리 후 출력값이 {magnitude} 증가함",
        })
    return rows[:max_rows]
