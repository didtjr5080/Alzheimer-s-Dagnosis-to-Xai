from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal
from uuid import uuid4
from zoneinfo import ZoneInfo

from PIL import Image as PILImage
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


KST = ZoneInfo("Asia/Seoul")


@dataclass(frozen=True)
class SlicePrediction:
    index: int
    filename: str
    probabilities: dict[str, float]
    predicted_class: str


@dataclass(frozen=True)
class SubjectReportPrediction:
    mr_id: str
    slice_count: int
    probabilities: dict[str, float]
    predicted_class: str
    aggregation_method: str = "mean_slice_probabilities"


@dataclass(frozen=True)
class ClassScore:
    class_index: int
    class_name: str
    probability: float


@dataclass(frozen=True)
class ExplanationConfig:
    version: str = "explanation-rule-v1"
    top1_low_threshold: float = 0.50
    margin_boundary_threshold: float = 0.10
    margin_moderate_threshold: float = 0.25
    clinical_validation: bool = False


@dataclass(frozen=True)
class PredictionEvidence:
    analysis_mode: Literal["single_slice", "multi_slice_subject"]
    mr_id: str
    input_filenames: list[str]
    slice_count: int
    ordered_scores: list[ClassScore]
    top1_class: str
    top1_probability: float
    top2_class: str
    top2_probability: float
    probability_margin: float
    representative_slice_indices: list[int]
    top_contributing_slice: str
    slice_prediction_agreement: float | None
    foreground_heatmap_mean: float | None
    background_heatmap_mean: float | None
    background_activation_warning: bool
    uncertainty_level: str
    uncertainty_rule_version: str
    warnings: list[str]


@dataclass(frozen=True)
class ReportPayload:
    report_id: str
    created_at: datetime
    timezone_name: str
    mr_id: str
    analysis_mode: Literal["single_slice", "multi_slice_subject"]
    slices: list[SlicePrediction]
    subject_prediction: SubjectReportPrediction | None
    representative_indices: list[int]
    xai_items: list[dict]
    is_single_slice_proxy: bool
    evidence: PredictionEvidence


def ensure_report_dir(artifacts_root: Path) -> Path:
    report_dir = artifacts_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


def infer_subject_id(filename: str) -> str | None:
    match = re.match(r"^(OAS\d+_MR_d\d+)_cor\d+\.png$", Path(filename).name)
    return match.group(1) if match else None


def validate_same_mr_id(slice_names: list[str]) -> str | None:
    inferred = [infer_subject_id(name) for name in slice_names]
    known = {value for value in inferred if value}
    if len(known) > 1:
        raise ValueError(f"Mixed MR_ID inputs are not allowed: {sorted(known)}")
    return next(iter(known)) if known else None


def classify_output_separation(top1: float, top2: float, config: ExplanationConfig) -> str:
    margin = top1 - top2
    if top1 < config.top1_low_threshold:
        return "높은 불확실성"
    if margin < config.margin_boundary_threshold:
        return "경계 영역"
    if margin < config.margin_moderate_threshold:
        return "중간 수준 분리"
    return "상대적으로 큰 클래스 분리"


def _ordered_scores(probabilities: dict[str, float], class_order: list[str]) -> list[ClassScore]:
    scores = [
        ClassScore(class_index=index, class_name=class_name, probability=float(probabilities[class_name]))
        for index, class_name in enumerate(class_order)
    ]
    return sorted(scores, key=lambda score: score.probability, reverse=True)


def _top_contributing_slice(
    slices: list[SlicePrediction],
    top1_class: str,
    representative_indices: list[int],
) -> str:
    if not slices:
        return "N/A"
    if len(slices) == 1:
        return slices[0].filename

    # "Contribution" is operationalized as the largest slice-level probability
    # for the final class, not as an anatomical or causal contribution.
    best = max(slices, key=lambda item: item.probabilities.get(top1_class, 0.0))
    return best.filename


def build_prediction_evidence(
    probabilities: dict[str, float],
    class_order: list[str],
    analysis_mode: Literal["single_slice", "multi_slice_subject"],
    mr_id: str,
    filenames: list[str],
    slices: list[SlicePrediction],
    representative_indices: list[int],
    config: ExplanationConfig,
) -> PredictionEvidence:
    ordered = _ordered_scores(probabilities, class_order)
    top1 = ordered[0]
    top2 = ordered[1]
    margin = top1.probability - top2.probability
    agreement = None
    warnings = []

    if analysis_mode == "multi_slice_subject":
        agreement = sum(item.predicted_class == top1.class_name for item in slices) / len(slices)
    else:
        warnings.append(
            "단일 슬라이스 결과이므로 subject 단위 슬라이스 예측 일치율은 계산하지 않았습니다."
        )

    warnings.append(
        "히트맵의 강조 영역은 선택된 클래스 logit에 대한 모델 내부 기여 영역이며 의학적 의미를 부여할 수 없습니다."
    )

    return PredictionEvidence(
        analysis_mode=analysis_mode,
        mr_id=mr_id,
        input_filenames=filenames,
        slice_count=len(filenames),
        ordered_scores=ordered,
        top1_class=top1.class_name,
        top1_probability=top1.probability,
        top2_class=top2.class_name,
        top2_probability=top2.probability,
        probability_margin=margin,
        representative_slice_indices=representative_indices,
        top_contributing_slice=_top_contributing_slice(slices, top1.class_name, representative_indices),
        slice_prediction_agreement=agreement,
        foreground_heatmap_mean=None,
        background_heatmap_mean=None,
        background_activation_warning=True,
        uncertainty_level=classify_output_separation(top1.probability, top2.probability, config),
        uncertainty_rule_version=config.version,
        warnings=warnings,
    )


def build_report_payload(
    mr_id: str,
    subject_prediction,
    slice_predictions,
    representative_indices: list[int],
    slice_names: list[str] | None = None,
    representative_images: list[dict] | None = None,
) -> ReportPayload:
    slice_names = slice_names or [f"slice_{i}" for i in range(slice_predictions.probs.shape[0])]
    inferred_mr_id = validate_same_mr_id(slice_names)
    resolved_mr_id = inferred_mr_id or mr_id

    slices = []
    for index, name in enumerate(slice_names):
        probabilities = {
            class_name: float(slice_predictions.probs[index, class_index])
            for class_index, class_name in enumerate(slice_predictions.class_names)
        }
        slices.append(SlicePrediction(
            index=index,
            filename=name,
            probabilities=probabilities,
            predicted_class=slice_predictions.predicted_classes[index],
        ))

    subject_report = None
    is_single_slice_proxy = len(slices) == 1
    analysis_mode: Literal["single_slice", "multi_slice_subject"] = "single_slice" if is_single_slice_proxy else "multi_slice_subject"
    class_order = list(slice_predictions.class_names)
    evidence_probabilities = slices[0].probabilities
    if len(slices) >= 2:
        subject_report = SubjectReportPrediction(
            mr_id=resolved_mr_id,
            slice_count=len(slices),
            probabilities={name: float(prob) for name, prob in subject_prediction.probs.items()},
            predicted_class=subject_prediction.predicted_class,
        )
        evidence_probabilities = subject_report.probabilities

    config = ExplanationConfig()
    evidence = build_prediction_evidence(
        probabilities=evidence_probabilities,
        class_order=class_order,
        analysis_mode=analysis_mode,
        mr_id=resolved_mr_id,
        filenames=slice_names,
        slices=slices,
        representative_indices=representative_indices,
        config=config,
    )

    created_at = datetime.now(KST)
    return ReportPayload(
        report_id=f"report_{created_at.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}",
        created_at=created_at,
        timezone_name="Asia/Seoul",
        mr_id=resolved_mr_id,
        analysis_mode=analysis_mode,
        slices=slices,
        subject_prediction=subject_report,
        representative_indices=representative_indices,
        xai_items=representative_images or [],
        is_single_slice_proxy=is_single_slice_proxy,
        evidence=evidence,
    )


def _register_korean_font() -> tuple[str, str]:
    font_path = Path("C:/Windows/Fonts/malgun.ttf")
    bold_path = Path("C:/Windows/Fonts/malgunbd.ttf")
    if font_path.exists():
        if "MalgunGothic" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("MalgunGothic", str(font_path)))
        if bold_path.exists() and "MalgunGothic-Bold" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("MalgunGothic-Bold", str(bold_path)))
        return "MalgunGothic", "MalgunGothic-Bold" if bold_path.exists() else "MalgunGothic"
    return "Helvetica", "Helvetica-Bold"


def _styles():
    font_name, bold_name = _register_korean_font()
    styles = getSampleStyleSheet()
    for style_name in ("Normal", "BodyText", "Title", "Heading1", "Heading2"):
        styles[style_name].fontName = font_name
    styles["Title"].fontName = bold_name
    styles["Title"].fontSize = 16
    styles["Title"].leading = 20
    styles["Heading1"].fontName = bold_name
    styles["Heading2"].fontName = bold_name
    styles["Heading2"].fontSize = 11
    styles["Heading2"].leading = 14
    styles["BodyText"].fontSize = 9
    styles["BodyText"].leading = 12
    styles["Normal"].fontSize = 9
    styles["Normal"].leading = 12
    styles.add(ParagraphStyle(
        name="SmallKorean",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=8,
        leading=10,
    ))
    return styles, font_name, bold_name


def _table_style(font_name: str, bold_name: str) -> TableStyle:
    return TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("FONTNAME", (0, 0), (-1, 0), bold_name),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#9aa4b2")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ])


def _footer(report_id: str, font_name: str):
    def draw(canvas, doc):
        canvas.saveState()
        canvas.setFont(font_name, 8)
        footer = f"Research Use Only | 보고서 ID: {report_id} | Page {doc.page}"
        canvas.drawCentredString(A4[0] / 2, 10 * mm, footer)
        canvas.restoreState()
    return draw


def _prob_rows(probabilities: dict[str, float]) -> list[list[str]]:
    return [[name, f"{prob:.4f}", f"{prob * 100:.2f}%"] for name, prob in probabilities.items()]


def _save_temp_image(image: PILImage.Image, path: Path) -> Path:
    image.convert("RGB").save(path, format="PNG")
    return path


def _model_info_table(table_style: TableStyle) -> Table:
    table = Table([
        ["항목", "값"],
        ["CLIP 모델", "openai/clip-vit-base-patch16"],
        ["이미지 크기", "224x224"],
        ["Patch/Grid", "16x16 / 14x14"],
        ["임베딩 차원", "512"],
        ["분류기", "sklearn Logistic Regression"],
        ["클래스 순서", "clf.classes_ 기준 CN, MCI, AD"],
        ["Subject 집계", "슬라이스별 모델 출력 확률 단순 평균"],
        ["용도", "연구용 프로토타입"],
    ], colWidths=[38 * mm, 126 * mm])
    table.setStyle(table_style)
    return table


def _slice_summary_table(payload: ReportPayload, table_style: TableStyle) -> Table:
    rows = [["인덱스", "파일명", "모델 예측 클래스", "CN", "MCI", "AD"]]
    for item in payload.slices:
        rows.append([
            str(item.index),
            item.filename,
            item.predicted_class,
            f"{item.probabilities.get('CN', 0.0):.4f}",
            f"{item.probabilities.get('MCI', 0.0):.4f}",
            f"{item.probabilities.get('AD', 0.0):.4f}",
        ])
    table = Table(rows, repeatRows=1, colWidths=[13 * mm, 62 * mm, 31 * mm, 19 * mm, 19 * mm, 19 * mm])
    table.setStyle(table_style)
    return table


def _single_slice_section(payload: ReportPayload, styles, table_style: TableStyle) -> list:
    item = payload.slices[0]
    top_prob = max(item.probabilities.values())
    table = Table(
        [["클래스", "모델 출력 확률", "표시 백분율"], *_prob_rows(item.probabilities)],
        colWidths=[35 * mm, 45 * mm, 45 * mm],
    )
    table.setStyle(table_style)
    return [
        Paragraph("단일 슬라이스 모델 예측", styles["Heading2"]),
        Paragraph("분석 단위: 단일 슬라이스", styles["BodyText"]),
        Paragraph(f"입력 파일: {item.filename}", styles["BodyText"]),
        Paragraph("모델 출력 확률", styles["Heading2"]),
        table,
        Spacer(1, 6),
        Paragraph(f"모델 예측 클래스: {item.predicted_class}", styles["BodyText"]),
        Paragraph(f"최대 모델 출력 확률: {top_prob * 100:.2f}%", styles["BodyText"]),
        Spacer(1, 6),
        Paragraph(
            "주의: 본 결과는 단일 슬라이스 예측입니다. 학습 및 검증에 사용된 동일 subject의 "
            "다중 슬라이스 확률 평균 방식과 동일하지 않으며 subject 단위 결과로 해석할 수 없습니다.",
            styles["BodyText"],
        ),
    ]


def _subject_section(payload: ReportPayload, styles, table_style: TableStyle) -> list:
    subject = payload.subject_prediction
    assert subject is not None
    table = Table(
        [["클래스", "Subject 평균 모델 출력 확률", "표시 백분율"], *_prob_rows(subject.probabilities)],
        colWidths=[35 * mm, 60 * mm, 45 * mm],
    )
    table.setStyle(table_style)
    top_prob = max(subject.probabilities.values())
    return [
        Paragraph("Subject 단위 모델 예측", styles["Heading2"]),
        Paragraph("분석 단위: 다중 슬라이스 subject 평균", styles["BodyText"]),
        Paragraph("집계 방식: 슬라이스별 클래스 확률의 단순 평균", styles["BodyText"]),
        Paragraph(f"입력 슬라이스 수: {subject.slice_count}", styles["BodyText"]),
        Paragraph("Subject 평균 모델 출력 확률", styles["Heading2"]),
        table,
        Spacer(1, 6),
        Paragraph(f"모델 예측 클래스: {subject.predicted_class}", styles["BodyText"]),
        Paragraph(f"최대 모델 출력 확률: {top_prob * 100:.2f}%", styles["BodyText"]),
    ]


def _prediction_evidence_section(payload: ReportPayload, styles) -> list:
    evidence = payload.evidence
    margin_percent = evidence.probability_margin * 100.0
    top1_percent = evidence.top1_probability * 100.0
    top2_percent = evidence.top2_probability * 100.0

    if evidence.analysis_mode == "single_slice":
        classification_text = (
            f"모델은 {evidence.top1_class} 클래스에 {top1_percent:.2f}%로 가장 높은 모델 출력 확률을 "
            f"보였습니다. 두 번째 후보인 {evidence.top2_class}의 출력 확률은 {top2_percent:.2f}%이며, "
            f"두 클래스의 차이는 {margin_percent:.2f}%p입니다."
        )
        compact_text = (
            f"요약: {evidence.top1_class} 클래스에 {top1_percent:.2f}%; "
            f"{evidence.top2_class}의 출력 확률은 {top2_percent:.2f}%; "
            f"차이는 {margin_percent:.2f}%p."
        )
        slice_text = (
            f"본 결과는 {evidence.input_filenames[0]} 1장을 기준으로 계산했습니다. 단일 슬라이스이므로 "
            "subject 내부 슬라이스 간 예측 일치율은 계산하지 않았습니다."
        )
    else:
        agree_count = round((evidence.slice_prediction_agreement or 0.0) * evidence.slice_count)
        classification_text = (
            f"전체 {evidence.slice_count}개 슬라이스의 클래스 확률을 단순 평균한 결과, "
            f"{evidence.top1_class}가 {top1_percent:.2f}%로 가장 높았습니다. 두 번째 후보인 "
            f"{evidence.top2_class}는 {top2_percent:.2f}%이며, 차이는 {margin_percent:.2f}%p입니다."
        )
        compact_text = (
            f"요약: {evidence.top1_class} 클래스에 {top1_percent:.2f}%; "
            f"{evidence.top2_class}의 출력 확률은 {top2_percent:.2f}%; "
            f"차이는 {margin_percent:.2f}%p."
        )
        slice_text = (
            f"전체 슬라이스 중 {agree_count}/{evidence.slice_count}장이 subject 최종 클래스와 동일한 "
            f"클래스를 예측하여 슬라이스 예측 일치율은 {(evidence.slice_prediction_agreement or 0.0) * 100:.2f}%입니다. "
            f"{evidence.top_contributing_slice}가 {evidence.top1_class} 클래스의 슬라이스별 출력 확률에 "
            "가장 크게 기여한 대표 슬라이스로 선택되었습니다."
        )

    heatmap_text = (
        f"히트맵의 밝은 영역은 {evidence.top1_class} 클래스의 Logistic Regression logit 계산 과정에서 "
        "상대적으로 크게 기여한 CLIP ViT 내부 영역입니다. 전경 마스크가 없는 현재 보고서에서는 "
        "전경/배경 평균값을 별도 정량값으로 만들지 않으며, 비뇌 배경에도 활성값이 나타날 수 있습니다."
    )
    uncertainty_text = (
        f"{evidence.top1_class}와 두 번째 후보 {evidence.top2_class}의 출력 확률 차이는 "
        f"{margin_percent:.2f}%p이며, 설정된 연구용 설명 규칙에서는 `{evidence.uncertainty_level}`로 "
        f"분류됩니다. 규칙 버전: {evidence.uncertainty_rule_version}. 이 분류는 모델 출력값의 "
        "상대적 차이를 설명하기 위한 연구용 규칙이며 임상적 확신이나 판단 근거를 의미하지 않습니다."
    )

    return [
        KeepTogether([
            Paragraph("모델 예측 근거", styles["Heading2"]),
            Paragraph("1. 확률 근거", styles["Heading2"]),
            Paragraph(classification_text, styles["BodyText"]),
            Paragraph(compact_text, styles["SmallKorean"]),
        ]),
        Paragraph("2. 영상 근거", styles["Heading2"]),
        Paragraph(heatmap_text, styles["BodyText"]),
        Paragraph("3. 슬라이스 근거", styles["Heading2"]),
        Paragraph(slice_text, styles["BodyText"]),
        Paragraph("4. 출력값 분리 및 불확실성", styles["Heading2"]),
        Paragraph(uncertainty_text, styles["BodyText"]),
        Paragraph("5. 해석 제한", styles["Heading2"]),
        Paragraph(
            "본 설명은 모델의 수학적 예측 과정을 요약한 것입니다. subject 단위 결과, 의학적 이상, "
            "또는 임상 판단을 의미하지 않습니다.",
            styles["BodyText"],
        ),
    ]


def _xai_sections(payload: ReportPayload, styles, table_style: TableStyle, temp_dir: Path) -> list:
    intro = KeepTogether([
        Paragraph("XAI 방법: LR class logit 기반 CLIP ViT gradient heatmap", styles["Heading2"]),
        Paragraph("분류: Grad-ECLIP 응용형 - attention 결합 제거", styles["BodyText"]),
        Paragraph(
            "본 히트맵은 선택된 Logistic Regression 클래스 logit에 기여한 CLIP ViT 내부 영역을 "
            "시각화합니다. 원 Grad-ECLIP 논문의 이미지-텍스트 유사도 및 attention 결합을 그대로 "
            "구현한 결과가 아닙니다.",
            styles["BodyText"],
        ),
        Paragraph(
            "히트맵은 병변, 진단, 인과관계 또는 의학적 근거를 의미하지 않습니다. 검은 배경을 "
            "포함한 비뇌 영역에도 활성값이 나타날 수 있습니다.",
            styles["BodyText"],
        ),
    ])
    story = [intro]
    if not payload.xai_items:
        return story

    for item_index, item in enumerate(payload.xai_items):
        image_paths = [
            _save_temp_image(item["original"], temp_dir / f"{item_index}_original.png"),
            _save_temp_image(item["heatmap"], temp_dir / f"{item_index}_heatmap.png"),
            _save_temp_image(item["overlay"], temp_dir / f"{item_index}_overlay.png"),
        ]
        image_table = Table([
            ["원본", "히트맵", "오버레이"],
            [Image(str(path), width=42 * mm, height=42 * mm) for path in image_paths],
        ], colWidths=[52 * mm, 52 * mm, 52 * mm])
        image_table.setStyle(table_style)
        block = []
        if item_index == 0:
            block.append(Paragraph("대표 슬라이스 XAI 이미지", styles["Heading2"]))
        block.extend([
            Paragraph(item["title"], styles["Heading2"]),
            image_table,
            Paragraph("오버레이는 원본 MRI 위에 모델 점수 기여 영역을 겹쳐 표시한 것입니다.", styles["SmallKorean"]),
            Spacer(1, 8),
        ])
        story.append(KeepTogether(block))
    return story


def _limitations(styles) -> list:
    return [
        Paragraph("검증 성능 및 알려진 제한", styles["Heading2"]),
        Paragraph("검증 성능: Subject Accuracy 72.2%, Macro-F1 0.435", styles["BodyText"]),
        Paragraph(
            "전체 정확도와 비교해 Macro-F1이 낮으므로 클래스별 성능 차이가 존재할 수 있습니다. "
            "현재 수치는 독립적인 임상 검증 결과가 아니며 모든 클래스에서 안정적인 분류 성능을 "
            "의미하지 않습니다.",
            styles["BodyText"],
        ),
        Paragraph(
            "현재 분류기는 최초 실행 객체가 아니라 동일 코드와 데이터로 재생성한 new_run 모델입니다. "
            "원본 실행과 예측 클래스는 100% 일치했으며 확률의 최대 절대 오차는 0.0088이었습니다.",
            styles["BodyText"],
        ),
        Paragraph(
            "원본 195명 중 OAS31378_MR_d0196이 누락되어 최종 194명을 기준으로 검증되었습니다. "
            "실제 학습에는 정제 후 manifest가 사용되었습니다. "
            "XAI 전경/배경 정량 검증은 6개 샘플 중 5개만 통과했으며 전체 슬라이스 XAI 일관성 및 "
            "의학적 타당성은 검증되지 않았습니다.",
            styles["BodyText"],
        ),
        Paragraph("학습 데이터: 정제 후 manifest 사용.", styles["SmallKorean"]),
        Spacer(1, 8),
        Paragraph("연구 결과 검토 기록", styles["Heading2"]),
        Paragraph("검토자: ____________________", styles["BodyText"]),
        Paragraph("검토 일시: __________________", styles["BodyText"]),
        Paragraph("서명: ______________________", styles["BodyText"]),
        Paragraph(
            "본 서명란은 모델 예측의 임상적 승인이나 확정을 의미하지 않습니다.",
            styles["SmallKorean"],
        ),
    ]


def _metadata_date(value: datetime) -> str:
    offset = value.strftime("%z")
    return f"D:{value.strftime('%Y%m%d%H%M%S')}{offset[:3]}'{offset[3:]}'"


def _rewrite_pdf_metadata(output_path: Path, payload: ReportPayload) -> None:
    reader = PdfReader(str(output_path))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.add_metadata({
        "/Title": "Alzheimer MRI XAI Research Report",
        "/Author": "CLIP XAI App",
        "/Subject": "Research Use Only",
        "/CreationDate": _metadata_date(payload.created_at),
        "/ModDate": _metadata_date(payload.created_at),
    })
    with output_path.open("wb") as f:
        writer.write(f)


def render_report_pdf(output_path: Path, payload: ReportPayload) -> Path:
    styles, font_name, bold_name = _styles()
    table_style = _table_style(font_name, bold_name)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="Alzheimer MRI XAI Research Report",
        author="CLIP XAI App",
    )

    created = payload.created_at.isoformat(timespec="seconds")
    story = [
        Paragraph("Alzheimer MRI XAI 연구용 보고서", styles["Title"]),
        Spacer(1, 8),
        Paragraph(f"보고서 ID: {payload.report_id}", styles["BodyText"]),
        Paragraph(f"생성 시각: {created} ({payload.timezone_name})", styles["BodyText"]),
        Paragraph(f"MR_ID: {payload.mr_id}", styles["BodyText"]),
        Paragraph(
            f"분석 단위: {'단일 슬라이스' if payload.analysis_mode == 'single_slice' else '다중 슬라이스 subject 평균'}",
            styles["BodyText"],
        ),
        Spacer(1, 8),
        Paragraph("연구용 경고", styles["Heading2"]),
        Paragraph(
            "본 결과는 연구용 프로토타입 출력입니다. 의료기기가 아니며 진단, 치료 결정, "
            "개별 환자에 대한 임상적 결론에 사용할 수 없습니다.",
            styles["BodyText"],
        ),
        Spacer(1, 8),
        Paragraph("모델 정보", styles["Heading2"]),
        _model_info_table(table_style),
        Spacer(1, 8),
    ]

    if payload.subject_prediction is None:
        story.extend(_single_slice_section(payload, styles, table_style))
    else:
        story.extend(_subject_section(payload, styles, table_style))

    story.extend([
        Spacer(1, 8),
        *_prediction_evidence_section(payload, styles),
        Spacer(1, 8),
        Paragraph("슬라이스별 요약", styles["Heading2"]),
        _slice_summary_table(payload, table_style),
        Spacer(1, 8),
    ])

    with TemporaryDirectory() as temp_dir_name:
        story.extend(_xai_sections(payload, styles, table_style, Path(temp_dir_name)))
        story.extend([Spacer(1, 8), *_limitations(styles)])
        doc.build(story, onFirstPage=_footer(payload.report_id, font_name), onLaterPages=_footer(payload.report_id, font_name))

    _rewrite_pdf_metadata(output_path, payload)
    return output_path


def create_basic_pdf_report(
    artifacts_root: Path,
    mr_id: str,
    subject_prediction,
    slice_predictions,
    representative_indices: list[int],
    slice_names: list[str] | None = None,
    representative_images: list[dict] | None = None,
) -> Path:
    payload = build_report_payload(
        mr_id=mr_id,
        subject_prediction=subject_prediction,
        slice_predictions=slice_predictions,
        representative_indices=representative_indices,
        slice_names=slice_names,
        representative_images=representative_images,
    )
    report_dir = ensure_report_dir(artifacts_root)
    output_path = report_dir / f"{payload.report_id}.pdf"
    return render_report_pdf(output_path, payload)
