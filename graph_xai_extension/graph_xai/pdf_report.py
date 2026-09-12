"""PDF report generation: model prediction + model-output-sensitivity
reasoning (support / suppress / absolute-sensitivity rankings, CAM
visualizations, masking-method comparison, CAM-perturbation agreement,
optional multi-sample summary), ending in a research-reviewer sign-off
section. Independent of `clip_xai_app/src/report.py` (never imported from
here), though it follows a similar reportlab/Korean-font/footer pattern for
visual consistency.

Wording rules enforced throughout this module (work order section 6):
- never call the model's output a "판단" (judgment) -- always "예측" (prediction).
- never describe the region-perturbation numbers as a "상관관계 기반 설명"
  (correlation-based explanation) or imply a region caused the prediction.
- a region where masking DECREASED the predicted class's probability is
  "지지 근거" (supporting evidence); one where masking INCREASED it is
  "억제 근거" (suppressing evidence). These are never merged into one
  "importance" list.

Report modes (work order: Graph_XAI_의료진용_보고서_개선_작업지시서, section 5):
- "clinical_summary": 2-3 plain-Korean pages for medical staff only.
- "technical_full": the full statistical/reproducibility report only.
- "combined" (default): clinical pages, then a clear divider page, then the
  full technical report as an appendix. Every technical number from the old
  single-mode report is preserved -- this only separates it from the
  clinical-facing body, never deletes it.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from PIL import Image as PILImage
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .cam_visuals import (
    render_cam_heatmap,
    render_grid_cam_overlay,
    render_mri_cam_overlay,
    render_region_masking_before_after,
    render_top_cam_regions_highlight,
)
from .clinical_wording import (
    CLINICAL_FIGURE_CAPTION,
    CLINICAL_MASKED_IMAGE_CAPTION_TEMPLATE,
    CLINICAL_TITLE,
    OUTPUT_VALUE_DISCLAIMER_TEMPLATE,
    RESEARCH_ONLY_HEADER_NOTE,
    STABILITY_LOW_WARNING,
    build_clinical_summary_sentence,
    build_top_changes_rows,
    format_percent,
    masking_method_label,
    region_plain_name,
    stability_plain,
)
from .exporter import MEDICAL_LIMITATION_NOTICE
from .ranking import rank_by_absolute_sensitivity, rank_supporting_regions, rank_suppressing_regions
from .visualization import render_clinical_region_graph_png, render_probability_bar_png

REPORT_MODES = ("clinical_summary", "technical_full", "combined")
DEFAULT_REPORT_MODE = "combined"

TECHNICAL_APPENDIX_TITLE = "연구자용 기술 부록"
TECHNICAL_APPENDIX_INTRO = (
    "다음은 모델 설정, 통계 지표, 재현 정보, 테스트·무결성 증거를 포함한 연구자·개발자용 상세 자료입니다. "
    "의료진용 요약과 동일한 실행 결과이며, 수치는 요약본과 완전히 동일합니다."
)

RESEARCH_USE_WARNING = (
    "본 결과는 연구용 프로토타입 출력입니다. 의료기기가 아니며 진단, 치료 결정, "
    "개별 환자에 대한 임상적 결론에 사용할 수 없습니다."
)

SENSITIVITY_EXPLANATION_SENTENCE = (
    "입력 영상의 각 공간 구역을 인위적으로 마스킹했을 때 발생한 모델 출력의 변화를 측정했습니다. "
    "이 값은 해당 구역에 대한 모델의 출력 민감도를 나타내며, 실제 질병의 원인·병변 또는 "
    "해부학적 중요성을 의미하지 않습니다."
)

GRAPH_XAI_UI_WARNING = (
    "이 분석은 모델이 특정 영상 구역을 얼마나 참고했는지 평가하는 설명가능 AI 결과입니다. "
    "표시된 구역은 실제 병변 위치, 알츠하이머병의 원인 또는 임상 진단을 의미하지 않습니다. "
    "3×3 구역은 해부학적 뇌 영역이 아닙니다. 그래프의 연결선은 두 구역이 공간적으로 이웃해 "
    "있다는 표시일 뿐이며, 실제 뇌의 배선이나 신경학적 경로를 뜻하지 않습니다."
)

MOCK_MODE_WARNING = (
    "*** 이 보고서는 테스트용 mock 모델로 생성되었습니다 (mock_mode: true). "
    "실제 의료영상 분석 결과가 아니며 어떤 목적으로도 사용할 수 없습니다. ***"
)

REVIEWER_SIGNATURE_DISCLAIMER = (
    "본 서명은 연구용 AI 출력물을 열람했다는 기록이며, 진단 확정·치료 결정·모델 승인 또는 "
    "의료기기 성능 검증을 의미하지 않습니다."
)

DEFAULT_REVIEWER_SECTION_TITLE = "연구 검토자 확인"


def _register_korean_font() -> tuple[str, str]:
    font_path = Path("C:/Windows/Fonts/malgun.ttf")
    bold_path = Path("C:/Windows/Fonts/malgunbd.ttf")
    if font_path.exists():
        if "GraphXAI-MalgunGothic" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("GraphXAI-MalgunGothic", str(font_path)))
        if bold_path.exists() and "GraphXAI-MalgunGothic-Bold" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("GraphXAI-MalgunGothic-Bold", str(bold_path)))
        return "GraphXAI-MalgunGothic", ("GraphXAI-MalgunGothic-Bold" if bold_path.exists() else "GraphXAI-MalgunGothic")
    return "Helvetica", "Helvetica-Bold"


def _styles():
    font_name, bold_name = _register_korean_font()
    styles = getSampleStyleSheet()
    for name in ("Normal", "BodyText", "Title", "Heading1", "Heading2"):
        styles[name].fontName = font_name
    styles["Title"].fontName = bold_name
    styles["Title"].fontSize = 15
    styles["Title"].leading = 19
    styles["Heading1"].fontName = bold_name
    styles["Heading1"].fontSize = 12
    styles["Heading2"].fontName = bold_name
    styles["Heading2"].fontSize = 10.5
    styles["Heading2"].leading = 13
    styles["BodyText"].fontSize = 8.5
    styles["BodyText"].leading = 12
    styles["Normal"].fontSize = 8.5
    styles["Normal"].leading = 12
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontName=font_name, fontSize=7.5, leading=9.5))
    styles.add(ParagraphStyle(
        name="Warning", parent=styles["BodyText"], fontName=bold_name, fontSize=8.5, leading=12,
        textColor=colors.HexColor("#8a1f11"),
    ))
    # Clinical (medical-staff-facing) styles: work order section 13 requires
    # >=10.5pt body text and >=16pt for key numbers on A4.
    styles.add(ParagraphStyle(name="ClinicalTitle", parent=styles["Title"], fontName=bold_name, fontSize=19, leading=23))
    styles.add(ParagraphStyle(name="ClinicalHeading", parent=styles["Heading1"], fontName=bold_name, fontSize=13.5, leading=17))
    styles.add(ParagraphStyle(name="ClinicalLabel", parent=styles["BodyText"], fontName=bold_name, fontSize=11, leading=14))
    styles.add(ParagraphStyle(name="ClinicalValue", parent=styles["BodyText"], fontName=bold_name, fontSize=16, leading=19))
    styles.add(ParagraphStyle(name="ClinicalBody", parent=styles["BodyText"], fontName=font_name, fontSize=11, leading=15))
    styles.add(ParagraphStyle(name="ClinicalSmall", parent=styles["BodyText"], fontName=font_name, fontSize=10.5, leading=13.5))
    styles.add(ParagraphStyle(
        name="ClinicalWarning", parent=styles["BodyText"], fontName=bold_name, fontSize=11, leading=15,
        textColor=colors.HexColor("#8a1f11"),
    ))
    return styles, font_name, bold_name


def _table_style(font_name: str, bold_name: str) -> TableStyle:
    return TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("FONTNAME", (0, 0), (-1, 0), bold_name),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#9aa4b2")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
    ])


def _clinical_table_style(font_name: str, bold_name: str) -> TableStyle:
    return TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("FONTNAME", (0, 0), (-1, 0), bold_name),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa4b2")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("FONTSIZE", (0, 0), (-1, -1), 10.5),
    ])


def _footer(report_id: str, font_name: str):
    def draw(canvas, doc):
        canvas.saveState()
        canvas.setFont(font_name, 8)
        canvas.drawCentredString(A4[0] / 2, 10 * mm, f"연구용(Research Use Only) | 보고서 ID: {report_id} | Page {doc.page}")
        canvas.restoreState()
    return draw


def _metadata_date(value: datetime) -> str:
    offset = value.strftime("%z") or "+0000"
    return f"D:{value.strftime('%Y%m%d%H%M%S')}{offset[:3]}'{offset[3:]}'"


def _rewrite_pdf_metadata(output_path: Path, report_id: str, created_at: datetime) -> None:
    reader = PdfReader(str(output_path))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.add_metadata({
        "/Title": "Graph XAI Model-Output Sensitivity Report",
        "/Author": "Graph XAI Extension",
        "/Subject": "Research Use Only",
        "/CreationDate": _metadata_date(created_at),
        "/ModDate": _metadata_date(created_at),
    })
    with output_path.open("wb") as handle:
        writer.write(handle)


def _cell(text, styles, style_name: str = "Small") -> Paragraph:
    """Wraps table cell text in a Paragraph so reportlab actually wraps it
    within the column width instead of overflowing past the page margin or
    into the neighboring cell (plain strings in a Table do not reliably wrap)."""
    return Paragraph(str(text), styles[style_name])


def _kv_table(rows: list[tuple[str, str]], table_style: TableStyle, styles, style_name: str = "Small") -> Table:
    table = Table([[_cell(k, styles, style_name), _cell(v, styles, style_name)] for k, v in rows], colWidths=[52 * mm, 98 * mm])
    table.setStyle(table_style)
    return table


# ---------------------------------------------------------------------------
# Clinical (medical-staff-facing) pages -- work order sections 6, 7, 8, 14.
# ---------------------------------------------------------------------------

def _clinical_page1(
    predicted_class: str, class_probabilities: dict, absolute_sensitivity, primary_masking_method: str,
    stability_report, styles, clinical_table_style,
) -> list:
    predicted_probability = class_probabilities[predicted_class]
    most_sensitive = absolute_sensitivity[0].result if absolute_sensitivity else None
    stability_verdict = stability_report.stability_verdict if stability_report is not None else "N/A"

    card_rows = [
        ("모델 예측 범주", predicted_class),
        (f"{predicted_class} 모델 출력값", format_percent(predicted_probability)),
        ("가장 민감하게 반응한 위치", region_plain_name(most_sensitive.region_name) if most_sensitive else "N/A"),
        ("설명 결과의 일관성", stability_plain(stability_verdict)),
    ]
    card_table = Table(
        [[_cell(k, styles, "ClinicalLabel"), _cell(v, styles, "ClinicalValue")] for k, v in card_rows],
        colWidths=[68 * mm, 96 * mm],
    )
    card_table.setStyle(clinical_table_style)

    block = [
        Paragraph(CLINICAL_TITLE, styles["ClinicalTitle"]),
        Spacer(1, 3),
        Paragraph(RESEARCH_ONLY_HEADER_NOTE, styles["ClinicalSmall"]),
        Spacer(1, 10),
        card_table,
        Spacer(1, 6),
        Paragraph(OUTPUT_VALUE_DISCLAIMER_TEMPLATE.format(cls=predicted_class), styles["ClinicalSmall"]),
        Spacer(1, 14),
    ]
    if most_sensitive is not None:
        summary_sentence = build_clinical_summary_sentence(
            predicted_class=predicted_class, predicted_probability=predicted_probability,
            primary_masking_method=primary_masking_method, most_sensitive_region=most_sensitive.region_name,
            probability_before=most_sensitive.original_class_probability,
            probability_after=most_sensitive.masked_original_class_probability,
            stability_verdict=stability_verdict,
        )
        block.append(Paragraph(summary_sentence, styles["ClinicalBody"]))
        block.append(Spacer(1, 12))
    if stability_verdict == "낮음":
        block.append(Paragraph(STABILITY_LOW_WARNING, styles["ClinicalWarning"]))
    return block


def _clinical_page2(
    original_image: PILImage.Image, cam, regions, most_sensitive_result, primary_masking_method: str,
    tmp_dir: Path, styles, clinical_table_style,
) -> list:
    original_path = tmp_dir / "clinical_original.png"
    original_image.convert("RGB").save(original_path)

    cam_only = render_cam_heatmap(cam, (original_image.height, original_image.width))
    cam_path = tmp_dir / "clinical_cam.png"
    cam_only["image"].convert("RGB").save(cam_path)

    overlay = render_mri_cam_overlay(original_image, cam)
    overlay_path = tmp_dir / "clinical_overlay.png"
    overlay["image"].save(overlay_path)

    highlight_region_info = None
    if most_sensitive_result is not None:
        highlight_region_info = next((r for r in regions if r.name == most_sensitive_result.region_name), None)
    if highlight_region_info is not None:
        highlight = render_top_cam_regions_highlight(original_image, [highlight_region_info], top_k=1)
        highlight_path = tmp_dir / "clinical_highlight.png"
        highlight["image"].save(highlight_path)
    else:
        highlight_path = original_path

    image_grid = Table([
        [_cell("1. 원본 MRI", styles, "ClinicalSmall"), _cell("2. 모델 반응 지도", styles, "ClinicalSmall")],
        [RLImage(str(original_path), width=68 * mm, height=68 * mm), RLImage(str(cam_path), width=68 * mm, height=68 * mm)],
        [_cell("3. MRI + 모델 반응 지도 중첩", styles, "ClinicalSmall"), _cell("4. 출력 변화가 가장 컸던 구역", styles, "ClinicalSmall")],
        [RLImage(str(overlay_path), width=68 * mm, height=68 * mm), RLImage(str(highlight_path), width=68 * mm, height=68 * mm)],
    ], colWidths=[76 * mm, 76 * mm])
    image_grid.setStyle(clinical_table_style)

    block = [
        Paragraph("영상으로 보는 설명", styles["ClinicalHeading"]),
        KeepTogether([image_grid, Spacer(1, 4), Paragraph(CLINICAL_FIGURE_CAPTION, styles["ClinicalSmall"])]),
    ]

    if most_sensitive_result is not None and highlight_region_info is not None:
        ba = render_region_masking_before_after(original_image, highlight_region_info, primary_masking_method)
        before_path, after_path = tmp_dir / "clinical_before.png", tmp_dir / "clinical_after.png"
        ba["before"].save(before_path)
        ba["after"].save(after_path)
        before_val = format_percent(most_sensitive_result.original_class_probability)
        after_val = format_percent(most_sensitive_result.masked_original_class_probability)
        delta_val = format_percent(
            most_sensitive_result.masked_original_class_probability - most_sensitive_result.original_class_probability
        )
        caption = CLINICAL_MASKED_IMAGE_CAPTION_TEMPLATE.format(
            before=before_val, after=after_val, delta=delta_val,
            method=masking_method_label(primary_masking_method),
        )
        ba_table = Table([
            [_cell("처리 전", styles, "ClinicalSmall"), _cell("처리 후", styles, "ClinicalSmall")],
            [RLImage(str(before_path), width=60 * mm, height=60 * mm), RLImage(str(after_path), width=60 * mm, height=60 * mm)],
        ], colWidths=[76 * mm, 76 * mm])
        ba_table.setStyle(clinical_table_style)
        block.append(Spacer(1, 8))
        block.append(KeepTogether([ba_table, Spacer(1, 4), Paragraph(caption, styles["ClinicalSmall"])]))
    return block


def _clinical_page3(
    supporting, suppressing, stability_report, reviewer_section_title: str, styles, clinical_table_style,
    include_signature: bool,
) -> list:
    rows = build_top_changes_rows(supporting, suppressing)
    block = [Paragraph("주요 변화 및 설명 일관성", styles["ClinicalHeading"])]
    if rows:
        table_rows = [["구분", "영상 위치", "모델 출력 변화", "쉬운 설명"]]
        for row in rows:
            table_rows.append([
                row["label"], row["region"],
                f"{format_percent(row['before'])} \u2192 {format_percent(row['after'])}",
                row["explanation"],
            ])
        table = Table(
            [[_cell(c, styles, "ClinicalSmall") for c in r] for r in table_rows],
            colWidths=[24 * mm, 30 * mm, 40 * mm, 70 * mm], repeatRows=1,
        )
        table.setStyle(clinical_table_style)
        block.append(table)
    else:
        block.append(Paragraph("표시할 만한 뚜렷한 변화가 관찰되지 않았습니다.", styles["ClinicalBody"]))
    block.append(Spacer(1, 10))

    stability_verdict = stability_report.stability_verdict if stability_report is not None else "N/A"
    block.append(Paragraph(f"설명 일관성: {stability_plain(stability_verdict)}", styles["ClinicalBody"]))
    if stability_verdict == "낮음":
        block.append(Spacer(1, 4))
        block.append(Paragraph(STABILITY_LOW_WARNING, styles["ClinicalWarning"]))

    if include_signature:
        block.append(Spacer(1, 16))
        block.extend(_signature_section(styles, reviewer_section_title, body_style="ClinicalBody", small_style="ClinicalSmall"))
    return block


def _technical_appendix_divider(styles) -> list:
    return [
        PageBreak(),
        Paragraph(TECHNICAL_APPENDIX_TITLE, styles["Title"]),
        Spacer(1, 6),
        Paragraph(TECHNICAL_APPENDIX_INTRO, styles["BodyText"]),
        Spacer(1, 10),
    ]


# ---------------------------------------------------------------------------
# Technical appendix sections (unchanged numeric content from the previous
# single-mode report; work order section 11 only requires separating these
# from the clinical body, never deleting any number).
# ---------------------------------------------------------------------------

def _section1_input_model_version(run_metadata: dict, class_mapping_report, input_filename: str, styles, table_style) -> list:
    rows = [
        ("입력 파일", Path(input_filename).name),
        ("run_id", run_metadata.get("run_id", "N/A")),
        ("생성 시각", run_metadata.get("timestamp", "N/A")),
        ("Git 커밋 / 작업트리 식별자", run_metadata.get("git_commit_or_worktree_identifier", "N/A")),
        ("소프트웨어 버전", run_metadata.get("software_version", "N/A")),
        ("입력 해시(sha256)", run_metadata.get("input_hash") or "N/A"),
        ("모델 ID", run_metadata.get("model_identifier", "N/A")),
        ("장치(device)", run_metadata.get("device", "N/A")),
        ("CAM 방법", run_metadata.get("cam_method", "N/A")),
        ("마스킹 방법(비교 대상)", ", ".join(run_metadata.get("masking_methods", []))),
        ("그리드 크기", f"{run_metadata.get('grid_size', 'N/A')}x{run_metadata.get('grid_size', 'N/A')}"),
        ("mock_mode", str(run_metadata.get("mock_mode", False))),
    ]
    block = [
        Paragraph("1. 입력 · 모델 · 버전 · 클래스 매핑", styles["Heading1"]),
        _kv_table(rows, table_style, styles),
        Spacer(1, 4),
    ]
    if class_mapping_report is not None:
        mapping_rows = [
            ("검증된 클래스 순서", ", ".join(class_mapping_report.class_names_ordered)),
            ("클래스 수", str(class_mapping_report.n_classes)),
            ("이진 분류 여부", str(class_mapping_report.is_binary)),
            ("예상치 못한 클래스", ", ".join(class_mapping_report.unexpected_classes) or "없음"),
            ("검증 결과", class_mapping_report.detail),
        ]
        block.extend([
            Paragraph(
                "클래스 순서는 분류기가 실제로 보고한 classes_ 및 클래스명 매핑에서 읽었으며, "
                "CN=0/MCI=1/AD=2를 임의로 가정하지 않았습니다.",
                styles["BodyText"],
            ),
            _kv_table(mapping_rows, table_style, styles),
        ])
    return [KeepTogether(block)]


def _section2_prediction(predicted_class: str, class_probabilities: dict, table_style, styles) -> list:
    rows = [["클래스", "모델 출력 확률", "표시 백분율"]]
    for name, prob in class_probabilities.items():
        marker = "  <-- 모델 예측" if name == predicted_class else ""
        rows.append([name, f"{prob:.4f}", f"{prob * 100:.2f}%{marker}"])
    table = Table(rows, colWidths=[35 * mm, 45 * mm, 65 * mm])
    table.setStyle(table_style)
    return [
        Paragraph("2. 모델 예측 확률", styles["Heading1"]),
        Paragraph(f"모델 예측: {predicted_class} (확률 {class_probabilities[predicted_class] * 100:.2f}%)", styles["BodyText"]),
        table,
        Spacer(1, 6),
    ]


def _region_ranking_table(ranked_regions, table_style) -> Table:
    rows = [["순위", "구역", "CAM 비율", "마스킹 전", "마스킹 후", "확률 변화"]]
    for ranked in ranked_regions:
        r = ranked.result
        rows.append([
            str(ranked.rank), r.region_name, f"{r.cam_ratio:.4f}",
            f"{r.original_class_probability:.4f}", f"{r.masked_original_class_probability:.4f}",
            f"{r.probability_drop:+.4f}",
        ])
    table = Table(rows, repeatRows=1, colWidths=[12 * mm, 30 * mm, 24 * mm, 24 * mm, 24 * mm, 24 * mm])
    table.setStyle(table_style)
    return table


def _section456_rankings(supporting, suppressing, absolute_sensitivity, predicted_class, masking_method, table_style, styles) -> list:
    story: list = [
        KeepTogether([
            Paragraph("모델 출력 민감도 분석", styles["Heading1"]),
            Paragraph(SENSITIVITY_EXPLANATION_SENTENCE, styles["BodyText"]),
            Paragraph(f"(현재 페이지 랭킹은 '{masking_method}' 마스킹 결과 기준; 3방식 비교는 7절 참고)", styles["Small"]),
        ]),
        Spacer(1, 6),
    ]

    support_block = [Paragraph("4. 지지 근거 순위 (probability_drop > 0)", styles["Heading2"])]
    if supporting:
        support_block.append(Paragraph(
            f"해당 구역을 가렸을 때 예측 클래스({predicted_class})의 확률이 감소한 구역입니다. "
            "모델의 예측을 지지하는 정보가 있었을 가능성을 나타냅니다.",
            styles["BodyText"],
        ))
    else:
        support_block.append(Paragraph("지지 근거로 분류되는 구역이 없습니다 (모든 구역의 probability_drop <= 0).", styles["BodyText"]))
    story.append(KeepTogether(support_block))
    if supporting:
        story.append(_region_ranking_table(supporting, table_style))
    story.append(Spacer(1, 6))

    suppress_block = [Paragraph("5. 억제 근거 순위 (probability_drop < 0)", styles["Heading2"])]
    if suppressing:
        suppress_block.append(Paragraph(
            f"해당 구역을 가렸을 때 예측 클래스({predicted_class})의 확률이 오히려 증가한 구역입니다. "
            "모델의 예측 확신을 낮추는 정보가 있었을 가능성을 나타냅니다.",
            styles["BodyText"],
        ))
    else:
        suppress_block.append(Paragraph("억제 근거로 분류되는 구역이 없습니다 (모든 구역의 probability_drop >= 0).", styles["BodyText"]))
    story.append(KeepTogether(suppress_block))
    if suppressing:
        story.append(_region_ranking_table(suppressing, table_style))
    story.append(Spacer(1, 6))

    story.append(KeepTogether([
        Paragraph("6. 절대 민감도 순위 (|probability_drop|, 방향 무관)", styles["Heading2"]),
        Paragraph("방향과 무관하게 모델 출력이 얼마나 민감하게 변했는지를 나타냅니다.", styles["BodyText"]),
    ]))
    story.append(_region_ranking_table(absolute_sensitivity, table_style))

    return story


def _section7_masking_comparison(comparison_table, stability_report, table_style, styles) -> list:
    block = [Paragraph("7. 세 마스킹 방식 비교 (zero / mean / blur)", styles["Heading1"])]
    if comparison_table:
        methods = [k[:-len("_probability_drop")] for k in comparison_table[0].keys() if k.endswith("_probability_drop")]
        header = [_cell("구역", styles)] + [_cell(f"{m} drop", styles) for m in methods]
        rows = [header]
        for row in comparison_table:
            rows.append([row["region_name"]] + [f"{row[f'{m}_probability_drop']:+.4f}" for m in methods])
        table = Table(rows, repeatRows=1, colWidths=[30 * mm] + [40 * mm] * len(methods))
        table.setStyle(table_style)
        block.append(Paragraph("동일 구역의 masking 방식별 probability_drop 비교 (양수=지지 근거, 음수=억제 근거):", styles["BodyText"]))
        block.append(table)
        block.append(Spacer(1, 6))

    if stability_report is not None:
        s = stability_report
        stability_rows = [
            ("비교한 마스킹 방식", ", ".join(s.masking_methods)),
            ("평균 Spearman (절대 민감도 기준)", f"{s.mean_spearman:.4f}" if s.mean_spearman == s.mean_spearman else "N/A"),
            ("부호 일치율", f"{s.sign_agreement_rate:.2%}" if s.sign_agreement_rate == s.sign_agreement_rate else "N/A"),
            ("부호가 바뀐 구역", ", ".join(s.sign_flipped_regions) or "없음"),
            ("가장 불안정한 구역", s.most_unstable_region or "N/A"),
            ("안정성 판정", s.stability_verdict),
            ("판정 기준", ", ".join(f"{k}={v}" for k, v in s.stability_thresholds.items())),
        ]
        block.append(Paragraph("마스킹 방식 간 안정성 지표", styles["Heading2"]))
        block.append(_kv_table(stability_rows, table_style, styles))
        block.append(Paragraph(
            "이 안정성 판정은 프로젝트 운영 기준이며 의학적으로 검증된 기준이 아닙니다. "
            "'mean' 결과 하나만으로 최종 설명을 확정하지 않았습니다.",
            styles["Small"],
        ))
    return [KeepTogether(block)]


def _section9_cam_agreement(agreement_reports: dict, table_style, styles) -> list:
    block = [
        Paragraph("9. CAM-Perturbation 일치도", styles["Heading1"]),
        Paragraph(
            "CAM 비율과 확률 변화(probability_drop, |probability_drop|)의 Spearman 상관을 마스킹 "
            "방식별로 계산했습니다. 표본이 구역 9개뿐이므로 모든 값은 탐색적 지표입니다.",
            styles["BodyText"],
        ),
    ]
    headers = ["마스킹 방식", "CAM vs drop", "CAM vs |drop|", "CAM∩지지 top3", "CAM/절대 top3 Jaccard", "CAM 높은데 억제(음수)인 구역"]
    rows = [[_cell(h, styles) for h in headers]]
    for method, report in agreement_reports.items():
        rows.append([
            _cell(method, styles),
            _cell(f"{report.spearman_cam_ratio_vs_drop:.3f}" if report.spearman_cam_ratio_vs_drop is not None else "N/A", styles),
            _cell(f"{report.spearman_cam_ratio_vs_absolute:.3f}" if report.spearman_cam_ratio_vs_absolute is not None else "N/A", styles),
            _cell(report.cam_top3_vs_support_top3_overlap, styles),
            _cell(f"{report.cam_top3_vs_absolute_top3_jaccard:.3f}", styles),
            _cell(", ".join(report.high_cam_negative_drop_regions) or "없음", styles),
        ])
    table = Table(rows, repeatRows=1, colWidths=[16 * mm, 22 * mm, 24 * mm, 22 * mm, 28 * mm, 38 * mm])
    table.setStyle(table_style)
    block.append(table)
    block.append(Paragraph(
        "CAM이 높지만 확률 변화가 음수(억제 근거)인 구역은 삭제하거나 부호를 바꾸지 않고 그대로 표시했습니다.",
        styles["Small"],
    ))
    return [KeepTogether(block)]


def _section10_batch_or_single(batch_summary, styles, table_style) -> list:
    block = [Paragraph("10. 다중 샘플 요약", styles["Heading1"])]
    if batch_summary is None:
        block.append(Paragraph("단일 샘플 분석입니다. 배치 평가는 실행되지 않았습니다.", styles["BodyText"]))
        return [KeepTogether(block)]

    b = batch_summary
    rows = [
        ("전체 이미지 수", str(b.total_images)),
        ("피험자 수", str(b.total_subjects)),
        ("클래스별 이미지 수", str(b.class_counts_images)),
        ("클래스별 피험자 수", str(b.class_counts_subjects)),
        ("피험자 단위 예측 클래스 분포", str(b.predicted_class_distribution_subjects)),
        ("실패/제외 샘플 수", str(len(b.failed_samples))),
    ]
    block.append(_kv_table(rows, table_style, styles))
    block.append(Paragraph(b.dedup_note, styles["Small"]))
    if b.classification_metrics:
        m = b.classification_metrics
        block.append(Paragraph(
            f"분류 성능 (정답 라벨 있는 피험자 {m['n_subjects_with_ground_truth']}명 기준): "
            f"accuracy={m['accuracy']:.3f}, balanced_accuracy={m['balanced_accuracy']:.3f}, macro_F1={m['macro_f1']:.3f}",
            styles["BodyText"],
        ))
    else:
        block.append(Paragraph("정답 라벨이 없어 분류 성능(accuracy 등)은 계산하지 않았습니다.", styles["Small"]))
    return [KeepTogether(block)]


def _section11_tests_and_reproduction(test_evidence: dict | None, styles) -> list:
    block = [Paragraph("11. 테스트 및 재현 정보", styles["Heading1"])]
    if test_evidence:
        for key, value in test_evidence.items():
            block.append(Paragraph(f"{key}: {value}", styles["BodyText"]))
    else:
        block.append(Paragraph(
            "이 실행 자체에는 테스트 증거가 첨부되지 않았습니다. 전체 단위/통합/회귀 테스트 결과와 "
            "원본 프로젝트 파일 무결성(Git/SHA-256) 검증 기록은 "
            "graph_xai_extension/docs/IMPLEMENTATION_REPORT.md 를 참고하세요.",
            styles["Small"],
        ))
    block.append(Paragraph(
        "재현 명령: python graph_xai_extension/run_graph_xai.py (또는 graph_xai.batch.run_batch_evaluation)",
        styles["Small"],
    ))
    return [KeepTogether(block)]


def _limitations_section(mock_mode: bool, styles) -> list:
    block = [
        Paragraph("12. 제한사항", styles["Heading1"]),
        Paragraph(MEDICAL_LIMITATION_NOTICE, styles["BodyText"]),
        Paragraph(GRAPH_XAI_UI_WARNING, styles["BodyText"]),
    ]
    if mock_mode:
        block.append(Spacer(1, 4))
        block.append(Paragraph(MOCK_MODE_WARNING, styles["Warning"]))
    return [KeepTogether(block)]


def _signature_section(styles, title: str, body_style: str = "BodyText", small_style: str = "Small") -> list:
    # The disclaimer is stated once here -- work order section 13 asks that
    # warning text not be repeated more than necessary on a page.
    block = [
        Paragraph(title, styles["Heading1"] if body_style == "BodyText" else styles["ClinicalHeading"]),
        Paragraph(REVIEWER_SIGNATURE_DISCLAIMER, styles[small_style]),
        Spacer(1, 10),
        Paragraph("검토자(성명): ______________________________", styles[body_style]),
        Spacer(1, 6),
        Paragraph("소속 / 직위: ______________________________", styles[body_style]),
        Spacer(1, 6),
        Paragraph("검토 일시: ______________________________", styles[body_style]),
        Spacer(1, 6),
        Paragraph("서명: ______________________________", styles[body_style]),
    ]
    return [KeepTogether(block)]


def build_graph_xai_pdf_report(
    output_path,
    *,
    run_metadata: dict,
    predicted_class: str,
    class_probabilities: dict,
    class_mapping_report,
    original_image: PILImage.Image,
    cam,
    regions,
    results_by_method: dict,
    primary_masking_method: str,
    graph,
    stability_report=None,
    agreement_reports: dict | None = None,
    batch_summary=None,
    test_evidence: dict | None = None,
    diagonal_edges: bool = False,
    mock_mode: bool = False,
    reviewer_section_title: str = DEFAULT_REVIEWER_SECTION_TITLE,
    report_id: str | None = None,
    report_mode: str = DEFAULT_REPORT_MODE,
) -> Path:
    if report_mode not in REPORT_MODES:
        raise ValueError(f"Unknown report_mode {report_mode!r}; expected one of {REPORT_MODES}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_id = report_id or f"graphxai_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    created_at = datetime.now()

    styles, font_name, bold_name = _styles()
    table_style = _table_style(font_name, bold_name)
    clinical_table_style = _clinical_table_style(font_name, bold_name)

    primary_results = results_by_method[primary_masking_method]
    supporting = rank_supporting_regions(primary_results)
    suppressing = rank_suppressing_regions(primary_results)
    absolute_sensitivity = rank_by_absolute_sensitivity(primary_results)

    include_clinical = report_mode in ("clinical_summary", "combined")
    include_technical = report_mode in ("technical_full", "combined")

    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
        title="Graph XAI Model-Output Sensitivity Report", author="Graph XAI Extension",
    )

    story: list = []

    with TemporaryDirectory() as tmp_name:
        tmp_dir = Path(tmp_name)

        if include_clinical:
            most_sensitive_result = absolute_sensitivity[0].result if absolute_sensitivity else None
            story.extend(_clinical_page1(
                predicted_class, class_probabilities, absolute_sensitivity, primary_masking_method,
                stability_report, styles, clinical_table_style,
            ))
            story.append(Spacer(1, 10))
            story.extend(_clinical_page2(
                original_image, cam, regions, most_sensitive_result, primary_masking_method,
                tmp_dir, styles, clinical_table_style,
            ))
            story.append(Spacer(1, 10))
            story.extend(_clinical_page3(
                supporting, suppressing, stability_report, reviewer_section_title, styles, clinical_table_style,
                include_signature=True,
            ))
            if mock_mode:
                story.append(Spacer(1, 8))
                story.append(Paragraph(MOCK_MODE_WARNING, styles["ClinicalWarning"]))

        if include_technical:
            if include_clinical:
                story.extend(_technical_appendix_divider(styles))

            story.extend([
                Paragraph(
                    "Graph XAI 모델 출력 민감도 분석 보고서" + (" (연구자용 기술 부록)" if include_clinical else ""),
                    styles["Title"],
                ),
                Spacer(1, 6),
                Paragraph(f"보고서 ID: {report_id}", styles["BodyText"]),
                Spacer(1, 4),
                Paragraph("연구용 경고", styles["Heading2"]),
                Paragraph(RESEARCH_USE_WARNING, styles["BodyText"]),
            ])
            if mock_mode:
                story.append(Spacer(1, 4))
                story.append(Paragraph(MOCK_MODE_WARNING, styles["Warning"]))
            story.append(Spacer(1, 8))

            story.extend(_section1_input_model_version(run_metadata, class_mapping_report, run_metadata.get("input_filename", ""), styles, table_style))
            story.append(Spacer(1, 8))
            story.extend(_section2_prediction(predicted_class, class_probabilities, table_style, styles))

            original_path = tmp_dir / "original.png"
            original_image.convert("RGB").save(original_path)
            cam_only = render_cam_heatmap(cam, (original_image.height, original_image.width))
            cam_only_path = tmp_dir / "cam_only.png"
            cam_only["image"].convert("RGB").save(cam_only_path)
            mri_cam = render_mri_cam_overlay(original_image, cam)
            mri_cam_path = tmp_dir / "mri_cam.png"
            mri_cam["image"].save(mri_cam_path)
            grid_cam = render_grid_cam_overlay(original_image, cam, regions)
            grid_cam_path = tmp_dir / "grid_cam.png"
            grid_cam["image"].save(grid_cam_path)
            top_cam = render_top_cam_regions_highlight(original_image, regions, top_k=3)
            top_cam_path = tmp_dir / "top_cam.png"
            top_cam["image"].save(top_cam_path)

            image_grid = Table([
                ["원본", "CAM 단독", "MRI+CAM 중첩"],
                [RLImage(str(original_path), width=45 * mm, height=45 * mm),
                 RLImage(str(cam_only_path), width=45 * mm, height=45 * mm),
                 RLImage(str(mri_cam_path), width=45 * mm, height=45 * mm)],
                ["3x3 경계+CAM 중첩", "CAM 상위 3구역 표시", ""],
                [RLImage(str(grid_cam_path), width=45 * mm, height=45 * mm),
                 RLImage(str(top_cam_path), width=45 * mm, height=45 * mm), ""],
            ], colWidths=[50 * mm, 50 * mm, 50 * mm])
            image_grid.setStyle(table_style)

            story.append(KeepTogether([
                Paragraph("3. 원본 · CAM · 중첩 영상", styles["Heading1"]),
                Paragraph(
                    f"CAM 값 범위=[{cam_only['cam_value_range'][0]:.4f}, {cam_only['cam_value_range'][1]:.4f}], "
                    f"colormap={cam_only['colormap']}, alpha={mri_cam['alpha']}, 보간법={cam_only['interpolation']}",
                    styles["Small"],
                ),
                image_grid,
                Paragraph(cam_only["caption"], styles["Small"]),
            ]))

            story.extend(_section456_rankings(supporting, suppressing, absolute_sensitivity, predicted_class, primary_masking_method, table_style, styles))
            story.append(Spacer(1, 8))

            before_after_blocks = []
            if supporting:
                top_support_region = next(r for r in regions if r.name == supporting[0].region_name)
                ba = render_region_masking_before_after(original_image, top_support_region, primary_masking_method)
                before_path, after_path = tmp_dir / "support_before.png", tmp_dir / "support_after.png"
                ba["before"].save(before_path)
                ba["after"].save(after_path)
                support_table = Table([
                    [f"지지 1위 '{ba['region_name']}' 마스킹 전", "마스킹 후"],
                    [RLImage(str(before_path), width=45 * mm, height=45 * mm), RLImage(str(after_path), width=45 * mm, height=45 * mm)],
                ], colWidths=[50 * mm, 50 * mm])
                support_table.setStyle(table_style)
                before_after_blocks.append(support_table)
            if suppressing:
                top_suppress_region = next(r for r in regions if r.name == suppressing[0].region_name)
                ba = render_region_masking_before_after(original_image, top_suppress_region, primary_masking_method)
                before_path, after_path = tmp_dir / "suppress_before.png", tmp_dir / "suppress_after.png"
                ba["before"].save(before_path)
                ba["after"].save(after_path)
                suppress_table = Table([
                    [f"억제 1위 '{ba['region_name']}' 마스킹 전", "마스킹 후"],
                    [RLImage(str(before_path), width=45 * mm, height=45 * mm), RLImage(str(after_path), width=45 * mm, height=45 * mm)],
                ], colWidths=[50 * mm, 50 * mm])
                suppress_table.setStyle(table_style)
                before_after_blocks.append(suppress_table)
            if before_after_blocks:
                story.append(KeepTogether([Paragraph("대표 구역 마스킹 전·후 영상", styles["Heading2"]), *before_after_blocks]))
                story.append(Spacer(1, 8))

            story.extend(_section7_masking_comparison(
                _comparison_table_or_none(results_by_method), stability_report, table_style, styles,
            ))
            story.append(Spacer(1, 8))

            if agreement_reports:
                story.extend(_section9_cam_agreement(agreement_reports, table_style, styles))
                story.append(Spacer(1, 8))

            story.extend(_section10_batch_or_single(batch_summary, styles, table_style))
            story.append(Spacer(1, 8))
            story.extend(_section11_tests_and_reproduction(test_evidence, styles))
            story.append(Spacer(1, 8))

            bar_path = tmp_dir / "bar.png"
            render_probability_bar_png(primary_results, bar_path)
            graph_path = tmp_dir / "graph.png"
            render_clinical_region_graph_png(graph, graph_path)

            story.extend([
                Paragraph("8. Graph XAI (영상 구역별 예측 변화 지도)", styles["Heading1"]),
                Paragraph(
                    "막대그래프와 그래프의 색상/부호는 probability_drop 기준입니다: 양수(+)=마스킹 시 "
                    "예측 클래스 확률 감소(지지 근거), 음수(-)=확률 증가(억제 근거). "
                    "노드 크기는 CAM 비율을 나타냅니다. 색상은 질환/정상을 의미하지 않습니다.",
                    styles["BodyText"],
                ),
                RLImage(str(bar_path), width=150 * mm, height=74 * mm),
                Spacer(1, 6),
                RLImage(str(graph_path), width=115 * mm, height=118 * mm),
                Paragraph(GRAPH_XAI_UI_WARNING, styles["Small"]),
                Spacer(1, 8),
            ])

            story.extend(_limitations_section(mock_mode, styles))
            story.append(Spacer(1, 10))
            if not include_clinical:
                story.extend(_signature_section(styles, reviewer_section_title))

        doc.build(story, onFirstPage=_footer(report_id, font_name), onLaterPages=_footer(report_id, font_name))

    _rewrite_pdf_metadata(output_path, report_id, created_at)
    return output_path


def _comparison_table_or_none(results_by_method: dict):
    if len(results_by_method) < 2:
        return None
    from .masking_compare import build_masking_comparison_table
    return build_masking_comparison_table(results_by_method)
