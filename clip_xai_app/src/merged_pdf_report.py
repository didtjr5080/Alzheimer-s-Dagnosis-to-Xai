"""Full clinical-summary + technical-appendix PDF report for the merged
(OASIS-3+ADNI) model, matching the structure/style of
`graph_xai_extension`'s combined report but covering BOTH spatial branches
(CLIP 2D 3x3=9 regions, 3D CNN 3D 3x3x3=27 regions) plus GBM baseline and
the ensemble. Replaces `report.py::create_merged_pdf_report`.

Wording rules (same as graph_xai_extension and this project's other PDF
reports): never call the prediction a "판단" (judgment) -- always "예측". A
region where masking DECREASED the predicted class's probability is "지지
근거"; one where it INCREASED it is "억제 근거" -- never merged into one list.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from PIL import Image as PILImage
from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
)

from .adapters.merged_cnn3d_gradcam import LATERALITY_NOT_VERIFIED_NOTE, render_overlay_image, render_plain_slice
from .adapters.merged_grad_eclip import (
    render_cam_only, render_grid_overlay, render_masked_overlay, render_top_regions_highlight,
)
from .merged_region_analysis import BranchRegionAnalysis
from .region_xai.clinical_wording import (
    OUTPUT_VALUE_DISCLAIMER_TEMPLATE, REGION_3D_DISCLAIMER, STABILITY_LOW_WARNING,
    build_before_after_caption, build_branch_summary_sentence, build_top_changes_rows, format_percent,
    masking_method_label, region_plain_name_2d, region_plain_name_3d, stability_plain,
)
from .region_xai.visualization import render_probability_bar_png, render_region_graph_2d_png, render_region_graph_3d_layers_png
from .report import (
    RESEARCH_USE_WARNING, _footer, _merged_branch_prob_table, _merged_cell, _merged_kv_table, _styles, _table_style,
    ensure_report_dir,
)
from .warnings import XAI_LIMITATION_TEXT

MERGED_FULL_XAI_LIMITATION_TEXT = (
    "히트맵/구역 순위는 각 모델의 예측 클래스 점수에 대한 근사적 민감도를 보여줍니다 "
    "(CLIP: class-embedding 코사인 유사도 logit, 3D CNN: 마지막 fc layer logit). "
    "실제 병변, 진단, 인과적 해부학적 근거를 의미하지 않습니다. 3x3(2D)/3x3x3(3D) 구역은 "
    "해부학적 영역이 아닌 격자 구획입니다."
)

REVIEWER_SIGNATURE_DISCLAIMER = (
    "본 서명은 연구용 AI 출력물을 열람했다는 기록이며, 진단 확정·치료 결정·모델 승인 또는 "
    "의료기기 성능 검증을 의미하지 않습니다."
)
DEFAULT_REVIEWER_SECTION_TITLE = "연구 검토자 확인"
TECHNICAL_APPENDIX_TITLE = "연구자용 기술 부록"
TECHNICAL_APPENDIX_INTRO = (
    "다음은 모델 설정, 통계 지표, 재현 정보를 포함한 연구자·개발자용 상세 자료입니다. "
    "의료진용 요약과 동일한 실행 결과이며, 수치는 요약본과 완전히 동일합니다."
)


def _region_ranking_table(ranked_regions, table_style, styles) -> Table:
    header = ["순위", "구역", "CAM 비율", "마스킹 전", "마스킹 후", "확률 변화"]
    rows = [[_merged_cell(h, styles) for h in header]]
    for ranked in ranked_regions:
        r = ranked.result
        rows.append([_merged_cell(c, styles) for c in (
            ranked.rank, r.region_name, f"{r.cam_ratio:.4f}",
            f"{r.original_class_probability:.4f}", f"{r.masked_original_class_probability:.4f}",
            f"{r.probability_drop:+.4f}",
        )])
    table = Table(rows, repeatRows=1, colWidths=[14 * mm, 34 * mm, 24 * mm, 26 * mm, 26 * mm, 26 * mm])
    table.setStyle(table_style)
    return table


def _masking_comparison_table(comparison_rows, region_names, table_style, styles) -> Table:
    methods = ["zero", "mean", "blur"]
    header = ["구역"] + [f"{m} drop" for m in methods]
    rows = [[_merged_cell(h, styles) for h in header]]
    for name in region_names:
        rows.append([_merged_cell(c, styles) for c in (
            name, *(f"{comparison_rows[m][name]:+.4f}" for m in methods),
        )])
    table = Table(rows, repeatRows=1, colWidths=[34 * mm] + [40 * mm] * 3)
    table.setStyle(table_style)
    return table


def _stability_table(stability, table_style, styles) -> Table:
    rows = [
        ("비교한 마스킹 방식", ", ".join(stability.masking_methods)),
        ("평균 Spearman (절대 민감도 기준)", f"{stability.mean_spearman:.4f}"),
        ("부호 일치율", f"{stability.sign_agreement_rate:.2%}"),
        ("부호가 바뀐 구역", ", ".join(stability.sign_flipped_regions) or "없음"),
        ("가장 불안정한 구역", stability.most_unstable_region or "N/A"),
        ("안정성 판정", stability.stability_verdict),
        ("판정 기준", ", ".join(f"{k}={v}" for k, v in stability.stability_thresholds.items())),
    ]
    return _merged_kv_table(rows, [55 * mm, 109 * mm], table_style, styles)


def _agreement_table(agreement_by_method, table_style, styles) -> Table:
    header = ["마스킹 방식", "CAM vs drop", "CAM vs |drop|", "top3 Jaccard(절대)", "CAM 높은데 억제인 구역"]
    rows = [[_merged_cell(h, styles) for h in header]]
    for method, report in agreement_by_method.items():
        rows.append([_merged_cell(c, styles) for c in (
            method,
            f"{report.spearman_cam_ratio_vs_drop:.3f}" if report.spearman_cam_ratio_vs_drop is not None else "N/A",
            f"{report.spearman_cam_ratio_vs_absolute:.3f}" if report.spearman_cam_ratio_vs_absolute is not None else "N/A",
            f"{report.top3_jaccard_absolute:.3f}",
            ", ".join(report.high_cam_negative_drop_regions) or "없음",
        )])
    table = Table(rows, repeatRows=1, colWidths=[24 * mm] + [30 * mm] * 3 + [40 * mm])
    table.setStyle(table_style)
    return table


# ---------------------------------------------------------------------------
# Clinical (medical-staff-facing) pages
# ---------------------------------------------------------------------------

def _clinical_page1(
    predicted_class_ensemble, ensemble_probs, branch_probs, clip_analysis: BranchRegionAnalysis,
    cnn3d_analysis: BranchRegionAnalysis | None, styles, clinical_table_style,
) -> list:
    clip_top = clip_analysis.absolute_sensitivity[0].result if clip_analysis.absolute_sensitivity else None
    cnn3d_top = (
        cnn3d_analysis.absolute_sensitivity[0].result
        if cnn3d_analysis is not None and cnn3d_analysis.absolute_sensitivity else None
    )
    clip_verdict = clip_analysis.stability.stability_verdict
    cnn3d_verdict = cnn3d_analysis.stability.stability_verdict if cnn3d_analysis is not None else None

    title_label = "모델 예측 범주 (앙상블)" if ensemble_probs is not None else "모델 예측 범주 (CLIP, 앙상블 불가)"
    card_rows = [(title_label, predicted_class_ensemble)]
    if ensemble_probs is not None:
        card_rows.append((f"앙상블 {predicted_class_ensemble} 출력값", format_percent(ensemble_probs[predicted_class_ensemble])))
    for branch in ("CLIP", "3D CNN", "GBM baseline"):
        probs = branch_probs.get(branch)
        if probs is not None:
            branch_pred = max(probs, key=probs.get)
            card_rows.append((f"{branch} 출력값", f"{branch_pred} {format_percent(probs[branch_pred])}"))
        else:
            card_rows.append((f"{branch} 출력값", "사용 불가"))
    if clip_top is not None:
        card_rows.append(("CLIP 가장 민감하게 반응한 위치", region_plain_name_2d(clip_top.region_name)))
    if cnn3d_top is not None:
        card_rows.append(("3D CNN 가장 민감하게 반응한 위치", region_plain_name_3d(cnn3d_top.region_name)))
    card_rows.append(("CLIP 설명 결과의 일관성", stability_plain(clip_verdict)))
    if cnn3d_verdict is not None:
        card_rows.append(("3D CNN 설명 결과의 일관성", stability_plain(cnn3d_verdict)))

    card_table = Table(
        [[_merged_cell(k, styles, "ClinicalLabel"), _merged_cell(v, styles, "ClinicalValue")] for k, v in card_rows],
        colWidths=[68 * mm, 96 * mm],
    )
    card_table.setStyle(clinical_table_style)

    block = [
        Paragraph("AI MRI 분석 요약 - 연구용", styles["ClinicalTitle"]),
        Spacer(1, 3),
        Paragraph("연구용(Research Use Only)", styles["ClinicalSmall"]),
        Spacer(1, 10),
        card_table,
        Spacer(1, 6),
        Paragraph(OUTPUT_VALUE_DISCLAIMER_TEMPLATE.format(cls=predicted_class_ensemble), styles["ClinicalSmall"]),
        Spacer(1, 14),
    ]

    if clip_top is not None:
        sentence = build_branch_summary_sentence(
            branch_label="CLIP", predicted_class=clip_top.original_predicted_class,
            predicted_probability=clip_top.original_class_probability, primary_masking_method=clip_analysis.primary_method,
            most_sensitive_region_plain=region_plain_name_2d(clip_top.region_name),
            probability_before=clip_top.original_class_probability, probability_after=clip_top.masked_original_class_probability,
            stability_verdict=clip_verdict,
        )
        block.append(Paragraph(sentence, styles["ClinicalBody"]))
        block.append(Spacer(1, 6))
    if cnn3d_top is not None:
        sentence = build_branch_summary_sentence(
            branch_label="3D CNN", predicted_class=cnn3d_top.original_predicted_class,
            predicted_probability=cnn3d_top.original_class_probability, primary_masking_method=cnn3d_analysis.primary_method,
            most_sensitive_region_plain=region_plain_name_3d(cnn3d_top.region_name),
            probability_before=cnn3d_top.original_class_probability, probability_after=cnn3d_top.masked_original_class_probability,
            stability_verdict=cnn3d_verdict,
        )
        block.append(Paragraph(sentence, styles["ClinicalBody"]))
        block.append(Spacer(1, 10))

    low_branches = [name for name, v in (("CLIP", clip_verdict), ("3D CNN", cnn3d_verdict)) if v == "낮음"]
    if low_branches:
        block.append(Paragraph(f"({', '.join(low_branches)}) {STABILITY_LOW_WARNING}", styles["ClinicalWarning"]))

    return block


def _clinical_before_after_panel(
    before_path: Path, after_path: Path, before_probability: float, after_probability: float,
    masking_method: str, styles, clinical_table_style,
) -> list:
    """Presentation-friendly before/after masking panel: two side-by-side
    images captioned in plain Korean with the actual before/after model
    output values, so a non-expert audience can read the effect directly
    off the page instead of having to interpret a heatmap."""
    table = Table([
        [_merged_cell("처리 전", styles, "ClinicalSmall"), _merged_cell("처리 후", styles, "ClinicalSmall")],
        [RLImage(str(before_path), width=65 * mm, height=65 * mm), RLImage(str(after_path), width=65 * mm, height=65 * mm)],
    ], colWidths=[76 * mm, 76 * mm])
    table.setStyle(clinical_table_style)
    caption = build_before_after_caption(before_probability, after_probability, masking_method)
    return [Spacer(1, 10), KeepTogether([table, Spacer(1, 4), Paragraph(caption, styles["ClinicalBody"])])]


def _clinical_page2(
    clip_original_image, clip_heatmap_224, clip_analysis: BranchRegionAnalysis,
    cnn3d_volume, cnn3d_cam_volume, cnn3d_analysis: BranchRegionAnalysis | None,
    tmp_dir: Path, styles, clinical_table_style,
) -> list:
    from .adapters.merged_cnn3d_gradcam import central_slice_overlays
    from .region_xai.masking import apply_region_mask
    import numpy as np

    block = [Paragraph("영상으로 보는 설명 (CLIP)", styles["ClinicalHeading"])]

    original_path = tmp_dir / "clip_original.png"
    clip_original_image.convert("RGB").resize((224, 224)).save(original_path)
    cam_path = tmp_dir / "clip_cam.png"
    render_cam_only(clip_heatmap_224).save(cam_path)
    overlay_path = tmp_dir / "clip_overlay.png"
    render_masked_overlay(clip_original_image, clip_heatmap_224).save(overlay_path)
    top_region = next(r for r in clip_analysis.regions if r.name == clip_analysis.absolute_sensitivity[0].region_name)
    highlight_path = tmp_dir / "clip_highlight.png"
    render_top_regions_highlight(clip_original_image, [top_region], top_k=1).save(highlight_path)

    grid = Table([
        [_merged_cell("1. 원본 MRI (axial)", styles, "ClinicalSmall"), _merged_cell("2. 모델 반응 지도", styles, "ClinicalSmall")],
        [RLImage(str(original_path), width=65 * mm, height=65 * mm), RLImage(str(cam_path), width=65 * mm, height=65 * mm)],
        [_merged_cell("3. MRI + 반응 지도 중첩", styles, "ClinicalSmall"), _merged_cell("4. 출력 변화가 가장 컸던 구역", styles, "ClinicalSmall")],
        [RLImage(str(overlay_path), width=65 * mm, height=65 * mm), RLImage(str(highlight_path), width=65 * mm, height=65 * mm)],
    ], colWidths=[76 * mm, 76 * mm])
    grid.setStyle(clinical_table_style)
    block.append(KeepTogether([grid, Spacer(1, 4), Paragraph(
        "색이 강하게 표시된 부분은 모델 계산에 상대적으로 많이 반영된 위치입니다. "
        "색상은 병변의 존재, 질환의 심각도 또는 조직 손상을 의미하지 않습니다.",
        styles["ClinicalSmall"],
    )]))

    clip_top_result = clip_analysis.absolute_sensitivity[0].result
    clip_gray = np.array(clip_original_image.convert("L"))
    clip_masked = apply_region_mask(clip_gray, top_region.bbox, method=clip_analysis.primary_method)
    clip_before_path, clip_after_path = tmp_dir / "clip_plain_before.png", tmp_dir / "clip_plain_after.png"
    PILImage.fromarray(clip_gray).convert("RGB").save(clip_before_path)
    PILImage.fromarray(clip_masked).convert("RGB").save(clip_after_path)
    block.extend(_clinical_before_after_panel(
        clip_before_path, clip_after_path, clip_top_result.original_class_probability,
        clip_top_result.masked_original_class_probability, clip_analysis.primary_method, styles, clinical_table_style,
    ))

    if cnn3d_analysis is not None and cnn3d_volume is not None:
        block.append(Spacer(1, 12))
        block.append(Paragraph("영상으로 보는 설명 (3D CNN)", styles["ClinicalHeading"]))
        overlays = central_slice_overlays(cnn3d_volume, cnn3d_cam_volume)
        cnn3d_imgs = []
        for i, (view, (vslice, cslice)) in enumerate(overlays.items()):
            path = tmp_dir / f"cnn3d_{view}.png"
            render_overlay_image(vslice, cslice).save(path)
            cnn3d_imgs.append((view, path))
        cnn3d_grid = Table(
            [[_merged_cell(v, styles, "ClinicalSmall") for v, _ in cnn3d_imgs],
             [RLImage(str(p), width=48 * mm, height=48 * mm) for _, p in cnn3d_imgs]],
            colWidths=[51 * mm] * 3,
        )
        cnn3d_grid.setStyle(clinical_table_style)
        block.append(KeepTogether([cnn3d_grid, Spacer(1, 4), Paragraph(REGION_3D_DISCLAIMER, styles["ClinicalSmall"])]))

        cnn3d_top_ranked = cnn3d_analysis.absolute_sensitivity[0]
        cnn3d_top_region = next(r for r in cnn3d_analysis.regions if r.name == cnn3d_top_ranked.region_name)
        cnn3d_masked = apply_region_mask(cnn3d_volume, cnn3d_top_region.bbox, method=cnn3d_analysis.primary_method)
        d = cnn3d_volume.shape[0] // 2
        cnn3d_before_path, cnn3d_after_path = tmp_dir / "cnn3d_plain_before.png", tmp_dir / "cnn3d_plain_after.png"
        render_plain_slice(cnn3d_volume[d, :, :]).save(cnn3d_before_path)
        render_plain_slice(cnn3d_masked[d, :, :]).save(cnn3d_after_path)
        block.extend(_clinical_before_after_panel(
            cnn3d_before_path, cnn3d_after_path, cnn3d_top_ranked.result.original_class_probability,
            cnn3d_top_ranked.result.masked_original_class_probability, cnn3d_analysis.primary_method,
            styles, clinical_table_style,
        ))
    else:
        block.append(Spacer(1, 12))
        block.append(Paragraph("영상으로 보는 설명 (3D CNN)", styles["ClinicalHeading"]))
        block.append(Paragraph(
            "업로드된 이미지는 3D CNN 분석에 필요한 MNI 정합 3D 볼륨이 아니므로 이 분석은 제공되지 않습니다.",
            styles["ClinicalSmall"],
        ))
    return block


def _clinical_page3(clip_analysis, cnn3d_analysis, reviewer_section_title, styles, clinical_table_style) -> list:
    block = [Paragraph("주요 변화 및 설명 일관성", styles["ClinicalHeading"])]

    branch_entries = [("CLIP", clip_analysis, region_plain_name_2d)]
    if cnn3d_analysis is not None:
        branch_entries.append(("3D CNN", cnn3d_analysis, region_plain_name_3d))

    for label, analysis, region_fn in branch_entries:
        rows = build_top_changes_rows(analysis.supporting, analysis.suppressing, region_fn)
        block.append(Paragraph(f"{label} 주요 변화", styles["ClinicalBody"]))
        if rows:
            table_rows = [["구분", "위치", "모델 출력 변화", "쉬운 설명"]]
            for row in rows:
                table_rows.append([
                    row["label"], row["region"],
                    f"{format_percent(row['before'])} → {format_percent(row['after'])}", row["explanation"],
                ])
            table = Table(
                [[_merged_cell(c, styles, "ClinicalSmall") for c in r] for r in table_rows],
                colWidths=[24 * mm, 34 * mm, 38 * mm, 68 * mm], repeatRows=1,
            )
            table.setStyle(clinical_table_style)
            block.append(table)
        else:
            block.append(Paragraph("표시할 만한 뚜렷한 변화가 관찰되지 않았습니다.", styles["ClinicalBody"]))
        verdict = analysis.stability.stability_verdict
        block.append(Paragraph(f"{label} 설명 일관성: {stability_plain(verdict)}", styles["ClinicalBody"]))
        if verdict == "낮음":
            block.append(Paragraph(STABILITY_LOW_WARNING, styles["ClinicalWarning"]))
        block.append(Spacer(1, 8))

    block.append(Spacer(1, 8))
    block.append(Paragraph(reviewer_section_title, styles["ClinicalHeading"]))
    block.append(Paragraph(REVIEWER_SIGNATURE_DISCLAIMER, styles["ClinicalSmall"]))
    block.append(Spacer(1, 10))
    for label in ("검토자(성명)", "소속 / 직위", "검토 일시", "서명"):
        block.append(Paragraph(f"{label}: ______________________________", styles["ClinicalBody"]))
        block.append(Spacer(1, 6))
    return block


# ---------------------------------------------------------------------------
# Technical appendix
# ---------------------------------------------------------------------------

def _section_input_model(run_metadata, scan_id, dataset_source, split, true_label, table_style, styles) -> list:
    rows = [
        ("scan_id", scan_id), ("dataset_source", dataset_source), ("split", split), ("true_label", true_label or "N/A"),
        ("run_id", run_metadata.get("run_id", "N/A")), ("생성 시각", run_metadata.get("timestamp", "N/A")),
        ("Git 커밋 / 작업트리 식별자", run_metadata.get("git_commit_or_worktree_identifier", "N/A")),
        ("CLIP 모델 ID", "clip_lr::merged_clip (openai/clip-vit-base-patch16 + class_embeds)"),
        ("3D CNN 모델 ID", "merged_cnn3d::Simple3DCNN_Stable"),
        ("GBM 모델 ID", "merged_gbm::gbm_baseline"),
        ("장치(device)", run_metadata.get("device", "N/A")),
        ("마스킹 방법(비교 대상)", "zero, mean, blur"),
        ("2D 그리드 크기", "3x3 (9구역)"), ("3D 그리드 크기", "3x3x3 (27구역)"),
    ]
    return [KeepTogether([
        Paragraph("1. 입력 · 모델 · 버전 · 클래스 매핑", styles["Heading1"]),
        _merged_kv_table(rows, [55 * mm, 109 * mm], table_style, styles),
        Paragraph(
            "클래스 순서는 각 모델이 실제로 보고한 순서에서 읽었으며(CLIP/3D CNN: 컬럼 순서 검증, "
            "GBM: booster.feature_name()/클래스 순서 검증), CN=0/MCI=1/AD=2를 임의로 가정하지 않았습니다.",
            styles["BodyText"],
        ),
    ])]


def _section_prediction(predicted_class_ensemble, ensemble_probs, branch_probs, table_style, styles) -> list:
    block = [
        Paragraph("2. 모델 예측 확률", styles["Heading1"]),
        _merged_branch_prob_table(branch_probs, table_style, styles),
        Spacer(1, 4),
    ]
    if ensemble_probs is not None:
        block.append(Paragraph(
            f"최종 앙상블 예측: <b>{predicted_class_ensemble}</b> "
            f"(CN={ensemble_probs['CN']:.4f}, MCI={ensemble_probs['MCI']:.4f}, AD={ensemble_probs['AD']:.4f}, "
            f"가중치 GBM 0.35 + 3D CNN 0.45 + CLIP 0.20)",
            styles["BodyText"],
        ))
    return [KeepTogether(block)]


def _section_clip_images(clip_original_image, clip_heatmap_224, clip_analysis, tmp_dir, table_style, styles) -> list:
    original_path = tmp_dir / "t_clip_original.png"
    clip_original_image.convert("RGB").resize((224, 224)).save(original_path)
    cam_path = tmp_dir / "t_clip_cam.png"
    cam_result = render_cam_only(clip_heatmap_224)
    cam_result.save(cam_path)
    overlay_path = tmp_dir / "t_clip_overlay.png"
    render_masked_overlay(clip_original_image, clip_heatmap_224).save(overlay_path)
    grid_path = tmp_dir / "t_clip_grid.png"
    render_grid_overlay(clip_original_image, clip_heatmap_224, clip_analysis.regions).save(grid_path)
    top3 = sorted(clip_analysis.regions, key=lambda r: r.cam_ratio, reverse=True)[:3]
    top3_path = tmp_dir / "t_clip_top3.png"
    render_top_regions_highlight(clip_original_image, clip_analysis.regions, top_k=3).save(top3_path)

    grid = Table([
        ["원본", "CAM 단독", "MRI+CAM 중첩"],
        [RLImage(str(original_path), width=45 * mm, height=45 * mm),
         RLImage(str(cam_path), width=45 * mm, height=45 * mm),
         RLImage(str(overlay_path), width=45 * mm, height=45 * mm)],
        ["3x3 경계+CAM 중첩", "CAM 상위 3구역 표시", ""],
        [RLImage(str(grid_path), width=45 * mm, height=45 * mm),
         RLImage(str(top3_path), width=45 * mm, height=45 * mm), ""],
    ], colWidths=[50 * mm, 50 * mm, 50 * mm])
    grid.setStyle(table_style)
    cam_min, cam_max = float(clip_heatmap_224.min()), float(clip_heatmap_224.max())
    return [KeepTogether([
        Paragraph("3. CLIP: 원본 · CAM · 중첩 영상 (axial 슬라이스, 배경 마스킹 적용)", styles["Heading1"]),
        Paragraph(
            f"CAM 값 범위=[{cam_min:.4f}, {cam_max:.4f}], colormap=jet, alpha=0.45, 보간법=bilinear. "
            "배경(뇌 밖)은 히트맵 표시에서 투명 처리했습니다(값 자체는 변경 없음).",
            styles["SmallKorean"],
        ),
        grid,
        Paragraph("CAM은 모델 출력에 대한 근사적 민감도 지도이며, 실제 병변 위치를 의미하지 않습니다. 좌우 미검증.", styles["SmallKorean"]),
    ])]


def _section_rankings(analysis: BranchRegionAnalysis, branch_label: str, section_prefix: str, table_style, styles) -> list:
    story: list = [KeepTogether([
        Paragraph(f"{branch_label} 모델 출력 민감도 분석", styles["Heading1"]),
        Paragraph(
            "입력의 각 공간 구역을 인위적으로 마스킹했을 때 발생한 모델 출력의 변화를 측정했습니다. "
            "이 값은 해당 구역에 대한 모델의 출력 민감도를 나타내며, 실제 질병의 원인·병변 또는 "
            "해부학적 중요성을 의미하지 않습니다.",
            styles["BodyText"],
        ),
        Paragraph(f"(현재 랭킹은 '{analysis.primary_method}' 마스킹 결과 기준; 마스킹 비교는 뒤 절 참고)", styles["SmallKorean"]),
    ])]
    predicted_class = analysis.results_by_method[analysis.primary_method][0].original_predicted_class

    support_block = [Paragraph(f"{section_prefix}-1. 지지 근거 순위 (probability_drop > 0)", styles["Heading2"])]
    if analysis.supporting:
        support_block.append(Paragraph(
            f"해당 구역을 가렸을 때 예측 클래스({predicted_class})의 확률이 감소한 구역입니다.", styles["BodyText"],
        ))
    else:
        support_block.append(Paragraph("지지 근거로 분류되는 구역이 없습니다.", styles["BodyText"]))
    story.append(KeepTogether(support_block))
    if analysis.supporting:
        story.append(_region_ranking_table(analysis.supporting, table_style, styles))
    story.append(Spacer(1, 6))

    suppress_block = [Paragraph(f"{section_prefix}-2. 억제 근거 순위 (probability_drop < 0)", styles["Heading2"])]
    if analysis.suppressing:
        suppress_block.append(Paragraph(
            f"해당 구역을 가렸을 때 예측 클래스({predicted_class})의 확률이 오히려 증가한 구역입니다.", styles["BodyText"],
        ))
    else:
        suppress_block.append(Paragraph("억제 근거로 분류되는 구역이 없습니다.", styles["BodyText"]))
    story.append(KeepTogether(suppress_block))
    if analysis.suppressing:
        story.append(_region_ranking_table(analysis.suppressing, table_style, styles))
    story.append(Spacer(1, 6))

    story.append(KeepTogether([
        Paragraph(f"{section_prefix}-3. 절대 민감도 순위 (|probability_drop|, 방향 무관)", styles["Heading2"]),
    ]))
    story.append(_region_ranking_table(analysis.absolute_sensitivity, table_style, styles))
    return story


def _section_clip_before_after(clip_analysis, clip_original_image, tmp_dir, table_style, styles) -> list:
    from .adapters.merged_clip import CLASS_NAMES
    from .region_xai.masking import apply_region_mask
    import numpy as np

    gray = np.array(clip_original_image.convert("L"))
    blocks = []
    for label, ranked_list in (("지지", clip_analysis.supporting), ("억제", clip_analysis.suppressing)):
        if not ranked_list:
            continue
        region = next(r for r in clip_analysis.regions if r.name == ranked_list[0].region_name)
        masked = apply_region_mask(gray, region.bbox, method=clip_analysis.primary_method)
        before_path, after_path = tmp_dir / f"clip_ba_before_{label}.png", tmp_dir / f"clip_ba_after_{label}.png"
        PILImage.fromarray(gray).convert("RGB").save(before_path)
        PILImage.fromarray(masked).convert("RGB").save(after_path)
        table = Table([
            [_merged_cell(f"{label} 1위 '{region.name}' 마스킹 전", styles), _merged_cell("마스킹 후", styles)],
            [RLImage(str(before_path), width=45 * mm, height=45 * mm), RLImage(str(after_path), width=45 * mm, height=45 * mm)],
        ], colWidths=[50 * mm, 50 * mm])
        table.setStyle(table_style)
        blocks.append(table)
    if not blocks:
        return []
    return [KeepTogether([Paragraph("CLIP 대표 구역 마스킹 전·후 영상", styles["Heading2"]), *blocks])]


def _section_cnn3d_images(cnn3d_volume, cnn3d_cam_volume, table_style, styles, tmp_dir) -> list:
    from .adapters.merged_cnn3d_gradcam import central_slice_overlays

    overlays = central_slice_overlays(cnn3d_volume, cnn3d_cam_volume)
    imgs = []
    for view, (vslice, cslice) in overlays.items():
        path = tmp_dir / f"t_cnn3d_{view}.png"
        render_overlay_image(vslice, cslice).save(path)
        imgs.append((view, path))
    grid = Table(
        [[view for view, _ in imgs], [RLImage(str(p), width=48 * mm, height=48 * mm) for _, p in imgs]],
        colWidths=[51 * mm] * 3,
    )
    grid.setStyle(table_style)
    return [KeepTogether([
        Paragraph("4. 3D CNN: CAM 중첩 영상 (coronal/axial/sagittal 중앙 단면, 배경 마스킹 적용)", styles["Heading1"]),
        grid,
        Paragraph(f"{REGION_3D_DISCLAIMER} {LATERALITY_NOT_VERIFIED_NOTE}", styles["SmallKorean"]),
    ])]


def _section_cnn3d_before_after(cnn3d_analysis, cnn3d_volume, table_style, styles, tmp_dir) -> list:
    from .region_xai.masking import apply_region_mask

    blocks = []
    for label, ranked_list in (("지지", cnn3d_analysis.supporting), ("억제", cnn3d_analysis.suppressing)):
        if not ranked_list:
            continue
        region = next(r for r in cnn3d_analysis.regions if r.name == ranked_list[0].region_name)
        masked = apply_region_mask(cnn3d_volume, region.bbox, method=cnn3d_analysis.primary_method)
        d = cnn3d_volume.shape[0] // 2
        before_path, after_path = tmp_dir / f"cnn3d_ba_before_{label}.png", tmp_dir / f"cnn3d_ba_after_{label}.png"
        render_plain_slice(cnn3d_volume[d, :, :]).save(before_path)
        render_plain_slice(masked[d, :, :]).save(after_path)
        table = Table([
            [_merged_cell(f"{label} 1위 '{region.name}' 마스킹 전 (coronal 중앙 단면)", styles), _merged_cell("마스킹 후", styles)],
            [RLImage(str(before_path), width=45 * mm, height=45 * mm), RLImage(str(after_path), width=45 * mm, height=45 * mm)],
        ], colWidths=[50 * mm, 50 * mm])
        table.setStyle(table_style)
        blocks.append(table)
    if not blocks:
        return []
    return [KeepTogether([Paragraph("3D CNN 대표 구역 마스킹 전·후 영상", styles["Heading2"]), *blocks])]


def _build_comparison_rows(results_by_method: dict) -> dict:
    return {
        method: {r.region_name: r.probability_drop for r in results}
        for method, results in results_by_method.items()
    }


def _section_masking_comparison(clip_analysis, cnn3d_analysis, table_style, styles) -> list:
    clip_names = [r.name for r in clip_analysis.regions]
    content = [
        Paragraph("5. 세 마스킹 방식 비교 (zero / mean / blur)", styles["Heading1"]),
        Paragraph("CLIP (9개 구역, 양수=지지 근거, 음수=억제 근거):", styles["BodyText"]),
        _masking_comparison_table(_build_comparison_rows(clip_analysis.results_by_method), clip_names, table_style, styles),
    ]
    if cnn3d_analysis is not None:
        cnn3d_names = sorted(
            (r.name for r in cnn3d_analysis.regions),
            key=lambda n: abs(_build_comparison_rows(cnn3d_analysis.results_by_method)[cnn3d_analysis.primary_method][n]),
            reverse=True,
        )
        content.extend([
            Spacer(1, 6),
            Paragraph("3D CNN (27개 구역, 기본 마스킹 방식 기준 절대값 내림차순 정렬):", styles["BodyText"]),
            _masking_comparison_table(_build_comparison_rows(cnn3d_analysis.results_by_method), cnn3d_names, table_style, styles),
        ])
    return [KeepTogether(content)]


def _section_stability(clip_analysis, cnn3d_analysis, table_style, styles) -> list:
    from .region_xai.stability import STABILITY_METHODOLOGY_NOTE

    content = [
        Paragraph("6. 마스킹 방식 간 안정성 지표", styles["Heading1"]),
        Paragraph("CLIP", styles["Heading2"]),
        _stability_table(clip_analysis.stability, table_style, styles),
    ]
    if cnn3d_analysis is not None:
        content.extend([
            Spacer(1, 6),
            Paragraph("3D CNN", styles["Heading2"]),
            _stability_table(cnn3d_analysis.stability, table_style, styles),
        ])
    content.append(Paragraph(STABILITY_METHODOLOGY_NOTE, styles["SmallKorean"]))
    return [KeepTogether(content)]


def _section_agreement(clip_analysis, cnn3d_analysis, table_style, styles) -> list:
    from .region_xai.agreement import SAMPLE_SIZE_NOTE

    content = [
        Paragraph("7. CAM-Perturbation 일치도", styles["Heading1"]),
        Paragraph(f"CLIP (n=9). {SAMPLE_SIZE_NOTE}", styles["BodyText"]),
        _agreement_table(clip_analysis.agreement_by_method, table_style, styles),
    ]
    if cnn3d_analysis is not None:
        content.extend([
            Spacer(1, 6),
            Paragraph(f"3D CNN (n=27). {SAMPLE_SIZE_NOTE}", styles["BodyText"]),
            _agreement_table(cnn3d_analysis.agreement_by_method, table_style, styles),
        ])
    content.append(Paragraph("CAM이 높지만 확률 변화가 음수(억제 근거)인 구역은 삭제하거나 부호를 바꾸지 않고 그대로 표시했습니다.", styles["SmallKorean"]))
    return [KeepTogether(content)]


def _section_graph_xai(clip_analysis, cnn3d_analysis, tmp_dir, styles) -> list:
    bar_clip_path = tmp_dir / "bar_clip.png"
    render_probability_bar_png(clip_analysis.results_by_method[clip_analysis.primary_method], bar_clip_path, "CLIP")
    graph_clip_path = tmp_dir / "graph_clip.png"
    render_region_graph_2d_png(clip_analysis.graph, graph_clip_path)

    content = [
        Paragraph("8. Graph XAI (영상 구역별 예측 변화 지도)", styles["Heading1"]),
        Paragraph(
            "그래프의 색상/부호는 probability_drop 기준입니다: 양수(+)=마스킹 시 예측 클래스 확률 감소(지지 근거), "
            "음수(-)=확률 증가(억제 근거). 노드 크기는 CAM 비율을 나타냅니다. 색상은 질환/정상을 의미하지 않습니다.",
            styles["BodyText"],
        ),
        Paragraph("CLIP", styles["Heading2"]),
        RLImage(str(bar_clip_path), width=150 * mm, height=70 * mm),
        Spacer(1, 4),
        RLImage(str(graph_clip_path), width=105 * mm, height=105 * mm),
    ]
    if cnn3d_analysis is not None:
        bar_cnn3d_path = tmp_dir / "bar_cnn3d.png"
        render_probability_bar_png(cnn3d_analysis.results_by_method[cnn3d_analysis.primary_method], bar_cnn3d_path, "3D CNN")
        graph_cnn3d_path = tmp_dir / "graph_cnn3d.png"
        render_region_graph_3d_layers_png(cnn3d_analysis.graph, graph_cnn3d_path)
        content.extend([
            Spacer(1, 8),
            Paragraph("3D CNN (axis0 3개 층으로 분할 표시)", styles["Heading2"]),
            RLImage(str(bar_cnn3d_path), width=150 * mm, height=70 * mm),
            Spacer(1, 4),
            RLImage(str(graph_cnn3d_path), width=150 * mm, height=56 * mm),
        ])
    return content


def _limitations_section(mock_mode: bool, styles) -> list:
    from .warnings import RESEARCH_USE_WARNING as _r  # noqa: F401

    block = [
        Paragraph("9. 제한사항", styles["Heading1"]),
        Paragraph(MERGED_FULL_XAI_LIMITATION_TEXT, styles["BodyText"]),
        Paragraph(
            "GBM/3D CNN 라이브 추론은 SynthSeg 부피 또는 MNI 정합 캐시가 있는 scan에만 가능하며, "
            "신규로 업로드한 원본 이미지에는 이 파이프라인의 전처리가 적용되어 있지 않아 사용할 수 없습니다.",
            styles["BodyText"],
        ),
    ]
    return [KeepTogether(block)]


def build_merged_full_pdf_report(
    artifacts_root: Path,
    *,
    scan_id: str,
    dataset_source: str,
    split: str,
    true_label: str | None,
    branch_probs: dict,
    ensemble_probs: dict | None,
    clip_original_image: PILImage.Image,
    clip_heatmap_224,
    clip_analysis: BranchRegionAnalysis,
    cnn3d_volume=None,
    cnn3d_cam_volume=None,
    cnn3d_analysis: BranchRegionAnalysis | None = None,
    run_metadata: dict,
    mock_mode: bool = False,
) -> Path:
    styles, font_name, bold_name = _styles()
    table_style = _table_style(font_name, bold_name)
    clinical_table_style = _table_style(font_name, bold_name)
    report_id = uuid4().hex[:12]

    report_dir = ensure_report_dir(artifacts_root)
    output_path = report_dir / f"merged_full_{report_id}.pdf"

    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
        title="Merged (OASIS-3+ADNI) Full Model-Output Sensitivity Report", author="clip_xai_app",
    )

    predicted_class_ensemble = max(ensemble_probs, key=ensemble_probs.get) if ensemble_probs else (
        max(branch_probs["CLIP"], key=branch_probs["CLIP"].get) if branch_probs.get("CLIP") else "N/A"
    )

    story: list = []

    with TemporaryDirectory() as tmp_name:
        tmp_dir = Path(tmp_name)

        story.extend(_clinical_page1(
            predicted_class_ensemble, ensemble_probs, branch_probs, clip_analysis, cnn3d_analysis,
            styles, clinical_table_style,
        ))
        story.append(Spacer(1, 10))
        story.extend(_clinical_page2(
            clip_original_image, clip_heatmap_224, clip_analysis, cnn3d_volume, cnn3d_cam_volume,
            cnn3d_analysis, tmp_dir, styles, clinical_table_style,
        ))
        story.append(Spacer(1, 10))
        story.extend(_clinical_page3(clip_analysis, cnn3d_analysis, DEFAULT_REVIEWER_SECTION_TITLE, styles, clinical_table_style))

        story.append(PageBreak())
        story.append(Paragraph(TECHNICAL_APPENDIX_TITLE, styles["Title"]))
        story.append(Spacer(1, 6))
        story.append(Paragraph(TECHNICAL_APPENDIX_INTRO, styles["BodyText"]))
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"보고서 ID: {report_id}", styles["BodyText"]))
        story.append(Paragraph(RESEARCH_USE_WARNING, styles["BodyText"]))
        story.append(Spacer(1, 8))

        story.extend(_section_input_model(run_metadata, scan_id, dataset_source, split, true_label, table_style, styles))
        story.append(Spacer(1, 8))
        story.extend(_section_prediction(predicted_class_ensemble, ensemble_probs, branch_probs, table_style, styles))
        story.append(Spacer(1, 8))
        story.extend(_section_clip_images(clip_original_image, clip_heatmap_224, clip_analysis, tmp_dir, table_style, styles))
        story.append(Spacer(1, 8))
        story.extend(_section_rankings(clip_analysis, "CLIP", "3", table_style, styles))
        story.append(Spacer(1, 8))
        story.extend(_section_clip_before_after(clip_analysis, clip_original_image, tmp_dir, table_style, styles))
        story.append(Spacer(1, 8))
        if cnn3d_analysis is not None and cnn3d_volume is not None:
            story.extend(_section_cnn3d_images(cnn3d_volume, cnn3d_cam_volume, table_style, styles, tmp_dir))
            story.append(Spacer(1, 8))
            story.extend(_section_rankings(cnn3d_analysis, "3D CNN", "4", table_style, styles))
            story.append(Spacer(1, 8))
            story.extend(_section_cnn3d_before_after(cnn3d_analysis, cnn3d_volume, table_style, styles, tmp_dir))
            story.append(Spacer(1, 8))
        story.extend(_section_masking_comparison(clip_analysis, cnn3d_analysis, table_style, styles))
        story.append(Spacer(1, 8))
        story.extend(_section_stability(clip_analysis, cnn3d_analysis, table_style, styles))
        story.append(Spacer(1, 8))
        story.extend(_section_agreement(clip_analysis, cnn3d_analysis, table_style, styles))
        story.append(Spacer(1, 8))
        story.extend(_section_graph_xai(clip_analysis, cnn3d_analysis, tmp_dir, styles))
        story.append(Spacer(1, 8))
        story.extend(_limitations_section(mock_mode, styles))

        doc.build(story, onFirstPage=_footer(report_id, font_name), onLaterPages=_footer(report_id, font_name))

    return output_path
