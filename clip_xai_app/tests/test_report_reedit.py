import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.merged_region_analysis import _summarize
from src.merged_pdf_report import build_merged_full_pdf_report
from src.region_xai.clinical_wording import build_before_after_caption, build_branch_summary_sentence
from src.region_xai.region_grid_2d import split_into_regions_2d
from src.region_xai.schemas import RegionPerturbationResult


def test_pdf_distinguishes_scan_from_slice_and_omits_evaluations(tmp_path, monkeypatch):
    import src.merged_pdf_report as report
    def unexpected(*args, **kwargs):
        pytest.fail('Excluded evaluation section was called')
    for name in ('_section_stability', '_section_agreement', '_section_masking_comparison'):
        monkeypatch.setattr(report, name, unexpected)
    cam = np.ones((224, 224), dtype=np.float32)
    regions = split_into_regions_2d(cam)
    methods = {}
    for method in ('zero', 'mean', 'blur'):
        methods[method] = [RegionPerturbationResult(r.name, 'MCI', .5021, 'MCI', .5844, -.0823, .0823, method, r.cam_mean, r.cam_ratio) for r in regions]
    analysis = _summarize(regions, methods, 'mean')
    pdf = build_merged_full_pdf_report(
        tmp_path, scan_id='test_scan', dataset_source='ADNI', split='test', true_label='AD',
        branch_probs={'CLIP': {'CN': .1425, 'MCI': .4162, 'AD': .4413}, '3D CNN': None, 'GBM baseline': None},
        ensemble_probs=None, clip_original_image=Image.new('L', (224, 224), 128),
        clip_heatmap_224=cam, clip_analysis=analysis, run_metadata={'timestamp': '2026-10-19'},
    )
    pages = PdfReader(pdf).pages
    first = pages[0].extract_text()
    text = '\n'.join(p.extract_text() for p in pages)
    assert all(word in first for word in ('MediOn', 'test_scan', '2026-10-19', '44.13%'))
    assert 'CLIP 스캔 단위 분류' in first
    for removed in ('점수는 모델 출력이며', 'CLIP 대표 MRI 단면의 XAI 분석 대상:', '3D CNN 분석 대상:', '50.21%', '58.44%'):
        assert removed not in first
    assert '50.21%' in text and '58.44%' in text
    assert '+8.23%p' in text
    assert text.count('보고서 검토 확인') == 1
    reviewer_page = next(i for i, page in enumerate(pages) if '보고서 검토 확인' in page.extract_text())
    detail_page = next(i for i, page in enumerate(pages) if '상세 분석 및 재현 정보' in page.extract_text())
    assert detail_page > reviewer_page
    assert '4. Graph XAI' in text and '5. 분석 범위 및 해석 안내' in text
    for forbidden in ('설명 결과의 일관성', '설명 일관성', '설명 안정성', '안정성 판정', '부호 일치율', '가장 불안정한', 'Spearman', 'CAM-Perturbation 일치도', '세 마스킹 방식 비교'):
        assert forbidden not in text
    assert len(list(tmp_path.rglob('*.pdf'))) == 1


def test_summary_accepts_legacy_stability_argument_without_displaying_it():
    text = build_branch_summary_sentence(branch_label='CLIP', predicted_class='MCI', predicted_probability=.5021, primary_masking_method='mean', most_sensitive_region_plain='화면 중앙', probability_before=.5021, probability_after=.5844, stability_verdict='낮음')
    assert '일관성' not in text and '낮음' not in text
    assert '50.21%' in text and '58.44%' in text
    assert '-79.38%p' in build_before_after_caption(.8573, .0635, 'mean')
