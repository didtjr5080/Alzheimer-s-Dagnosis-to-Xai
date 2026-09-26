"""Fast structural/small-sample sanity tests for the merged (OASIS-3+ADNI)
live-inference adapters. The full 573-scan test-set reproduction (the
authoritative evidence for `통합(OASIS-3+ADNI) 모델 라이브 추론 전환
작업지시서.md`'s pass/fail criteria) is in test_merged_full_validation.py,
which is slow and excluded from routine runs the same way
graph_xai_extension's real-model batch test is."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "clip_xai_app"
sys.path.insert(0, str(APP_ROOT))

MERGED_ROOT = REPO_ROOT / "merged_project"
_merged_available = (MERGED_ROOT / "unified_manifest.csv").exists()

pytestmark = pytest.mark.skipif(not _merged_available, reason="merged_project/ not available locally")


SAMPLE_SCAN_ID = "OAS30564_MR_d0000"  # OASIS3, test split, true_label=CN


def test_merged_clip_bundle_loads_with_expected_shapes():
    from src.adapters.merged_clip import load_merged_clip_bundle

    bundle = load_merged_clip_bundle(REPO_ROOT, device="cpu")
    assert tuple(bundle.class_embeds.shape) == (3, 512)
    assert bundle.logit_scale > 0
    assert isinstance(bundle.epoch, int)
    # class_embeds must be L2-normalized (norm ~= 1 per row)
    norms = bundle.class_embeds.norm(dim=-1).cpu().numpy()
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_merged_clip_predicts_a_valid_probability_distribution():
    from src.adapters.merged_clip import CLASS_NAMES, load_merged_clip_bundle, predict_slice_batch
    import torch

    bundle = load_merged_clip_bundle(REPO_ROOT, device="cpu")
    path = MERGED_ROOT / "slices_multi" / f"{SAMPLE_SCAN_ID}_cor080.png"
    with torch.no_grad():
        probs = predict_slice_batch(bundle, [Image.open(path)]).cpu().numpy()[0]
    assert probs.shape == (3,)
    assert np.isfinite(probs).all()
    assert abs(probs.sum() - 1.0) < 1e-5
    assert len(CLASS_NAMES) == 3


def test_merged_grad_eclip_heatmap_is_finite_and_not_all_zero():
    from src.adapters.merged_clip import load_merged_clip_bundle
    from src.adapters.merged_grad_eclip import explain_image

    bundle = load_merged_clip_bundle(REPO_ROOT, device="cpu")
    path = MERGED_ROOT / "slices_multi" / f"{SAMPLE_SCAN_ID}_cor080.png"
    result = explain_image(bundle, Image.open(path))
    heatmap = result["heatmap_224"]
    assert heatmap.shape == (224, 224)
    assert np.isfinite(heatmap).all()
    assert not (heatmap == 0).all()
    assert result["predicted_class"] in ("CN", "MCI", "AD")


def test_merged_gbm_booster_feature_order_matches_work_order():
    from src.adapters.merged_gbm import FEATURE_COLS, load_merged_gbm_bundle

    bundle = load_merged_gbm_bundle(REPO_ROOT)
    assert bundle.booster.num_model_per_iteration() == 3
    assert bundle.booster.feature_name() == FEATURE_COLS
    assert FEATURE_COLS[-1] == "is_adni"


def test_merged_gbm_predicts_and_derives_is_adni_from_dataset_source():
    from src.adapters.merged_gbm import load_merged_gbm_bundle

    bundle = load_merged_gbm_bundle(REPO_ROOT)
    assert "is_adni" in bundle.features.columns
    row = bundle.features.loc[SAMPLE_SCAN_ID]
    assert row["dataset_source"] == "OASIS3"
    assert row["is_adni"] == 0  # derived: 1 only when dataset_source == "ADNI"

    probs = bundle.predict_by_scan_id(SAMPLE_SCAN_ID)
    assert probs is not None
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert bundle.predict_by_scan_id("not_a_real_scan_id_xyz") is None


def test_merged_cnn3d_state_dict_loads_strictly():
    from src.adapters.merged_cnn3d import load_merged_cnn3d_bundle

    bundle = load_merged_cnn3d_bundle(REPO_ROOT, device="cpu")
    assert isinstance(bundle.epoch, int)
    n_params = sum(p.numel() for p in bundle.model.parameters())
    assert n_params > 0


def test_merged_cnn3d_predicts_and_reports_unavailable_for_missing_scan():
    from src.adapters.merged_cnn3d import load_merged_cnn3d_bundle

    bundle = load_merged_cnn3d_bundle(REPO_ROOT, device="cpu")
    probs = bundle.predict_by_scan_id(SAMPLE_SCAN_ID)
    assert probs is not None
    assert abs(sum(probs.values()) - 1.0) < 1e-5
    assert bundle.predict_by_scan_id("not_a_real_scan_id_xyz") is None


def test_merged_cnn3d_grad_cam_matches_volume_shape_and_has_no_nan():
    from src.adapters.merged_cnn3d import load_merged_cnn3d_bundle
    from src.adapters.merged_cnn3d_gradcam import central_slice_overlays, grad_cam_3d

    bundle = load_merged_cnn3d_bundle(REPO_ROOT, device="cpu")
    volume = np.load(bundle.cache_dir / f"{SAMPLE_SCAN_ID}.npy")
    result = grad_cam_3d(bundle, volume)
    cam = result["cam_volume"]
    assert cam.shape == volume.shape
    assert np.isfinite(cam).all()
    assert not (cam == 0).all()
    assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-6

    overlays = central_slice_overlays(volume, cam)
    assert set(overlays.keys()) == {"axial", "coronal", "sagittal"}
    for volume_slice, cam_slice in overlays.values():
        assert volume_slice.shape == cam_slice.shape


def test_merged_inference_orchestrator_end_to_end_on_one_scan():
    from src.merged_inference import analyze_merged_scan

    result = analyze_merged_scan(SAMPLE_SCAN_ID)
    assert result.scan_meta.scan_id == SAMPLE_SCAN_ID
    assert set(result.branch_probs.keys()) == {"CLIP", "3D CNN", "GBM baseline"}
    assert all(probs is not None for probs in result.branch_probs.values())
    assert result.ensemble_probs is not None
    assert result.ensemble_predicted_class in ("CN", "MCI", "AD")
    assert "heatmap_224" in result.clip_xai
    assert set(result.cnn3d_overlays.keys()) == {"axial", "coronal", "sagittal"}


def test_merged_inference_orchestrator_rejects_unknown_scan_id():
    from src.merged_inference import analyze_merged_scan

    with pytest.raises(ValueError):
        analyze_merged_scan("definitely_not_a_real_scan_id")


def test_build_merged_full_pdf_report_produces_a_valid_multi_branch_pdf(tmp_path):
    from pypdf import PdfReader

    from src.merged_inference import analyze_merged_scan
    from src.merged_pdf_report import build_merged_full_pdf_report

    result = analyze_merged_scan(SAMPLE_SCAN_ID)
    assert result.clip_analysis is not None
    assert result.cnn3d_analysis is not None

    output_path = build_merged_full_pdf_report(
        tmp_path,
        scan_id=result.scan_meta.scan_id, dataset_source=result.scan_meta.dataset_source,
        split=result.scan_meta.split, true_label=result.scan_meta.label,
        branch_probs=result.branch_probs, ensemble_probs=result.ensemble_probs,
        clip_original_image=result.clip_xai["rep_image"], clip_heatmap_224=result.clip_xai["heatmap_224"],
        clip_analysis=result.clip_analysis, cnn3d_volume=result.cnn3d_volume,
        cnn3d_cam_volume=result.cnn3d_cam_volume, cnn3d_analysis=result.cnn3d_analysis,
        run_metadata={"run_id": "test", "timestamp": "2026-01-01", "device": "cpu"},
    )
    assert output_path.exists()
    reader = PdfReader(str(output_path))
    assert len(reader.pages) >= 8  # clinical (3) + technical appendix sections
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert SAMPLE_SCAN_ID in text
    assert "지지 근거 순위" in text
    assert "억제 근거 순위" in text
    assert "절대 민감도 순위" in text
    assert "연구자용 기술 부록" in text
    assert text.count("연구 검토자 확인") == 1  # never duplicated


def test_analyze_merged_uploaded_image_is_clip_only_with_no_ensemble():
    from src.merged_inference import analyze_merged_uploaded_image

    path = MERGED_ROOT / "slices_multi" / f"{SAMPLE_SCAN_ID}_cor080.png"
    result = analyze_merged_uploaded_image(str(path))

    assert result.clip_analysis is not None
    assert result.cnn3d_analysis is None
    assert result.cnn3d_volume is None
    assert result.cnn3d_cam_volume is None
    assert result.ensemble_probs is None
    assert result.ensemble_predicted_class is None
    assert result.branch_probs["CLIP"] is not None
    assert result.branch_probs["3D CNN"] is None
    assert result.branch_probs["GBM baseline"] is None
    assert abs(sum(result.branch_probs["CLIP"].values()) - 1.0) < 1e-5


def test_build_merged_full_pdf_report_handles_clip_only_upload_path(tmp_path):
    from pypdf import PdfReader

    from src.merged_inference import analyze_merged_uploaded_image
    from src.merged_pdf_report import build_merged_full_pdf_report

    path = MERGED_ROOT / "slices_multi" / f"{SAMPLE_SCAN_ID}_cor080.png"
    result = analyze_merged_uploaded_image(str(path))

    output_path = build_merged_full_pdf_report(
        tmp_path,
        scan_id=result.scan_meta.scan_id, dataset_source=result.scan_meta.dataset_source,
        split=result.scan_meta.split, true_label=result.scan_meta.label,
        branch_probs=result.branch_probs, ensemble_probs=result.ensemble_probs,
        clip_original_image=result.clip_xai["rep_image"], clip_heatmap_224=result.clip_xai["heatmap_224"],
        clip_analysis=result.clip_analysis, cnn3d_volume=result.cnn3d_volume,
        cnn3d_cam_volume=result.cnn3d_cam_volume, cnn3d_analysis=result.cnn3d_analysis,
        run_metadata={"run_id": "test", "timestamp": "2026-01-01", "device": "cpu"},
    )
    assert output_path.exists()
    reader = PdfReader(str(output_path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "지지 근거 순위" in text
    assert "절대 민감도 순위" in text
    assert "3D CNN 분석에 필요한" in text or "사용할 수 없습니다" in text
    assert text.count("연구 검토자 확인") == 1
