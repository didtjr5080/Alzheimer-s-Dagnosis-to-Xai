from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "clip_xai_app"
SAMPLE_ROOT = REPO_ROOT / "clip_lr_grad_eclip_handoff_v1_20260812_105027" / "data" / "xai_samples"
sys.path.insert(0, str(APP_ROOT))


def _manifest_row(filename: str) -> dict[str, str]:
    with (SAMPLE_ROOT / "xai_sample_manifest.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if Path(row["slice_path_portable"]).name == filename:
                return row
    raise AssertionError(f"missing manifest row: {filename}")


def test_cor084_and_cor088_manifest_probabilities_are_distinct():
    cor084 = _manifest_row("OAS30100_MR_d0158_cor084.png")
    cor088 = _manifest_row("OAS30100_MR_d0158_cor088.png")

    assert cor084["true_class"] == "AD"
    assert cor088["true_class"] == "AD"
    assert abs(float(cor084["CN_prob_slice"]) - 0.41905890880325103) < 1e-12
    assert abs(float(cor084["MCI_prob_slice"]) - 0.45292135763144564) < 1e-12
    assert abs(float(cor088["CN_prob_slice"]) - 0.5537639592790997) < 1e-12
    assert abs(float(cor088["MCI_prob_slice"]) - 0.35217214265480157) < 1e-12


def test_slice_filename_parser_uses_full_axis_tokens():
    from src.xai_artifacts import parse_slice_filename

    assert parse_slice_filename("SUBJECT_scan_cor084.png") == ("SUBJECT_scan", "cor", 84)
    assert parse_slice_filename("SUBJECT_scan_sag012.png") == ("SUBJECT_scan", "sag", 12)
    assert parse_slice_filename("SUBJECT_scan_ax101.png") == ("SUBJECT_scan", "ax", 101)
    assert parse_slice_filename("SUBJECT_scan_axi102.png") == ("SUBJECT_scan", "ax", 102)
    assert parse_slice_filename("SUBJECT_scan_cor084_overlay.png") == ("SUBJECT_scan_cor084_overlay", None, None)
    assert parse_slice_filename("SUBJECT_scan_prefixcor084.png") == ("SUBJECT_scan_prefixcor084", None, None)


def test_cor084_and_cor088_create_distinct_artifact_stems_and_analysis_ids(tmp_path):
    from src.xai_artifacts import save_xai_artifact

    artifacts = []
    for filename, probs, pred in [
        ("OAS30100_MR_d0158_cor084.png", {"CN": 0.41905890880325103, "MCI": 0.45292135763144564, "AD": 0.12801973356530316}, "MCI"),
        ("OAS30100_MR_d0158_cor088.png", {"CN": 0.5537639592790997, "MCI": 0.35217214265480157, "AD": 0.09406389806609877}, "CN"),
    ]:
        path = SAMPLE_ROOT / "images" / filename
        image = Image.open(path).convert("RGB")
        heatmap = np.zeros((224, 224), dtype=np.float32)
        heatmap[70:150, 70:150] = 1.0
        fake_xai = SimpleNamespace(predicted_class=pred, target_class=pred)
        fake_bundle = SimpleNamespace(root=REPO_ROOT / "clip_lr_grad_eclip_handoff_v1_20260812_105027")
        artifacts.append(
            save_xai_artifact(
                artifacts_root=tmp_path,
                repo_root=REPO_ROOT,
                source_path=path,
                image=image,
                heatmap=heatmap,
                xai_result=fake_xai,
                slice_probability_row=probs,
                subject_probabilities={"CN": 0.7132610106724648, "MCI": 0.21862336810264757, "AD": 0.06811562122488768},
                class_names=["CN", "MCI", "AD"],
                target_class_index=["CN", "MCI", "AD"].index(pred),
                bundle=fake_bundle,
                run_id="test-run",
            )
        )

    assert artifacts[0].provenance.analysis_id != artifacts[1].provenance.analysis_id
    assert "cor084" in artifacts[0].metadata_json_path.name
    assert "cor088" in artifacts[1].metadata_json_path.name
    assert artifacts[0].provenance.true_label == "AD"
    assert artifacts[0].provenance.predicted_class == "MCI"
    assert artifacts[0].provenance.prediction_correct is False
    assert artifacts[1].provenance.predicted_class == "CN"
    assert artifacts[1].provenance.prediction_correct is False
    assert artifacts[1].provenance.code_commit
    assert artifacts[1].provenance.original_sha256
    assert artifacts[1].provenance.raw_heatmap_sha256
    assert artifacts[1].provenance.heatmap_png_sha256
    assert artifacts[1].provenance.overlay_sha256
    assert (tmp_path / "xai_artifact_index.csv").exists()


def test_analysis_id_changes_with_target_checkpoint_and_config_dimensions():
    from src.xai_artifacts import analysis_id_for, identity_from_path

    path = SAMPLE_ROOT / "images" / "OAS30100_MR_d0158_cor088.png"
    identity = identity_from_path(path, REPO_ROOT)
    baseline = analysis_id_for(identity, "legacy_clip_lr", "ckpt-a", "prep-a", "CN", "LR class logit", "run-a")

    assert baseline != analysis_id_for(identity, "legacy_clip_lr", "ckpt-b", "prep-a", "CN", "LR class logit", "run-a")
    assert baseline != analysis_id_for(identity, "legacy_clip_lr", "ckpt-a", "prep-a", "AD", "LR class logit", "run-a")
    assert baseline != analysis_id_for(identity, "legacy_clip_lr", "ckpt-a", "prep-b", "CN", "LR class logit", "run-a")
    assert baseline != analysis_id_for(identity, "legacy_clip_lr", "ckpt-a", "prep-a", "CN", "LR class logit", "run-b")


def test_sample_identity_manifest_has_no_unexpected_duplicates():
    from src.xai_artifacts import validate_sample_identity_rows

    result = validate_sample_identity_rows(REPO_ROOT)
    assert result["status"] == "PASS"
    assert result["rows"] == 152
    assert result["duplicates"] == []


def test_legacy_grad_eclip_migration_maps_only_probability_verified_sample():
    from src.xai_artifacts import migrate_legacy_grad_eclip_samples

    result = migrate_legacy_grad_eclip_samples(REPO_ROOT)
    by_name = {item["legacy_file"]: item for item in result["results"]}

    mapped = by_name["OAS30100_MR_d0158_incorrect_gradeclip.png"]
    assert mapped["status"] == "MAPPED"
    assert mapped["mapped_slice"] == "OAS30100_MR_d0158_cor084.png"

    ambiguous = [item for item in result["results"] if item["status"] == "AMBIGUOUS"]
    assert len(ambiguous) == 5
