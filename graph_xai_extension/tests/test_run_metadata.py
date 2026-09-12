from __future__ import annotations

from graph_xai.run_metadata import build_run_metadata, hash_input_bytes


def test_build_run_metadata_has_all_required_fields():
    metadata = build_run_metadata(
        model_identifier="clip_lr::handoff",
        classifier_classes=["CN", "MCI", "AD"],
        cam_method="grad_eclip_lr_logit_applied",
        masking_methods=["zero", "mean", "blur"],
        grid_size=3,
        device="cpu",
        input_bytes=b"fake-image-bytes",
        mock_mode=False,
    )
    for key in (
        "run_id", "timestamp", "git_commit_or_worktree_identifier", "model_identifier",
        "classifier_classes", "cam_method", "masking_methods", "grid_size",
        "software_version", "device", "input_hash", "mock_mode",
    ):
        assert key in metadata

    assert metadata["mock_mode"] is False
    assert metadata["input_hash"] == hash_input_bytes(b"fake-image-bytes")
    assert len(metadata["run_id"]) == 32


def test_run_ids_are_unique_across_calls():
    a = build_run_metadata(
        model_identifier="m", classifier_classes=["CN", "MCI", "AD"], cam_method="x",
        masking_methods=["mean"], grid_size=3, device="cpu", input_bytes=None, mock_mode=True,
    )
    b = build_run_metadata(
        model_identifier="m", classifier_classes=["CN", "MCI", "AD"], cam_method="x",
        masking_methods=["mean"], grid_size=3, device="cpu", input_bytes=None, mock_mode=True,
    )
    assert a["run_id"] != b["run_id"]
    assert a["input_hash"] is None


def test_git_identifier_is_a_real_commit_hash_in_this_repo():
    metadata = build_run_metadata(
        model_identifier="m", classifier_classes=["CN", "MCI", "AD"], cam_method="x",
        masking_methods=["mean"], grid_size=3, device="cpu", input_bytes=None, mock_mode=True,
    )
    identifier = metadata["git_commit_or_worktree_identifier"]
    assert identifier != "unknown"
    commit_part = identifier.split("-dirty")[0]
    assert len(commit_part) == 40  # full git SHA-1 length
