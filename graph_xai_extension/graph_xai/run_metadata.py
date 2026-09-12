"""Per-run reproducibility metadata for CSV/JSON/PDF exports: enough to trace
or reproduce a result (run id, git state, software version, model/CAM/masking
settings, device, and a content hash of the input) without ever storing a
patient identifier or a full local filesystem path.
"""
from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from . import __version__ as SOFTWARE_VERSION

__all__ = ["build_run_metadata", "hash_input_bytes", "SOFTWARE_VERSION"]


def _git_commit_or_worktree_identifier(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return "unknown"
        commit = result.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, timeout=5,
        )
        dirty = bool(status.stdout.strip())
        return f"{commit}{'-dirty' if dirty else ''}"
    except Exception:
        return "unknown"


def hash_input_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_run_metadata(
    *,
    model_identifier: str,
    classifier_classes: list,
    cam_method: str,
    masking_methods: list,
    grid_size: int,
    device: str,
    input_bytes: bytes | None,
    mock_mode: bool,
    repo_root: Path | None = None,
) -> dict:
    repo_root = repo_root or Path(__file__).resolve().parents[2]
    return {
        "run_id": uuid4().hex,
        "timestamp": datetime.now().astimezone().isoformat(),
        "git_commit_or_worktree_identifier": _git_commit_or_worktree_identifier(repo_root),
        "model_identifier": model_identifier,
        "classifier_classes": list(classifier_classes),
        "cam_method": cam_method,
        "masking_methods": list(masking_methods),
        "grid_size": grid_size,
        "software_version": SOFTWARE_VERSION,
        "device": device,
        "input_hash": hash_input_bytes(input_bytes) if input_bytes is not None else None,
        "mock_mode": bool(mock_mode),
    }
