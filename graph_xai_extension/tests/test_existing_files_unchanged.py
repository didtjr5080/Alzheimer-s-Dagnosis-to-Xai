"""Protection test: compares the working tree against the pre-work baseline
captured in docs/preexisting_files_manifest.json. Any modification, deletion,
move, or unexpected new file outside this extension folder fails the whole
test module -- per the work order this counts as a hard integrity failure.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

EXTENSION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXTENSION_ROOT.parent
MANIFEST_PATH = EXTENSION_ROOT / "docs" / "preexisting_files_manifest.json"
EXTENSION_DIR_NAME = EXTENSION_ROOT.name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def manifest():
    assert MANIFEST_PATH.exists(), "Baseline manifest missing; the pre-work baseline was never captured."
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_no_preexisting_file_modified_deleted_or_moved(manifest):
    changed, missing = [], []
    for rel_path, expected_hash in manifest["file_sha256"].items():
        full_path = REPO_ROOT / rel_path
        if not full_path.exists():
            missing.append(rel_path)
            continue
        if _sha256(full_path) != expected_hash:
            changed.append(rel_path)
    assert not missing, f"Pre-existing files deleted or moved: {missing}"
    assert not changed, f"Pre-existing files modified: {changed}"


def test_large_excluded_data_dumps_file_count_unchanged(manifest):
    mismatched = []
    for name, info in manifest["excluded_large_directories"].items():
        directory = REPO_ROOT / name
        if not directory.exists():
            mismatched.append(f"{name}: directory deleted")
            continue
        count = sum(1 for p in directory.rglob("*") if p.is_file())
        if count != info["file_count"]:
            mismatched.append(f"{name}: file_count {info['file_count']} -> {count}")
    assert not mismatched, f"Large pre-existing data directories changed: {mismatched}"


def _scan_current_files_excluding_git_and_large_dumps(manifest) -> set[str]:
    excluded_top_level = {".git", EXTENSION_DIR_NAME, *manifest["excluded_large_directories"].keys()}
    current = set()
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        if Path(dirpath) == REPO_ROOT:
            dirnames[:] = [d for d in dirnames if d not in excluded_top_level]
        for filename in filenames:
            rel = os.path.relpath(os.path.join(dirpath, filename), REPO_ROOT)
            current.add(rel.replace(os.sep, "/"))
    return current


def test_no_new_files_appeared_outside_extension_folder(manifest):
    # git status --porcelain silently skips gitignored paths (e.g. __pycache__),
    # so this walks the filesystem directly to catch new files git would miss.
    baseline_files = set(manifest["file_sha256"].keys())
    current_files = _scan_current_files_excluding_git_and_large_dumps(manifest)
    new_files = sorted(current_files - baseline_files)
    assert not new_files, f"New files appeared outside {EXTENSION_DIR_NAME}/ (including gitignored paths): {new_files}"


def test_no_new_or_changed_git_status_entries_outside_extension_folder(manifest):
    baseline_lines = set(manifest["git_status_short"])
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    current_lines = [line for line in result.stdout.splitlines() if line.strip()]
    new_lines = [line for line in current_lines if line not in baseline_lines]
    offending = [line for line in new_lines if not line[3:].strip('"').startswith(EXTENSION_DIR_NAME)]
    assert not offending, (
        f"Unexpected new/changed git status entries outside {EXTENSION_DIR_NAME}/: {offending}"
    )


def test_git_diff_stat_of_tracked_files_matches_baseline(manifest):
    result = subprocess.run(
        ["git", "diff", "--stat"], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    assert result.stdout == manifest["git_diff_stat_before"], (
        "git diff --stat changed since baseline for pre-existing tracked files.\n"
        f"--- baseline ---\n{manifest['git_diff_stat_before']}\n--- current ---\n{result.stdout}"
    )
