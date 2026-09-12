"""One-off helper (not part of the test suite) to regenerate
docs/preexisting_files_manifest.json after the user adds new content outside
graph_xai_extension/. Not imported anywhere; run manually with `python`."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

EXTENSION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXTENSION_ROOT.parent
EXTENSION_DIR_NAME = EXTENSION_ROOT.name
EXCLUDED_LARGE_DIRS = [
    "merged_project",
    "merged_project-20260825T072246Z-1-001",
    "merged_project-20260825T072246Z-1-002",
    "merged_project-20260825T072246Z-1-003",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dir_stats(directory: Path) -> dict:
    file_count = 0
    total_bytes = 0
    for p in directory.rglob("*"):
        if p.is_file():
            file_count += 1
            total_bytes += p.stat().st_size
    return {"file_count": file_count, "total_bytes": total_bytes}


def main(note: str, baseline_version: int) -> None:
    excluded_top_level = {".git", EXTENSION_DIR_NAME, *EXCLUDED_LARGE_DIRS}
    file_sha256 = {}
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        if Path(dirpath) == REPO_ROOT:
            dirnames[:] = [d for d in dirnames if d not in excluded_top_level]
        for filename in filenames:
            full = Path(dirpath) / filename
            rel = os.path.relpath(full, REPO_ROOT).replace(os.sep, "/")
            file_sha256[rel] = _sha256(full)

    status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    diff_stat = subprocess.run(["git", "diff", "--stat"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_version": baseline_version,
        "note": note,
        "repo_root": str(REPO_ROOT).replace("\\", "/"),
        "git_branch": branch,
        "git_head": head,
        "git_status_short": [line for line in status.splitlines() if line.strip()],
        "git_diff_stat_before": diff_stat,
        "tracked_file_count": None,
        "hashed_file_count": len(file_sha256),
        "hash_algorithm": "sha256",
        "hashed_files_note": (
            f"All files in the repo except .git/, {EXTENSION_DIR_NAME}/ (the allowed-to-modify "
            "extension folder), and the large data-dump directories listed in "
            "excluded_large_directories (hashed individually would be too slow; tracked by "
            "file_count/total_bytes instead)."
        ),
        "excluded_large_directories": {name: _dir_stats(REPO_ROOT / name) for name in EXCLUDED_LARGE_DIRS},
        "file_sha256": file_sha256,
    }

    out_path = EXTENSION_ROOT / "docs" / "preexisting_files_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path} with {len(file_sha256)} hashed files")


if __name__ == "__main__":
    main(
        note=(
            "Refreshed for the '의료진용_보고서_개선' work order. v2 (before this work order existed) "
            "is archived at preexisting_files_manifest_v2_20260912.json; v1 at "
            "preexisting_files_manifest_v1_20260912.json. This v3 baseline additionally includes "
            "WorkOrder/Graph_XAI_의료진용_보고서_개선_작업지시서.md, which the user added between work orders."
        ),
        baseline_version=3,
    )
