"""Fast read-only inventory and collision check for merged_project split folders.

Usage:
    python scripts/inspect_merged_parts_fast.py --root E:\\Develop\\CLIPtoXAI

Exit codes:
    0 = no blocking relative-path content conflicts
    2 = at least one same-relative-path file has different content
    3 = required part root is missing
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


PART_NAMES = [
    "merged_project-20260825T072246Z-1-001",
    "merged_project-20260825T072246Z-1-002",
    "merged_project-20260825T072246Z-1-003",
]


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(root: Path):
    for dirpath, _, filenames in os.walk(root):
        base = Path(dirpath)
        for filename in filenames:
            yield base / filename


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()

    root = Path(args.root).resolve()
    artifact_dir = root / "artifacts" / "merge_validation"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = artifact_dir / f"fast_part_summary_{stamp}.csv"
    rel_collision_path = artifact_dir / f"fast_relative_path_collisions_{stamp}.csv"
    basename_collision_path = artifact_dir / f"fast_basename_collisions_{stamp}.csv"

    part_roots = [(name, root / name / "merged_project") for name in PART_NAMES]
    missing = [path for _, path in part_roots if not path.is_dir()]
    if missing:
        for path in missing:
            print(f"Missing part root: {path}")
        return 3

    by_rel: dict[str, list[dict]] = defaultdict(list)
    by_name: dict[str, list[dict]] = defaultdict(list)
    summary_rows = []
    for part_name, part_root in part_roots:
        print(f"Scanning {part_name}")
        file_count = 0
        total_bytes = 0
        ext_counts: Counter[str] = Counter()
        for path in iter_files(part_root):
            st = path.stat()
            rel = path.relative_to(part_root).as_posix()
            row = {
                "part": part_name,
                "relative_path": rel,
                "file_name": path.name,
                "full_path": str(path),
                "bytes": st.st_size,
                "mtime": st.st_mtime,
            }
            by_rel[rel].append(row)
            by_name[path.name].append(row)
            file_count += 1
            total_bytes += st.st_size
            ext_counts[path.suffix.lower() or "[no_ext]"] += 1
        summary_rows.append(
            {
                "part": part_name,
                "file_count": file_count,
                "total_bytes": total_bytes,
                "extension_counts": "; ".join(f"{k}={v}" for k, v in sorted(ext_counts.items())),
            }
        )

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["part", "file_count", "total_bytes", "extension_counts"])
        writer.writeheader()
        writer.writerows(summary_rows)

    rel_collisions = {key: rows for key, rows in by_rel.items() if len(rows) > 1}
    blocking = []
    with rel_collision_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["relative_path", "part", "full_path", "bytes", "mtime", "sha256", "identical_content"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for relative_path, rows in sorted(rel_collisions.items()):
            hashes = []
            for row in rows:
                row_hash = sha256(Path(row["full_path"]))
                hashes.append(row_hash)
            identical = len(set(hashes)) == 1
            if not identical:
                blocking.append(relative_path)
            for row, row_hash in zip(rows, hashes):
                writer.writerow(
                    {
                        "relative_path": relative_path,
                        "part": row["part"],
                        "full_path": row["full_path"],
                        "bytes": row["bytes"],
                        "mtime": row["mtime"],
                        "sha256": row_hash,
                        "identical_content": identical,
                    }
                )

    basename_groups = {key: rows for key, rows in by_name.items() if len(rows) > 1}
    with basename_collision_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["file_name", "part", "relative_path", "full_path", "bytes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for file_name, rows in sorted(basename_groups.items()):
            for row in rows:
                writer.writerow(
                    {
                        "file_name": file_name,
                        "part": row["part"],
                        "relative_path": row["relative_path"],
                        "full_path": row["full_path"],
                        "bytes": row["bytes"],
                    }
                )

    print(f"Summary: {summary_path}")
    print(f"Relative collisions: {len(rel_collisions)} -> {rel_collision_path}")
    print(f"Same filename groups: {len(basename_groups)} -> {basename_collision_path}")
    print(f"Blocking content conflicts: {len(blocking)}")
    return 2 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
