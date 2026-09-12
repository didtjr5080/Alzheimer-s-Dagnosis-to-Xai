from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


CLASS_NAMES = ["CN", "MCI", "AD"]
RESULT_FILES = [
    "clip_test_probs.csv",
    "3dcnn_test_probs.csv",
    "gbm_hierarchical_test_probs.csv",
    "final_ensemble_results.csv",
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
        for name in filenames:
            yield base / name


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        return sum(1 for _ in reader)


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _class_order_from_columns(columns: list[str], prefixes: list[str]) -> list[str]:
    for prefix in prefixes:
        names = []
        for class_name in CLASS_NAMES:
            if f"{prefix}{class_name}" in columns:
                names.append(class_name)
        if names == CLASS_NAMES:
            return names
    return []


def audit_merged_project(repo_root: Path) -> dict[str, Any]:
    merged_root = repo_root / "merged_project"
    if not merged_root.is_dir():
        raise FileNotFoundError(f"missing merged_project: {merged_root}")

    files = list(iter_files(merged_root))
    ext_counts = Counter(path.suffix.lower() or "[no_ext]" for path in files)
    csv_contracts: dict[str, Any] = {}
    for name in [
        "unified_manifest.csv",
        "unified_slice_manifest.csv",
        "unified_final_features_gbm.csv",
        *RESULT_FILES,
    ]:
        path = merged_root / name
        header = read_header(path)
        duplicates = [key for key, count in Counter(header).items() if count > 1]
        csv_contracts[name] = {
            "exists": path.exists(),
            "rows": csv_row_count(path),
            "columns": header,
            "duplicate_columns": duplicates,
            "sha256": sha256(path),
        }

    unified = _read_csv_dicts(merged_root / "unified_manifest.csv")
    slice_manifest = _read_csv_dicts(merged_root / "unified_slice_manifest.csv")
    final_rows = _read_csv_dicts(merged_root / "final_ensemble_results.csv")
    scan_ids = {row["scan_id"] for row in unified}
    slice_scan_ids = {row["scan_id"] for row in slice_manifest}
    final_scan_ids = {row["scan_id"] for row in final_rows}
    split_by_scan = {row["scan_id"]: row["split"] for row in unified}
    split_counts = Counter(split_by_scan.values())
    source_counts = Counter(row["dataset_source"] for row in unified)
    class_counts = Counter(row["label"] for row in unified)

    result_scan_sets = {}
    for name in RESULT_FILES:
        rows = _read_csv_dicts(merged_root / name)
        result_scan_sets[name] = {row["scan_id"] for row in rows}

    overlap_checks = {
        "slice_scans_missing_from_manifest": len(slice_scan_ids - scan_ids),
        "final_scans_missing_from_manifest": len(final_scan_ids - scan_ids),
        "result_scan_set_equal": {
            name: sorted(final_scan_ids.symmetric_difference(ids))[:25]
            for name, ids in result_scan_sets.items()
        },
    }

    result_schema = {}
    for name in RESULT_FILES:
        columns = csv_contracts[name]["columns"]
        result_schema[name] = {
            "class_order": _class_order_from_columns(columns, ["clip_", "prob_", "gbm_", "cnn_"]),
            "label_columns": [column for column in columns if column in {"label", "true_label", "label_idx", "pred"}],
            "scan_id_column": "scan_id" in columns,
        }

    subject_prefixes = Counter()
    for row in unified:
        subject = row.get("subject_id", "")
        if re.match(r"^OAS\d+$", subject):
            subject_prefixes["OASIS"] += 1
        elif re.match(r"^\d{3}_S_\d{4}$", subject):
            subject_prefixes["ADNI"] += 1
        else:
            subject_prefixes["other"] += 1

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "merged_root": str(merged_root),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "extension_counts": dict(sorted(ext_counts.items())),
        "csv_contracts": csv_contracts,
        "split_counts": dict(split_counts),
        "dataset_source_counts": dict(source_counts),
        "class_counts": dict(class_counts),
        "subject_id_format_counts": dict(subject_prefixes),
        "contains_adni": subject_prefixes["ADNI"] > 0,
        "contains_oasis": subject_prefixes["OASIS"] > 0,
        "overlap_checks": overlap_checks,
        "result_schema": result_schema,
        "dynamic_model_contracts": {
            "merged_clip_checkpoint": "BLOCKED_MODEL_CONTRACT: clip_best.pt stores class_embeds only; CLIP backbone/preprocessing linkage is not present.",
            "cnn3d_checkpoint": "BLOCKED_MODEL_CONTRACT: state_dict is readable, but model class and preprocessing contract are not present.",
            "gbm_models": "BLOCKED_PREPROCESSING_CONTRACT for live inference unless feature order and is_adni construction are supplied by caller.",
            "ensemble": "BLOCKED_MODEL_CONTRACT: final_ensemble_results.csv is available, but combining rule/weights are not fully recoverable from artifacts.",
        },
    }


def write_audit_artifacts(repo_root: Path) -> tuple[Path, Path]:
    audit = audit_merged_project(repo_root)
    out_dir = repo_root / "clip_xai_app" / "artifacts" / "verification"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "merged_artifact_audit.json"
    md_path = out_dir / "merged_artifact_audit.md"
    json_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Merged Artifact Audit",
        "",
        f"- merged_root: `{audit['merged_root']}`",
        f"- file_count: {audit['file_count']}",
        f"- total_bytes: {audit['total_bytes']}",
        f"- extension_counts: `{audit['extension_counts']}`",
        f"- split_counts: `{audit['split_counts']}`",
        f"- dataset_source_counts: `{audit['dataset_source_counts']}`",
        f"- class_counts: `{audit['class_counts']}`",
        f"- contains_adni: {audit['contains_adni']}",
        f"- contains_oasis: {audit['contains_oasis']}",
        "",
        "## CSV Contracts",
    ]
    for name, contract in audit["csv_contracts"].items():
        lines.append(f"- `{name}`: rows={contract['rows']}, columns={len(contract['columns'])}, sha256={contract['sha256'][:16]}...")
    lines.extend(["", "## Dynamic Model Contract Blocks"])
    for name, reason in audit["dynamic_model_contracts"].items():
        lines.append(f"- {name}: {reason}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
