"""Validate the merged_project folder without loading full image/model datasets into memory.

Usage:
    python scripts/validate_merged_project.py --root E:\\Develop\\CLIPtoXAI

Exit codes:
    0 = validation completed with no blocking errors
    1 = validation completed with errors
    3 = required folders are missing
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


PART_NAMES = [
    "merged_project-20260825T072246Z-1-001",
    "merged_project-20260825T072246Z-1-002",
    "merged_project-20260825T072246Z-1-003",
]
IMPORTANT_FILES = [
    "checkpoints/clip_best.pt",
    "checkpoints/clip_latest.pt",
    "checkpoints/3dcnn_best.pt",
    "checkpoints/3dcnn_latest.pt",
    "models/gbm_stage1.txt",
    "models/gbm_stage2.txt",
    "models/gbm_baseline.txt",
    "gradcam_results/gradcam_summary.csv",
    "clip_test_probs.csv",
    "3dcnn_test_probs.csv",
    "gbm_hierarchical_test_probs.csv",
    "final_ensemble_results.csv",
    "unified_manifest.csv",
    "unified_slice_manifest.csv",
    "unified_final_features_gbm.csv",
    "task_A_column_summary.txt",
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


def rel(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def inventory(root: Path):
    rows = {}
    for path in iter_files(root):
        st = path.stat()
        rows[rel(path, root)] = {"bytes": st.st_size, "mtime": st.st_mtime}
    return rows


def csv_probe(path: Path) -> dict:
    result = {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
        "ok": False,
    }
    if not path.exists():
        result["error"] = "missing"
        return result
    if result["bytes"] == 0:
        result["error"] = "empty"
        return result
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            duplicates = [name for name, count in Counter(header).items() if count > 1]
            count = 0
            for count, _ in enumerate(reader, 1):
                pass
        result.update(
            ok=True,
            rows=count,
            columns=len(header),
            duplicate_columns=duplicates,
            header=header[:50],
        )
    except Exception as exc:  # noqa: BLE001
        result["error"] = repr(exc)
    return result


def checkpoint_probe(path: Path) -> dict:
    result = {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256(path) if path.exists() else None,
        "ok": False,
    }
    if not path.exists():
        result["error"] = "missing"
        return result
    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"torch unavailable: {exc!r}"
        return result
    try:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
            result["load_mode"] = "weights_only=True"
        except TypeError:
            obj = torch.load(path, map_location="cpu")
            result["load_mode"] = "legacy torch.load"
        except Exception as first_exc:  # noqa: BLE001
            obj = torch.load(path, map_location="cpu", weights_only=False)
            result["load_mode"] = f"weights_only=False after {type(first_exc).__name__}"
        result["object_type"] = type(obj).__name__
        tensors = []
        if isinstance(obj, dict):
            result["top_keys"] = list(obj.keys())[:30]
            source = obj.get("state_dict", obj.get("model_state", obj))
            if isinstance(source, dict):
                for key, value in source.items():
                    if hasattr(value, "shape") and hasattr(value, "dtype"):
                        tensors.append(
                            {"key": str(key), "shape": list(value.shape), "dtype": str(value.dtype)}
                        )
                    if len(tensors) >= 25:
                        break
        result["tensor_samples"] = tensors
        result["ok"] = True
    except Exception as exc:  # noqa: BLE001
        result["error"] = repr(exc)
    return result


def npy_probe(paths: list[Path], sample_size: int) -> list[dict]:
    sample = paths[:]
    random.Random(20260826).shuffle(sample)
    sample = sample[:sample_size]
    results = []
    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return [{"ok": False, "error": f"numpy unavailable: {exc!r}", "sample_count": len(sample)}]
    for path in sample:
        item = {"path": str(path), "bytes": path.stat().st_size, "ok": False}
        try:
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            finite_sample = arr
            if arr.size > 50000:
                finite_sample = arr.reshape(-1)[:: max(1, arr.size // 50000)]
            item.update(
                ok=True,
                shape=list(arr.shape),
                dtype=str(arr.dtype),
                has_nan=bool(np.isnan(finite_sample).any()) if np.issubdtype(arr.dtype, np.number) else None,
                has_inf=bool(np.isinf(finite_sample).any()) if np.issubdtype(arr.dtype, np.number) else None,
                min=float(np.nanmin(finite_sample)) if np.issubdtype(arr.dtype, np.number) else None,
                max=float(np.nanmax(finite_sample)) if np.issubdtype(arr.dtype, np.number) else None,
            )
        except Exception as exc:  # noqa: BLE001
            item["error"] = repr(exc)
        results.append(item)
    return results


def png_distribution(paths: list[Path]) -> dict:
    sizes = [p.stat().st_size for p in paths]
    if not sizes:
        return {"count": 0}
    sizes.sort()
    return {
        "count": len(sizes),
        "min_bytes": sizes[0],
        "median_bytes": sizes[len(sizes) // 2],
        "max_bytes": sizes[-1],
    }


def scan_code(root: Path) -> dict:
    patterns = {
        "readmes": ["README*"],
        "requirements": ["requirements*.txt", "pyproject.toml", "environment.yml", "setup.py"],
    }
    found = defaultdict(list)
    for pattern in patterns["readmes"]:
        found["readmes"].extend(str(p) for p in root.rglob(pattern))
    for pattern in patterns["requirements"]:
        found["requirements"].extend(str(p) for p in root.rglob(pattern))
    code_hits = []
    terms = re.compile(r"(torch|clip|grad.?cam|lightgbm|xgboost|checkpoint|predict|infer)", re.I)
    for path in root.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if terms.search(text):
            code_hits.append(str(path))
    found["python_code_hits"] = code_hits[:100]
    return dict(found)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--sample-npy", type=int, default=12)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    target = root / "merged_project"
    artifact_dir = root / "artifacts" / "merge_validation"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = artifact_dir / f"validate_{stamp}.json"

    part_roots = [root / name / "merged_project" for name in PART_NAMES]
    errors = []
    if not target.is_dir():
        errors.append(f"missing target: {target}")
    missing_parts = [str(path) for path in part_roots if not path.is_dir()]
    if missing_parts:
        errors.extend(f"missing part: {path}" for path in missing_parts)
    if errors:
        report_path.write_text(json.dumps({"errors": errors}, indent=2), encoding="utf-8")
        print(f"Validation failed before scan. Report: {report_path}")
        return 3

    part_inventories = [inventory(path) for path in part_roots]
    expected = {}
    for inv in part_inventories:
        expected.update(inv)
    actual = inventory(target)
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    size_mismatches = sorted(
        key for key in set(expected) & set(actual) if expected[key]["bytes"] != actual[key]["bytes"]
    )

    ext_counts = Counter(Path(name).suffix.lower() or "[no_ext]" for name in actual)
    important = {}
    for name in IMPORTANT_FILES:
        path = target / name
        important[name] = {
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else None,
            "sha256": sha256(path) if path.exists() else None,
        }

    csv_files = [target / name for name in IMPORTANT_FILES if name.endswith(".csv")]
    csv_results = [csv_probe(path) for path in csv_files]
    checkpoint_results = [
        checkpoint_probe(target / name) for name in IMPORTANT_FILES if name.startswith("checkpoints/")
    ]

    npy_paths = sorted((target / "mri_3d_cache").glob("*.npy")) if (target / "mri_3d_cache").is_dir() else []
    png_paths = sorted((target / "slices_multi").glob("*.png")) if (target / "slices_multi").is_dir() else []
    gradcam_png = (
        sorted((target / "gradcam_results").glob("*.png"))
        if (target / "gradcam_results").is_dir()
        else []
    )
    gradcam_summary = target / "gradcam_results" / "gradcam_summary.csv"
    gradcam_refs = {"checked": False}
    if gradcam_summary.exists():
        summary = csv_probe(gradcam_summary)
        refs_found = 0
        if summary.get("ok"):
            text = gradcam_summary.read_text(encoding="utf-8-sig", errors="ignore")
            gradcam_names = {p.name for p in gradcam_png}
            refs_found = sum(1 for name in gradcam_names if name in text)
        gradcam_refs = {"checked": True, "summary": summary, "png_names_referenced": refs_found}

    names = list(actual)
    has_adni = any(re.search(r"\d{3}_S_\d{4}", name) for name in names)
    has_oasis = any("OAS" in name for name in names)

    report = {
        "root": str(root),
        "target": str(target),
        "part_file_counts": [len(inv) for inv in part_inventories],
        "expected_unique_files": len(expected),
        "actual_files": len(actual),
        "actual_total_bytes": sum(item["bytes"] for item in actual.values()),
        "extension_counts": dict(sorted(ext_counts.items())),
        "missing_files": missing[:200],
        "missing_count": len(missing),
        "extra_files": extra[:200],
        "extra_count": len(extra),
        "size_mismatches": size_mismatches[:200],
        "size_mismatch_count": len(size_mismatches),
        "contains_adni_names": has_adni,
        "contains_oasis_names": has_oasis,
        "important_files": important,
        "csv_results": csv_results,
        "checkpoint_results": checkpoint_results,
        "npy_count": len(npy_paths),
        "npy_samples": npy_probe(npy_paths, args.sample_npy),
        "slices_multi_png": png_distribution(png_paths),
        "gradcam_png": png_distribution(gradcam_png),
        "gradcam_summary_refs": gradcam_refs,
        "code_and_dependency_files": scan_code(root),
        "errors": errors,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Validation report: {report_path}")
    print(
        "Files expected/actual/missing/extra/size_mismatch: "
        f"{len(expected)}/{len(actual)}/{len(missing)}/{len(extra)}/{len(size_mismatches)}"
    )
    print(f"slices_multi PNG: {len(png_paths)}; mri_3d_cache NPY: {len(npy_paths)}; Grad-CAM PNG: {len(gradcam_png)}")
    print(f"ADNI names: {has_adni}; OASIS names: {has_oasis}")
    blocking = bool(missing or extra or size_mismatches or errors)
    return 1 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
