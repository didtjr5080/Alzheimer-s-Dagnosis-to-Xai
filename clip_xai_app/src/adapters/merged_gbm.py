"""Live-inference adapter for the merged (OASIS-3+ADNI) GBM baseline branch.

Loads `merged_project/models/gbm_baseline.txt` (a LightGBM multiclass
Booster, `num_model_per_iteration() == 3`) and scores it against structural
volume-ratio features looked up by `scan_id` from
`merged_project/unified_final_features_gbm.csv`. `is_adni` is not a column
in that CSV -- it is derived from `dataset_source` at call time, per work
order section 2. Missing-value handling is intentionally absent (the
original notebook cell 30 did not impute), matching the shipped
`unified_final_features_gbm.csv`, which has 0 missing values among the
required columns for every split.
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import pandas as pd

CLASS_NAMES = ["CN", "MCI", "AD"]

FEATURE_COLS = [
    "Hippocampus_ratio", "Amygdala_ratio", "LateralVentricle_ratio",
    "InfLatVent_ratio", "Entorhinal_ratio", "Thalamus_ratio",
    "Accumbens_ratio", "BrainStem_ratio", "AgeatEntry", "GENDER", "is_adni",
]


class MergedGbmBundle:
    def __init__(self, booster: lgb.Booster, features: pd.DataFrame):
        self.booster = booster
        self.features = features  # indexed by scan_id, columns = FEATURE_COLS

    def available_scan_ids(self) -> set[str]:
        return set(self.features.index)

    def predict_by_scan_id(self, scan_id: str):
        if scan_id not in self.features.index:
            return None  # "사용 불가": no SynthSeg volume features for this scan
        row = self.features.loc[[scan_id], FEATURE_COLS]
        probs = self.booster.predict(row)[0]
        return {name: float(p) for name, p in zip(CLASS_NAMES, probs)}


def load_merged_gbm_bundle(repo_root: Path) -> MergedGbmBundle:
    merged_root = repo_root / "merged_project"
    booster_path = merged_root / "models" / "gbm_baseline.txt"
    features_path = merged_root / "unified_final_features_gbm.csv"

    if not booster_path.exists():
        raise FileNotFoundError(f"missing GBM baseline model: {booster_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"missing GBM feature table: {features_path}")

    booster = lgb.Booster(model_file=str(booster_path))
    if booster.num_model_per_iteration() != 3:
        raise ValueError(
            f"unexpected num_model_per_iteration={booster.num_model_per_iteration()} (expected 3)"
        )
    booster_features = booster.feature_name()
    if booster_features != FEATURE_COLS:
        raise ValueError(f"booster feature_name() {booster_features} != expected {FEATURE_COLS}")

    df = pd.read_csv(features_path, encoding="utf-8-sig")
    df["is_adni"] = (df["dataset_source"] == "ADNI").astype(int)
    df = df.set_index("scan_id")

    return MergedGbmBundle(booster=booster, features=df)
