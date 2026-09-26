"""Fixed-weight ensemble combining rule for the merged (OASIS-3+ADNI)
branches, per work order section 4 (notebook cell 46). This is NOT a
rediscovered rule -- the weights below are given directly by the work order
and are not tuned or searched for in this codebase.

GBM must be the *baseline* branch (`merged_gbm.py` / `gbm_baseline.txt`),
not the hierarchical stage1/stage2 model -- using the hierarchical
probabilities does not reproduce `final_ensemble_results.csv`.
"""
from __future__ import annotations

CLASS_NAMES = ["CN", "MCI", "AD"]

ENSEMBLE_WEIGHTS = {"gbm": 0.35, "cnn": 0.45, "clip": 0.20}


def combine_ensemble(gbm_probs: dict, cnn_probs: dict, clip_probs: dict) -> dict:
    """Each *_probs is a {"CN": p, "MCI": p, "AD": p} dict from the baseline
    GBM, 3D CNN, and merged CLIP branches respectively. Returns the same
    shape. Callers must treat a missing branch as "사용 불가" for the whole
    ensemble (no re-weighting of the remaining branches), per section 4."""
    w = ENSEMBLE_WEIGHTS
    return {
        name: w["gbm"] * gbm_probs[name] + w["cnn"] * cnn_probs[name] + w["clip"] * clip_probs[name]
        for name in CLASS_NAMES
    }


def ensemble_predicted_class(ensemble_probs: dict) -> str:
    return max(CLASS_NAMES, key=lambda name: ensemble_probs[name])
