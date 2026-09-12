from __future__ import annotations

from .contracts import XAIResult


def gbm_feature_contributions_unavailable(model_id: str = "gbm_reference") -> XAIResult:
    result = XAIResult(
        model_id=model_id,
        target_class="N/A",
        target_score_type="raw_tree_output",
        method_name="feature_contribution_blocked",
        source_level="subject",
        feature_contributions=[],
        warnings=[
            "GBM feature contribution is blocked until the runtime environment provides LightGBM and the validated feature schema."
        ],
    )
    return result


def ensemble_contributions_unavailable(model_id: str = "ensemble_reference") -> XAIResult:
    result = XAIResult(
        model_id=model_id,
        target_class="N/A",
        target_score_type="final_ensemble_probability",
        method_name="model_contribution_blocked",
        source_level="subject",
        model_contributions=[],
        warnings=[
            "Ensemble model contribution is blocked because final probabilities and exact combining rule are absent."
        ],
    )
    return result
