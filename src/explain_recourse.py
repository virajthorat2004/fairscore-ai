import shap
import pandas as pd

from ensemble_predictor import ADAPTERS, load_model, build_input_row, run_ensemble

# Known, previously-validated calibration caveats per dataset, from this
# project's own confidence/calibration analysis on each dataset's held-out
# test set. These are dataset-level properties, not recomputed per
# applicant -- shown as honest caveats on how much to trust a model's
# raw probability.
CALIBRATION_NOTES = {
    "german_credit": "Reasonably well-calibrated; minor noise in the 0.35-0.55 probability band due to limited test-set size.",
    "gmsc": "Broadly well-calibrated with mild deviations in the mid-probability range.",
    "homecredit": "Systematically overconfident across all probability bins -- class-weighting used to fix recall materially hurt calibration. Treat its probability as a relative risk ranking, not a true likelihood.",
    "africa": "Not independently calibration-tested (n=101 is too small for a reliable calibration curve). Apparent near-perfect discrimination on this dataset is itself a sign of overfitting risk, not strong performance.",
}

# Fields the applicant can realistically and immediately change without
# re-uploading a new statement -- things under their control for THIS loan
# request, not their financial history.
MODIFIABLE_CORE_FIELDS = ["requested_amount", "requested_duration_months"]
RECOURSE_STEPS = [0.1, 0.2, 0.3]


def compute_shap_breakdown(core, stmt_features, stmt_df, gate_answer, ensemble_result, top_n=5):
    """
    For each model that voted, computes that model's own SHAP explanation for
    this specific applicant. Kept separate per model (not merged into one
    list) because each dataset's features are a different, non-comparable
    space -- merging them numerically would be misleading.
    """
    breakdown = {}
    for dataset_name in ensemble_result["models_used"]:
        X, tags, _ = build_input_row(dataset_name, core, stmt_features, stmt_df, gate_answer)
        model = load_model(dataset_name)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)[0]

        rows = []
        for feat, val in zip(X.columns, shap_values):
            if abs(val) < 1e-6:
                continue
            rows.append({"feature": feat, "shap_value": float(val), "source": tags.get(feat, "imputed")})
        rows.sort(key=lambda r: abs(r["shap_value"]), reverse=True)
        breakdown[dataset_name] = rows[:top_n]

    return breakdown


def compute_confidence(ensemble_result):
    """
    Combines two signals: how far the consensus sits from the 0.5 decision
    boundary, and -- when more than one model voted -- how much those models
    agree with each other. A result near 0.5, or where qualifying models
    disagree substantially, is flagged lower-confidence even if a single
    model alone looked decisive.
    """
    consensus = ensemble_result["consensus_risk_probability"]
    qualifying = {k: v for k, v in ensemble_result["details"].items() if v["qualifies"]}

    if consensus is None or not qualifying:
        return {"label": "Low", "combined_score": 0.0, "manual_review": True,
                "explanation": "No model had enough information to vote."}

    distance_score = abs(consensus - 0.5) * 2  # 0 (at boundary) to 1 (certain)

    probs = [v["risk_probability"] for v in qualifying.values()]
    if len(probs) > 1:
        spread = max(probs) - min(probs)
        agreement_score = max(0.0, 1 - spread / 0.5)
        combined = (distance_score + agreement_score) / 2
        explanation = f"{len(probs)} models voted; spread between their risk estimates was {spread:.3f}."
    else:
        agreement_score = None
        combined = distance_score * 0.8  # single-vote cap -- never "High" on one model alone
        explanation = "Only one model qualified to vote -- confidence is capped accordingly."

    label = "High" if combined >= 0.65 else ("Medium" if combined >= 0.35 else "Low")

    return {
        "label": label, "combined_score": round(combined, 3),
        "distance_score": round(distance_score, 3),
        "agreement_score": round(agreement_score, 3) if agreement_score is not None else None,
        "manual_review": label == "Low", "explanation": explanation,
    }


def get_calibration_notes(models_used):
    return {name: CALIBRATION_NOTES[name] for name in models_used}


def generate_recourse(core, stmt_features, stmt_df, gate_answer, original_result):
    """
    Only meaningful when the applicant was assessed as higher-risk. Re-runs
    the FULL ensemble (not a single model in isolation) with modified loan
    terms, so a suggestion reflects the real, final decision logic --
    including cases where a smaller loan changes which models even qualify.
    """
    if original_result["consensus_risk_probability"] is None:
        return []
    if original_result["consensus_risk_probability"] < 0.5:
        return []

    original_risk = original_result["consensus_risk_probability"]
    suggestions = []
    fields = [f for f in MODIFIABLE_CORE_FIELDS if core.get(f) is not None]

    for field in fields:
        for pct in RECOURSE_STEPS:
            modified_core = dict(core)
            modified_core[field] = core[field] * (1 - pct)
            new_result = run_ensemble(modified_core, stmt_features, stmt_df, gate_answer)
            new_risk = new_result["consensus_risk_probability"]
            if new_risk is None:
                continue
            reduction = original_risk - new_risk
            if reduction > max(0.02, 0.05 * original_risk):
                suggestions.append({
                    "change": f"{field.replace('_', ' ')} -{int(pct*100)}%",
                    "new_risk": round(new_risk, 4), "reduction": round(reduction, 4),
                    "flips_to_lower_risk": new_risk < 0.5,
                })

    if len(fields) == 2:
        for pct in RECOURSE_STEPS:
            modified_core = dict(core)
            for field in fields:
                modified_core[field] = core[field] * (1 - pct)
            new_result = run_ensemble(modified_core, stmt_features, stmt_df, gate_answer)
            new_risk = new_result["consensus_risk_probability"]
            if new_risk is None:
                continue
            reduction = original_risk - new_risk
            if reduction > max(0.02, 0.05 * original_risk):
                suggestions.append({
                    "change": f"both fields -{int(pct*100)}%",
                    "new_risk": round(new_risk, 4), "reduction": round(reduction, 4),
                    "flips_to_lower_risk": new_risk < 0.5,
                })

    suggestions.sort(key=lambda s: s["reduction"], reverse=True)
    return suggestions[:5]