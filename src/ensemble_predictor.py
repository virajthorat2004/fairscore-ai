import pickle
from functools import lru_cache

import pandas as pd

from feature_adapters import ADAPTERS, load_reference
from config import (APPLICABILITY_THRESHOLD, TAG_WEIGHTS, CALIBRATION_WEIGHT,
                    VOTING_MODELS, ADVISORY_MODELS, MODEL_PATHS, SHAP_WEIGHTS_PATH)

# Top global-SHAP features per dataset (from each dataset's global summary).
# Used ONLY for the applicability gate.
TOP_SHAP_FEATURES = {
    "german_credit": [
        "checking_account_no_checking_account", "duration", "credit_history_no_credits_all_paid",
        "credit_amount", "checking_account_lt0_DM", "savings_account_lt100_DM",
        "installment_rate", "other_installment_plans_A143", "purpose_A41", "purpose_A43",
        "employment_since_lt1_year", "personal_status_sex_A93", "savings_account_unknown_none",
        "employment_since_4-7_years", "age",
    ],
    "gmsc": [
        "RevolvingUtilizationOfUnsecuredLines", "NumberOfTime30-59DaysPastDueNotWorse", "age",
        "NumberOfTimes90DaysLate", "NumberOfOpenCreditLinesAndLoans", "DebtRatio",
        "NumberOfTime60-89DaysPastDueNotWorse", "NumberRealEstateLoansOrLines",
        "monthly_income_was_missing", "MonthlyIncome", "NumberOfDependents",
    ],
    "homecredit": [
        "EXT_SOURCE_3", "EXT_SOURCE_2", "AMT_GOODS_PRICE", "AMT_CREDIT", "CODE_GENDER_M",
        "age", "DAYS_EMPLOYED", "AMT_ANNUITY", "DAYS_ID_PUBLISH", "FLAG_OWN_CAR_Y",
        "NAME_EDUCATION_TYPE_Higher_education", "DAYS_LAST_PHONE_CHANGE",
        "NAME_FAMILY_STATUS_Married", "NAME_CONTRACT_TYPE_Revolving_loans", "FLAG_DOCUMENT_3",
    ],
    "africa": [
        "overdraft_frequency", "overdraft_event_count", "risk_behavior_score", "avg_balance",
        "min_balance_observed", "cashflow_stability_score", "discretionary_spend_ratio",
        "betting_ratio", "min_balance", "avg_txn_amount",
    ],
}


def _validate_top_features():
    """Drop (and loudly report) any hardcoded feature name that doesn't exist
    in the real encoded columns, so a typo can't silently corrupt applicability."""
    for dataset_name, feature_list in TOP_SHAP_FEATURES.items():
        real_cols = set(load_reference(dataset_name)["all_columns"])
        invalid = [f for f in feature_list if f not in real_cols]
        if invalid:
            print(f"WARNING [{dataset_name}]: unknown SHAP feature names ignored: {invalid}")
        TOP_SHAP_FEATURES[dataset_name] = [f for f in feature_list if f in real_cols]


def _load_gate_weights():
    """Per-dataset feature weights for the applicability gate, normalised to sum to 1.
    Uses mean |SHAP| when models/shap_weights.pkl exists, else equal weights."""
    raw = {}
    if SHAP_WEIGHTS_PATH.exists():
        with open(SHAP_WEIGHTS_PATH, "rb") as f:
            raw = pickle.load(f)

    weights, mode = {}, {}
    for ds, feats in TOP_SHAP_FEATURES.items():
        if ds in raw:
            w = {f: abs(float(raw[ds].get(f, 0.0))) for f in feats}
            mode[ds] = "shap"
        else:
            w = {f: 1.0 for f in feats}
            mode[ds] = "equal"
        total = sum(w.values()) or 1.0
        weights[ds] = {f: v / total for f, v in w.items()}
    return weights, mode


_validate_top_features()
GATE_WEIGHTS, GATE_MODE = _load_gate_weights()


@lru_cache(maxsize=None)
def load_model(dataset_name):
    """Loaded once per process, then reused (recourse calls this many times)."""
    with open(MODEL_PATHS[dataset_name], "rb") as f:
        return pickle.load(f)


def compute_applicability(dataset_name, tags):
    weights = GATE_WEIGHTS[dataset_name]
    return sum(w * TAG_WEIGHTS.get(tags.get(f, "imputed"), 0.0) for f, w in weights.items())


def build_input_row(dataset_name, core, stmt_features, stmt_df, gate_answer):
    """Returns (X_row_df, tags, applicability). Shared with explain_recourse so
    explanations use the exact vector the model was scored on."""
    features, tags = ADAPTERS[dataset_name](core, stmt_features, stmt_df, gate_answer)
    applicability = compute_applicability(dataset_name, tags)
    ref_cols = load_reference(dataset_name)["all_columns"]
    X = pd.DataFrame([[features.get(c, 0) for c in ref_cols]], columns=ref_cols)
    return X, tags, applicability


def check_consistency(stmt_features, gate_answer):
    """Declared credit status vs statement evidence. Mismatch -> manual review."""
    detected = stmt_features.get("detected_existing_credit", False)
    declared_has_credit = (gate_answer == "Yes")

    if detected and not declared_has_credit:
        return True, ("Applicant declared no existing credit, but the statement shows "
                      f"{stmt_features.get('detected_credit_signal_count', 1)} recurring "
                      "EMI/credit-card-style debit(s). Flagged for manual review.")
    if declared_has_credit and not detected and stmt_features.get("statement_months", 0) >= 3:
        return True, ("Applicant declared existing credit, but no matching recurring debit "
                      "was found in the statement. Flagged for manual review.")
    return False, None


def run_ensemble(core, stmt_features, stmt_df, gate_answer):
    """Scores every model, but only VOTING_MODELS that clear the threshold
    influence the consensus. Advisory models are reported, never counted."""
    results = {}
    consistency_flagged, consistency_message = check_consistency(stmt_features, gate_answer)

    for dataset_name in ADAPTERS:
        X, tags, applicability = build_input_row(dataset_name, core, stmt_features, stmt_df, gate_answer)
        is_voter = dataset_name in VOTING_MODELS
        clears = applicability >= APPLICABILITY_THRESHOLD
        qualifies = is_voter and clears

        entry = {
            "role": "voter" if is_voter else "advisory",
            "applicability": round(applicability, 3),
            "gate_mode": GATE_MODE[dataset_name],
            "qualifies": qualifies,
            "tags": tags,
            "risk_probability": None,
            "weight": 0.0,
        }

        if qualifies:
            prob = float(load_model(dataset_name).predict_proba(X)[0][1])
            entry["risk_probability"] = round(prob, 4)
            entry["weight"] = round(applicability * CALIBRATION_WEIGHT[dataset_name], 4)
        elif not is_voter:
            if clears:
                entry["advisory_risk_probability"] = round(
                    float(load_model(dataset_name).predict_proba(X)[0][1]), 4)
            entry["exclusion_reason"] = (
                "Explanation-only model: trained on 101 synthetic records, and its near-perfect "
                "test score indicates overfitting. Shown for behavioural insight; never counted "
                "in the decision.")
        else:
            entry["exclusion_reason"] = (
                f"Applicability {applicability:.2f} is below the {APPLICABILITY_THRESHOLD:.2f} "
                "threshold: too many of this model's key features could not be genuinely "
                "derived from the applicant's inputs.")

        results[dataset_name] = entry

    qualifying = {k: v for k, v in results.items() if v["qualifies"]}
    consensus = None
    if qualifying:
        total_weight = sum(v["weight"] for v in qualifying.values())
        if total_weight > 0:
            consensus = sum(v["risk_probability"] * v["weight"] for v in qualifying.values()) / total_weight

    return {
        "consensus_risk_probability": round(consensus, 4) if consensus is not None else None,
        "models_used": list(qualifying.keys()),
        "models_excluded": [k for k in results if k not in qualifying],
        "models_advisory": [k for k, v in results.items()
                            if v["role"] == "advisory" and "advisory_risk_probability" in v],
        "details": results,
        "consistency_flagged": consistency_flagged,
        "consistency_message": consistency_message,
        "manual_review_required": consistency_flagged or (consensus is None),
    }


if __name__ == "__main__":
    from statement_parser import parse_statement

    stmt_df, stmt_features, _ = parse_statement("data/raw/sample_statement.csv")
    core = {"age": 29, "monthly_income": 42000, "requested_amount": 200000,
            "requested_duration_months": 24, "total_savings": 25000,
            "dependents": 1, "employment_days": 800}

    result = run_ensemble(core, stmt_features, stmt_df, gate_answer="Yes")
    print("Threshold:", APPLICABILITY_THRESHOLD)
    print("Consensus:", result["consensus_risk_probability"])
    print("Voted:", result["models_used"], "| Advisory:", result["models_advisory"])
    for name, d in result["details"].items():
        line = f"{name:14s} role={d['role']:8s} applic={d['applicability']:.3f} gate={d['gate_mode']}"
        if d["qualifies"]:
            line += f" risk={d['risk_probability']}"
        elif "advisory_risk_probability" in d:
            line += f" advisory_risk={d['advisory_risk_probability']}"
        print(line)