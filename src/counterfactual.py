import pandas as pd
import pickle
import os
import numpy as np
from itertools import combinations

from data_loader import load_german_credit
from preprocessing import preprocess, get_train_test_split


def load_model(model_name="CatBoost"):
    with open(f"models/{model_name}.pkl", "rb") as f:
        return pickle.load(f)


# Features we consider "modifiable" — things an applicant could realistically change.
# We will NOT touch age, personal_status, foreign_worker, etc. — those aren't actionable
# or ethically appropriate to suggest changing.
MODIFIABLE_NUMERIC = ["credit_amount", "duration", "installment_rate", "existing_credits"]
PERCENT_STEPS = [0.1, 0.2, 0.3]


def generate_counterfactual(model, applicant_row, feature_names):
    """
    Single-feature search: tries reducing ONE modifiable feature at a time
    by 10%, 20%, 30% and checks the effect on predicted risk.
    """
    original = applicant_row.copy()
    original_prob = model.predict_proba(original.values.reshape(1, -1))[0][1]

    suggestions = []
    available_features = [f for f in MODIFIABLE_NUMERIC if f in feature_names]

    for feature in available_features:
        col_idx = list(feature_names).index(feature)
        original_value = original.iloc[col_idx]

        for pct_reduction in PERCENT_STEPS:
            modified = original.copy()
            new_value = original_value * (1 - pct_reduction)
            modified.iloc[col_idx] = new_value

            new_prob = model.predict_proba(modified.values.reshape(1, -1))[0][1]
            improvement = original_prob - new_prob

            if improvement > 0.02:  # only report meaningful improvements
                suggestions.append({
                    "features": feature,
                    "change": f"-{int(pct_reduction*100)}%",
                    "original_risk_prob": round(original_prob, 4),
                    "new_risk_prob": round(new_prob, 4),
                    "risk_reduction": round(improvement, 4),
                    "flips_to_approved": new_prob < 0.5,
                })

    suggestions_df = pd.DataFrame(suggestions).sort_values("risk_reduction", ascending=False)
    return original_prob, suggestions_df


def generate_combined_counterfactual(model, applicant_row, feature_names):
    """
    Combined search: tries changing TWO modifiable features TOGETHER, at matching
    percentage steps. More realistic — an applicant might improve multiple things at once.
    NOTE: this checks a fixed, limited search space (pairs of 4 features x 3 percentages),
    not every possible combination of features/percentages.
    """
    original = applicant_row.copy()
    original_prob = model.predict_proba(original.values.reshape(1, -1))[0][1]

    available_features = [f for f in MODIFIABLE_NUMERIC if f in feature_names]
    suggestions = []

    for feat1, feat2 in combinations(available_features, 2):
        idx1 = list(feature_names).index(feat1)
        idx2 = list(feature_names).index(feat2)
        val1 = original.iloc[idx1]
        val2 = original.iloc[idx2]

        for pct in PERCENT_STEPS:
            modified = original.copy()
            modified.iloc[idx1] = val1 * (1 - pct)
            modified.iloc[idx2] = val2 * (1 - pct)

            new_prob = model.predict_proba(modified.values.reshape(1, -1))[0][1]
            improvement = original_prob - new_prob

            if improvement > 0.02:
                suggestions.append({
                    "features": f"{feat1} + {feat2}",
                    "change": f"both -{int(pct*100)}%",
                    "original_risk_prob": round(original_prob, 4),
                    "new_risk_prob": round(new_prob, 4),
                    "risk_reduction": round(improvement, 4),
                    "flips_to_approved": new_prob < 0.5,
                })

    suggestions_df = pd.DataFrame(suggestions).sort_values("risk_reduction", ascending=False)
    return original_prob, suggestions_df


def run_counterfactual_batch(model, X_test, y_test, n_examples=5, run_label="baseline"):
    os.makedirs("results/counterfactual", exist_ok=True)

    preds = model.predict(X_test)
    # Find applicants predicted as "bad credit" (rejected) to generate recourse for
    rejected_idx = np.where(preds == 1)[0][:n_examples]

    all_results = []
    for idx in rejected_idx:
        applicant = X_test.iloc[idx]
        original_prob, single_df = generate_counterfactual(model, applicant, X_test.columns)
        _, combined_df = generate_combined_counterfactual(model, applicant, X_test.columns)

        print(f"\n--- Applicant {idx} ---")
        print(f"Original risk probability: {original_prob:.4f}")

        print("\nSingle-feature changes:")
        if len(single_df) > 0:
            print(single_df.head(3).to_string(index=False))
        else:
            print("No meaningful single-feature improvement found.")

        print("\nCombined (2-feature) changes:")
        if len(combined_df) > 0:
            print(combined_df.head(3).to_string(index=False))
            n_flips = combined_df["flips_to_approved"].sum()
            print(f"({n_flips} combined-change option(s) flip this applicant to 'approved')")
        else:
            print("No meaningful combined-feature improvement found.")

        # Pick the best overall suggestion (single or combined, whichever reduces risk more)
        best_single = single_df.iloc[0] if len(single_df) > 0 else None
        best_combined = combined_df.iloc[0] if len(combined_df) > 0 else None

        if best_combined is not None and (best_single is None or best_combined["risk_reduction"] > best_single["risk_reduction"]):
            best = best_combined
            best_type = "combined"
        elif best_single is not None:
            best = best_single
            best_type = "single"
        else:
            best = None
            best_type = None

        all_results.append({
            "applicant_idx": idx,
            "original_risk_prob": original_prob,
            "best_type": best_type,
            "best_features": best["features"] if best is not None else None,
            "best_change": best["change"] if best is not None else None,
            "best_new_risk_prob": best["new_risk_prob"] if best is not None else None,
            "risk_reduction": best["risk_reduction"] if best is not None else None,
            "flips_to_approved": best["flips_to_approved"] if best is not None else False,
        })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"results/counterfactual/recourse_{run_label}.csv", index=False)
    print(f"\nSaved: results/counterfactual/recourse_{run_label}.csv")
    return results_df


if __name__ == "__main__":
    df = load_german_credit()
    df_readable, df_encoded = preprocess(df)
    X_train, X_test, y_train, y_test, age_train, age_test = get_train_test_split(df_encoded)

    model = load_model("CatBoost")
    run_counterfactual_batch(model, X_test, y_test, n_examples=5, run_label="baseline")