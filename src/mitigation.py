import pandas as pd
import pickle
import os
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from fairlearn.metrics import (
    MetricFrame, demographic_parity_difference, demographic_parity_ratio,
    equalized_odds_difference, true_positive_rate, false_positive_rate
)

from data_loader import load_german_credit
from preprocessing import preprocess, get_train_test_split


def load_model(model_name="CatBoost"):
    with open(f"models/{model_name}.pkl", "rb") as f:
        return pickle.load(f)


def evaluate(y_true, preds, probs, age_group, label):
    """Compute both performance and fairness metrics for a given prediction set."""
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds)
    rec = recall_score(y_true, preds)
    f1 = f1_score(y_true, preds)
    auc = roc_auc_score(y_true, probs)

    dp_diff = demographic_parity_difference(y_true, preds, sensitive_features=age_group)
    dp_ratio = demographic_parity_ratio(y_true, preds, sensitive_features=age_group)
    eo_diff = equalized_odds_difference(y_true, preds, sensitive_features=age_group)

    result = {
        "stage": label, "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "roc_auc": auc,
        "demographic_parity_difference": dp_diff,
        "demographic_parity_ratio": dp_ratio,
        "equalized_odds_difference": eo_diff,
    }
    print(f"\n--- {label} ---")
    for k, v in result.items():
        if k != "stage":
            print(f"{k}: {v:.4f}")
    return result


def run_mitigation(X_train, X_test, y_train, y_test, age_train, age_test):
    os.makedirs("results/mitigation", exist_ok=True)

    # ---- BEFORE: baseline CatBoost (already trained) ----
    baseline_model = load_model("CatBoost")
    baseline_preds = baseline_model.predict(X_test)
    baseline_probs = baseline_model.predict_proba(X_test)[:, 1]
    before = evaluate(y_test, baseline_preds, baseline_probs, age_test, "Before Mitigation")

    # ---- AFTER: Fairlearn's ExponentiatedGradient with Demographic Parity constraint ----
    # This trains a NEW model that explicitly optimizes for both accuracy and fairness together
    base_estimator = CatBoostClassifier(verbose=0, random_state=42)
    mitigator = ExponentiatedGradient(
        estimator=base_estimator,
        constraints=DemographicParity(),
    )
    mitigator.fit(X_train, y_train, sensitive_features=age_train)

    mitigated_preds = mitigator.predict(X_test)
    # ExponentiatedGradient doesn't directly give predict_proba, so we use predict as a 0/1 proxy for AUC
    after = evaluate(y_test, mitigated_preds, mitigated_preds, age_test, "After Mitigation")

    # ---- Save comparison ----
    comparison_df = pd.DataFrame([before, after])
    comparison_df.to_csv("results/mitigation/before_after_comparison.csv", index=False)
    print("\nSaved: results/mitigation/before_after_comparison.csv")

    print("\n=== SUMMARY: What changed? ===")
    print(f"Accuracy:  {before['accuracy']:.4f} -> {after['accuracy']:.4f}  "
          f"({after['accuracy']-before['accuracy']:+.4f})")
    print(f"DP Ratio:  {before['demographic_parity_ratio']:.4f} -> {after['demographic_parity_ratio']:.4f}  "
          f"({after['demographic_parity_ratio']-before['demographic_parity_ratio']:+.4f})  "
          f"[closer to 1.0 = better]")

    return comparison_df


if __name__ == "__main__":
    df = load_german_credit()
    df_readable, df_encoded = preprocess(df)
    X_train, X_test, y_train, y_test, age_train, age_test = get_train_test_split(df_encoded)

    run_mitigation(X_train, X_test, y_train, y_test, age_train, age_test)