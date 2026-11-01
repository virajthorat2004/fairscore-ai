import pandas as pd
import pickle
import os
from fairlearn.metrics import (
    MetricFrame, demographic_parity_difference, demographic_parity_ratio,
    equalized_odds_difference, true_positive_rate, false_positive_rate
)
from sklearn.metrics import accuracy_score

from data_loader import load_german_credit
from preprocessing import preprocess, get_train_test_split


def load_model(model_name="CatBoost"):
    with open(f"models/{model_name}.pkl", "rb") as f:
        return pickle.load(f)


def run_fairness_analysis(model, X_test, y_test, age_test, run_label="baseline"):
    os.makedirs("results", exist_ok=True)

    preds = model.predict(X_test)

    # Group sizes — report this every time, since small subgroups make metrics unreliable
    group_counts = age_test.value_counts()
    print("Test set group sizes:\n", group_counts, "\n")

    MIN_GROUP_SIZE = 20
    reliable = group_counts.min() >= MIN_GROUP_SIZE
    if not reliable:
        print(f"⚠ WARNING: smallest group has only {group_counts.min()} samples "
              f"(threshold={MIN_GROUP_SIZE}). Fairness metrics below may be noisy/unreliable.\n")

    # Demographic Parity: are approval rates similar across age groups?
    dp_diff = demographic_parity_difference(y_test, preds, sensitive_features=age_test)
    dp_ratio = demographic_parity_ratio(y_test, preds, sensitive_features=age_test)

    # Equalized Odds: are TPR and FPR similar across age groups?
    eo_diff = equalized_odds_difference(y_test, preds, sensitive_features=age_test)

    # Per-group breakdown: accuracy, TPR (equal opportunity), FPR
    metric_frame = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "selection_rate": lambda y_true, y_pred: (y_pred == 1).mean(),  # predicted "bad credit" rate
            "true_positive_rate": true_positive_rate,
            "false_positive_rate": false_positive_rate,
        },
        y_true=y_test,
        y_pred=preds,
        sensitive_features=age_test
    )

    print("Per-group metrics:\n", metric_frame.by_group, "\n")
    print(f"Demographic Parity Difference: {dp_diff:.4f}  (0 = perfectly equal)")
    print(f"Demographic Parity Ratio:      {dp_ratio:.4f}  (1 = perfectly equal)")
    print(f"Equalized Odds Difference:     {eo_diff:.4f}  (0 = perfectly equal)")

    # Save results
    summary = pd.DataFrame([{
        "run_label": run_label,
        "smallest_group_size": group_counts.min(),
        "reliable": reliable,
        "demographic_parity_difference": dp_diff,
        "demographic_parity_ratio": dp_ratio,
        "equalized_odds_difference": eo_diff,
    }])
    filename = f"results/fairness_{run_label}.csv"
    summary.to_csv(filename, index=False)
    metric_frame.by_group.to_csv(f"results/fairness_by_group_{run_label}.csv")
    print(f"\nSaved: {filename}")

    return metric_frame, dp_diff, dp_ratio, eo_diff


if __name__ == "__main__":
    df = load_german_credit()
    df_readable, df_encoded = preprocess(df)
    X_train, X_test, y_train, y_test, age_train, age_test = get_train_test_split(df_encoded)

    model = load_model("CatBoost")
    run_fairness_analysis(model, X_test, y_test, age_test, run_label="baseline")