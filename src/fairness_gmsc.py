import pandas as pd
import pickle
import os
from fairlearn.metrics import (
    MetricFrame, demographic_parity_difference, demographic_parity_ratio,
    equalized_odds_difference, true_positive_rate, false_positive_rate
)
from sklearn.metrics import accuracy_score

from data_loader_gmsc import load_gmsc
from preprocessing_gmsc import preprocess_gmsc, get_train_test_split_gmsc


def load_model(model_name="CatBoost"):
    with open(f"models/gmsc/{model_name}.pkl", "rb") as f:
        return pickle.load(f)


def run_fairness_analysis(model, X_test, y_test, age_test, run_label="gmsc_baseline"):
    os.makedirs("results", exist_ok=True)

    preds = model.predict(X_test)

    group_counts = age_test.value_counts()
    print("Test set group sizes:\n", group_counts, "\n")

    MIN_GROUP_SIZE = 20
    reliable = group_counts.min() >= MIN_GROUP_SIZE
    if not reliable:
        print(f"⚠ WARNING: smallest group has only {group_counts.min()} samples — "
              f"metrics below may be noisy.\n")
    else:
        print(f"✓ Smallest group has {group_counts.min()} samples — large enough for reliable metrics.\n")

    dp_diff = demographic_parity_difference(y_test, preds, sensitive_features=age_test)
    dp_ratio = demographic_parity_ratio(y_test, preds, sensitive_features=age_test)
    eo_diff = equalized_odds_difference(y_test, preds, sensitive_features=age_test)

    metric_frame = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "selection_rate": lambda y_true, y_pred: (y_pred == 1).mean(),
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
    df = load_gmsc()
    df_processed = preprocess_gmsc(df)
    X_train, X_test, y_train, y_test, age_train, age_test = get_train_test_split_gmsc(df_processed)

    model = load_model("CatBoost")
    run_fairness_analysis(model, X_test, y_test, age_test, run_label="gmsc_baseline")