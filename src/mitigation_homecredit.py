import pandas as pd
import pickle
import os
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from fairlearn.metrics import (
    demographic_parity_difference, demographic_parity_ratio, equalized_odds_difference
)

from data_loader_homecredit import load_home_credit
from preprocessing_homecredit import preprocess_home_credit, get_train_test_split_homecredit


def load_model(model_name="CatBoost"):
    with open(f"models/homecredit/{model_name}.pkl", "rb") as f:
        return pickle.load(f)


def evaluate(y_true, preds, probs, age_group, label):
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
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

    baseline_model = load_model("CatBoost")
    baseline_preds = baseline_model.predict(X_test)
    baseline_probs = baseline_model.predict_proba(X_test)[:, 1]
    before = evaluate(y_test, baseline_preds, baseline_probs, age_test, "Before Mitigation (Home Credit)")

    # Home Credit has 246,008 training rows. ExponentiatedGradient retrains the
    # base estimator many times internally (dozens of passes), so running it on
    # the FULL training set is impractical (could realistically take many hours
    # to over a day). We use a larger, more representative sample than GMSC's
    # 20,000 (here: 40,000) since accuracy of the mitigation estimate matters —
    # but this is still a sample, and worth stating explicitly in the paper as
    # a computational necessity, not a shortcut taken carelessly.
    sample_size = 40000
    X_train_sample = X_train.sample(n=sample_size, random_state=42)
    y_train_sample = y_train.loc[X_train_sample.index]
    age_train_sample = age_train.loc[X_train_sample.index]

    print(f"\nTraining mitigated model on a {sample_size}-row sample of the training set. "
          f"This is a large, iterative computation — expect a long runtime "
          f"(likely 30 minutes to a few hours depending on your machine). Let it run.")

    base_estimator = CatBoostClassifier(verbose=0, random_state=42, auto_class_weights="Balanced")
    mitigator = ExponentiatedGradient(
        estimator=base_estimator,
        constraints=DemographicParity(),
    )
    mitigator.fit(X_train_sample, y_train_sample, sensitive_features=age_train_sample)

    mitigated_preds = mitigator.predict(X_test)
    after = evaluate(y_test, mitigated_preds, mitigated_preds, age_test, "After Mitigation (Home Credit)")

    comparison_df = pd.DataFrame([before, after])
    comparison_df.to_csv("results/mitigation/before_after_comparison_homecredit.csv", index=False)
    print("\nSaved: results/mitigation/before_after_comparison_homecredit.csv")

    print("\n=== SUMMARY: What changed? ===")
    print(f"Accuracy:  {before['accuracy']:.4f} -> {after['accuracy']:.4f}  "
          f"({after['accuracy']-before['accuracy']:+.4f})")
    print(f"Recall:    {before['recall']:.4f} -> {after['recall']:.4f}  "
          f"({after['recall']-before['recall']:+.4f})")
    print(f"DP Ratio:  {before['demographic_parity_ratio']:.4f} -> {after['demographic_parity_ratio']:.4f}  "
          f"({after['demographic_parity_ratio']-before['demographic_parity_ratio']:+.4f})  "
          f"[closer to 1.0 = better]")
    print(f"EO Diff:   {before['equalized_odds_difference']:.4f} -> {after['equalized_odds_difference']:.4f}  "
          f"({after['equalized_odds_difference']-before['equalized_odds_difference']:+.4f})  "
          f"[closer to 0 = better]")
    print("\nNOTE: roc_auc 'After' uses 0/1 predictions as a proxy (ExponentiatedGradient "
          "doesn't output calibrated probabilities) — do NOT directly compare it to 'Before' AUC.")

    return comparison_df


if __name__ == "__main__":
    df = load_home_credit()
    df_encoded = preprocess_home_credit(df)
    X_train, X_test, y_train, y_test, age_train, age_test = get_train_test_split_homecredit(df_encoded)

    run_mitigation(X_train, X_test, y_train, y_test, age_train, age_test)