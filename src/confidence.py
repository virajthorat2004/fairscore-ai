import pandas as pd
import pickle
import os
import numpy as np
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

from data_loader import load_german_credit
from preprocessing import preprocess, get_train_test_split


def load_model(model_name="CatBoost"):
    with open(f"models/{model_name}.pkl", "rb") as f:
        return pickle.load(f)


def assign_confidence(prob, low_thresh=0.4, high_thresh=0.6):
    """
    Probability near 0 or 1 = high confidence.
    Probability near 0.5 = low confidence (model is unsure).
    """
    if prob <= low_thresh or prob >= high_thresh:
        # distance from 0.5, scaled to a 0-1 confidence score
        confidence_score = abs(prob - 0.5) * 2
        if confidence_score >= 0.6:
            label = "High"
        else:
            label = "Medium"
    else:
        confidence_score = abs(prob - 0.5) * 2
        label = "Low"
    return confidence_score, label


def run_confidence_analysis(model, X_test, y_test, run_label="baseline"):
    os.makedirs("results/confidence", exist_ok=True)

    probs = model.predict_proba(X_test)[:, 1]  # probability of "bad credit"

    confidence_scores = []
    confidence_labels = []
    for p in probs:
        score, label = assign_confidence(p)
        confidence_scores.append(score)
        confidence_labels.append(label)

    results_df = pd.DataFrame({
        "predicted_probability": probs,
        "confidence_score": confidence_scores,
        "confidence_label": confidence_labels,
        "manual_review_flag": [l == "Low" for l in confidence_labels],
        "actual_target": y_test.values,
    })

    print("Confidence label distribution:\n", results_df["confidence_label"].value_counts())
    print(f"\nApplicants flagged for manual review (Low confidence): "
          f"{results_df['manual_review_flag'].sum()} / {len(results_df)}")

    # Calibration check: does "70% probability" actually mean ~70% are correct?
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="CatBoost")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of actual positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/confidence/calibration_curve_{run_label}.png", dpi=150)
    plt.close()
    print(f"\nSaved: results/confidence/calibration_curve_{run_label}.png")

    results_df.to_csv(f"results/confidence/confidence_scores_{run_label}.csv", index=False)
    print(f"Saved: results/confidence/confidence_scores_{run_label}.csv")

    return results_df


if __name__ == "__main__":
    df = load_german_credit()
    df_readable, df_encoded = preprocess(df)
    X_train, X_test, y_train, y_test, age_train, age_test = get_train_test_split(df_encoded)

    model = load_model("CatBoost")
    run_confidence_analysis(model, X_test, y_test, run_label="baseline")