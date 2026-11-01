import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import os

from data_loader import load_german_credit
from preprocessing import preprocess, get_train_test_split

def load_model(model_name="CatBoost"):
    with open(f"models/{model_name}.pkl", "rb") as f:
        return pickle.load(f)

def run_shap_analysis(model, X_train, X_test):
    os.makedirs("results/shap_plots", exist_ok=True)

    # TreeExplainer is fast and exact for tree-based models like CatBoost
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # 1. Global explanation: which features matter most overall, across all applicants
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig("results/shap_plots/global_summary.png", dpi=150)
    plt.close()
    print("Saved: results/shap_plots/global_summary.png")

    # 2. Local explanation: why did the model decide THIS specific applicant's outcome?
    # We'll explain the first test applicant as an example
    sample_idx = 0
    plt.figure()
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[sample_idx],
            base_values=explainer.expected_value,
            data=X_test.iloc[sample_idx],
            feature_names=X_test.columns.tolist()
        ),
        show=False
    )
    plt.tight_layout()
    plt.savefig("results/shap_plots/local_applicant_0.png", dpi=150)
    plt.close()
    print("Saved: results/shap_plots/local_applicant_0.png")

    return shap_values, explainer


if __name__ == "__main__":
    df = load_german_credit()
    df_readable, df_encoded = preprocess(df)
    X_train, X_test, y_train, y_test, age_train, age_test = get_train_test_split(df_encoded)

    model = load_model("CatBoost")
    shap_values, explainer = run_shap_analysis(model, X_train, X_test)

    print("\nSHAP analysis complete.")
    print("Shape of shap_values:", shap_values.shape)