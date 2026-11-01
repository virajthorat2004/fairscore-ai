import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import os

from data_loader_africa import load_africa_synth
from preprocessing_africa import preprocess_africa, get_train_test_split_africa


def load_model():
    with open("models/africa/CatBoost.pkl", "rb") as f:
        return pickle.load(f)


def run_shap_analysis(model, X_test, run_label="africa"):
    os.makedirs("results/shap_plots", exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(f"results/shap_plots/global_summary_{run_label}.png", dpi=150)
    plt.close()
    print(f"Saved: results/shap_plots/global_summary_{run_label}.png")

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
    plt.savefig(f"results/shap_plots/local_applicant_0_{run_label}.png", dpi=150)
    plt.close()
    print(f"Saved: results/shap_plots/local_applicant_0_{run_label}.png")

    return shap_values, explainer


if __name__ == "__main__":
    df = load_africa_synth()
    df_encoded = preprocess_africa(df)
    X_train, X_test, y_train, y_test, age_train, age_test = get_train_test_split_africa(df_encoded)

    model = load_model()
    # Full test set used (only 21 rows — no need to sample)
    shap_values, explainer = run_shap_analysis(model, X_test, run_label="africa")
    print("\nSHAP analysis complete.")