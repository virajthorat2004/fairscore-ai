import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import os

from data_loader_homecredit import load_home_credit
from preprocessing_homecredit import preprocess_home_credit, get_train_test_split_homecredit


def load_model(model_name="CatBoost"):
    with open(f"models/homecredit/{model_name}.pkl", "rb") as f:
        return pickle.load(f)


def run_shap_analysis(model, X_test, run_label="homecredit"):
    os.makedirs("results/shap_plots", exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # 1. Global explanation
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(f"results/shap_plots/global_summary_{run_label}.png", dpi=150)
    plt.close()
    print(f"Saved: results/shap_plots/global_summary_{run_label}.png")

    # 2. Local explanation for one example applicant
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
    df = load_home_credit()
    df_encoded = preprocess_home_credit(df)
    X_train, X_test, y_train, y_test, age_train, age_test = get_train_test_split_homecredit(df_encoded)

    model = load_model("CatBoost")

    # Since time isn't a constraint, we use a larger, more statistically robust
    # sample than the 2,000-3,000 rows used for GMSC — 8,000 rows still runs in
    # reasonable time but gives a much more stable, defensible global explanation
    # on a dataset this wide (174 features).
    X_test_sample = X_test.sample(n=8000, random_state=42)

    shap_values, explainer = run_shap_analysis(model, X_test_sample, run_label="homecredit")
    print("\nSHAP analysis complete.")