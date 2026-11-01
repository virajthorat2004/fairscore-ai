import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import os

from data_loader_gmsc import load_gmsc
from preprocessing_gmsc import preprocess_gmsc, get_train_test_split_gmsc


def load_model(model_name="CatBoost"):
    with open(f"models/gmsc/{model_name}.pkl", "rb") as f:
        return pickle.load(f)


def run_shap_analysis(model, X_test, run_label="gmsc"):
    os.makedirs("results/shap_plots", exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # 1. Global explanation
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
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
    df = load_gmsc()
    df_processed = preprocess_gmsc(df)
    X_train, X_test, y_train, y_test, age_train, age_test = get_train_test_split_gmsc(df_processed)

    model = load_model("CatBoost")

    # SHAP on 120k+ rows can be slow — use a sample of the test set (2000 rows) for the global plot.
    # This is standard practice and doesn't meaningfully change the interpretation.
    X_test_sample = X_test.sample(n=2000, random_state=42)

    shap_values, explainer = run_shap_analysis(model, X_test_sample, run_label="gmsc")
    print("\nSHAP analysis complete.")