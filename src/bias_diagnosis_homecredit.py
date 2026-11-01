import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import os

from data_loader_homecredit import load_home_credit
from preprocessing_homecredit import preprocess_home_credit, get_train_test_split_homecredit


def load_model(model_name="CatBoost"):
    with open(f"models/homecredit/{model_name}.pkl", "rb") as f:
        return pickle.load(f)


def run_bias_diagnosis(model, X_test, age_test, run_label="homecredit_baseline", top_n=15):
    os.makedirs("results/bias_diagnosis", exist_ok=True)

    # Larger sample since time isn't a constraint — more stable divergence estimates
    sample_idx = X_test.sample(n=8000, random_state=42).index
    X_sample = X_test.loc[sample_idx]
    age_sample = age_test.loc[sample_idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    shap_df = pd.DataFrame(shap_values, columns=X_sample.columns, index=X_sample.index)
    shap_df["age_group"] = age_sample.values

    feature_cols = X_sample.columns.tolist()
    group_means = shap_df.groupby("age_group")[feature_cols].apply(
        lambda g: g.abs().mean()
    ).T

    group_means["divergence"] = (group_means["under_25"] - group_means["25_and_over"]).abs()
    group_means = group_means.sort_values("divergence", ascending=False)

    print(f"\nTop {top_n} features by attribution divergence between age groups:\n")
    print(group_means.head(top_n).round(4))

    # Specifically call out where the EXT_SOURCE features rank, since we have a
    # hypothesis about them driving the larger equalized-odds gap on this dataset
    ext_source_ranks = {}
    for i, feat in enumerate(group_means.index):
        if "EXT_SOURCE" in feat:
            ext_source_ranks[feat] = i + 1
    print("\nEXT_SOURCE feature ranks by divergence:", ext_source_ranks)

    filename = f"results/bias_diagnosis/divergence_{run_label}.csv"
    group_means.to_csv(filename)
    print(f"\nSaved: {filename}")

    top_features = group_means.head(top_n)
    fig, ax = plt.subplots(figsize=(9, 7))
    x = np.arange(len(top_features))
    width = 0.35
    ax.barh(x - width/2, top_features["25_and_over"], height=width, label="25_and_over", color="#2166ac")
    ax.barh(x + width/2, top_features["under_25"], height=width, label="under_25", color="#b2182b")
    ax.set_yticks(x)
    ax.set_yticklabels(top_features.index)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Attribution Divergence (Home Credit): under_25 vs 25_and_over")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"results/bias_diagnosis/divergence_plot_{run_label}.png", dpi=150)
    plt.close()
    print(f"Saved: results/bias_diagnosis/divergence_plot_{run_label}.png")

    return group_means


if __name__ == "__main__":
    df = load_home_credit()
    df_encoded = preprocess_home_credit(df)
    X_train, X_test, y_train, y_test, age_train, age_test = get_train_test_split_homecredit(df_encoded)

    model = load_model("CatBoost")
    run_bias_diagnosis(model, X_test, age_test, run_label="homecredit_baseline")