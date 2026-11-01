import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import os

from data_loader import load_german_credit
from preprocessing import preprocess, get_train_test_split


def load_model(model_name="CatBoost"):
    with open(f"models/{model_name}.pkl", "rb") as f:
        return pickle.load(f)


def run_bias_diagnosis(model, X_test, age_test, run_label="baseline", top_n=15):
    os.makedirs("results/bias_diagnosis", exist_ok=True)

    # Compute SHAP values for the full test set
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap_df = pd.DataFrame(shap_values, columns=X_test.columns, index=X_test.index)
    shap_df["age_group"] = age_test.values

    # Mean ABSOLUTE SHAP value per feature, per group
    # (absolute, because we care about the SIZE of each feature's influence, not its direction)
    feature_cols = X_test.columns.tolist()
    group_means = shap_df.groupby("age_group")[feature_cols].apply(
        lambda g: g.abs().mean()
    ).T  # features as rows, groups as columns

    # Divergence = how differently a feature is weighted between groups
    group_means["divergence"] = (group_means["under_25"] - group_means["25_and_over"]).abs()
    group_means = group_means.sort_values("divergence", ascending=False)

    print(f"\nTop {top_n} features by attribution divergence between age groups:\n")
    print(group_means.head(top_n).round(4))

    # Save full table
    filename = f"results/bias_diagnosis/divergence_{run_label}.csv"
    group_means.to_csv(filename)
    print(f"\nSaved: {filename}")

    # Plot top divergent features
    top_features = group_means.head(top_n)
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(top_features))
    width = 0.35
    ax.barh(x - width/2, top_features["25_and_over"], height=width, label="25_and_over", color="#2166ac")
    ax.barh(x + width/2, top_features["under_25"], height=width, label="under_25", color="#b2182b")
    ax.set_yticks(x)
    ax.set_yticklabels(top_features.index)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Attribution Divergence: under_25 vs 25_and_over")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"results/bias_diagnosis/divergence_plot_{run_label}.png", dpi=150)
    plt.close()
    print(f"Saved: results/bias_diagnosis/divergence_plot_{run_label}.png")

    return group_means


if __name__ == "__main__":
    df = load_german_credit()
    df_readable, df_encoded = preprocess(df)
    X_train, X_test, y_train, y_test, age_train, age_test = get_train_test_split(df_encoded)

    model = load_model("CatBoost")
    run_bias_diagnosis(model, X_test, age_test, run_label="baseline")