import pandas as pd
import pickle
import os
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from data_loader_homecredit import load_home_credit
from preprocessing_homecredit import preprocess_home_credit, get_train_test_split_homecredit


def get_models(scale_pos_weight):
    return {
        # Logistic Regression: needs scaled features + balanced class weights,
        # otherwise it collapses to predicting the majority class for everything.
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            random_state=42, n_jobs=-1, class_weight="balanced"
        ),
        "XGBoost": XGBClassifier(
            eval_metric="logloss", random_state=42, n_jobs=-1,
            scale_pos_weight=scale_pos_weight  # XGBoost's own imbalance handling
        ),
        "LightGBM": LGBMClassifier(
            random_state=42, verbose=-1, n_jobs=-1, class_weight="balanced"
        ),
        "CatBoost": CatBoostClassifier(
            verbose=0, random_state=42, auto_class_weights="Balanced"
        ),
    }


def train_and_evaluate(X_train, X_test, y_train, y_test, run_label="homecredit_baseline_balanced"):
    results = []
    os.makedirs("models/homecredit", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Compute class imbalance ratio for XGBoost's scale_pos_weight
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos
    print(f"Class imbalance ratio (neg/pos): {scale_pos_weight:.2f}")

    # Scale features — needed for Logistic Regression, harmless for tree models
    # if we only apply it to the LR copy (tree models are scale-invariant, but
    # we scale a shared copy for simplicity and speed).
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    for name, model in get_models(scale_pos_weight).items():
        print(f"Training {name}...")

        # Logistic Regression uses scaled features; tree-based models use raw features
        if name == "LogisticRegression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            probs = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]

        results.append({
            "model": name,
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probs),
        })

        with open(f"models/homecredit/{name}.pkl", "wb") as f:
            pickle.dump(model, f)

    # Save the scaler too — we'll need it if we ever re-run Logistic Regression later
    with open("models/homecredit/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/{run_label}_{timestamp}.csv"
    results_df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")

    return results_df


if __name__ == "__main__":
    df = load_home_credit()
    df_encoded = preprocess_home_credit(df)
    X_train, X_test, y_train, y_test, age_train, age_test = get_train_test_split_homecredit(df_encoded)

    results_df = train_and_evaluate(X_train, X_test, y_train, y_test, run_label="homecredit_baseline_balanced")
    print("\nModel comparison (sorted by ROC-AUC):\n")
    print(results_df.to_string(index=False))