import pandas as pd
import pickle
import os
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from data_loader_africa import load_africa_synth
from preprocessing_africa import preprocess_africa, get_train_test_split_africa


def train_model(X_train, X_test, y_train, y_test):
    os.makedirs("models/africa", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Simpler CatBoost settings than our other datasets, deliberately —
    # fewer trees, shallower depth, to reduce overfitting risk on 80 rows / 55 features.
    model = CatBoostClassifier(
        verbose=0, random_state=42,
        iterations=200, depth=3, learning_rate=0.05,
        auto_class_weights="Balanced"
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probs),
    }

    print("--- Model performance (n_test=21 — indicative only, not statistically reliable) ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    with open("models/africa/CatBoost.pkl", "wb") as f:
        pickle.dump(model, f)

    pd.DataFrame([metrics]).to_csv("results/africa_baseline_metrics.csv", index=False)
    print("\nSaved: models/africa/CatBoost.pkl, results/africa_baseline_metrics.csv")

    return model, metrics


if __name__ == "__main__":
    df = load_africa_synth()
    df_encoded = preprocess_africa(df)
    X_train, X_test, y_train, y_test, age_train, age_test = get_train_test_split_africa(df_encoded)

    model, metrics = train_model(X_train, X_test, y_train, y_test)