import pandas as pd
import pickle
import os
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from data_loader_gmsc import load_gmsc
from preprocessing_gmsc import preprocess_gmsc, get_train_test_split_gmsc

def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
        "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
    }

def train_and_evaluate(X_train, X_test, y_train, y_test, run_label="gmsc_baseline"):
    results = []
    os.makedirs("models/gmsc", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    for name, model in get_models().items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        results.append({
            "model": name,
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, probs),
        })

        with open(f"models/gmsc/{name}.pkl", "wb") as f:
            pickle.dump(model, f)

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/{run_label}_{timestamp}.csv"
    results_df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")

    return results_df


if __name__ == "__main__":
    df = load_gmsc()
    df_processed = preprocess_gmsc(df)
    X_train, X_test, y_train, y_test, age_train, age_test = get_train_test_split_gmsc(df_processed)

    results_df = train_and_evaluate(X_train, X_test, y_train, y_test, run_label="gmsc_baseline")
    print("\nModel comparison (sorted by ROC-AUC):\n")
    print(results_df.to_string(index=False))