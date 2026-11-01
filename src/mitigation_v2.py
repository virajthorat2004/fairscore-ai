"""
Corrected bias-mitigation experiment (replaces the numbers from mitigation.py,
mitigation_gmsc.py and mitigation_homecredit.py, which stay untouched).

Three fixes over v1:
  1. ROC-AUC after mitigation is computed from the mitigated model's real
     probabilities (ExponentiatedGradient._pmf_predict), not from 0/1
     predictions. v1's "after" AUCs (0.711 / 0.601 / 0.641) were artefacts.
  2. The mitigated model's randomized predict() is seeded, so results are
     reproducible (v1 gave a different DP ratio on every run).
  3. GMSC / Home Credit train the mitigated model on a subsample. v1 compared
     it against a baseline trained on the FULL training set, mixing the effect
     of mitigation with the effect of less data. v2 adds an unmitigated
     CatBoost trained on the SAME subsample, so the mitigation effect is
     isolated.

Usage (from the project root):
    python src\\mitigation_v2.py --dataset german_credit
    python src\\mitigation_v2.py --dataset gmsc          (slow: ~10-30 min)
    python src\\mitigation_v2.py --dataset homecredit    (slow: can take hours)
"""
import argparse
import pickle

import pandas as pd
from catboost import CatBoostClassifier
from fairlearn.metrics import (demographic_parity_difference, demographic_parity_ratio,
                               equalized_odds_difference)
from fairlearn.reductions import DemographicParity, ExponentiatedGradient
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from config import MODEL_PATHS, PROJECT_ROOT
from train_deployable import catboost_params, get_split

RANDOM_STATE = 42
SAMPLE_SIZE = {"german_credit": None, "gmsc": 20000, "homecredit": 40000}  # same as v1


def evaluate(label, y, preds, probs, age):
    row = {
        "stage": label,
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "roc_auc": roc_auc_score(y, probs),
        "demographic_parity_difference": demographic_parity_difference(y, preds, sensitive_features=age),
        "demographic_parity_ratio": demographic_parity_ratio(y, preds, sensitive_features=age),
        "equalized_odds_difference": equalized_odds_difference(y, preds, sensitive_features=age),
    }
    print(f"  {label:42s} AUC={row['roc_auc']:.4f}  recall={row['recall']:.4f}  "
          f"DP ratio={row['demographic_parity_ratio']:.4f}  EO diff={row['equalized_odds_difference']:.4f}")
    return row


def run(name):
    print(f"\n=== {name} ===")
    (X_train, X_test, y_train, y_test, age_train, age_test), _ = get_split(name)
    rows = []

    with open(MODEL_PATHS[name], "rb") as f:
        baseline = pickle.load(f)
    rows.append(evaluate("Baseline (full training set)", y_test, baseline.predict(X_test),
                         baseline.predict_proba(X_test)[:, 1], age_test))

    n = SAMPLE_SIZE[name]
    if n:
        Xs = X_train.sample(n=n, random_state=RANDOM_STATE)
        ys, ages = y_train.loc[Xs.index], age_train.loc[Xs.index]
        control = CatBoostClassifier(**catboost_params(name)).fit(Xs, ys)
        rows.append(evaluate(f"Unmitigated control ({n:,}-row sample)", y_test, control.predict(X_test),
                             control.predict_proba(X_test)[:, 1], age_test))
    else:
        Xs, ys, ages = X_train, y_train, age_train

    mitigator = ExponentiatedGradient(estimator=CatBoostClassifier(**catboost_params(name)),
                                      constraints=DemographicParity())
    mitigator.fit(Xs, ys, sensitive_features=ages)
    probs = mitigator._pmf_predict(X_test)[:, 1]
    preds = mitigator.predict(X_test, random_state=RANDOM_STATE)
    label = "Mitigated (ExpGrad + DemographicParity)" + (f", {n:,}-row sample" if n else "")
    rows.append(evaluate(label, y_test, preds, probs, age_test))

    out_dir = PROJECT_ROOT / "results" / "mitigation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"before_after_v2_{name}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  Saved {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(SAMPLE_SIZE))
    run(ap.parse_args().dataset)
