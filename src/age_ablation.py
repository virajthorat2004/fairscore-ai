"""
Age ablation: what does removing AGE from the deployable models cost in
accuracy, and what does it buy in fairness between age groups?

Each deployable model is trained twice on the same split -- with age and
age-blind (the live configuration) -- and compared on the same test rows.
Threshold metrics are taken at the research model's flag rate, as in
train_deployable.py. German Credit also reports 5-fold CV AUC because its
200-row test set is too small to trust on its own.

Usage (from the project root):  python src\\age_ablation.py
Writes results/age_ablation.csv
"""
import pickle

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from fairlearn.metrics import demographic_parity_ratio, equalized_odds_difference
from sklearn.metrics import roc_auc_score

import train_deployable as td
from deployable_features import select_deployable


def run(name):
    (X_train, X_test, y_train, y_test, _, age_test), enc = td.get_split(name)
    with open(td.MODEL_PATHS[name], "rb") as f:
        research = pickle.load(f)
    flag_rate = float(np.mean(research.predict(X_test[list(research.feature_names_)])))
    groups = np.asarray(age_test)
    rows = []
    for variant, with_age in (("with age", True), ("age-blind (live)", False)):
        A = select_deployable(name, X_train, include_age=with_age)
        B = select_deployable(name, X_test, include_age=with_age)
        model = CatBoostClassifier(**td.catboost_params(name, deployable=True))
        model.fit(A, y_train)
        p = model.predict_proba(B)[:, 1]
        pred = (p >= np.quantile(p, 1 - flag_rate)).astype(int)
        row = {"dataset": name, "variant": variant, "auc": roc_auc_score(y_test, p),
               "dp_ratio": demographic_parity_ratio(y_test, pred, sensitive_features=groups),
               "eo_diff": equalized_odds_difference(y_test, pred, sensitive_features=groups)}
        for g in sorted(set(groups)):
            row[f"flag_rate_{g}"] = float(pred[groups == g].mean())
        if name == "german_credit":
            X_all, y_all = enc.drop(columns=["target", "age_group"]), enc["target"]
            row["cv_auc"], row["cv_auc_sd"] = td.cv_auc(
                name, select_deployable(name, X_all, include_age=with_age), y_all, deployable=True)
        rows.append(row)
        print(f"  {name:14s} {variant:17s} AUC {row['auc']:.4f}  DP ratio {row['dp_ratio']:.3f}  "
              f"EO diff {row['eo_diff']:.3f}")
    return rows


if __name__ == "__main__":
    out = []
    for ds in ("german_credit", "gmsc", "homecredit"):
        out += run(ds)
    td.RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(out).to_csv(td.RESULTS_DIR / "age_ablation.csv", index=False)
    print("Saved results/age_ablation.csv")
