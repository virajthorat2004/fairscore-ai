"""
Fairness study of the FAIRScore 2.0 AI engine on 679 real loans (Berka), with gender and age group as the
protected attributes. Neither is ever a model input; they are used here only to MEASURE and, for the
constrained variants, to CONSTRAIN training (fairlearn exponentiated-gradient reduction).

For each variant, out-of-fold (5 folds x 2 repeats):
    AUC                       ranking quality
    EO gap                    equalized-odds difference: largest gap between groups in the rate at which
                              repayers are wrongly flagged as risky / defaulters are missed
    approval gap (repayers)   equal opportunity in lending: gap in the share of REPAYERS who would be approved
                              at the bank's approval threshold
    DP ratio                  lowest / highest approval rate between groups

    python src\\ai_fairness.py          writes results/ai_engine_fairness_tradeoff.csv
"""
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from fairlearn.metrics import demographic_parity_ratio, equalized_odds_difference
from fairlearn.reductions import EqualizedOdds, ExponentiatedGradient
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import ai_engine as ae
from config import PROJECT_ROOT

warnings.filterwarnings("ignore")
RESULTS = PROJECT_ROOT / "results"


class _CB:
    """CatBoost with the engine's settings, in the estimator form fairlearn expects (accepts sample_weight)."""
    def __init__(self):
        self.m = CatBoostClassifier(**ae.CATBOOST_PARAMS, monotone_constraints=ae.MONOTONE)

    def fit(self, X, y, sample_weight=None):
        self.m.fit(np.asarray(X), np.asarray(y), sample_weight=sample_weight)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        return (self.m.predict_proba(np.asarray(X))[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X):
        return self.m.predict_proba(np.asarray(X))

    def get_params(self, deep=True):
        return {}

    def set_params(self, **p):
        return self


def main():
    d = pd.read_csv(RESULTS / "berka_backtest_loans.csv")
    for k in ae.EXPERT_FACTORS:
        d[f"sc_{k}"] = d[f"sc_{k}"].fillna(0.5)
    X, y = d[ae.FEATURES].astype(float).values, d["bad"].values.astype(int)
    groups = {"gender": np.where(d["female"] == 1, "female", "male"),
              "age group": np.where(d["age"] < 30, "under 30", np.where(d["age"] < 45, "30-44", "45 and over"))}
    approve_share = 0.40          # compare variants at the same approval rate (about version 2's)
    rows = []
    for attr, g in groups.items():
        for eps in (None, 0.05, 0.02):
            name = "unconstrained (deployed)" if eps is None else f"equalized-odds constraint, eps={eps}"
            for rep in range(2):
                score = np.zeros(len(y))
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=200 + rep).split(X, y):
                    if eps is None:
                        score[te] = _CB().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
                    else:
                        eg = ExponentiatedGradient(_CB(), EqualizedOdds(difference_bound=eps), max_iter=8)
                        eg.fit(X[tr], y[tr], sensitive_features=g[tr])
                        score[te] = eg._pmf_predict(X[te])[:, 1]
                thr = np.quantile(score, approve_share)
                approve = (score <= thr).astype(int)
                flag = 1 - approve
                rep_ = y == 0
                appr = pd.Series(approve[rep_]).groupby(g[rep_]).mean()
                rows.append({"attribute": attr, "variant": name, "rep": rep, "auc": roc_auc_score(y, score),
                             "eo_gap": equalized_odds_difference(y, flag, sensitive_features=g),
                             "approval_gap_repayers": float(appr.max() - appr.min()),
                             "dp_ratio": demographic_parity_ratio(y, approve, sensitive_features=g),
                             "default_rate_approved": y[approve == 1].mean()})
            print(f"  {attr:10s} {name} done")
    t = pd.DataFrame(rows).groupby(["attribute", "variant"], sort=False).mean(numeric_only=True).drop(columns="rep")
    t = t.reset_index()
    t.to_csv(RESULTS / "ai_engine_fairness_tradeoff.csv", index=False)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
