"""
Train and validate the FAIRScore cash-flow model on REAL loan outcomes, and
backtest the whole decision system (see berka_data.py, berka_backtest.py,
cashflow_model.py).

    python src\\train_cashflow.py            (about 3 minutes)
    python src\\train_cashflow.py --reuse    (reuse results/berka_backtest_loans.csv)

Writes
    models/deployable/cashflow/CatBoost.pkl (+ reference, background, score reference, SHAP weights)
    results/cashflow_model_metrics.csv     cross-validated AUC of each candidate
    results/berka_backtest_summary.csv     default rate by recommendation / score band / check
    results/cashflow_fairness.csv          outcomes by age group and gender
"""
import argparse
import pickle

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import berka_backtest
import cashflow_model as cm
from config import PROJECT_ROOT
from train_deployable import DEPLOY_DIR, decile_table, mean_abs_shap, save_reference, update_pickle_dict

RESULTS = PROJECT_ROOT / "results"
NAME = "cashflow"


def age_group(a):
    return "under 30" if a < 30 else "30-44" if a < 45 else "45-59" if a < 60 else "60 and over"


def cv_auc(make, X, y, repeats=5):
    aucs = []
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=42).split(X, y):
        m = make()
        m.fit(X.iloc[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X.iloc[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def oof(make, X, y):
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=7).split(X, y):
        m = make()
        m.fit(X.iloc[tr], y[tr])
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
    return p


def group_table(d, flag, by):
    rows = []
    for g, s in d.groupby(by):
        bad, good = s["bad"] == 1, s["bad"] == 0
        rows.append({"attribute": by, "group": g, "loans": len(s), "default_rate": s["bad"].mean(),
                     "flagged_share": s[flag].mean(),
                     "flagged_among_defaulters": s.loc[bad, flag].mean() if bad.any() else np.nan,
                     "flagged_among_repayers": s.loc[good, flag].mean() if good.any() else np.nan,
                     "approved_share": (s["verdict_with_cf"] == "approve").mean(),
                     # equal opportunity in lending: of the people who DID repay, how many were approved
                     "approved_among_repayers": (s.loc[good, "verdict_with_cf"] == "approve").mean(),
                     # the same, leaving out the legal / product age rule (minimum age, age at maturity)
                     "approved_among_repayers_excl_age_rule": (
                         s.loc[good & (s["check_eligibility"] != "fail"), "verdict_with_cf"] == "approve").mean(),
                     "age_rule_declines": (s["check_eligibility"] == "fail").mean(),
                     "default_rate_if_approved": s.loc[s["verdict_with_cf"] == "approve", "bad"].mean()})
    t = pd.DataFrame(rows)
    big = t[t["loans"] >= 30]
    t["dp_ratio_vs_best"] = t["approved_share"] / big["approved_share"].max() if len(big) else np.nan
    return t


def main(reuse=False):
    path = RESULTS / "berka_backtest_loans.csv"
    d = pd.read_csv(path) if reuse and path.exists() else berka_backtest.build()
    RESULTS.mkdir(exist_ok=True)
    d.to_csv(path, index=False)
    X, y = d[cm.FEATURES].astype(float), d["bad"].values
    print(f"{len(d)} loans with a rebuilt pre-loan statement, {int(y.sum())} defaulted ({y.mean():.1%})")

    make_cb = lambda: CatBoostClassifier(**cm.CATBOOST_PARAMS, monotone_constraints=dict(cm.MONOTONE))  # noqa: E731
    make_lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))  # noqa: E731
    metrics = []
    for label, make in (("Cash-flow model (monotone CatBoost) - deployed", make_cb),
                        ("Logistic regression, same inputs", make_lr)):
        m, s = cv_auc(make, X, y)
        metrics.append({"model": label, "cv_auc": m, "cv_auc_sd": s})
    folds = [roc_auc_score(y[te], -d["fairscore"].values[te])
             for _, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42).split(X, y)]
    metrics.append({"model": "FAIRScore expert scorecard (not trained on this data)", "cv_auc": np.mean(folds),
                    "cv_auc_sd": np.std(folds)})
    rp = d["risk_percentile"].fillna(d["risk_percentile"].median())
    metrics.append({"model": "Foreign-data models consensus (German / GMSC / Home Credit)",
                    "cv_auc": roc_auc_score(y, rp), "cv_auc_sd": np.nan})
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(RESULTS / "cashflow_model_metrics.csv", index=False)
    print(metrics.to_string(index=False))

    # --- out-of-fold predictions: percentile reference, backtest of the full system, fairness
    p = oof(make_cb, X, y)
    d["cf_prob_oof"] = p
    ranks = pd.Series(p).rank(pct=True).values * 100
    d["cf_pct_oof"] = ranks
    d["verdict_with_cf"] = np.where(d["verdict"] == "decline", "decline",
                                    np.where((d["verdict"] == "refer") | (ranks >= 80), "refer", "approve"))
    d["cf_flag"] = (ranks >= 100 * (1 - y.mean())).astype(int)       # flag as many as actually defaulted

    summary = []
    for col, label in (("verdict", "Recommendation (policy + scorecard + foreign models)"),
                       ("verdict_with_cf", "Recommendation with the cash-flow model added")):
        for v, s in d.groupby(col):
            summary.append({"table": label, "group": v, "loans": len(s), "default_rate": s["bad"].mean()})
    for b, s in d.groupby("band"):
        summary.append({"table": "FAIRScore band", "group": b, "loans": len(s), "default_rate": s["bad"].mean()})
    for c in [c for c in d if c.startswith("check_")]:
        for v, s in d.groupby(c):
            if v != "n/a":
                summary.append({"table": f"Check: {c[6:]}", "group": v, "loans": len(s), "default_rate": s["bad"].mean()})
    q = pd.qcut(d["cf_prob_oof"], 5, labels=["1 (safest)", "2", "3", "4", "5 (riskiest)"])
    for g, s in d.groupby(q, observed=True):
        summary.append({"table": "Cash-flow model quintile (out-of-fold)", "group": g, "loans": len(s),
                        "default_rate": s["bad"].mean()})
    summary = pd.DataFrame(summary)
    summary.to_csv(RESULTS / "berka_backtest_summary.csv", index=False)

    d["age_group"] = d["age"].map(age_group)
    d["gender"] = np.where(d["female"] == 1, "female", "male")
    fair = pd.concat([group_table(d, "cf_flag", "age_group"), group_table(d, "cf_flag", "gender")])
    fair.to_csv(RESULTS / "cashflow_fairness.csv", index=False)
    print("\n" + summary[summary["table"].str.startswith(("Recommendation", "Cash-flow"))].to_string(index=False))
    print("\n" + fair[["attribute", "group", "loans", "default_rate", "flagged_among_defaulters", "flagged_among_repayers",
                       "approved_among_repayers", "approved_among_repayers_excl_age_rule", "age_rule_declines",
                       "default_rate_if_approved"]].round(3).to_string(index=False))

    # --- final model on all loans
    model = make_cb()
    model.fit(X, y)
    out = DEPLOY_DIR / NAME
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "CatBoost.pkl", "wb") as f:
        pickle.dump(model, f)
    save_reference(X, DEPLOY_DIR / f"{NAME}_reference.pkl")
    X.sample(min(2000, len(X)), random_state=42).to_pickle(DEPLOY_DIR / f"{NAME}_background.pkl")
    update_pickle_dict(DEPLOY_DIR / "score_reference.pkl", NAME, {
        "test_scores_sorted": np.sort(p), "n_test": int(len(y)), "n_train": int(len(y)),
        "base_rate": float(y.mean()), "auc": float(metrics.iloc[0]["cv_auc"]), "deciles": decile_table(p, y),
        "features": cm.FEATURES, "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "source": "Berka / PKDD'99 Czech bank: 679 loans with real outcomes, statements rebuilt from transactions"})
    update_pickle_dict(DEPLOY_DIR / "shap_weights.pkl", NAME, mean_abs_shap(model, X))
    print(f"\nSaved {out / 'CatBoost.pkl'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    main(ap.parse_args().reuse)
