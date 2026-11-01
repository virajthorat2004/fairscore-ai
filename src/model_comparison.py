"""
Standard data-science evaluation of the FAIRScore 2.0 AI engine on 679 real loans (Berka): every algorithm on the
same folds, significance tests, overfitting check, tuning sensitivity, ROC / calibration / confusion matrix /
precision-recall at the operating point, global feature importance, and version 2 vs version 1 with bootstrap
intervals.

    python src\\model_comparison.py      (about 10 minutes)   ->  results/eval_*.csv, results/model_zoo*.csv
"""
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import ai_engine as ae
from config import PROJECT_ROOT

warnings.filterwarnings("ignore")
RESULTS = PROJECT_ROOT / "results"
REPEATS = 3


# ---------------------------------------------------------------- DeLong test (Sun & Xu, 2014)
def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def delong(y, p1, p2):
    """Two-sided p-value that two correlated AUCs (same loans) differ."""
    pos, neg = p1[y == 1], p1[y == 0]
    m, n = (y == 1).sum(), (y == 0).sum()
    preds = np.vstack([p1, p2])
    order = np.r_[np.where(y == 1)[0], np.where(y == 0)[0]]
    preds = preds[:, order]
    tx = np.array([_midrank(r[:m]) for r in preds])
    ty = np.array([_midrank(r[m:]) for r in preds])
    tz = np.array([_midrank(r) for r in preds])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(max(var, 1e-12))
    return float(aucs[0] - aucs[1]), float(2 * stats.norm.sf(abs(z)))


def ks(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def models():
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    return {
        "Logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=4000)),
        "Decision tree (depth 4)": lambda: DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=0),
        "Random forest": lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=5, random_state=0, n_jobs=2),
        "XGBoost": lambda: XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                                         colsample_bytree=0.8, reg_lambda=5, random_state=0, verbosity=0, n_jobs=2),
        "LightGBM": lambda: LGBMClassifier(n_estimators=300, num_leaves=8, learning_rate=0.03, min_child_samples=20,
                                           subsample=0.8, subsample_freq=1, reg_lambda=5, random_state=0, verbose=-1),
        "Neural network (MLP 32-16)": lambda: make_pipeline(StandardScaler(), MLPClassifier(
            (32, 16), alpha=1e-2, max_iter=2000, early_stopping=True, random_state=0)),
        "CatBoost (no constraints)": lambda: CatBoostClassifier(**ae.CATBOOST_PARAMS),
        "CatBoost monotone (deployed)": lambda: CatBoostClassifier(**ae.CATBOOST_PARAMS,
                                                                   monotone_constraints=ae.MONOTONE),
    }


def main():
    t0 = time.time()
    d = pd.read_csv(RESULTS / "berka_backtest_loans.csv")
    for k in ae.EXPERT_FACTORS:
        d[f"sc_{k}"] = d[f"sc_{k}"].fillna(0.5)
    d["regular_income_share"] = d["regular_income_share"].clip(upper=1.0)
    X, y = d[ae.FEATURES].astype(float).values, d["bad"].values.astype(int)
    M = models()
    oof = {k: np.zeros((REPEATS, len(y))) for k in M}
    fold_auc = {k: [] for k in M}
    train_auc = {k: [] for k in M}
    for rep in range(REPEATS):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=100 + rep).split(X, y):
            for name, make in M.items():
                m = make().fit(X[tr], y[tr])
                p = m.predict_proba(X[te])[:, 1]
                oof[name][rep, te] = p
                fold_auc[name].append(roc_auc_score(y[te], p))
                train_auc[name].append(roc_auc_score(y[tr], m.predict_proba(X[tr])[:, 1]))
        print(f"  repeat {rep + 1}/{REPEATS} ({time.time() - t0:.0f}s)", flush=True)
    oof["FAIRScore expert scorecard (no training)"] = np.tile(-d["fairscore"].values.astype(float), (REPEATS, 1))
    fold_auc["FAIRScore expert scorecard (no training)"] = [roc_auc_score(y, -d["fairscore"])]
    train_auc["FAIRScore expert scorecard (no training)"] = [np.nan]

    deployed = oof["CatBoost monotone (deployed)"][0]
    rows, rng = [], np.random.default_rng(0)
    for name, P in oof.items():
        p0 = P[0]
        b = [roc_auc_score(y[i], p0[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(1000))]
        diff, pval = (0.0, 1.0) if name == "CatBoost monotone (deployed)" else delong(y, deployed, p0)
        pr = 1 / (1 + np.exp(-p0)) if name.startswith("FAIRScore") else p0
        rows.append({"model": name, "auc_mean": np.mean(fold_auc[name]), "auc_sd": np.std(fold_auc[name]),
                     "auc_ci_low": np.percentile(b, 2.5), "auc_ci_high": np.percentile(b, 97.5),
                     "gini": 2 * np.mean(fold_auc[name]) - 1, "ks": ks(y, p0), "pr_auc": average_precision_score(y, p0),
                     "brier": np.nan if name.startswith("FAIRScore") else brier_score_loss(y, pr),
                     "train_auc": np.nanmean(train_auc[name]),
                     "overfit_gap": np.nanmean(train_auc[name]) - np.mean(fold_auc[name]),
                     "deployed_minus_this": diff, "delong_p": pval})
    zoo = pd.DataFrame(rows).sort_values("auc_mean", ascending=False)
    zoo.to_csv(RESULTS / "model_zoo.csv", index=False)
    print(zoo.round(3).to_string(index=False))

    # ---- tuning sensitivity of the deployed model
    from catboost import CatBoostClassifier
    tune = []
    for depth in (2, 3, 4, 6):
        for iters, lr in ((300, 0.03), (600, 0.03), (300, 0.1)):
            a = []
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=100).split(X, y):
                p = {**ae.CATBOOST_PARAMS, "depth": depth, "iterations": iters, "learning_rate": lr}
                m = CatBoostClassifier(**p, monotone_constraints=ae.MONOTONE).fit(X[tr], y[tr])
                a.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
            tune.append({"depth": depth, "iterations": iters, "learning_rate": lr, "cv_auc": np.mean(a),
                         "deployed": depth == 3 and iters == 300 and lr == 0.03})
    pd.DataFrame(tune).to_csv(RESULTS / "model_tuning.csv", index=False)
    print(pd.DataFrame(tune).round(3).to_string(index=False))

    # ---- diagnostics of the deployed engine (calibrated, nested; from train_ai_engine.py)
    o = pd.read_csv(RESULTS / "ai_engine_oof.csv")
    g = o[o.rep == 0]
    roc = []
    for name, col, sign in (("AI engine (deployed)", "pd", 1), ("Logistic regression", "p_lr12", 1),
                            ("FAIRScore scorecard", "fairscore", -1)):
        fpr, tpr, _ = roc_curve(g["y"], sign * g[col])
        idx = np.unique(np.linspace(0, len(fpr) - 1, 60).astype(int))
        roc += [{"model": name, "fpr": fpr[i], "tpr": tpr[i]} for i in idx]
    pd.DataFrame(roc).to_csv(RESULTS / "eval_roc.csv", index=False)
    cal = o.assign(bin=pd.qcut(o["pd"], 10, labels=False, duplicates="drop")).groupby("bin").agg(
        predicted=("pd", "mean"), observed=("y", "mean"), loans=("y", "size")).reset_index()
    cal["loans"] = cal["loans"] / REPEATS
    cal.to_csv(RESULTS / "eval_calibration.csv", index=False)

    conf = []
    for name, flag in (("Version 2 decision (flag = refer or decline)", o["v2"] != "approve"),
                       ("Version 1 decision (flag = refer or decline)", o["v1"] != "approve"),
                       ("AI threshold only (flag = PD above the approval limit)", o["pd"] > o["approve_max_pd"])):
        yy, ff = o["y"].values == 1, flag.values
        tp, fp, fn, tn = (ff & yy).sum(), (ff & ~yy).sum(), (~ff & yy).sum(), (~ff & ~yy).sum()
        prec, rec = tp / (tp + fp), tp / (tp + fn)
        conf.append({"rule": name, "defaulters_flagged_TP": tp / REPEATS, "repayers_flagged_FP": fp / REPEATS,
                     "defaulters_approved_FN": fn / REPEATS, "repayers_approved_TN": tn / REPEATS,
                     "accuracy": (tp + tn) / len(ff), "precision": prec, "recall_defaulters_caught": rec,
                     "specificity_repayers_approved": tn / (tn + fp), "f1": 2 * prec * rec / (prec + rec),
                     "default_rate_among_approved": fn / (fn + tn)})
    conf = pd.DataFrame(conf)
    conf.to_csv(RESULTS / "eval_confusion.csv", index=False)
    print(conf.round(3).to_string(index=False))

    # ---- version 2 vs version 1: bootstrap over loans (paired)
    bs = []
    by_loan = {k: v.index.values for k, v in o.groupby("i")}
    loans = np.array(list(by_loan))
    for _ in range(2000):
        pick = rng.choice(loans, len(loans), replace=True)          # resample LOANS; keep all repeats of each
        s = o.loc[np.concatenate([by_loan[k] for k in pick])]
        a1, a2 = s["v1"] == "approve", s["v2"] == "approve"
        bs.append({"approval_gain": a2.mean() - a1.mean(),
                   "default_rate_change": s["y"][a2].mean() - s["y"][a1].mean(),
                   "referral_change": (s["v2"] == "refer").mean() - (s["v1"] == "refer").mean()})
    bs = pd.DataFrame(bs)
    sig = pd.DataFrame([{"measure": c, "estimate": bs[c].mean(), "ci_low": bs[c].quantile(0.025),
                         "ci_high": bs[c].quantile(0.975), "share_of_resamples_v2_better":
                         float((bs[c] > 0).mean() if c == "approval_gain" else (bs[c] < 0).mean())} for c in bs])
    sig.to_csv(RESULTS / "eval_v1_v2_bootstrap.csv", index=False)
    print(sig.round(4).to_string(index=False))

    # ---- global feature importance (mean |SHAP| of the deployed engine on all loans)
    from catboost import Pool
    b = ae.load()
    sv = b["model"].get_feature_importance(Pool(X), type="ShapValues")[:, :-1]
    imp = pd.DataFrame({"feature": [ae.LABELS[f] for f in ae.FEATURES], "mean_abs_shap": np.abs(sv).mean(0),
                        "direction_corr": [np.corrcoef(X[:, j], sv[:, j])[0, 1] if X[:, j].std() > 0 else 0
                                           for j in range(X.shape[1])]})
    group = {f: g_ for g_, fs in ae.GROUPS.items() for f in fs}
    imp["group"] = [group[f] for f in ae.FEATURES]
    imp.sort_values("mean_abs_shap", ascending=False).to_csv(RESULTS / "eval_shap_global.csv", index=False)
    print(imp.sort_values("mean_abs_shap", ascending=False).head(8).round(3).to_string(index=False))
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
