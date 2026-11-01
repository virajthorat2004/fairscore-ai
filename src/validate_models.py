"""
Independent-style validation of FAIRScore's statement-based risk components, in the
form a bank's model-validation team (and RBI's draft Model Risk Management guidance,
2026) expects: discrimination with confidence intervals, a challenger model, an
out-of-time test, calibration, stability, label sensitivity and what drives the model.

It re-uses the backtest built by train_cashflow.py (679 Berka loans whose pre-loan
statement was rebuilt and read by FAIRScore's own pipeline) and never looks at a
loan's own outcome when predicting it (out-of-fold / out-of-time predictions only).

    python src\\validate_models.py          (about 2 minutes; uses results/berka_backtest_loans.csv from
                                             train_cashflow.py, so the raw Berka files are not needed)

Writes
    results/model_validation.csv              every metric, with 95% bootstrap intervals
    results/model_validation_calibration.csv  predicted vs actual default rate by decile
    results/model_validation_features.csv     each input on its own, and input-group ablations
"""
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import berka_data
import cashflow_model as cm
from config import PROJECT_ROOT
from monitoring import psi

warnings.filterwarnings("ignore")
RESULTS = PROJECT_ROOT / "results"
SEEDS = 5
BOOT = 1000
CHAMPION = "Cash-flow model (monotone CatBoost)"
CHALLENGER = "Logistic regression (same inputs)"


def champion(cols=None):
    cols = cols or cm.FEATURES
    return CatBoostClassifier(**cm.CATBOOST_PARAMS, monotone_constraints={c: v for c, v in cm.MONOTONE.items()
                                                                          if c in cols})


def challenger(cols=None):
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))


def ks(y, p):
    f, t, _ = roc_curve(y, p)
    return float(np.max(t - f))


def oof(make, X, y, seed):
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = make(list(X.columns))
        m.fit(X.iloc[tr], y[tr])
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
    return p


def oof_avg(make, X, y):
    """Out-of-fold probabilities averaged over several fold splits (a steadier estimate)."""
    return np.mean([oof(make, X, y, s) for s in range(SEEDS)], axis=0)


def boot_ci(y, scores, stat=roc_auc_score, n=BOOT, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            vals.append(stat(y[i], *[s[i] for s in scores]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def load():
    d = pd.read_csv(RESULTS / "berka_backtest_loans.csv")
    if "loan_date" in d:
        d["loan_date"] = pd.to_datetime(d["loan_date"])
    else:                                    # backtest file from before loan dates were saved
        loan, _ = berka_data.load_tables()
        d = d.merge(loan[["loan_id", "loan_date"]], on="loan_id")
    return d.sort_values("loan_date", kind="stable").reset_index(drop=True)


def main():
    d = load()
    X, y = d[cm.FEATURES].astype(float), d["bad"].values
    rows = []

    def add(section, metric, value, lo=None, hi=None, n=None, note=""):
        rows.append({"section": section, "metric": metric, "value": round(float(value), 4),
                     "ci_low": None if lo is None else round(lo, 4), "ci_high": None if hi is None else round(hi, 4),
                     "loans": n, "note": note})

    n, bad = len(y), int(y.sum())
    add("Data", "Loans with a rebuilt pre-loan statement", n, n=n, note="Berka / PKDD'99, Czech bank 1993-98")
    add("Data", "Defaults (status B or D)", bad, n=n, note=f"{y.mean():.1%} default rate")

    # --- 1. discrimination with intervals ------------------------------------------------
    p_ch, p_lr = oof_avg(champion, X, y), oof_avg(challenger, X, y)
    sc = -d["fairscore"].values
    fm = d["risk_percentile"].fillna(d["risk_percentile"].median()).values
    for name, s in ((CHAMPION, p_ch), (CHALLENGER, p_lr), ("FAIRScore expert scorecard (not trained)", sc),
                    ("Foreign-data models (German / GMSC / Home Credit)", fm)):
        auc = roc_auc_score(y, s)
        lo, hi = boot_ci(y, [s])
        add("Discrimination (out-of-fold)", f"AUC: {name}", auc, lo, hi, n)
        add("Discrimination (out-of-fold)", f"Gini: {name}", 2 * auc - 1, 2 * lo - 1, 2 * hi - 1, n)
        add("Discrimination (out-of-fold)", f"KS: {name}", ks(y, s), *boot_ci(y, [s], ks), n)
    diff = lambda yy, a, b: roc_auc_score(yy, a) - roc_auc_score(yy, b)      # noqa: E731
    add("Champion vs challenger", "AUC gain over logistic regression", diff(y, p_ch, p_lr),
        *boot_ci(y, [p_ch, p_lr], diff), n, "paired bootstrap; an interval above 0 means the gain is real")
    add("Champion vs challenger", "AUC gain over the expert scorecard", diff(y, p_ch, sc),
        *boot_ci(y, [p_ch, sc], diff), n)

    # --- 2. out-of-time: train on the earliest 70% of loans, test on the latest 30% -------
    cut = int(0.7 * n)
    tr, te = np.arange(cut), np.arange(cut, n)
    note = (f"trained on loans granted up to {d['loan_date'].iloc[cut - 1]:%b %Y}, tested on loans granted "
            f"{d['loan_date'].iloc[cut]:%b %Y} - {d['loan_date'].iloc[-1]:%b %Y} ({int(y[te].sum())} defaults)")
    for name, make in ((CHAMPION, champion), (CHALLENGER, challenger)):
        m = make(list(X.columns))
        m.fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        add("Out-of-time", f"AUC: {name}", roc_auc_score(y[te], p), *boot_ci(y[te], [p]), len(te), note)
        if name == CHAMPION:
            p_oot, p_in = p, m.predict_proba(X.iloc[tr])[:, 1]
    add("Out-of-time", "AUC: FAIRScore expert scorecard", roc_auc_score(y[te], sc[te]), *boot_ci(y[te], [sc[te]]),
        len(te), note)
    add("Stability", "Score PSI, later loans vs earlier loans", psi(p_in, p_oot), n=n,
        note="below 0.10 stable, 0.10-0.25 moderate shift, above 0.25 major shift")
    worst = max(cm.FEATURES, key=lambda f: psi(X[f].values[tr], X[f].values[te]))
    add("Stability", f"Largest input PSI ({worst})", psi(X[worst].values[tr], X[worst].values[te]), n=n)

    # --- 3. label maturity: only loans observed for 12+ months ---------------------------------
    mature = (d["loan_date"] <= d["loan_date"].max() - pd.DateOffset(months=12)).values
    pm = oof_avg(champion, X[mature].reset_index(drop=True), y[mature])
    add("Label sensitivity", "AUC on loans observed 12+ months only", roc_auc_score(y[mature], pm),
        *boot_ci(y[mature], [pm]), int(mature.sum()),
        "loans granted in the last year of the data have had less time to default")
    fin = d["status"].isin(["A", "B"]).values
    add("Label sensitivity", "AUC on finished loans only (repaid vs not repaid)", roc_auc_score(y[fin], p_ch[fin]),
        *boot_ci(y[fin], [p_ch[fin]]), int(fin.sum()), "same out-of-fold predictions, finished loans only")

    # --- 4. calibration ------------------------------------------------------------------------
    add("Calibration", "Brier score (lower is better)", brier_score_loss(y, p_ch), n=n,
        note=f"a model that always predicts the base rate scores {brier_score_loss(y, np.full(n, y.mean())):.4f}")
    cal = pd.DataFrame({"p": p_ch, "y": y})
    cal["decile"] = pd.qcut(cal["p"].rank(method="first"), 10, labels=range(1, 11))
    cal = cal.groupby("decile", observed=True).agg(loans=("y", "size"), predicted=("p", "mean"),
                                                   actual=("y", "mean"), defaults=("y", "sum")).reset_index()
    add("Calibration", "Expected calibration error",
        float((cal["loans"] * (cal["predicted"] - cal["actual"]).abs()).sum() / n), n=n,
        note="average gap between predicted and actual default rate across risk deciles")
    cal.round(4).to_csv(RESULTS / "model_validation_calibration.csv", index=False)

    # --- 5. what drives the model ------------------------------------------------------------
    feats = []
    for f in cm.FEATURES:
        sgn = cm.MONOTONE.get(f, 1)
        feats.append({"kind": "single input", "inputs": f, "auc": roc_auc_score(y, sgn * X[f]),
                      "nonzero_share": float((X[f] != 0).mean())})
    balance = sorted(cm.NEEDS_BALANCE)
    groups = {"all 12 inputs (deployed)": cm.FEATURES, "balance inputs only": balance,
              "without balance inputs": [f for f in cm.FEATURES if f not in cm.NEEDS_BALANCE],
              "without lowest balance vs income": [f for f in cm.FEATURES if f != "min_balance_to_income"],
              "affordability inputs only": ["foir", "emi_to_income", "loan_to_annual_income", "term_months"]}
    for g, cols in groups.items():
        feats.append({"kind": "input group", "inputs": g, "auc": roc_auc_score(y, oof_avg(champion, X[cols], y)),
                      "nonzero_share": None})
    feats = pd.DataFrame(feats)
    feats["auc"] = feats["auc"].round(4)
    feats.to_csv(RESULTS / "model_validation_features.csv", index=False)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "model_validation.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.drop(columns="note").to_string(index=False))
    print("\n" + cal.round(3).to_string(index=False))
    print("\n" + feats.to_string(index=False))
    return out


if __name__ == "__main__":
    main()
