"""
Train and evaluate the FAIRScore 2.0 AI decision engine (src/ai_engine.py) on 679 real loans whose
pre-loan bank statement was rebuilt from the Berka bank data (results/berka_backtest_loans.csv, made by
src/train_cashflow.py).

Every number is NESTED: the whole training recipe (model, calibration, conformal scores, approval
threshold, unusual-applicant detector) is re-run inside each outer fold, and only then applied to the
held-out loans. Nothing about a held-out loan is used to decide that loan.

    python src\\train_ai_engine.py          (about 5 minutes)

Writes
    models/ai_engine.pkl                       the engine used by the dashboard / API
    results/ai_engine_metrics.csv              AUC, Brier, calibration error, with 95% intervals
    results/ai_engine_decisions.csv            version 1 (rules) vs version 2 (AI): approve / refer / decline
    results/ai_engine_conformal.csv            guarantee check: coverage and automation at each alpha
    results/ai_engine_tradeoff.csv             approval rate vs default rate (AI vs scorecard)
    results/ai_engine_fairness.csv             outcomes by gender and age group, version 1 vs 2
    results/ai_engine_profit.csv               lender profit per 100 applicants, version 1 vs 2
"""
import pickle
import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import ai_engine as ae
import cashflow_model as cm
from config import PROJECT_ROOT
from policy_config import POLICY

warnings.filterwarnings("ignore")
RESULTS = PROJECT_ROOT / "results"
REPEATS = 3
SETTINGS = POLICY["ai"]


def load_data():
    d = pd.read_csv(RESULTS / "berka_backtest_loans.csv")
    for k in ae.EXPERT_FACTORS:
        d[f"sc_{k}"] = d[f"sc_{k}"].fillna(0.5)
    d["regular_income_share"] = d["regular_income_share"].clip(upper=1.0)      # as in live use
    X = d[ae.FEATURES].astype(float).values
    return d, X, d["bad"].values.astype(int)


def make_model():
    return CatBoostClassifier(**ae.CATBOOST_PARAMS, monotone_constraints=ae.MONOTONE)


def oof_raw(X, y, seeds=(0, 1)):
    p = np.zeros(len(y))
    for s in seeds:
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            p[te] += make_model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1] / len(seeds)
    return p


def approve_threshold(pd_, y, good_set, target):
    """Largest PD at which the default rate among auto-approvable loans stays within the risk appetite."""
    idx = np.where(good_set)[0]
    if not len(idx):
        return 0.0
    o = idx[np.argsort(pd_[idx])]
    cum = np.cumsum(y[o]) / np.arange(1, len(o) + 1)
    ok = np.where(cum <= target)[0]
    return float(pd_[o][ok.max()]) if len(ok) else float(pd_[o][0])


def fit_bundle(X, y, settings=SETTINGS):
    """The full training recipe. Returns the engine bundle."""
    raw = oof_raw(X, y)
    cal = LogisticRegression(C=100.0).fit(ae.logit(raw)[:, None], y)
    pd_oof = cal.predict_proba(ae.logit(raw)[:, None])[:, 1]
    s0, s1 = np.sort(pd_oof[y == 0]), np.sort(1 - pd_oof[y == 1])
    bundle = {"model": make_model().fit(X, y), "calibrator": cal, "conformal": {"s0": s0, "s1": s1}}
    p0 = np.array([(np.sum(s0 >= p) + 1) / (len(s0) + 1) for p in pd_oof])
    p1 = np.array([(np.sum(s1 >= 1 - p) + 1) / (len(s1) + 1) for p in pd_oof])
    good = (p0 > settings["alpha"]) & ~(p1 > settings["alpha"])
    iso = IsolationForest(n_estimators=300, random_state=0).fit(X)
    bundle["ood"] = {"model": iso, "train_scores": np.sort(iso.score_samples(X))}
    bundle["thresholds"] = {"approve_max_pd": approve_threshold(pd_oof, y, good, settings["target_default_rate"])}
    bundle["oof"] = {"pd": pd_oof, "y": y}
    bundle["cal_X"] = np.asarray(X, float)         # inputs of the calibration cases: drift reference and the
    return bundle                                  # density-ratio weights of shift-aware conformal (learning_loop)


def apply_bundle(bundle, X, settings=SETTINGS):
    pd_ = ae.predict_pd(bundle, X)
    out = []
    for i, p in enumerate(pd_):
        p0, p1 = ae.conformal(bundle, p)
        in0, in1 = p0 > settings["alpha"], p1 > settings["alpha"]
        typ = ae.typicality(bundle, X[i:i + 1])
        out.append({"pd": p, "p0": p0, "p1": p1,
                    "outcome_set": "repays" if in0 and not in1 else "defaults" if in1 and not in0 else "uncertain",
                    "typicality": typ, "in_experience": typ >= settings["min_typicality"]})
    return out


CHECK_COLS = None


def statuses(row):
    return {c[6:]: row[c] for c in CHECK_COLS if row[c] not in ("n/a", None) and isinstance(row[c], str)}


def ece(p, y, bins=10):
    q = pd.qcut(p, bins, duplicates="drop")
    g = pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q", observed=True)
    return float((g.size() / len(p) * (g["p"].mean() - g["y"].mean()).abs()).sum())


def boot_ci(y, p, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(fn(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def age_group(a):
    return "under 30" if a < 30 else "30-44" if a < 45 else "45 and over"


def profit(verdicts, y, term, rate, settings):
    """Lender's profit per 100 applicants, per Rs 1 lent (indicative): a repaid loan earns the interest
    margin over its life on the average outstanding balance (about 55% of the amount); a default loses
    LGD of the amount."""
    margin = rate - settings["cost_of_funds"] - settings["operating_cost"]
    gain = margin * 0.55 * term / 12.0
    ap = verdicts == "approve"
    return float(100 * (np.sum(gain[ap & (y == 0)]) - settings["lgd"] * np.sum(ap & (y == 1))) / len(y))


def main():
    global CHECK_COLS
    t0 = time.time()
    d, X, y = load_data()
    CHECK_COLS = [c for c in d if c.startswith("check_")]
    F12 = d[cm.FEATURES].astype(float).values
    mon12 = [cm.MONOTONE.get(f, 0) for f in cm.FEATURES]
    print(f"{len(y)} loans, {y.sum()} defaults ({y.mean():.1%}); {len(ae.FEATURES)} inputs")

    rows, metrics_rows = [], []
    for rep in range(REPEATS):
        for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=100 + rep).split(X, y)):
            b = fit_bundle(X[tr], y[tr])
            res = apply_bundle(b, X[te])
            cb12 = CatBoostClassifier(**cm.CATBOOST_PARAMS, monotone_constraints=mon12).fit(F12[tr], y[tr])
            lr12 = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)).fit(F12[tr], y[tr])
            p_cb12, p_lr12 = cb12.predict_proba(F12[te])[:, 1], lr12.predict_proba(F12[te])[:, 1]
            for j, i in enumerate(te):
                r = res[j]
                verdict, by, _ = ae.decide(statuses(d.iloc[i]), r, SETTINGS, b["thresholds"]["approve_max_pd"])
                if by == "defer":
                    verdict = d["verdict"].iloc[i]           # the version-1 expert policy path
                rows.append({"rep": rep, "i": i, "y": y[i], "pd": r["pd"], "p0": r["p0"], "p1": r["p1"],
                             "set": r["outcome_set"], "typicality": r["typicality"], "v2": verdict, "by": by,
                             "p_cb12": p_cb12[j], "p_lr12": p_lr12[j], "approve_max_pd": b["thresholds"]["approve_max_pd"]})
        print(f"  repeat {rep + 1}/{REPEATS} done ({time.time() - t0:.0f}s)")
    o = pd.DataFrame(rows)
    o["v1"] = d["verdict"].values[o["i"]]
    # "AI only": the same engine with NO lending rules at all (no age limit, no affordability cap, no bounce or
    # fraud stops) -- to measure what the rules add on top of the AI
    good, bad = (o["p0"] > SETTINGS["alpha"]) & ~(o["p1"] > SETTINGS["alpha"]), (o["p1"] > SETTINGS["alpha"]) & ~(o["p0"] > SETTINGS["alpha"])
    o["ai_only"] = np.where(good & (o["pd"] <= o["approve_max_pd"]), "approve",
                            np.where(bad & (o["pd"] >= SETTINGS["decline_min_pd"]), "decline", "refer"))
    o["fairscore"] = d["fairscore"].values[o["i"]]
    o.to_csv(RESULTS / "ai_engine_oof.csv", index=False)

    # ---- 1. discrimination and calibration (pooled out-of-fold, averaged over repeats)
    for name, col, sign in (("FAIRScore 2.0 AI engine (knowledge-guided monotone CatBoost, calibrated)", "pd", 1),
                            ("Cash-flow model v1 (CatBoost, 12 ratios)", "p_cb12", 1),
                            ("Logistic regression (12 ratios)", "p_lr12", 1),
                            ("FAIRScore expert scorecard (no training)", "fairscore", -1)):
        aucs = [roc_auc_score(g["y"], sign * g[col]) for _, g in o.groupby("rep")]
        g0 = o[o.rep == 0]
        lo, hi = boot_ci(g0["y"].values, sign * g0[col].values, roc_auc_score)
        row = {"model": name, "auc": np.mean(aucs), "auc_ci_low": lo, "auc_ci_high": hi, "gini": 2 * np.mean(aucs) - 1}
        if col != "fairscore":
            row.update({"brier": np.mean([brier_score_loss(g["y"], g[col]) for _, g in o.groupby("rep")]),
                        "ece": np.mean([ece(g[col].values, g["y"].values) for _, g in o.groupby("rep")])})
        metrics_rows.append(row)
    metrics_rows.append({"model": "Always predict the average default rate", "brier": y.mean() * (1 - y.mean()),
                         "auc": 0.5})
    met = pd.DataFrame(metrics_rows)
    met.to_csv(RESULTS / "ai_engine_metrics.csv", index=False)
    print("\n" + met.round(3).to_string(index=False))

    # ---- 2. decisions: version 1 (rules) vs version 2 (AI)
    dec = []
    for ver in ("v1", "ai_only", "v2"):
        for v in ("approve", "refer", "decline"):
            s = o[o[ver] == v]
            dec.append({"version": ver, "verdict": v, "share": len(s) / len(o), "default_rate": s["y"].mean() if len(s) else np.nan})
    stops = d[[c for c in CHECK_COLS]].eq("fail")
    behaviour = stops[["check_affordability", "check_conduct", "check_stability", "check_lenders"]].any(axis=1).values[o["i"]]
    age_rule = (d["check_eligibility"] == "fail").values[o["i"]]
    s = o[behaviour]
    dec.append({"version": "rules check", "verdict": "loans stopped by behaviour rules", "share": len(s) / len(o),
                "default_rate": s["y"].mean()})
    dec.append({"version": "rules check", "verdict": "...of which AI alone would approve", "share":
                (s["ai_only"] == "approve").mean(), "default_rate": s["y"][s["ai_only"] == "approve"].mean()})
    x = o[(o["ai_only"] == "approve") & (o["v2"] != "approve")]
    dec.append({"version": "rules check", "verdict": "extra approvals of AI alone", "share": len(x) / len(o),
                "default_rate": x["y"].mean()})
    dec.append({"version": "rules check", "verdict": "...of which blocked by the legal age rule", "share":
                age_rule[x.index].mean() if len(x) else 0, "default_rate": np.nan})
    for by, s in o.groupby("by"):
        dec.append({"version": "v2 decided by", "verdict": by, "share": len(s) / len(o), "default_rate": s["y"].mean()})
    dec = pd.DataFrame(dec)
    dec.to_csv(RESULTS / "ai_engine_decisions.csv", index=False)
    print("\n" + dec.round(3).to_string(index=False))

    # ---- 3. conformal guarantee check at several significance levels
    conf = []
    for a in (0.05, 0.10, 0.15, 0.20):
        in0, in1 = o["p0"] > a, o["p1"] > a
        good, bad = in0 & ~in1, in1 & ~in0
        conf.append({"alpha": a, "coverage_repayers": in0[o.y == 0].mean(), "coverage_defaulters": in1[o.y == 1].mean(),
                     "auto_approve_share": good.mean(), "default_rate_auto_approved": o.y[good].mean(),
                     "defaulters_auto_approved": (good & (o.y == 1)).sum() / (o.y == 1).sum(),
                     "auto_decline_share": bad.mean(), "uncertain_share": (~(good | bad)).mean()})
    conf = pd.DataFrame(conf)
    conf.to_csv(RESULTS / "ai_engine_conformal.csv", index=False)
    print("\n" + conf.round(3).to_string(index=False))

    # ---- 4. approval-rate vs default-rate trade-off
    trade = []
    for name, col, sign in (("AI engine", "pd", 1), ("FAIRScore scorecard", "fairscore", -1)):
        for share in np.arange(0.1, 1.0001, 0.05):
            vals = []
            for _, g in o.groupby("rep"):
                k = int(round(share * len(g)))
                top = g.iloc[np.argsort(sign * g[col].values)[:k]]
                vals.append(top["y"].mean())
            trade.append({"ranking": name, "approval_rate": round(share, 2), "default_rate": float(np.mean(vals))})
    trade.append({"ranking": "Version 1 rules (actual)", "approval_rate": (o.v1 == "approve").mean(),
                  "default_rate": o.y[o.v1 == "approve"].mean()})
    trade.append({"ranking": "Version 2 AI decision (actual)", "approval_rate": (o.v2 == "approve").mean(),
                  "default_rate": o.y[o.v2 == "approve"].mean()})
    pd.DataFrame(trade).to_csv(RESULTS / "ai_engine_tradeoff.csv", index=False)

    # ---- 5. fairness by gender and age group
    o["gender"] = np.where(d["female"].values[o["i"]] == 1, "female", "male")
    o["age_group"] = [age_group(a) for a in d["age"].values[o["i"]]]
    o["age_rule"] = d["check_eligibility"].values[o["i"]] == "fail"
    fair = []
    for attr in ("gender", "age_group"):
        for g, s in o.groupby(attr):
            rep_ = s[s.y == 0]
            for ver in ("v1", "v2"):
                fair.append({"attribute": attr, "group": g, "version": ver, "loans": len(s) // REPEATS,
                             "approved_share": (s[ver] == "approve").mean(),
                             "approved_among_repayers": (rep_[ver] == "approve").mean(),
                             "approved_among_repayers_excl_age_rule": (rep_[~rep_.age_rule][ver] == "approve").mean(),
                             "default_rate_if_approved": s.y[s[ver] == "approve"].mean()})
    fair = pd.DataFrame(fair)
    fair.to_csv(RESULTS / "ai_engine_fairness.csv", index=False)
    print("\n" + fair.round(3).to_string(index=False))

    # ---- 6. lender profit (indicative)
    term = d["term_months"].values[o["i"]].astype(float)
    rate = POLICY["annual_interest_rate"]
    prof = [{"version": ver, "profit_per_100_applicants_per_rupee": profit(o[ver].values, o["y"].values, term, rate,
                                                                            SETTINGS),
             "approved_share": (o[ver] == "approve").mean()} for ver in ("v1", "v2")]
    pd.DataFrame(prof).to_csv(RESULTS / "ai_engine_profit.csv", index=False)
    print("\n" + pd.DataFrame(prof).round(4).to_string(index=False))

    # ---- final engine on all loans
    b = fit_bundle(X, y)
    m0 = met.iloc[0]
    v1a, v2a = dec[(dec.version == "v1") & (dec.verdict == "approve")].iloc[0], dec[(dec.version == "v2") & (dec.verdict == "approve")].iloc[0]
    b.update({"version": f"ai-{pd.Timestamp.now():%Y%m%d}", "features": ae.FEATURES, "trained_on": "Berka, 679 loans",
              "metrics": {"auc": float(m0["auc"]), "auc_ci": [float(m0["auc_ci_low"]), float(m0["auc_ci_high"])],
                          "brier": float(m0["brier"]), "ece": float(m0["ece"]),
                          "v1_approve": float(v1a["share"]), "v1_default": float(v1a["default_rate"]),
                          "v2_approve": float(v2a["share"]), "v2_default": float(v2a["default_rate"]),
                          "conformal": conf.to_dict("records")}})
    with open(ae.MODEL_PATH, "wb") as fh:
        pickle.dump(b, fh)
    print(f"\nEngine saved to {ae.MODEL_PATH} (approve if PD <= {b['thresholds']['approve_max_pd']:.3f}) "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
