"""
Indian evidence for FAIRScore: what a lender can learn about NEW-TO-CREDIT applicants, on 233,154
real Indian vehicle loans (L&T Financial Services, 2018; see ltfs_data.py).

Questions
  1. Do new-to-credit borrowers (no bureau history) default more than bureau-scored ones?
  2. Without a bureau record AND without a bank statement, how well can the application alone rank
     their risk? (age-blind, as FAIRScore's models are; with age and with the lender's own dealer /
     branch identifiers only as ablations)
  3. How big is the information gap: the bureau score's power on scored borrowers vs the
     application's power on new-to-credit ones?
  4. Is an age-blind model fair across age groups and employment types on Indian outcomes?
  5. Which application facts carry the risk? (the loan-to-value ratio: the borrower's own
     contribution to the price)

    python src\\india_ntc_study.py          (about 5 minutes)

Writes results/india_ntc_{overview,models,fairness,ltv,features}.csv
"""
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

import ltfs_data as L
from config import PROJECT_ROOT

warnings.filterwarnings("ignore")
RESULTS = PROJECT_ROOT / "results"

APPLICATION = ["ltv", "disbursed_amount", "asset_cost", "employment", "aadhaar", "pan", "voter_id",
               "driving_licence", "passport"]
BUREAU = ["PERFORM_CNS.SCORE", "PRI.NO.OF.ACCTS", "PRI.ACTIVE.ACCTS", "PRI.OVERDUE.ACCTS", "PRI.CURRENT.BALANCE",
          "PRI.SANCTIONED.AMOUNT", "PRIMARY.INSTAL.AMT", "NEW.ACCTS.IN.LAST.SIX.MONTHS",
          "DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS", "credit_history_years", "NO.OF_INQUIRIES"]
CB = {"depth": 6, "iterations": 400, "learning_rate": 0.08, "l2_leaf_reg": 5, "random_seed": 42, "verbose": 0,
      "thread_count": -1}


def catboost(cols):
    cats = [c for c in cols if c in ("employment", "State_ID") or c in L.LENDER_SPECIFIC]
    return CatBoostClassifier(**CB, cat_features=cats)


def logistic(cols):
    cats = [c for c in cols if c == "employment"]
    nums = [c for c in cols if c not in cats]
    pre = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), cats), ("n", StandardScaler(), nums)])
    return make_pipeline(pre, LogisticRegression(max_iter=2000, C=1.0))


def prep(d, cols):
    X = d[cols].copy()
    for c in cols:
        if c in ("employment", "State_ID") or c in L.LENDER_SPECIFIC:
            X[c] = X[c].astype(str)
    return X


def cv(make, d, cols, folds=5, seed=0):
    X, y = prep(d, cols), d["bad"].values
    p, aucs = np.zeros(len(y)), []
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        m = make(cols)
        m.fit(X.iloc[tr], y[tr])
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p[te]))
    return p, float(np.mean(aucs)), float(np.std(aucs))


def out_of_time(make, d, cols):
    """Train on August-September 2018, test on October 2018."""
    tr, te = d[d["month"] < "2018-10"], d[d["month"] == "2018-10"]
    m = make(cols)
    m.fit(prep(tr, cols), tr["bad"].values)
    return float(roc_auc_score(te["bad"], m.predict_proba(prep(te, cols))[:, 1])), m


def fairness(d, p, by, label):
    """Flag the riskiest share equal to the default rate; who is flagged, by group."""
    flag = p >= np.quantile(p, 1 - d["bad"].mean())
    rows = []
    for g, idx in d.groupby(by).groups.items():
        s = d.loc[idx]
        f = flag[d.index.get_indexer(idx)]
        bad, good = s["bad"].values == 1, s["bad"].values == 0
        rows.append({"model": label, "attribute": by, "group": g, "loans": len(s), "default_rate": s["bad"].mean(),
                     "flagged_share": f.mean(), "defaulters_caught": f[bad].mean(),
                     "repayers_flagged": f[good].mean(), "repayers_cleared": 1 - f[good].mean()})
    t = pd.DataFrame(rows)
    t["clear_ratio_vs_best"] = t["repayers_cleared"] / t["repayers_cleared"].max()
    return t


def main():
    d = L.load()
    ntc = d[d["new_to_credit"]].reset_index(drop=True)
    scored = d[d["bureau_scored"]].reset_index(drop=True)
    print(f"{len(d):,} loans, {d['bad'].mean():.1%} defaulted on the first EMI; new-to-credit {len(ntc):,}")

    # --- 1. overview -------------------------------------------------------------------------
    ov = [{"segment": "All borrowers", "loans": len(d), "default_rate": d["bad"].mean()},
          {"segment": "New to credit (no bureau history)", "loans": len(ntc), "default_rate": ntc["bad"].mean()},
          {"segment": "Bureau-scored", "loans": len(scored), "default_rate": scored["bad"].mean()},
          {"segment": "Bureau record but not scored", "loans": int((~d["new_to_credit"] & ~d["bureau_scored"]).sum()),
           "default_rate": d.loc[~d["new_to_credit"] & ~d["bureau_scored"], "bad"].mean()}]
    for b, s in scored.groupby("bureau_band"):
        ov.append({"segment": f"Bureau band {b} ({s['PERFORM_CNS.SCORE.DESCRIPTION'].iloc[0].split('-', 1)[1]})",
                   "loans": len(s), "default_rate": s["bad"].mean()})
    for e, s in ntc.groupby("employment"):
        ov.append({"segment": f"New to credit, {e}", "loans": len(s), "default_rate": s["bad"].mean()})
    pd.DataFrame(ov).round(4).to_csv(RESULTS / "india_ntc_overview.csv", index=False)

    # --- 2/3. models ----------------------------------------------------------------------------
    models = []

    def add(pop, name, cols, make, data, oot=True, note=""):
        p, m, s = cv(make, data, cols)
        o = out_of_time(make, data, cols)[0] if oot else None
        models.append({"population": pop, "model": name, "inputs": len(cols), "cv_auc": m, "cv_auc_sd": s,
                       "oot_auc": o, "note": note})
        print(f"  {pop:22s} {name:55s} CV {m:.3f} ± {s:.3f}" + (f"  OOT {o:.3f}" if o else ""))
        return p

    p_blind = add("New to credit", "Application only, age-blind (CatBoost)", APPLICATION, catboost, ntc,
                  note="what FAIRScore's form-based models could know")
    add("New to credit", "Application only, age-blind (logistic regression)", APPLICATION, logistic, ntc)
    p_age = add("New to credit", "Application + age (ablation)", APPLICATION + ["age"], catboost, ntc,
                note="age is not allowed as a model input in FAIRScore")
    add("New to credit", "Application + state (ablation)", APPLICATION + ["State_ID"], catboost, ntc,
        note="location can stand in for protected attributes")
    add("New to credit", "Application + lender's dealer / branch IDs (ablation)",
        APPLICATION + L.LENDER_SPECIFIC, catboost, ntc, note="this lender only; not transferable")
    add("New to credit", "Loan-to-value ratio alone", ["ltv"], catboost, ntc, oot=False)
    add("Bureau-scored", "Bureau score alone (CNS score)", ["PERFORM_CNS.SCORE"], catboost, scored, oot=False)
    add("Bureau-scored", "Application only, age-blind", APPLICATION, catboost, scored, oot=False)
    add("Bureau-scored", "Application + bureau record", APPLICATION + BUREAU, catboost, scored, oot=False,
        note="the information a new-to-credit applicant does not have")
    mt = pd.DataFrame(models)
    mt.round(4).to_csv(RESULTS / "india_ntc_models.csv", index=False)

    # --- 4. fairness on new-to-credit borrowers ------------------------------------------------------
    ntc["age_group"] = ntc["age"].map(L.age_group)
    fair = pd.concat([fairness(ntc, p_blind, "age_group", "age-blind"), fairness(ntc, p_age, "age_group", "with age"),
                      fairness(ntc, p_blind, "employment", "age-blind")])
    fair.round(4).to_csv(RESULTS / "india_ntc_fairness.csv", index=False)

    # --- 5. loan-to-value ---------------------------------------------------------------------------
    bins = [0, 60, 70, 75, 80, 85, 90, 100]
    rows = []
    for pop, s in (("New to credit", ntc), ("Bureau-scored", scored), ("All", d)):
        for b, g in s.groupby(pd.cut(s["ltv"], bins), observed=True):
            rows.append({"population": pop, "ltv_band": f"{int(b.left)}-{int(b.right)}%", "loans": len(g),
                         "default_rate": g["bad"].mean()})
    for emp, s in ntc.groupby("employment"):
        for lo, hi in ((0, 80), (80, 85), (85, 100)):
            g = s[(s["ltv"] > lo) & (s["ltv"] <= hi)]
            rows.append({"population": f"New to credit, {emp}", "ltv_band": f"{lo}-{hi}%", "loans": len(g),
                         "default_rate": g["bad"].mean()})
    pd.DataFrame(rows).round(4).to_csv(RESULTS / "india_ntc_ltv.csv", index=False)

    # --- what the age-blind application model relies on -------------------------------------------------
    _, m = out_of_time(catboost, ntc, APPLICATION)
    imp = pd.DataFrame({"input": APPLICATION, "importance": m.get_feature_importance()}).sort_values(
        "importance", ascending=False)
    imp.round(2).to_csv(RESULTS / "india_ntc_features.csv", index=False)
    print(imp.to_string(index=False))
    return mt


if __name__ == "__main__":
    main()
