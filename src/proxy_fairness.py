"""
FAIRScore 3.0 proxy-fairness audit: does the AI treat people differently by WHERE they live or how well-off their
area is, even though neither is an input?

In India the dangerous proxies are the pin code (which can stand in for community, caste or religion through
residential patterns) and area income. A model that never sees them can still learn them from spending
patterns. Berka records each borrower's district (77 districts, 8 regions) with its average salary and
unemployment, so the same question can be tested on real loans:

  1 leakage   can the engine's 24 inputs predict the region, or a poor vs rich district? (cross-validated AUC;
              0.5 = the inputs carry no trace of it). The same test for gender and age group.
  2 outcomes  for each region and each district-income tercile: approval rate among borrowers who REPAID (equal
              opportunity), approval rate overall, and default rate among approved loans, from the nested
              out-of-fold decisions (results/ai_engine_oof.csv), with bootstrap 95% intervals, for the AI decision
              (version 2) and the rules (version 1). The four-fifths ratio is reported against the best-treated group.

    python src/proxy_fairness.py         (about 1 minute)

Writes results/proxy_leakage.csv and results/proxy_fairness.csv
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import train_ai_engine as tr
from config import PROJECT_ROOT

RESULTS = PROJECT_ROOT / "results"
BERKA = PROJECT_ROOT / "data" / "raw" / "berka"


def area(d):
    L = pd.read_csv(BERKA / "loan.csv", sep=";")[["loan_id", "account_id"]]
    A = pd.read_csv(BERKA / "account.csv", sep=";")[["account_id", "district_id"]]
    D = pd.read_csv(BERKA / "district.csv", sep=";")[["A1", "A3", "A11", "A13"]]
    D["A13"] = pd.to_numeric(D["A13"], errors="coerce")
    m = d[["loan_id"]].merge(L, on="loan_id", how="left").merge(A, on="account_id", how="left").merge(
        D, left_on="district_id", right_on="A1", how="left")
    out = pd.DataFrame({"region": m["A3"].values, "district": m["district_id"].values,
                        "avg_salary": m["A11"].values, "unemployment": m["A13"].values})
    terc = pd.qcut(out["avg_salary"].rank(method="first"), 3, labels=["poorer districts", "middle districts",
                                                                      "richer districts"])
    out["district_income"] = terc.astype(str)
    out["moravia"] = out["region"].str.contains("Moravia").astype(int)
    return out


def leak_auc(X, t, seed=0, probe="linear"):
    """Cross-validated AUC of a probe that tries to recover attribute t from the engine's inputs."""
    if probe == "linear":
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000))
    else:                                   # a nonlinear probe catches proxies that a linear one misses
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05, random_state=seed)
    Z = np.sign(X) * np.log1p(np.abs(X))
    p = cross_val_predict(clf, Z, t, cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    return float(roc_auc_score(t, p))


def leak_test(X, t, probe, repeats=3, perms=100, seed=0):
    """Observed CV AUC (mean of `repeats` splits) and a permutation p-value: the same probe on shuffled labels
    gives the AUC expected with NO information (cross-validated AUC under no signal sits slightly below 0.5,
    so 0.5 is not the right reference). Two-sided: a probe doing reliably WORSE than chance also means signal."""
    obs = float(np.mean([leak_auc(X, t, s, probe) for s in range(repeats)]))
    rng = np.random.default_rng(seed)
    null = np.array([leak_auc(X, rng.permutation(t), int(k), probe) for k in range(perms)])
    p = (np.sum(null >= obs) + 1) / (perms + 1)     # one-sided: can the inputs recover the attribute BETTER than chance?
    return obs, float(null.mean()), float(np.percentile(null, 97.5)), float(p)


def boot(values, n=1000, seed=0):
    values = np.asarray(values, float)
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    d, X, y = tr.load_data()
    ar = area(d)
    # ---- 1. leakage
    targets = {"region: Moravia vs Bohemia": ar["moravia"].values,
               "district income: poorer third vs the rest": (ar["district_income"] == "poorer districts").astype(int).values,
               "gender (female)": d["female"].values.astype(int),
               "age under 30": (d["age"].values < 30).astype(int)}
    leak = []
    for name, t in targets.items():
        row = {"attribute": name, "share": float(t.mean())}
        for probe in ("linear", "boosted"):
            obs, null_mean, null_hi, p = leak_test(X, t, probe)
            print(f"    {name} / {probe}: AUC {obs:.3f} vs no-information {null_mean:.3f}, one-sided p {p:.3f}", flush=True)
            row.update({f"{probe}_auc": obs, f"{probe}_null_auc": null_mean, f"{probe}_null_97_5": null_hi, f"{probe}_p": p})
        row["leakage_auc"] = max(row["linear_auc"], row["boosted_auc"])
        leak.append(row)
    # Holm correction over all attribute x probe tests (8): several tests at once find "something" by chance
    tests = sorted(((r[f"{pr}_p"], i, pr) for i, r in enumerate(leak) for pr in ("linear", "boosted")))
    m, running = len(tests), 0.0
    for k, (p, i, pr) in enumerate(tests):
        running = max(running, min(1.0, (m - k) * p))
        leak[i][f"{pr}_p_holm"] = running
    for r in leak:
        worst = min(r["linear_p_holm"], r["boosted_p_holm"])
        raw = min(r["linear_p"], r["boosted_p"])
        r["reading"] = ("no detectable trace" if raw >= 0.05 else
                        "no detectable trace after correction (raw p < 0.05: keep monitoring)" if worst >= 0.05 else
                        "weak trace: audit outcomes" if r["leakage_auc"] < 0.65 else "clear trace: audit outcomes")
        print(f"  leakage {r['attribute']}: linear {r['linear_auc']:.3f} (p {r['linear_p']:.3f}), boosted "
              f"{r['boosted_auc']:.3f} (p {r['boosted_p']:.3f}, Holm {r['boosted_p_holm']:.2f}) -> {r['reading']}", flush=True)
    leak = pd.DataFrame(leak)
    leak.to_csv(RESULTS / "proxy_leakage.csv", index=False)

    # ---- 2. outcomes by area
    o = pd.read_csv(RESULTS / "ai_engine_oof.csv")
    for c in ("region", "district_income"):
        o[c] = ar[c].values[o["i"]]
    rows = []
    for attr in ("district_income", "region"):
        for ver in ("v1", "v2"):
            best = None
            grp = []
            for g, s in o.groupby(attr):
                rep_ = s[s.y == 0]
                ap_rep = (rep_[ver] == "approve").values.astype(float)
                # bootstrap LOANS (each loan's approval averaged over the cross-validation repeats)
                per_loan = rep_.assign(ap=rep_[ver] == "approve").groupby("i")["ap"].mean().values
                lo, hi = boot(per_loan)
                grp.append({"attribute": attr, "group": g, "version": ver, "loans": len(s) // o["rep"].nunique(),
                            "default_rate": s["y"].mean(), "approved_share": (s[ver] == "approve").mean(),
                            "approved_among_repayers": ap_rep.mean(), "ci_low": lo, "ci_high": hi,
                            "default_rate_if_approved": s["y"][s[ver] == "approve"].mean()})
            best = max(r["approved_among_repayers"] for r in grp)
            for r in grp:
                r["ratio_to_best"] = r["approved_among_repayers"] / best if best else np.nan
                r["four_fifths_ok"] = bool(r["ratio_to_best"] >= 0.8)
            rows += grp
    fair = pd.DataFrame(rows)
    fair.to_csv(RESULTS / "proxy_fairness.csv", index=False)
    pd.set_option("display.width", 220)
    print(leak.round(3).to_string(index=False))
    print(fair.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
