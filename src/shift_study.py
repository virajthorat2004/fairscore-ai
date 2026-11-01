"""
FAIRScore 3.0 research study: what happens to the AI engine when it is used on a DIFFERENT population, and how
many local repayment outcomes are needed to fix it?

This is the engine's biggest real-world risk: it learned from Czech loans, a lender would use it in India. No
public data links Indian statements to repayments, so the study recreates the situation inside Berka, where the
truth is known:

  source  = the loans the engine is trained on
  target  = a different population: the "lender's market"

Eight scenarios
  1 no shift (control)            random split
  2 prior shift                   target default rate doubled (p(x | y) unchanged)
  3 covariate shift, thin cushion target tilted towards low balances / irregular income (gig-economy-like)
  4 covariate shift, thick cushion target tilted towards comfortable balances (hidden defaulters look good)
  5 the same, strong tilt
  6 covariate + prior shift       thin cushion and 1.5x default rate
  7 natural: out of time          trained on loans up to 1996, used on 1997-98 loans
  8 natural: other region         trained on Bohemia (incl. Prague), used in Moravia

Applications arrive as a stream; outcomes become known for the first n of them (n = 0 means day 0: applications
only). Each adaptation method is fitted on the stream and judged on a separate evaluation sample of the target.

Day 0           two label-free estimates of the new default rate: mean PD (right under covariate shift) and the
                EM prior-shift estimate (right under pure prior shift); shift detection from unlabelled stream
PD methods      none | Platt on n labels | Bayesian recalibration on n labels (prior centred on "no change";
                FAIRScore 3.0) | retrain on the n local loans | retrain on source + local
Conformal       standard (source calibration) | weighted Mondrian always | FAIRScore 3.0: weighted only when a
                covariate shift is detected (density-ratio weights from the unlabelled stream)
Decisions       approve if conformal set = {repays} and PD <= the engine's threshold

    python src/shift_study.py            (about 15 minutes)

Writes results/shift_{scenarios,day0,calibration,conformal,decisions}.csv
"""
import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

import ai_engine as ae
import learning_loop as ll
import train_ai_engine as tr
from config import PROJECT_ROOT

warnings.filterwarnings("ignore")
RESULTS = PROJECT_ROOT / "results"
BERKA = PROJECT_ROOT / "data" / "raw" / "berka"
REPEATS = 30
N_LABELS = (0, 25, 50, 100, 200, 400)
N_STREAM, N_EVAL = 400, 3000
ALPHA = tr.SETTINGS["alpha"]
TARGET_DR = tr.SETTINGS["target_default_rate"]


def regions(d):
    L = pd.read_csv(BERKA / "loan.csv", sep=";")[["loan_id", "account_id"]]
    A = pd.read_csv(BERKA / "account.csv", sep=";")[["account_id", "district_id"]]
    D = pd.read_csv(BERKA / "district.csv", sep=";")[["A1", "A3"]]
    m = d[["loan_id"]].merge(L, on="loan_id", how="left").merge(A, on="account_id", how="left").merge(
        D, left_on="district_id", right_on="A1", how="left")
    return m["A3"].fillna("unknown").values


def cushion(X):
    """Financial-cushion index (standardised): balance cover, regular income, steady income."""
    f = {n: i for i, n in enumerate(ae.FEATURES)}
    cols = [np.log1p(np.clip(X[:, f["abb_to_emi"]], 0, None)), X[:, f["sc_balance_cover"]],
            X[:, f["regular_income_share"]], -X[:, f["income_volatility"]], np.log1p(np.clip(X[:, f["min_balance_to_income"]], 0, None))]
    Z = np.column_stack(cols)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    return Z.mean(1)


SYNTHETIC = {
    "1 no shift": dict(cov=0.0, prior=1.0),
    "2 prior shift (default rate x2)": dict(cov=0.0, prior=2.0),
    "3 covariate shift: thin cushion": dict(cov=-0.6, prior=1.0),
    "4 covariate shift: thick cushion": dict(cov=+0.6, prior=1.0),
    "5 strong covariate shift: thick cushion": dict(cov=+1.2, prior=1.0),
    "6 covariate + prior shift": dict(cov=-0.6, prior=1.5),
}


def shift_weights(z, y, cov, prior):
    w = np.exp(cov * z)
    return w * np.where(y == 1, prior, 1.0)


def threshold_for_target(pd_, approvable, target):
    """Largest PD threshold such that the mean calibrated PD (= expected default rate) of the approved cases stays
    within the risk appetite -- needs no outcomes."""
    p = np.sort(pd_[approvable])
    if not len(p):
        return 0.0
    ok = np.where(np.cumsum(p) / np.arange(1, len(p) + 1) <= target)[0]
    return float(p[ok.max()]) if len(ok) else 0.0


def ece(p, y):
    return ll.ece(p, y)


def cat(n_iter=300):
    return CatBoostClassifier(**{**ae.CATBOOST_PARAMS, "iterations": n_iter}, monotone_constraints=ae.MONOTONE)


def evaluate_pd(name, n, p, y, rows, scen, rep):
    rows.append({"scenario": scen, "rep": rep, "method": name, "n_labels": n,
                 "brier": brier_score_loss(y, p), "ece": ece(p, y), "logloss": log_loss(y, np.clip(p, 1e-4, 1 - 1e-4)),
                 "auc": roc_auc_score(y, p), "mean_pd": float(np.mean(p)), "default_rate": float(np.mean(y))})


def run_one(scen, rep, b, Xs_, ys_, Xst, yst, Xev, yev, rows_cal, rows_conf, rows_dec, rows_day0, gst=None):
    pd_cal, y_cal = b["oof"]["pd"], b["oof"]["y"]
    pd_st, pd_ev = ae.predict_pd(b, Xst), ae.predict_pd(b, Xev)
    t_star = b["thresholds"]["approve_max_pd"]

    # ---- day 0: what can be known from applications alone
    pi_em, _ = ll.em_prior_shift(pd_st, float(np.mean(y_cal)))
    dens = ll.fit_density_ratio(b["cal_X"], Xst, groups_target=gst)
    rows_day0.append({"scenario": scen, "rep": rep, "true_default_rate": float(np.mean(yev)),
                      "estimate_mean_pd": float(np.mean(pd_st)), "estimate_em_prior_shift": pi_em,
                      "separability_auc": dens.separability_auc, "shift_detected": dens.shift_detected,
                      "max_psi": float(ll.drift_table(b["cal_X"], Xst, ae.FEATURES)["psi"].max())})

    # ---- conformal: standard vs weighted vs FAIRScore 3.0 (weighted only when a shift is detected)
    w_cal = dens.source_weights()
    variants = {"standard (source calibration)": ll.mondrian_pvalues(pd_cal, y_cal, pd_ev),
                "weighted Mondrian, always": ll.mondrian_pvalues(pd_cal, y_cal, pd_ev, w_cal, dens(Xev)),
                "FAIRScore 3.0 (weighted if shift detected)": ll.shift_aware_pvalues(pd_cal, y_cal, pd_ev, Xev, dens)}
    sets = {}
    for variant, (p0, p1) in variants.items():
        sets[variant] = ll.outcome_sets(p0, p1, ALPHA)
        rows_conf.append({"scenario": scen, "rep": rep, "variant": variant,
                          "defaulters_not_flagged": float(np.mean(p1[yev == 1] <= ALPHA)),
                          "repayers_not_flagged": float(np.mean(p0[yev == 0] <= ALPHA)),
                          "auto_approvable_share": float(np.mean(sets[variant] == "repays")),
                          "shift_detected": dens.shift_detected, "ess_share": ll.effective_sample_size(w_cal) / len(w_cal)})

    # ---- PD adaptation with n local outcomes
    evaluate_pd("none (source engine)", 0, pd_ev, yev, rows_cal, scen, rep)
    pds = {("none (source engine)", 0): (pd_ev, "standard (source calibration)")}
    for n in N_LABELS[1:]:
        if n > len(yst):
            continue
        yl, pl, Xl = yst[:n], pd_st[:n], Xst[:n]
        if yl.sum() < 1:
            continue
        p_pl = ll.apply_recalibration(pd_ev, *ll.bayes_recalibrate(pl, yl, prior_sd=(np.inf, np.inf)))
        evaluate_pd("Platt on local outcomes", n, p_pl, yev, rows_cal, scen, rep)
        p_by = ll.apply_recalibration(pd_ev, *ll.bayes_recalibrate(pl, yl))
        evaluate_pd("Bayesian recalibration (FAIRScore 3.0)", n, p_by, yev, rows_cal, scen, rep)
        pds[("Platt on local outcomes", n)] = (p_pl, "FAIRScore 3.0 (weighted if shift detected)")
        pds[("Bayesian recalibration (FAIRScore 3.0)", n)] = (p_by, "FAIRScore 3.0 (weighted if shift detected)")
        if n >= 50 and yl.sum() >= 3:
            m = cat(200).fit(Xl, yl)
            evaluate_pd("retrain on local loans only", n, m.predict_proba(Xev)[:, 1], yev, rows_cal, scen, rep)
            wp = np.r_[np.ones(len(ys_)), np.full(n, 3.0)]
            m = cat().fit(np.vstack([Xs_, Xl]), np.r_[ys_, yl], sample_weight=wp)
            evaluate_pd("retrain on source + local (local x3)", n, m.predict_proba(Xev)[:, 1], yev, rows_cal, scen, rep)

    # ---- decisions: approve if the conformal set is {repays} and the (adapted) PD <= the engine's threshold
    for (name, n), (p_ev, variant) in pds.items():
        ap = (sets[variant] == "repays") & (p_ev <= t_star)
        rows_dec.append({"scenario": scen, "rep": rep, "method": name, "n_labels": n,
                         "approve_share": float(ap.mean()),
                         "default_rate_approved": float(yev[ap].mean()) if ap.any() else np.nan,
                         "within_appetite": bool(ap.any() and yev[ap].mean() <= TARGET_DR)})


def main():
    t0 = time.time()
    d, X, y = tr.load_data()
    z = cushion(X)
    reg = regions(d)
    date = pd.to_datetime(d["loan_date"]).values
    rows_cal, rows_conf, rows_dec, scen_rows, rows_day0 = [], [], [], [], []

    # ---- synthetic shifts: one source engine per repeat, shared by scenarios 1-5
    for rep in range(REPEATS):
        rng = np.random.default_rng(1000 + rep)
        S, T = train_test_split(np.arange(len(y)), test_size=0.5, stratify=y, random_state=rep)
        Tst, Tev = train_test_split(T, test_size=0.4, stratify=y[T], random_state=rep)
        b = tr.fit_bundle(X[S], y[S])
        for scen, cfg in SYNTHETIC.items():
            ws, we = shift_weights(z[Tst], y[Tst], **cfg), shift_weights(z[Tev], y[Tev], **cfg)
            ist = rng.choice(Tst, N_STREAM, p=ws / ws.sum())
            iev = rng.choice(Tev, N_EVAL, p=we / we.sum())
            run_one(scen, rep, b, X[S], y[S], X[ist], y[ist], X[iev], y[iev], rows_cal, rows_conf, rows_dec, rows_day0, ist)
            if rep == 0:
                scen_rows.append({"scenario": scen, "source_loans": len(S), "source_default_rate": y[S].mean(),
                                  "target_default_rate": float(y[iev].mean()),
                                  "max_psi": float(ll.drift_table(X[S], X[iev], ae.FEATURES)["psi"].max())})
        print(f"  synthetic repeat {rep + 1}/{REPEATS} ({time.time() - t0:.0f}s)", flush=True)

    # ---- natural shifts: fixed source and target, repeats re-draw the stream / evaluation split
    natural = {"7 natural: out of time (to 1996 -> 1997-98)": date < np.datetime64("1997-01-01"),
               "8 natural: other region (Bohemia -> Moravia)": ~pd.Series(reg).str.contains("Moravia").values}
    for scen, src in natural.items():
        S, T = np.where(src)[0], np.where(~src)[0]
        b = tr.fit_bundle(X[S], y[S])
        scen_rows.append({"scenario": scen, "source_loans": len(S), "source_default_rate": y[S].mean(),
                          "target_loans": len(T), "target_default_rate": y[T].mean(),
                          "max_psi": float(ll.drift_table(X[S], X[T], ae.FEATURES)["psi"].max())})
        for rep in range(REPEATS):
            rng = np.random.default_rng(2000 + rep)
            Tst, Tev = train_test_split(T, test_size=0.4, stratify=y[T], random_state=rep)
            Tst = rng.permutation(Tst)                 # arrival order
            run_one(scen, rep, b, X[S], y[S], X[Tst], y[Tst], X[Tev], y[Tev], rows_cal, rows_conf, rows_dec, rows_day0)
        print(f"  {scen} done ({time.time() - t0:.0f}s)", flush=True)

    def summarise(rows, keys, vals):
        df = pd.DataFrame(rows)
        g = df.groupby(keys, sort=False)
        out = g[vals].mean()
        sd = g[vals].std().add_suffix("_sd")
        out = out.join(sd)
        out["repeats"] = g.size()
        return out.reset_index()

    pd.DataFrame(scen_rows).to_csv(RESULTS / "shift_scenarios.csv", index=False)
    cal = summarise(rows_cal, ["scenario", "method", "n_labels"], ["brier", "ece", "logloss", "auc", "mean_pd", "default_rate"])
    cal.to_csv(RESULTS / "shift_calibration.csv", index=False)
    conf = summarise(rows_conf, ["scenario", "variant"], ["defaulters_not_flagged", "repayers_not_flagged",
                                                          "auto_approvable_share", "shift_detected", "ess_share"])
    day0 = summarise(rows_day0, ["scenario"], ["true_default_rate", "estimate_mean_pd", "estimate_em_prior_shift",
                                               "separability_auc", "shift_detected", "max_psi"])
    day0.to_csv(RESULTS / "shift_day0.csv", index=False)
    conf.to_csv(RESULTS / "shift_conformal.csv", index=False)
    dec = pd.DataFrame(rows_dec)
    decs = dec.groupby(["scenario", "method", "n_labels"], sort=False).agg(
        approve_share=("approve_share", "mean"), default_rate_approved=("default_rate_approved", "mean"),
        default_rate_approved_sd=("default_rate_approved", "std"), within_appetite=("within_appetite", "mean")).reset_index()
    decs.to_csv(RESULTS / "shift_decisions.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(scen_rows).round(3).to_string(index=False))
    print(conf.round(3).to_string(index=False))
    print(day0.round(3).to_string(index=False))
    print(cal[cal.method.isin(["none (source engine)",
                               "Bayesian recalibration (FAIRScore 3.0)", "Platt on local outcomes"])].round(3).to_string(index=False))
    print(decs.round(3).to_string(index=False))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
