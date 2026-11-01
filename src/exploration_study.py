"""
FAIRScore 3.0 research study: the "rejected applicants are never seen" problem (selection bias / reject inference),
and why a small, logged exploration budget of starter loans fixes it.

A lender only learns repayment outcomes for the loans it grants. A model retrained on those alone learns about
the people the OLD policy liked and is blind (and over-optimistic) exactly where new-to-credit applicants sit.
Berka is unusual: every applicant in it received the loan, so the outcome of each one is known and the selection
can be simulated.

Simulation (repeated 5-fold cross-validation, 10 repeats)
    * the lender's historical policy is the expert FAIRScore scorecard (no AI): it accepts the best-scored
      share of applicants (60%, and 40% for a stricter lender); outcomes are observed only for them.
    * strategies to learn the new model from the training folds:
        oracle                       every outcome known (the unattainable upper bound)
        accepted only                the usual retrain (selection bias)
        fuzzy augmentation           a classic reject-inference method: each rejected applicant is added twice,
                                     as a default and as a repayer, weighted by the accepted-only model's PD
        exploration e (unweighted)   in addition a random share e of the rejected applicants received a starter
                                     loan, so their outcomes are known; trained without correction
        exploration e + IPW          the explored outcomes weighted by 1 / e (inverse propensity), which makes the
                                     training sample representative of all applicants again
        exploration e + clipped IPW  FAIRScore 3.0: the same with weights capped at 5 (less variance, a little bias)
    * judged on the held-out fold: ALL applicants who came through the door (accepted and rejected), by AUC,
      Brier score, calibration error, and AUC / calibration among the applicants the old policy rejects (where
      the new-to-credit growth is).
    * cost of learning: defaults among the explored starter loans.

    python src/exploration_study.py        (about 10 minutes)

Writes results/exploration_study.csv
"""
import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import ai_engine as ae
import learning_loop as ll
import train_ai_engine as tr
from config import PROJECT_ROOT

warnings.filterwarnings("ignore")
RESULTS = PROJECT_ROOT / "results"
REPEATS = 10
ACCEPT_SHARES = (0.6, 0.4)
EXPLORE = (0.05, 0.10, 0.20)
IPW_CAP = 5.0          # weights 1 / propensity are capped: with a 5% budget, 1 / 0.05 = 20 lets a handful of loans dominate


def model():
    return CatBoostClassifier(**ae.CATBOOST_PARAMS, monotone_constraints=ae.MONOTONE)


def fit(X, y, w=None):
    if y.min() == y.max():
        return None
    return model().fit(X, y, sample_weight=w)


def metrics(p, y, rej):
    out = {"auc_all": roc_auc_score(y, p), "brier_all": brier_score_loss(y, p), "ece_all": ll.ece(p, y),
           "mean_pd_rejected": float(np.mean(p[rej])), "default_rate_rejected": float(np.mean(y[rej]))}
    out["auc_rejected"] = roc_auc_score(y[rej], p[rej]) if 0 < y[rej].sum() < rej.sum() else np.nan
    out["calibration_gap_rejected"] = out["mean_pd_rejected"] - out["default_rate_rejected"]
    return out


def main():
    t0 = time.time()
    d, X, y = tr.load_data()
    score = d["fairscore"].values.astype(float)
    rows = []
    for acc_share in ACCEPT_SHARES:
        for rep in range(REPEATS):
            rng = np.random.default_rng(rep)
            for k, (tri, tei) in enumerate(StratifiedKFold(5, shuffle=True, random_state=50 + rep).split(X, y)):
                cut = np.quantile(score[tri], 1 - acc_share)          # the old policy's cut-off, set on its own data
                acc_tr = score[tri] >= cut
                rej_te = score[tei] < cut
                Xtr, ytr, Xte, yte = X[tri], y[tri], X[tei], y[tei]
                strategies = {"oracle (all outcomes known)": (Xtr, ytr, None)}
                strategies["accepted only (selection bias)"] = (Xtr[acc_tr], ytr[acc_tr], None)
                base = fit(Xtr[acc_tr], ytr[acc_tr])
                if base is not None:
                    pr = base.predict_proba(Xtr[~acc_tr])[:, 1]
                    Xa = np.vstack([Xtr[acc_tr], Xtr[~acc_tr], Xtr[~acc_tr]])
                    ya = np.r_[ytr[acc_tr], np.ones((~acc_tr).sum()), np.zeros((~acc_tr).sum())]
                    wa = np.r_[np.ones(acc_tr.sum()), pr, 1 - pr]
                    strategies["fuzzy augmentation (classic reject inference)"] = (Xa, ya.astype(int), wa)
                u = rng.random(len(tri))
                for e in EXPLORE:
                    explored = ~acc_tr & (u < e)
                    keep = acc_tr | explored
                    w = np.where(explored[keep], 1.0 / e, 1.0)
                    strategies[f"exploration {e:.0%}, unweighted"] = (Xtr[keep], ytr[keep], None)
                    strategies[f"exploration {e:.0%} + IPW (1 / e)"] = (Xtr[keep], ytr[keep], w)
                    strategies[f"exploration {e:.0%} + clipped IPW (FAIRScore 3.0)"] = (Xtr[keep], ytr[keep],
                                                                                        np.minimum(w, IPW_CAP))
                    rows.append({"accept_share": acc_share, "rep": rep, "fold": k, "strategy": f"cost of exploration {e:.0%}",
                                 "explored_share_of_applicants": explored.mean(),
                                 "explored_default_rate": ytr[explored].mean() if explored.any() else np.nan,
                                 "accepted_default_rate": ytr[acc_tr].mean()})
                for name, (Xs, ys, ws) in strategies.items():
                    m = fit(Xs, ys, ws)
                    if m is None:
                        continue
                    p = m.predict_proba(Xte)[:, 1]
                    rows.append({"accept_share": acc_share, "rep": rep, "fold": k, "strategy": name,
                                 "training_outcomes": int(len(ys) if ws is None or name.startswith("explor") else acc_tr.sum()),
                                 **metrics(p, yte, rej_te)})
            print(f"  accept {acc_share:.0%} repeat {rep + 1}/{REPEATS} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    num = [c for c in df.columns if c not in ("accept_share", "rep", "fold", "strategy")]
    out = df.groupby(["accept_share", "strategy"], sort=False)[num].mean().reset_index()
    sd = df.groupby(["accept_share", "strategy"], sort=False)[["auc_all", "auc_rejected", "calibration_gap_rejected"]].std()
    out = out.merge(sd.add_suffix("_sd").reset_index(), on=["accept_share", "strategy"])
    out.to_csv(RESULTS / "exploration_study.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string(index=False))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
