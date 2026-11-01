"""
Recalibrate the FAIRScore scorecard from REAL repayment outcomes (closing the loop).

The scorecard starts as an expert scorecard because no Indian repayment data
was available to train on. Every assessment stores its factor shares; once
loans have been disbursed and their outcomes recorded (History tab or the
API), this script:

  1. builds the training set: factor shares -> bad (defaulted 90+ days /
     written off) vs good (repaid / repaying for 12+ months),
  2. fits a logistic regression with NON-NEGATIVE weights on the "goodness"
     shares (so a better habit can never cost points) and checks it by
     5-fold cross-validated AUC against the current expert score,
  3. writes the proposed points to results/scorecard_recalibration.csv.

Nothing is applied automatically: a credit-risk team reviews the proposal and
copies it into config/policy.toml (model governance: a challenger only
replaces the champion after review).

    python src\\train_from_outcomes.py
"""
import json
import sys

import numpy as np
import pandas as pd

import audit_store
from config import PROJECT_ROOT
from policy_config import POLICY

MIN_LOANS, MIN_BAD = 200, 30
BAD = {"defaulted 90+ days", "written off"}
GOOD = {"repaid", "repaying"}


def load(path=None):
    h = audit_store.history(path)
    h = h[h["outcome"].isin(BAD | GOOD) & h["score_factors"].notna()]
    X = pd.DataFrame([json.loads(v) for v in h["score_factors"]], index=h.index)
    y = h["outcome"].isin(BAD).astype(int).values
    return X.fillna(0.5), y, h["fairscore"].values         # a factor that did not apply counts as neutral


def main(path=None):
    X, y, expert = load(path)
    print(f"{len(y)} loans with outcomes, {int(y.sum())} bad")
    if len(y) < MIN_LOANS or y.sum() < MIN_BAD:
        print(f"Not enough outcomes yet: need at least {MIN_LOANS} loans and {MIN_BAD} bad ones. "
              f"Keep recording outcomes in the History tab.")
        return 0
    from scipy.optimize import minimize
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    def fit(Xa, ya):
        # P(bad) = sigmoid(b - sum(w_i * share_i)), w >= 0  (more "goodness" -> lower risk)
        def loss(p):
            z = p[0] - Xa @ p[1:]
            return np.mean(np.logaddexp(0, z) - ya * z) + 1e-3 * np.sum(p[1:] ** 2)
        res = minimize(loss, np.r_[0.0, np.ones(Xa.shape[1])], method="L-BFGS-B",
                       bounds=[(None, None)] + [(0, None)] * Xa.shape[1])
        return res.x

    aucs, expert_aucs = [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        p = fit(X.values[tr], y[tr])
        aucs.append(roc_auc_score(y[te], p[0] - X.values[te] @ p[1:]))
        expert_aucs.append(roc_auc_score(y[te], -expert[te]))
    p = fit(X.values, y)
    total = sum(POLICY["score_points"].values())
    proposed = p[1:] / p[1:].sum() * total if p[1:].sum() else np.zeros(len(p) - 1)
    out = pd.DataFrame({"factor": X.columns, "current_points": [POLICY["score_points"].get(c) for c in X.columns],
                        "proposed_points": proposed.round(0)})
    path = PROJECT_ROOT / "results" / "scorecard_recalibration.csv"
    out.to_csv(path, index=False)
    print(out.to_string(index=False))
    print(f"\nCross-validated AUC  expert scorecard {np.mean(expert_aucs):.3f}   recalibrated {np.mean(aucs):.3f}")
    print(f"Proposal saved to {path}. Review it before copying the points into config/policy.toml.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
