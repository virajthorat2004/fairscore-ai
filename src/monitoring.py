"""
Model monitoring: are today's applicants like the people the cash-flow model
learned from?

Population Stability Index (PSI) per model input, live applications vs the
training data, on the training data's deciles -- the standard drift measure
in bank model-risk management:

    PSI < 0.10   stable          0.10 - 0.25   moderate shift: watch
    PSI > 0.25   major shift: the model's percentiles no longer describe these
                 applicants -> recalibrate (record outcomes, then
                 src/train_from_outcomes.py) before relying on it more

Only the scale-free ratios the model uses are stored per assessment (no
amounts, names or transactions).
"""
import json

import numpy as np
import pandas as pd

from config import PROJECT_ROOT

BACKGROUND = PROJECT_ROOT / "models" / "deployable" / "cashflow_background.pkl"
MIN_APPLICATIONS = 30
LABELS = {"foir": "All EMIs vs income", "emi_to_income": "New EMI vs income",
          "loan_to_annual_income": "Loan vs annual income", "term_months": "Loan term",
          "income_volatility": "Income variation", "regular_income_share": "Share of income that is regular",
          "overdraft_share": "Overdraft share", "penalty_charges": "Penalty charges",
          "low_balance_days": "Days at a very low balance", "negative_days": "Days below zero",
          "abb_to_emi": "Average balance vs EMI", "min_balance_to_income": "Lowest balance vs income"}


def psi(expected, actual, bins=10):
    """PSI of `actual` against `expected`, on the deciles of `expected`."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                       # (nearly) constant in training: compare the share at that value
        same_e = np.mean(np.isclose(expected, edges[0]))
        same_a = np.mean(np.isclose(actual, edges[0]))
        e, a = np.array([same_e, 1 - same_e]), np.array([same_a, 1 - same_a])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(expected, edges)[0] / len(expected)
        a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(hist):
    """DataFrame: input, PSI, status, training median, recent median. None if too few assessments."""
    if hist is None or "model_inputs" not in hist or not BACKGROUND.exists():
        return None
    rows = [json.loads(v) for v in hist["model_inputs"].dropna() if v and v != "null"]
    live = pd.DataFrame([r for r in rows if r])
    if len(live) < 1:
        return None
    train = pd.read_pickle(BACKGROUND)
    out = []
    for col in train.columns:
        if col not in live:
            continue
        v = psi(train[col].values, live[col].dropna().values)
        out.append({"input": LABELS.get(col, col), "psi": round(v, 3),
                    "status": "stable" if v < 0.10 else "moderate shift" if v < 0.25 else "major shift",
                    "training_median": round(float(train[col].median()), 3),
                    "applicants_median": round(float(live[col].median()), 3)})
    rep = pd.DataFrame(out).sort_values("psi", ascending=False)
    rep.attrs["applications"] = len(live)
    rep.attrs["enough"] = len(live) >= MIN_APPLICATIONS
    return rep
