"""
Backtest FAIRScore on REAL loan outcomes (Berka / PKDD'99, see berka_data.py).

Every loan's pre-loan statement goes through the live system unchanged --
statement features, lending-policy checks, eligibility engine and the
FAIRScore scorecard -- and the recommendation is compared with what actually
happened to the loan. This answers the question an examiner (or a bank's
model-risk team) asks first: "do your rules actually separate people who
repay from people who don't?"

Applicant fields the bank data does not contain are filled neutrally:
employment type is inferred the way an officer would read the statement
(pension credits -> pensioner; a steady monthly salary-like credit -> salaried;
otherwise self-employed), 5 years in work, rented home, 1 dependent, no
vehicle, savings = average balance. The loan's REAL monthly payment is used
as the new EMI. Age is taken from the data only for the eligibility rule and
the fairness audit.

    python src\\berka_backtest.py            -> results/berka_backtest_loans.csv + summary tables
"""
import sys
import warnings

import numpy as np
import pandas as pd

import berka_data
import cashflow_model
import live_scoring as ls
from config import PROJECT_ROOT
from statement_parser import features_from_transactions

RESULTS = PROJECT_ROOT / "results"
warnings.filterwarnings("ignore")


def applicant_for(ln, stmt):
    """The form an officer would fill for this borrower (fields the data lacks are neutral)."""
    share = stmt.get("regular_income_months_share") or 0
    regular = stmt.get("avg_monthly_regular_income") or 0
    allin = stmt.get("avg_monthly_all_inflows") or 0
    if stmt.get("_pension"):
        emp = "Pensioner"
    elif regular > 0 and share >= 0.8 and regular >= 0.5 * allin:
        emp = "Salaried (private sector)"
    else:
        emp = "Self-employed / business owner"
    return {"age": int(ln["age"]), "employment_type": emp, "employment_years": 5.0, "housing": "Rented home",
            "dependents": 1, "owns_car": False, "total_savings": float(max((stmt.get("avg_balance") or 0), 0)),
            "requested_amount": float(ln["amount"]) * berka_data.DEFAULT_SCALE,
            "requested_duration_months": int(ln["duration"]), "purpose": "Other",
            "proposed_emi": float(ln["payments"]) * berka_data.DEFAULT_SCALE,
            "application_date": ln["loan_date"].strftime("%Y-%m-%d"),
            "monthly_income": None, "active_credit_lines": None, "property_loans": None, "education": None}


def build(limit=None):
    # the backtest records the VERSION 1 (policy-first) decision; the version 2 AI engine is trained on these
    # rows and evaluated separately with nested cross-validation (src/train_ai_engine.py), so it must not
    # decide here (it would be judging loans it was trained on)
    ls.POLICY["ai"] = {**ls.POLICY["ai"], "mode": "policy_first"}
    rows = []
    for k, (ln, df, period) in enumerate(berka_data.statements()):
        if limit and k >= limit:
            break
        info = {"printed_period": period, "reconciled": 1.0, "checked_rows": len(df) - 1, "method": "bank data",
                "file_origin": {"category": "bank-generated"}}
        tx, st, _ = features_from_transactions(df.copy(), info)
        st["_pension"] = bool(tx["description"].str.contains("pension", na=False).any())
        a = applicant_for(ln, st)
        q = ls.quick_decision(a, st, tx, exclude_models=("cashflow",))   # never judged by a model trained on itself
        checks = {c["key"]: c["status"] for c in q["policy"]["checks"]}
        b = st.get("banking") or {}
        view = ls.employment_view(st, a)
        row = {"loan_id": ln["loan_id"], "loan_date": ln["loan_date"].strftime("%Y-%m-%d"),
               "bad": int(ln["bad"]), "status": ln["status"], "age": a["age"],
               "female": int(ln["female"]), "employment_type": a["employment_type"],
               "months": st["months_covered"], "verdict": q["verdict"],
               "fairscore": q["fairscore"]["score"], "band": q["fairscore"]["band"],
               "confidence": q["decision_confidence"]["score"], "risk_percentile": q["consensus_percentile"],
               "n_fail": sum(v == "fail" for v in checks.values()),
               "n_caution": sum(v == "caution" for v in checks.values()),
               "eligible_amount": (q.get("eligibility") or {}).get("eligible_amount"),
               "bill_months_share": st.get("bill_months_share"), "round_trip_in_share": st.get("round_trip_in_share"),
               }
        cf, _ = cashflow_model.features(a, view, q["policy"]["new_emi"], q["policy"]["existing_emi"])
        row.update(cf)
        for key in ("affordability", "surplus", "stability", "conduct", "income_source", "evidence", "take_home",
                    "balance_cover", "loan_size", "eligibility", "salary_trend", "lenders", "pass_through",
                    "source_of_funds"):
            row[f"check_{key}"] = checks.get(key, "n/a")
        row.update({f"sc_{k}": v for k, v in q["fairscore"]["fractions"].items()})
        rows.append(row)
        if (k + 1) % 100 == 0:
            print(f"  {k + 1} loans processed", flush=True)
    return pd.DataFrame(rows)


def summarise(d):
    from sklearn.metrics import roc_auc_score
    out = []
    base = d["bad"].mean()
    out.append(f"{len(d)} loans, {int(d['bad'].sum())} bad ({base:.1%})")
    v = d.groupby("verdict")["bad"].agg(["size", "mean"]).reindex(["approve", "refer", "decline"])
    out.append("\nDefault rate by FAIRScore recommendation:\n" + v.rename(columns={"size": "loans", "mean": "default_rate"}).to_string())
    out.append(f"\nAUC of the FAIRScore score (higher score = safer): {roc_auc_score(d['bad'], -d['fairscore']):.3f}")
    bands = d.groupby("band")["bad"].agg(["size", "mean"])
    out.append("\nDefault rate by score band:\n" + bands.to_string())
    return "\n".join(out)


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    d = build(lim)
    RESULTS.mkdir(exist_ok=True)
    d.to_csv(RESULTS / "berka_backtest_loans.csv", index=False)
    print(summarise(d))
