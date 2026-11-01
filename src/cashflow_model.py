"""
FAIRScore cash-flow model: a credit-risk model that needs NO credit history.

Trained on REAL loan outcomes: the 679 Berka loans whose pre-loan bank
statement could be rebuilt (berka_data.py) -- each statement read by
FAIRScore's own pipeline, then labelled with what actually happened to the
loan (repaid / defaulted). Inputs are only what a 6-month bank statement and
the loan request show, as SCALE-FREE ratios (no currency amounts), so the
model can be applied to rupee statements:

    affordability   all EMIs / income, new EMI / income, loan / annual income, term
    income          month-to-month variation, share that is regular (salary-like)
    liquidity       overdraft share, days near zero / negative, lowest balance vs income,
                    average balance vs EMI
    conduct         penalty / bounce charges

No age, gender or other protected attribute is an input. A CatBoost model with
MONOTONE constraints guarantees that a worse habit can never lower the risk
(e.g. more overdrafts, higher EMI share). Its quality is measured by 5 x 5
repeated cross-validation; fairness by age group and gender is audited on
out-of-fold predictions.

    python src\\train_cashflow.py      (builds the backtest, trains, writes models + results)
"""
import numpy as np

FEATURES = ["foir", "emi_to_income", "loan_to_annual_income", "term_months", "income_volatility",
            "regular_income_share", "overdraft_share", "penalty_charges", "low_balance_days", "negative_days",
            "abb_to_emi", "min_balance_to_income"]
# Left out on purpose: cash share of money in / out, transactions per month and balance trend.
# They describe 1990s cash banking (Berka median: 99% of deposits in cash, 6 transactions a
# month) rather than credit risk, and today's UPI statements fall outside their training range.
# Dropping them cost 0.007 cross-validated AUC (0.872 -> 0.865, well within one standard
# deviation) and makes the model transfer to Indian statements. Cash dependence is still
# assessed by the lending policy and the FAIRScore scorecard.
DROPPED_FOR_TRANSFER = ["cash_deposit_share", "cash_withdrawal_share", "txn_per_month", "abb_trend"]

# +1: higher can only raise risk; -1: higher can only lower risk. Left free where the
# direction is not clear-cut in lending practice (term, cash use, activity, trend).
MONOTONE = {"foir": 1, "emi_to_income": 1, "loan_to_annual_income": 1, "income_volatility": 1,
            "regular_income_share": -1, "overdraft_share": 1, "penalty_charges": 1, "low_balance_days": 1,
            "negative_days": 1, "abb_to_emi": -1, "min_balance_to_income": -1}

# Inputs that need a running balance in the statement.
NEEDS_BALANCE = {"overdraft_share", "low_balance_days", "negative_days", "abb_to_emi", "min_balance_to_income"}

CATBOOST_PARAMS = {"depth": 3, "iterations": 300, "learning_rate": 0.03, "l2_leaf_reg": 5,
                   "random_seed": 42, "verbose": 0}


def features(a, stmt, new_emi, existing_emi):
    """The model's inputs for one application. `stmt` is the employment view of the
    statement features (live_scoring.employment_view), so 'income' is the income the
    lending policy relies on. Returns (values, tags) where tags mark inputs that could
    not be evidenced (no running balance)."""
    stmt = stmt or {}
    b = stmt.get("banking") or {}
    inc = stmt.get("avg_monthly_income") or 0
    allin = stmt.get("avg_monthly_all_inflows") or 0
    amount = float(a["requested_amount"])
    abb = b.get("abb")
    v = {
        "foir": min((existing_emi + new_emi) / inc, 5.0) if inc else 5.0,
        "emi_to_income": min(new_emi / inc, 5.0) if inc else 5.0,
        "loan_to_annual_income": min(amount / (12 * inc), 20.0) if inc else 20.0,
        "term_months": float(a["requested_duration_months"]),
        "income_volatility": min(stmt.get("income_volatility") or 0.0, 3.0),
        "regular_income_share": min((stmt.get("avg_monthly_regular_income") or 0) / max(allin, 1.0), 1.0),
        "overdraft_share": stmt.get("overdraft_frequency") or 0.0,
        "penalty_charges": float(b.get("penalty_charges", 0) or 0),
        "low_balance_days": b.get("low_balance_days_share") or 0.0,
        "negative_days": b.get("negative_days_share") or 0.0,
        "abb_to_emi": min((abb or 0) / max(new_emi, 1.0), 50.0),
        "min_balance_to_income": float(np.clip((stmt.get("min_balance") or 0) / max(inc, 1.0), -5, 20)),
        # reported in the backtest, not used by the model (see DROPPED_FOR_TRANSFER)
        "abb_trend": float(np.clip(((b.get("abb_last_3_months") or 0) - (abb or 0)) / max(inc, 1.0), -10, 10)),
        "cash_deposit_share": b.get("cash_deposit_share", 0.0) or 0.0,
        "cash_withdrawal_share": b.get("cash_withdrawal_share", 0.0) or 0.0,
        "txn_per_month": stmt.get("transaction_frequency") or 0.0,
    }
    has_balance = abb is not None and stmt.get("min_balance") is not None
    tags = {k: ("imputed" if (k in NEEDS_BALANCE and not has_balance) else "real") for k in FEATURES}
    v = {k: v[k] for k in list(FEATURES) + DROPPED_FOR_TRANSFER}
    return v, tags
