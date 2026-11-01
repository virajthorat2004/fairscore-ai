"""
Turn a bank statement into a sequence of transaction EVENTS for the FAIRScore 2.0 sequence model.

The same code reads a rebuilt Berka statement (training) and an uploaded Indian statement (live use), so the
model sees one representation. Every number is scale-free (amounts and balances relative to the account's
own typical monthly money-in), so Czech crowns and rupees, small and large earners, look alike.

One event = [category id] + 6 numbers:
    direction (1 = money in), log(1 + amount / income), balance after / income (clipped to -2..10),
    balance below zero (0/1), log(1 + days since previous event), day of month / 31
"""
import re

import numpy as np
import pandas as pd

from banking_analysis import CASH_DEPOSIT, CASH_WITHDRAWAL, PENALTY_PATTERN
from statement_parser import (BETTING_PATTERN, BILL_PATTERN, INVESTMENT_PATTERN, LOAN_EMI_PATTERN, OWN_ACCOUNT_DEBIT,
                              SALARY_PATTERN)

CATEGORIES = ["pad", "salary_pension", "cash_in", "transfer_in", "interest", "refund", "other_in",
              "cash_out", "emi_loan", "bill_household", "insurance", "charge_penalty", "investment", "betting",
              "card_upi_spend", "transfer_out", "own_transfer_out", "other_out"]
CAT_ID = {c: i for i, c in enumerate(CATEGORIES)}
N_NUM = 6
MAX_EVENTS = 300

_R = {k: re.compile(v, re.I) for k, v in {
    "salary": SALARY_PATTERN, "cash_dep": CASH_DEPOSIT, "cash_wdl": CASH_WITHDRAWAL, "emi": LOAN_EMI_PATTERN,
    "bill": BILL_PATTERN + r"|household bill", "insurance": r"insurance|\blic\b|premium", "penalty": PENALTY_PATTERN +
    r"|maintenance fee|\bcharges?\b|\bchgs\b|sms chg", "invest": INVESTMENT_PATTERN, "bet": BETTING_PATTERN,
    "interest": r"interest|\bint\.? ?(?:cr|pd|paid)\b", "refund": r"refund|reversal|\brev\b|cashback",
    "card": r"\bpos\b|card|\bupi\b|swiggy|zomato|amazon|flipkart|paytm|phonepe|gpay", "own": OWN_ACCOUNT_DEBIT,
    "transfer": r"neft|imps|rtgs|\btrf\b|transfer|\bft\b"}.items()}


def category(desc, amount):
    d = str(desc)
    if amount >= 0:
        if _R["salary"].search(d):
            return "salary_pension"
        if _R["interest"].search(d):
            return "interest"
        if _R["refund"].search(d):
            return "refund"
        if _R["cash_dep"].search(d):
            return "cash_in"
        if _R["transfer"].search(d) or _R["card"].search(d):
            return "transfer_in"
        return "other_in"
    for key, cat in (("bet", "betting"), ("penalty", "charge_penalty"), ("emi", "emi_loan"), ("insurance", "insurance"),
                     ("bill", "bill_household"), ("invest", "investment"), ("cash_wdl", "cash_out"),
                     ("own", "own_transfer_out"), ("card", "card_upi_spend"), ("transfer", "transfer_out")):
        if _R[key].search(d):
            return cat
    return "other_out"


def income_reference(df):
    """Typical monthly money-in of the account (median over months), the scale for every amount."""
    t = pd.to_datetime(df["date"])
    credits = df["signed_amount"].clip(lower=0).groupby(t.dt.to_period("M")).sum()
    ref = float(credits.median()) if len(credits) else 0.0
    return max(ref, 1.0)


def events(df, income_ref=None, max_events=MAX_EVENTS):
    """df: date, description, signed_amount, balance (balance may be missing). Returns (cat_ids, numbers)."""
    df = df.sort_values("date", kind="stable").tail(max_events)
    if not len(df):
        return np.zeros(0, dtype=np.int64), np.zeros((0, N_NUM), dtype=np.float32)
    ref = income_ref or income_reference(df)
    amt = df["signed_amount"].astype(float).values
    bal = df["balance"].astype(float).values if "balance" in df and df["balance"].notna().any() else np.full(len(df), np.nan)
    bal = np.where(np.isnan(bal), 0.0, bal)
    dates = pd.to_datetime(df["date"]).values
    gaps = np.r_[0.0, np.diff(dates).astype("timedelta64[D]").astype(float)]
    cats = np.array([CAT_ID[category(d, a)] for d, a in zip(df["description"].values, amt)], dtype=np.int64)
    num = np.c_[(amt > 0).astype(float), np.log1p(np.abs(amt) / ref), np.clip(bal / ref, -2, 10), (bal < 0).astype(float),
                np.log1p(np.clip(gaps, 0, 365)), pd.to_datetime(df["date"]).dt.day.values / 31.0].astype(np.float32)
    return cats, num
