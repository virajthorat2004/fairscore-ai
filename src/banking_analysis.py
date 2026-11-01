"""
Banking analysis: the figures a credit officer reads off a bank statement
before deciding (the same ground that commercial bank-statement analysers
such as those used by Indian NBFCs cover), computed from the parsed
transactions.

    daily balances and ABB (average bank balance), month-wise banking summary,
    bounces (only the applicant's own EMI / cheque returns, not a cheque they
    deposited that bounced), penalty charges, digital-lender / loan-app usage
    (loan stacking), cash dependence, salary trend (stopped / dropped in the
    latest month), salary from more than one employer, large one-off credits
    (window dressing / unexplained source of funds), balance just before payday,
    days at a very low balance.

Every figure is computed from the statement alone: no credit-bureau data.
"""
import re

import numpy as np
import pandas as pd

from statement_parser import (INVESTMENT_PATTERN, is_loan_payment, narration_names, _complete_cycles,
                              DAYS_PER_MONTH)

# Own EMI / cheque / mandate returned unpaid (inward returns). An OUTWARD return (a
# cheque the applicant deposited that bounced) is not the applicant's default.
BOUNCE_PATTERN = (r"(?:\bach\b|\bnach\b|\becs\b|\bchq\b|cheque|\bi/?w\b|inward|mandate|\bemi\b|\bsi\b|auto ?debit)"
                  r"[^a-z]{0,4}(?:[a-z]+[^a-z]{1,3}){0,2}(?:rtn|ret|return|returned|reject\w*|dishono\w*)\b"
                  r"|\bbounc|dishono|insuff|funds? insuf|\brtn chg|return chg|\bret chg|unpaid emi")
OUTWARD_PATTERN = r"\bo/?w\b|outward|deposited chq|chq dep|clg dep"
PENALTY_PATTERN = (r"min(?:imum)?\.? ?bal|non[- ]?maint|\bpenal|negative bal|\bneg bal|overdue int|late (?:fee|pay)"
                   r"|\bod int|\bmab chg|\bamb chg")

# Banks, NBFCs and loan apps that lend to individuals in India (whole words).
LENDERS = {
    "Bajaj Finance": r"bajaj ?fin", "KreditBee": r"kredit ?bee|krazybee", "MoneyView": r"money ?view|whizdm",
    "Navi": r"navi ?(?:finserv|technologies|fin\b)", "Slice": r"\bslice ?(?:card|fin|pay)|quadrillion", "LazyPay": r"lazy ?pay", "Simpl": r"\bsimpl\b",
    "Fibe / EarlySalary": r"\bfibe\b|early ?salary|social worth", "CASHe": r"\bcashe\b|aeries",
    "mPokket": r"m ?pokket", "StashFin": r"stash ?fin|akara", "Kissht": r"kissht|onemi|si creva",
    "Dhani": r"dhani ?(?:loan|service|app)|indiabulls consumer", "PaySense": r"pay ?sense", "Home Credit": r"home ?credit", "Tata Capital": r"tata ?capital",
    "Aditya Birla Finance": r"aditya ?birla ?fin|\babfl\b", "Muthoot": r"muthoot ?(?:fin|gold|capital|home ?fin|micro|loan|mercantile)",
    "Manappuram": r"manappuram ?(?:fin|gold|home|loan|asirvad)|\bmafil\b|asirvad micro",
    # surnames alone (Muthoot, Manappuram, Bajaj, Tata...) are not lenders: a company marker is required
    "IIFL": r"\biifl\b", "SMFG / Fullerton": r"fullerton|\bsmfg\b|smicc", "Poonawalla": r"poonawalla",
    "Hero FinCorp": r"hero ?fin ?corp", "L&T Finance": r"l ?& ?t fin|\blnt fin|\bltfs\b", "ZestMoney": r"zest ?money",
    "Paytm Postpaid": r"paytm ?postpaid", "Amazon Pay Later": r"amazon ?pay ?later", "Flipkart Pay Later":
    r"flipkart ?pay ?later", "Axio / Capital Float": r"\baxio\b|capital ?float", "Lendingkart": r"lending ?kart",
    "TrueBalance": r"true ?balance", "SmartCoin": r"smart ?coin", "Prefr": r"\bprefr\b", "Nira": r"nira ?fin",
    "Indiabulls": r"india ?bulls", "Shriram Finance": r"shriram fin|shriram city", "Cholamandalam": r"cholamandalam|chola ?(?:fin|ms)",
    "Mahindra Finance": r"mahindra ?fin|\bmmfsl\b", "HDB Financial": r"\bhdb ?fin|hdbfs", "Credila": r"credila",
}
# "ATM" alone is not enough: some banks print the branch name ("Branch : ATM ...") in every narration.
CASH_WITHDRAWAL = (r"\batm ?(?:wdl|wd|withdrawal|cash|txn)|\batm/|atm-wdl|\batw\b|\batl\b|\bnfs\b|cash ?wdl"
                   r"|cash withdrawal|\bcash wd\b|\bcwdr\b|self chq|cash paid")
CASH_DEPOSIT = r"cash ?dep|by cash|\bcdm\b|cash deposit|\bbna\b|\bcash cr\b|csh dep"


def _f(x, nd=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def daily_balances(df, start, end):
    """End-of-day balance for every calendar day of the period (forward-filled),
    summed over accounts when several statements are combined."""
    if "balance" not in df.columns or df["balance"].notna().sum() < 3:
        return None
    days = pd.date_range(start, end, freq="D")
    total = pd.Series(0.0, index=days)
    for _, acc in (df.groupby("account") if "account" in df.columns else [(None, df)]):
        acc = acc[acc["balance"].notna()]
        if not len(acc):
            continue
        eod = acc.groupby(acc["date"].dt.normalize())["balance"].last()
        first = acc.iloc[0]
        opening = float(first["balance"] - first["signed_amount"])
        series = eod.reindex(days).ffill().fillna(opening)
        total = total.add(series, fill_value=0.0)
    return total


def lenders_seen(df):
    """{lender: {"disbursals": n, "repayments": n}} from narrations."""
    out = {}
    desc = df["description"].str.replace(r"@[a-z0-9._-]+", " ", regex=True)   # UPI handles (@naviaxis) are not lenders
    for name, pat in LENDERS.items():
        hit = desc.str.contains(pat, na=False, regex=True)
        if hit.any():
            cr = int((hit & (df["signed_amount"] > 0)).sum())
            dr = int((hit & (df["signed_amount"] < 0)).sum())
            out[name] = {"disbursals": cr, "repayments": dr}
    return out


def bounce_events(df):
    """Days on which one of the applicant's own payments was returned unpaid
    (the return and its charge on the same day count once)."""
    d = df["description"]
    hit = d.str.contains(BOUNCE_PATTERN, na=False, regex=True) & ~d.str.contains(OUTWARD_PATTERN, na=False, regex=True)
    outward = d.str.contains(BOUNCE_PATTERN, na=False, regex=True) & d.str.contains(OUTWARD_PATTERN, na=False, regex=True)
    return int(df.loc[hit, "date"].dt.normalize().nunique()), int(outward.sum()), hit


def analyse(df, st):
    """Returns the banking-analysis dict (JSON-safe) for a parsed statement."""
    start, end = pd.Timestamp(st["period_start"]), pd.Timestamp(st["period_end"])
    months_covered = st["months_covered"]
    income = st.get("avg_monthly_income") or 0.0
    credits = df["signed_amount"] > 0
    debits = df["signed_amount"] < 0
    total_in = float(df.loc[credits, "signed_amount"].sum())
    total_out = float(-df.loc[debits, "signed_amount"].sum())
    out = {}

    # --- balances -------------------------------------------------------------
    daily = daily_balances(df, start, end)
    if daily is not None:
        out["abb"] = _f(daily.mean())
        out["abb_last_3_months"] = _f(daily[daily.index > end - pd.DateOffset(months=3)].mean())
        out["low_balance_days_share"] = _f((daily < 1000).mean(), 4)
        out["negative_days_share"] = _f((daily < 0).mean(), 4)
        out["min_eod_balance"] = _f(daily.min())
        out["closing_balance"] = _f(daily.iloc[-1])
    else:
        out.update(abb=None, abb_last_3_months=None, low_balance_days_share=None, negative_days_share=None,
                   min_eod_balance=None, closing_balance=None)

    # --- bounces and penalties --------------------------------------------------
    events, outward, bounce_rows = bounce_events(df)
    out["bounce_events"] = events
    out["outward_returns"] = outward
    pen = df["description"].str.contains(PENALTY_PATTERN, na=False, regex=True) & debits & ~bounce_rows
    out["penalty_charges"] = int(pen.sum())

    # --- lenders / loan apps -----------------------------------------------------
    lenders = lenders_seen(df)
    out["lenders"] = lenders
    out["lenders_count"] = len(lenders)
    out["new_loans_in_period"] = sum(1 for v in lenders.values() if v["disbursals"]) + \
        int((df["inflow_type"] == "excluded:Loan disbursals").sum() > 0 and not lenders)

    # --- cash ---------------------------------------------------------------------
    cash_wd = debits & df["description"].str.contains(CASH_WITHDRAWAL, na=False, regex=True)
    cash_dep = credits & df["description"].str.contains(CASH_DEPOSIT, na=False, regex=True)
    out["cash_withdrawal_share"] = _f(-df.loc[cash_wd, "signed_amount"].sum() / total_out, 4) if total_out else 0.0
    out["cash_deposit_share"] = _f(df.loc[cash_dep, "signed_amount"].sum() / total_in, 4) if total_in else 0.0

    # --- salary trend, employers -------------------------------------------------------
    series = st.get("regular_income_series") or []
    out["salary_latest_vs_usual"] = None
    out["salary_stopped"] = False
    if len(series) >= 3 and np.median(series[:-1]) > 0:
        usual = float(np.median(series[:-1]))
        out["salary_latest_vs_usual"] = _f(series[-1] / usual, 3)
        out["salary_stopped"] = bool(series[-1] == 0 and sum(1 for v in series[:-1] if v > 0) >= 2)
    sal = df[df["inflow_type"].isin(["salary", "regular"])]
    names = set()
    for d in sal["description"]:
        n = narration_names(d)
        if n:
            names.add(" ".join(n[0].split()[:2]))
    out["salary_payers"] = len(names)

    # --- balance just before payday --------------------------------------------------
    out["pre_salary_balance_ratio"] = None
    if daily is not None and len(sal) and income:
        big = sal[sal["signed_amount"] >= 0.5 * sal["signed_amount"].median()]
        ratios = []
        for d, amt in zip(big["date"], big["signed_amount"]):
            prev = d.normalize() - pd.Timedelta(days=1)
            if prev in daily.index:
                ratios.append(daily[prev] / amt)
        if ratios:
            out["pre_salary_balance_ratio"] = _f(np.median(ratios), 3)

    # --- large one-off credits (source of funds / window dressing) -----------------------
    large = []
    if income > 0:
        cand = df[credits & ~df["inflow_type"].isin(["salary", "regular"])
                  & ~df["inflow_type"].eq("excluded:Transfers from own accounts")
                  & (df["signed_amount"] >= max(1.5 * income, 25000))]
        for _, r in cand.iterrows():
            late = r["date"] >= end - pd.Timedelta(days=30)
            large.append({"date": r["date"].date().isoformat(), "amount": _f(r["signed_amount"]),
                          "last_30_days": bool(late), "type": str(r["inflow_type"])})
    out["large_credits"] = large
    closing = out.get("closing_balance") or 0
    out["window_dressing"] = bool(any(c["last_30_days"] and closing >= 0.7 * c["amount"] for c in large))

    # --- month-wise banking summary --------------------------------------------------------
    counted_types = ["salary", "regular", "other"] if str(st.get("income_basis", "")).startswith("all") \
        else ["salary", "regular"]
    emis = debits & is_loan_payment(df["description"])
    invest = debits & df["description"].str.contains(INVESTMENT_PATTERN, na=False, regex=True)
    rows = []
    for m, g in df.groupby("month"):
        idx = g.index
        row = {"month": str(m), "transactions": int(len(g)),
               "credits": _f(g.loc[g["signed_amount"] > 0, "signed_amount"].sum()),
               "income_counted": _f(g.loc[(g["signed_amount"] > 0) & g["inflow_type"].isin(counted_types),
                                          "signed_amount"].sum()),
               "debits": _f(-g.loc[g["signed_amount"] < 0, "signed_amount"].sum()),
               "emis": _f(-df.loc[idx][emis.loc[idx]]["signed_amount"].sum()),
               "investments": _f(-df.loc[idx][invest.loc[idx]]["signed_amount"].sum()),
               "cash_withdrawals": _f(-df.loc[idx][cash_wd.loc[idx]]["signed_amount"].sum()),
               "bounces": int(df.loc[idx][bounce_rows.loc[idx]]["date"].dt.normalize().nunique())}
        if daily is not None:
            dm = daily[(daily.index >= m.start_time) & (daily.index <= m.end_time)]
            row.update(abb=_f(dm.mean()) if len(dm) else None, min_balance=_f(dm.min()) if len(dm) else None,
                       closing_balance=_f(dm.iloc[-1]) if len(dm) else None)
        m_start, m_end = max(m.start_time, start), min(m.end_time.normalize(), end)
        row["partial"] = bool(m_start > m.start_time + pd.Timedelta(days=3) or m_end < m.end_time.normalize()
                              - pd.Timedelta(days=3))
        rows.append(row)
    out["monthly"] = rows
    out["inflow_outflow_ratio"] = _f(total_in / total_out, 3) if total_out else None
    out["months_covered"] = months_covered
    return out
