"""
Real bank statements with real loan outcomes: the Berka / PKDD'99 financial data
set (a Czech bank, 1993-1998; 4,500 accounts, 1,056,320 transactions, 682 loans
whose repayment outcome is known).

It is the only public data set where a bank account's full transaction history
can be lined up with whether a loan was later repaid -- exactly the question
FAIRScore answers from a bank statement, without any credit-bureau data.

For every loan this module rebuilds the statement an underwriter would have
seen: the (up to) 6 months of transactions BEFORE the loan date, with the
running balance, Czech transaction codes translated into the narrations Indian
statements use (so FAIRScore's own reader rules classify them), and amounts
rescaled to rupee-like magnitudes. The statement then goes through exactly the
same feature pipeline, policy checks and scorecard as an uploaded PDF.

Data: copy loan.csv, trans.csv, account.csv, disp.csv, client.csv into
data/raw/berka/ (e.g. from github.com/Kusainov/czech-banking-fin-analysis or the
PKDD'99 Discovery Challenge site). Outcome codes: A = finished, repaid;
B = finished, NOT repaid; C = running, up to date; D = running, in debt.
Bad = B or D.
"""
import string

import numpy as np
import pandas as pd

from config import PROJECT_ROOT

BERKA_DIR = PROJECT_ROOT / "data" / "raw" / "berka"

# Amounts are scaled so the median verified income of Berka borrowers lands near a typical
# Indian retail borrower's (about Rs 40,000 a month). Every ratio-based rule (FOIR, money left,
# volatility, balance cover, overdrafts, bounces) is unaffected; the scale only matters for the
# few rupee floors (minimum take-home, Rs 1,000 low-balance mark, minimum loan amount).
DEFAULT_SCALE = 3.0
WINDOW_MONTHS = 6


def _payee(account):
    """A stable letters-only name for a partner account (so counterparty rules can match it)."""
    if pd.isna(account) or not account:
        return "UNKNOWN"
    n, out = int(account), ""
    while n:
        n, r = divmod(n, 26)
        out += string.ascii_uppercase[r]
    return "PARTY " + out[:6]


def narration(r):
    """Czech codes -> Indian-style narration (what the statement reader classifies)."""
    t, op, k = r["type"], r["operation"], r["k_symbol"] if isinstance(r["k_symbol"], str) else ""
    k = k.strip()
    who = _payee(r["account"])
    bank = r["bank"] if isinstance(r["bank"], str) else "XX"
    if t == "PRIJEM":                                   # money in
        if k == "UROK":
            return "INTEREST CREDIT"
        if k == "DUCHOD":
            return f"PENSION CREDIT NEFT CR-{bank}-{who}"
        if op == "VKLAD":
            return "CASH DEPOSIT"
        if op == "PREVOD Z UCTU":
            return f"NEFT CR-{bank}-{who}"
        return "CREDIT"
    # money out (VYDAJ) and cash withdrawals (VYBER)
    if k == "SANKC. UROK":
        return "PENALTY INTEREST NEGATIVE BALANCE CHGS"
    if k == "SLUZBY":
        return "ACCOUNT MAINTENANCE FEE"
    if k == "SIPO":
        return f"HOUSEHOLD BILL ELECTRICITY RENT {who}"
    if k == "POJISTNE":
        return f"INSURANCE PREMIUM {who}"
    if k == "UVER":
        return f"LOAN EMI {who}"
    if op == "VYBER KARTOU":
        return "POS CARD PURCHASE"
    if op == "VYBER" or t == "VYBER":
        return "CASH WITHDRAWAL ATM"
    if op == "PREVOD NA UCET":
        return f"NEFT DR-{bank}-{who}"
    return "DEBIT"


def _date(s):
    return pd.to_datetime(s.astype(int).astype(str), format="%y%m%d")


def load_tables(path=BERKA_DIR):
    if not (path / "trans.csv").exists():
        raise FileNotFoundError(f"Berka data not found in {path}: see the module docstring.")
    read = lambda n: pd.read_csv(path / n, sep=";", low_memory=False)  # noqa: E731
    loan, trans, acc, disp, client = (read(f"{n}.csv") for n in ("loan", "trans", "account", "disp", "client"))
    loan["loan_date"], acc["open_date"], trans["date"] = _date(loan["date"]), _date(acc["date"]), _date(trans["date"])
    owner = disp[disp["type"] == "OWNER"][["account_id", "client_id"]]
    bn = client["birth_number"].astype(str).str.zfill(6)
    mm = bn.str[2:4].astype(int)
    client["female"] = (mm > 50).astype(int)              # Berka encodes women as month + 50
    client["birth"] = pd.to_datetime("19" + bn.str[:2] + (mm % 50).astype(str).str.zfill(2) + bn.str[4:6],
                                     format="%Y%m%d", errors="coerce")
    loan = (loan.merge(acc[["account_id", "open_date"]], on="account_id")
                .merge(owner, on="account_id").merge(client[["client_id", "female", "birth"]], on="client_id"))
    loan["age"] = ((loan["loan_date"] - loan["birth"]).dt.days / 365.25).astype(int)
    loan["bad"] = loan["status"].isin(["B", "D"]).astype(int)
    return loan, trans


def _chain_order(t):
    """Berka lists same-day transactions in no particular order; a bank statement prints them in
    posting order, where every running balance follows from the previous one. Rebuild that order
    by following the balance chain within each day."""
    out, prev = [], None
    for _, day in t.groupby("date", sort=True):
        rows = day.to_dict("records")
        while rows:
            signed = lambda r: r["amount"] if r["type"] == "PRIJEM" else -r["amount"]   # noqa: E731
            start = 0.0 if prev is None else prev              # a new account starts from zero
            k = next((i for i, r in enumerate(rows) if abs(start + signed(r) - r["balance"]) < 0.3), 0)
            r = rows.pop(k)
            out.append(r)
            prev = r["balance"]
    return pd.DataFrame(out)


def statements(path=BERKA_DIR, scale=DEFAULT_SCALE, window_months=WINDOW_MONTHS):
    """Yields (loan_row, transactions_df, period) -- the statement before each loan."""
    loan, trans = load_tables(path)
    trans = trans.sort_values(["account_id", "date", "trans_id"])
    by_acc = dict(tuple(trans.groupby("account_id")))
    for _, ln in loan.iterrows():
        end = ln["loan_date"] - pd.Timedelta(days=1)
        start = max(ln["open_date"], ln["loan_date"] - pd.DateOffset(months=window_months))
        t = _chain_order(by_acc[ln["account_id"]])          # whole history, so the first day chains too
        t = t[(t["date"] >= start) & (t["date"] <= end)]
        if len(t) < 10:
            continue
        signed = np.where(t["type"] == "PRIJEM", t["amount"], -t["amount"]) * scale
        df = pd.DataFrame({"date": t["date"].values,
                           "description": [narration(r).lower() for _, r in t.iterrows()],
                           "signed_amount": np.round(signed, 2), "balance": np.round(t["balance"].values * scale, 2)})
        yield ln, df, [start.strftime("%d/%m/%Y"), end.strftime("%d/%m/%Y")]
