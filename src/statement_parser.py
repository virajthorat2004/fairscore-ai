import hashlib
import re

import numpy as np
import pandas as pd

# Common header variants across Indian bank CSV exports — mapped to our canonical names
COLUMN_SYNONYMS = {
    "date": ["date", "txn date", "transaction date", "tran date", "value date", "value dt", "posting date",
             "post date", "post dt", "txn dt", "trans date", "transaction dt", "tran dt", "book date", "entry date"],
    "description": ["description", "narration", "particulars", "remarks", "transaction details",
                    "transaction remarks", "details"],
    "debit": ["debit", "debits", "withdrawal", "withdrawals", "debit amount", "withdrawal amt", "withdrawal amount",
              "dr", "dr amount", "debit amt"],
    "credit": ["credit", "credits", "deposit", "deposits", "credit amount", "deposit amt", "deposit amount",
               "cr", "cr amount", "credit amt"],
    "amount": ["amount", "transaction amount", "txn amount"],  # one signed column
    "balance": ["balance", "running balance", "closing balance", "balance amt", "available balance"],
}

# Whole-word patterns: real UPI narrations are full of personal names, so plain
# substrings misfire ("bet" in Elizabeth, "emi" in premium/Emily, "rent" in current).
LOAN_EMI_PATTERN = r"\bemi\b|\bloan\b|\bnach\b|\bach\s?d|\becs\b|bajaj fin|home fin|\bfinance ltd"
# Home / property loans among the EMIs (used when the applicant leaves "property loans" blank).
HOME_LOAN_PATTERN = r"home loan|housing loan|\bhl\b|housing fin|home fin|mortgage|\blap\b|property loan"
# Any credit product (loans and credit cards): evidence that the applicant has credit.
RECURRING_CREDIT_PATTERN = LOAN_EMI_PATTERN + r"|card payment|credit card|\bcc payment"
BETTING_PATTERN = (r"\bbet\b|betting|\bcasino\b|lottery|\brummy|\bpoker|dream11|dream 11|my11circle|\bmpl\b"
                   r"|winzo|junglee|pokerbaazi|parimatch|1xbet|betway|\bstake\.com|a23 games|head digital")
ESSENTIAL_PATTERN = (r"\brent\b|house rent|room rent|electricity|\bmsedcl\b|\bbest\b undertaking|water bill"
                     r"|grocery|\bdmart\b|school fee|\bfees?\b|insurance|\blic\b|gas bill|\bmahanagar gas")
# Household bills paid from the account (utilities, phone / internet / DTH, cooking gas, rent,
# insurance, local taxes). Paid month after month they are the "alternative data" lenders use
# for new-to-credit customers. On 679 real Berka loans the share of months with a household-bill
# payment separated defaulters on its own (AUC 0.64). Loan EMIs and card bills are not counted.
BILL_PATTERN = (r"electricity|\belec(?:tric)? ?bill|\bmsedcl\b|\bmseb\b|\bbescom\b|tangedco|\btneb\b|\bbses\b"
                r"|tata ?power|adani ?(?:electricity|energy)|torrent ?power|\bcesc\b|\buppcl\b|\bpspcl\b|\bjvvnl\b"
                r"|\bavvnl\b|\b[mudp]gvcl\b|\bkseb\b|\bapspdcl\b|\btsspdcl\b|\bwbsedcl\b|\btpddl\b|\bbrpl\b|\bbypl\b"
                r"|\bbest\b undertaking|water bill|water tax|property tax|\bgas bill|mahanagar gas|\bigl\b|indane"
                r"|\bhp ?gas\b|bharat ?gas|gujarat gas|adani gas|\bbbps\b|\bairtel\b|\bjio\b|vodafone|\bbsnl\b|\bmtnl\b"
                r"|postpaid|prepaid|mobile recharge|broadband|fibernet|hathway|tata ?(?:sky|play)|dish ?tv|\bd2h\b"
                r"|sun direct|\brent\b|house rent|room rent|society maint|insurance|\blic\b")

# --- Income classification -------------------------------------------------
SALARY_PATTERN = r"\bsalary\b|\bsal\b|payroll|\bwages\b|stipend|pension"
# Money coming in that is NOT earnings. Counting it as income is the easiest
# way to make an application look stronger than it is.
EXCLUDED_INFLOWS = {
    "Refunds / reversals": r"return|reversal|reversed|refund|cashback|chargeback",
    "Transfers from own accounts": r"\bself\b|own a/?c|own account|sweep|from sb a/?c",
    "Loan disbursals": r"disburs|loan credit|loan a/?c cr",
    "Deposit maturities": r"\bfd\b|fixed deposit|\brd\b|recurring deposit|maturity|premature",
    "Bank interest": r"int\.?\s?pd|interest|int\.? ?paid|\bsb ?int\b|\bint\.? ?cr\b|\bint\.? ?coll",
    # money from family is support, not earnings (pocket money, gifts): a student's or anyone's
    "Family support / gifts": r"pocket ?money|\bgifts?\b|family support|\bshagun\b|birthday",
    "Round-trip transfers": None,             # assigned by counterparty matching, see classify_inflows
}
# Moving money to one's own savings/FD is not spending.
OWN_ACCOUNT_DEBIT = r"\bself\b|own a/?c|own account|sweep|to sb a/?c|\bfd\b|fixed deposit"
# Auto-debits that are savings / investments / insurance, NOT loan EMIs (NACH is used for both).
INVESTMENT_PATTERN = (r"\bmut\b|mutual|\bsip\b|\bmf\b|\bamc\b|nippon|groww|zerodha|kuvera|\bppf\b|\bnps\b"
                      r"|\bsgb\b|indian clearing|\biccl\b|\bbse\b|\bnse\b|insurance|\blic\b|life ins|\brd\b")


def is_loan_payment(desc_series):
    """Loan EMIs only (for affordability). Investment / insurance auto-debits and
    credit-card bills are excluded: a card paid each month is spending, not a fixed EMI."""
    return desc_series.str.contains(LOAN_EMI_PATTERN, na=False) & \
        ~desc_series.str.contains(INVESTMENT_PATTERN, na=False)


def is_credit_product_payment(desc_series):
    """Loans or credit cards: evidence that the applicant already has credit."""
    return desc_series.str.contains(RECURRING_CREDIT_PATTERN, na=False) & \
        ~desc_series.str.contains(INVESTMENT_PATTERN, na=False)
PAYMENT_RAIL_WORDS = {"neft", "imps", "rtgs", "upi", "ach", "nach", "ecs", "cr", "dr", "by", "to", "from",
                      "transfer", "trf", "credit", "received", "inb", "mb", "ib"}


def _counterparty(desc):
    """Rough counterparty key: description without digits, payment-rail words and IDs."""
    words = [w for w in re.sub(r"[^a-z ]", " ", desc).split() if w not in PAYMENT_RAIL_WORDS and len(w) > 1]
    return " ".join(words[:3])


def upi_counterparty(desc):
    """Who the money came from / went to, from UPI / IMPS narrations: the VPA handle
    ('name123@okaxis' -> 'name123'), else the handle after the bank code ('/SBIN/name123/'),
    else the payer's name ('UPI/RAVI KUMAR/1234/...' -> 'ravi ku'). Pure reference numbers
    are not counterparties."""
    m = re.search(r"([a-z0-9][a-z0-9._-]{2,})@[a-z]*", desc)
    if m:
        # 'upi-ravi kumar-ravik12@okaxis' (HDFC style): the handle is after the last hyphen
        handle = m.group(1).rsplit("-", 1)[-1] if len(m.group(1).rsplit("-", 1)[-1]) >= 3 else m.group(1)
        return handle[:12]
    m = re.search(r"/[a-z]{4}/([a-z0-9._-]*[a-z][a-z0-9._-]*)", desc)
    if m and len(m.group(1)) >= 3:
        return m.group(1)[:12]
    names = narration_names(desc)
    return _short_name(names[0]) if names else None


# Words in narrations that are payment plumbing, not names.
RAIL_WORDS = set("""upi upiab upiar upiint neft imps rtgs ach nach ecs p2a p2m mmt inb ib mb cr dr by to from
transfer trf payment paid pay received sent collect request refund reversal saving savings ok note bank
ltd pvt private limited bil billpay bharatpe paytm phonepe gpay googlepay ybl axl ibl okaxis oksbi okhdfcbank
okicici apl int pd sal salary chg charges fee""".split())
# IFSC bank codes that appear as path segments in UPI narrations.
BANK_CODES = set("""sbin hdfc icic utib kkbk ubin barb cnrb punb ioba idib bkid ucba mahb cbin yesb indb idfb
fdrl kvbl karb sibl jaka rbla psib dbss scbl citi hsbc aubl esfb ujvn airp pytm jsfb fino ippb""".split())


def narration_names(desc):
    """Person / business names in a narration: alphabetic path segments that are not
    payment-rail words or bank codes ('upi/ravi kumar s/1234/upi' -> ['ravi kumar s'])."""
    out = []
    for seg in re.split(r"[/\-:|]", desc.lower()):
        words = [w for w in re.findall(r"[a-z]+", seg) if w not in RAIL_WORDS]
        if not words or seg.strip() in BANK_CODES or re.search(r"\d{4,}|@", seg):
            continue
        if sum(len(w) for w in words) >= 4:
            out.append(" ".join(words))
    return out


def _short_name(name):
    """'ravi kumar s' and 'ravi ku' -> 'ravi ku' (banks truncate names differently)."""
    w = name.split()
    return (w[0] + (" " + w[1][:2] if len(w) > 1 else ""))[:12]


HONORIFICS = {"mr", "mrs", "ms", "miss", "shri", "smt", "kumari", "km", "dr", "m/s", "master", "sri", "kum"}


def name_tokens(name):
    """'Ms. Anaya V. Desai' -> ['anaya', 'v', 'desai']."""
    toks = re.findall(r"[a-z]+", str(name).lower()) if isinstance(name, str) else list(name)
    return [t for t in toks if t not in HONORIFICS]


def name_keys(name):
    """Privacy-preserving keys for comparing names: hashes of 'first last' and 'initial last'
    (statements often print initials). The names themselves are never stored."""
    t = name_tokens(name)
    if len(t) < 2:
        return set()
    h = lambda x: hashlib.sha256(x.encode()).hexdigest()[:16]   # noqa: E731
    return {h(f"{t[0]} {t[-1]}"), h(f"{t[0][0]} {t[-1]}")}


def own_account_mask(df, holder_names):
    """Rows that move money between the applicant's OWN accounts: the narration names
    the account holder (as printed on the statement), or a UPI handle built from the
    holder's first name and surname. Such credits are not income, such debits not spending."""
    if not holder_names or not len(df):
        return pd.Series(False, index=df.index)
    holders = [h for h in holder_names if len(h) >= 2]

    def own_name(name):
        w = name.split()
        if len(w) < 2:
            return False
        for h in holders:
            if w[0] == h[0] and any(x.startswith(w[1]) or w[1].startswith(x) for x in h[1:] if len(w[1]) >= 2):
                return True
            if w[0] == h[-1] and len(w[1]) >= 2 and h[0].startswith(w[1]):          # surname first
                return True
        return False

    def own_handle(desc):
        for m in re.finditer(r"([a-z][a-z0-9._-]{4,})@", desc.replace(" ", "")):
            for part in {m.group(1), m.group(1).rsplit("-", 1)[-1]}:      # 'upi-name-handle@bank'
                handle = re.sub(r"[^a-z]", "", part)
                if any(handle.startswith(h[0]) and h[-1] in handle[len(h[0]):] for h in holders):
                    return True
        return False

    def own(desc):
        # A salary / pension credit names the employee (the holder) as the beneficiary
        # ('NEFT CR-<IFSC>-ACME LTD-<HOLDER NAME>-SALARY'): it is income, never an own transfer.
        if re.search(SALARY_PATTERN, desc):
            return False
        # Only the COUNTERPARTY counts: the first name in the narration. NEFT / RTGS credits also
        # print the beneficiary -- the holder -- after the remitter, on every incoming transfer.
        names = narration_names(desc)
        return (bool(names) and own_name(names[0])) or own_handle(desc)

    return df["description"].map(own)


def classify_inflows(df, n_months):
    """Adds `inflow_type` to credit rows: salary / regular / other / excluded:<label>."""
    df["inflow_type"] = None
    credit = df["signed_amount"] > 0
    # Round-trip transfers: money received from someone the applicant also sends
    # back at least half as much to (family / friends passing money through).
    df["cp"] = df["description"].str.replace(" ", "", regex=False).map(upi_counterparty)  # IDs never contain spaces
    flows = df.dropna(subset=["cp"]).groupby("cp")["signed_amount"]
    received = flows.apply(lambda x: x[x > 0].sum())
    sent = flows.apply(lambda x: -x[x < 0].sum())
    circular = set(received[(received > 0) & (sent >= 0.5 * received)].index)
    # The money sent back is not spending either -- but only up to what came in from that party,
    # so rent paid to a landlord who once refunded a deposit still counts as spending.
    df["round_trip_out"] = False
    for cp in circular:
        budget = float(received[cp]) + 1.0
        for i in df.index[(df["cp"] == cp) & (df["signed_amount"] < 0)]:
            amt = -float(df.at[i, "signed_amount"])
            if amt <= budget:
                df.at[i, "round_trip_out"] = True
                budget -= amt
    if "own_account" in df.columns:
        df.loc[credit & df["own_account"], "inflow_type"] = "excluded:Transfers from own accounts"
    df.loc[credit & df["inflow_type"].isna() & df["cp"].isin(circular), "inflow_type"] = "excluded:Round-trip transfers"
    for label, pattern in EXCLUDED_INFLOWS.items():
        if pattern is None:
            continue
        hit = credit & df["inflow_type"].isna() & df["description"].str.contains(pattern, na=False, regex=True)
        df.loc[hit, "inflow_type"] = "excluded:" + label
    open_ = credit & df["inflow_type"].isna()
    df.loc[open_ & df["description"].str.contains(SALARY_PATTERN, na=False), "inflow_type"] = "salary"
    rest = df[open_ & df["inflow_type"].isna()].copy()
    if len(rest):
        rest["cp"] = rest["description"].map(_counterparty)
        need = max(2, int(np.ceil(0.6 * n_months)))
        for cp, g in rest.groupby("cp"):
            amounts = g["signed_amount"]
            cv = amounts.std() / amounts.mean() if len(amounts) > 1 and amounts.mean() > 0 else 1.0
            per_month = g.groupby("month").size().median()
            # regular = a salary-like payer: about one steady payment a month. A client paying
            # several times a month is business income, not a regular income source.
            if cp and g["month"].nunique() >= need and cv <= 0.25 and amounts.median() >= 1000 \
                    and per_month <= 2:
                df.loc[g.index, "inflow_type"] = "regular"
    df.loc[credit & df["inflow_type"].isna(), "inflow_type"] = "other"
    return df


def _clean_header(h):
    """'Withdrawal Amt.' / 'Withdrawal_Amt (INR)' -> 'withdrawal amt'."""
    h = re.sub(r"\(.*?\)", " ", str(h).lower())
    h = re.sub(r"[^a-z]+", " ", h)
    return " ".join(h.split())


def _normalize_columns(df):
    """Maps whatever headers the uploaded CSV has onto our canonical column names.
    The first matching column wins (e.g. 'Date' is preferred over a later 'Value Dt')."""
    col_map, used = {}, set()
    cleaned = {c: _clean_header(c) for c in df.columns}
    for canonical, variants in COLUMN_SYNONYMS.items():
        for col, name in cleaned.items():
            if name in variants and col not in used:
                col_map[col] = canonical
                used.add(col)
                break
    return df.rename(columns=col_map)


def _to_number(series):
    """Indian formats: '1,45,000.00', 'INR 500.00', 'Rs. 500', '1,200.00 Dr', '20.00(Dr)', '(500.00)',
    '500.00-', '-' for empty."""
    from pdf_statement import parse_money
    return pd.Series([parse_money(v)[0] if not isinstance(v, (int, float)) or pd.isna(v) else float(v)
                      for v in series], index=series.index, dtype="float64")


def _parse_dates(series):
    """ISO dates (2026-03-05) are parsed as ISO. Everything else is read
    day-first (05/03/26 = 5 March), the convention of Indian bank statements.
    Parsing day-first blindly would swap day/month in ISO dates (a bug found
    earlier), and parsing month-first would swap them in Indian dates."""
    s = series.astype(str).str.strip()
    iso = s.str.match(r"^\d{4}-\d{1,2}-\d{1,2}")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if iso.any():
        out[iso] = pd.to_datetime(s[iso].str[:10], format="%Y-%m-%d", errors="coerce")
    if (~iso).any():
        out[~iso] = pd.to_datetime(s[~iso], dayfirst=True, format="mixed", errors="coerce")
    return out


def parse_statement(filepath, password=None, progress=None, account_label="A1"):
    """
    Reads an uploaded bank statement CSV and returns:
      - a cleaned transaction-level DataFrame
      - a dict of universal behavioral features computed from it
      - a warnings list (e.g. insufficient duration)
    """
    warnings = []
    read_info = None
    holder_names = None
    if str(filepath).lower().endswith(".pdf"):
        from pdf_statement import read_pdf_statement
        raw = read_pdf_statement(filepath, password, progress)
        read_info = dict(raw.attrs.get("pdf_read", {}))
        holder_names = raw.attrs.pop("holder_names", None)        # in memory only
    else:
        raw = pd.read_csv(filepath)
    df = _normalize_columns(raw)

    required_any = {"date"}
    if not required_any.issubset(df.columns):
        raise ValueError("Could not find a recognizable date column in this statement. "
                          "Expected a header like 'Date', 'Txn Date', or 'Value Date'.")

    df["date"] = _parse_dates(df["date"])
    dropped = int(df["date"].isna().sum())
    if dropped:
        warnings.append(f"{dropped} row(s) had unreadable dates and were ignored.")
    df = df.dropna(subset=["date"])

    # Build a single signed 'amount' column: positive = credit (money in), negative = debit
    if "amount" in df.columns:
        df["signed_amount"] = _to_number(df["amount"]).fillna(0)
    elif "debit" in df.columns or "credit" in df.columns:
        zeros = pd.Series(0.0, index=df.index)
        debit = _to_number(df["debit"]).fillna(0).abs() if "debit" in df.columns else zeros
        credit = _to_number(df["credit"]).fillna(0).abs() if "credit" in df.columns else zeros
        df["signed_amount"] = credit - debit
    else:
        raise ValueError("Could not find debit/credit/amount columns in this statement.")

    if "description" not in df.columns:
        df["description"] = ""
    df["description"] = df["description"].fillna("").str.lower()

    if "balance" in df.columns:
        df["balance"] = _to_number(df["balance"])

    return features_from_transactions(df, read_info, holder_names, account_label, warnings)


def without_recent_large_credits(raw_df, large_credits, read_info, holder_names, account_label):
    """Features of the statement as if the recent unexplained large credits had never arrived."""
    df = raw_df.copy()
    if len(df) > 1 and df["date"].iloc[0] > df["date"].iloc[-1]:
        df = df.iloc[::-1]
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    for c in large_credits:
        if not c["last_30_days"]:
            continue
        hit = df.index[(df["date"].dt.date.astype(str) == c["date"]) & ((df["signed_amount"] - c["amount"]).abs() < 0.01)]
        if not len(hit):
            continue
        k = hit[0]
        if "balance" in df:
            df.loc[k + 1:, "balance"] = df.loc[k + 1:, "balance"] - c["amount"]
        df = df.drop(index=k).reset_index(drop=True)
    cdf, feats, _ = features_from_transactions(df, read_info, holder_names, account_label, _counterfactual=True)
    feats["_df"] = cdf
    return feats


def features_from_transactions(df, read_info=None, holder_names=None, account_label="A1", warnings=None,
                               _counterfactual=False):
    """Everything after reading: classify, analyse and summarise a statement's transactions.
    `df` needs date (datetime), description (lower case), signed_amount and, if known, balance.
    Shared by PDF / CSV statements, Account Aggregator data and the Berka backtest, so every
    route is scored by exactly the same rules."""
    warnings = list(warnings or [])
    raw_df, cf_names = df[[c for c in ("date", "description", "signed_amount", "balance") if c in df]].copy(), holder_names
    df["own_account"] = own_account_mask(df, holder_names)
    import hashlib
    holder_keys = sorted(set().union(*[name_keys(h) for h in (holder_names or [])])) if holder_names else []
    holder_names = None
    integrity = _balance_integrity(df)
    if len(df) > 1 and df["date"].iloc[0] > df["date"].iloc[-1]:
        df = df.iloc[::-1]          # newest-first export: restore time order within each day too
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    df["account"] = account_label
    df["month"] = df["date"].dt.to_period("M")

    n_months = df["month"].nunique()
    period = _statement_period(df, (read_info or {}).get("printed_period"))
    if n_months < 3:
        warnings.append(
            f"Only {n_months} month(s) of transactions found. Behavioral features "
            f"(volatility, consistency) are statistically unreliable below 3 months; "
            f"a 6-month statement is recommended."
        )

    features = _compute_features(df, n_months, period)
    features["balance_breaks"] = integrity
    features["transactions"] = len(df)
    features["read_info"] = read_info
    features["holder_keys"] = holder_keys          # compared across statements in memory; never stored
    features["accounts"] = 1
    from banking_analysis import analyse
    features["banking"] = analyse(df, features)
    if features["banking"].get("window_dressing") and not _counterfactual:
        # anti-gaming: the same statement WITHOUT the unexplained large credits of the last 30 days (and with the
        # balances they inflated reduced). The AI scores both and keeps the higher risk, so borrowed money parked
        # in the account just before applying can never make an applicant look safer.
        features["_without_unexplained_credits"] = without_recent_large_credits(
            raw_df, features["banking"]["large_credits"], read_info, cf_names, account_label)
    del cf_names
    if read_info and read_info.get("checked_rows") and read_info["reconciled"] < 0.98:
        warnings.append(f"Only {read_info['reconciled']:.0%} of this statement's rows reconcile with the running "
                        f"balance, so some figures may be misread. Please verify against the PDF.")
    return df, features, warnings


DAYS_PER_MONTH = 30.4375


def _statement_period(df, printed=None):
    """(start, end, source). The printed statement period when the PDF states one
    and the transactions fall inside it; otherwise first to last transaction date.
    Averages are taken over the period actually covered, so a statement running
    14 May - 4 Oct is 4.7 months, not "6 months" because it touches 6 calendar months."""
    first, last = df["date"].min().normalize(), df["date"].max().normalize()
    if printed:
        p = _parse_dates(pd.Series(printed))
        if p.notna().all() and p.iloc[0] <= first + pd.Timedelta(days=5) and \
                p.iloc[1] >= last - pd.Timedelta(days=5) and p.iloc[0] < p.iloc[1]:
            return p.iloc[0].normalize(), p.iloc[1].normalize(), "printed"
    return first, last, "transactions"


def _salary_day(dates):
    """Typical day of the month a regular income arrives (circular mean, so
    credits on the 30th and the 1st average to the turn of the month)."""
    angle = 2 * np.pi * (dates.dt.day - 1) / 30.0
    mean = np.arctan2(np.sin(angle).mean(), np.cos(angle).mean())
    return int(round((mean % (2 * np.pi)) / (2 * np.pi) * 30.0)) + 1


def _complete_cycles(start, end, shift_days, tol_days):
    """Cycle months (calendar months shifted by `shift_days`) that lie entirely
    inside the statement period."""
    shift = pd.Timedelta(days=shift_days)
    out = []
    for c in pd.period_range((start + shift).to_period("M"), (end + shift).to_period("M"), freq="M"):
        lo, hi = c.start_time - shift, c.end_time.normalize() - shift
        if lo >= start - pd.Timedelta(days=tol_days) and hi <= end + pd.Timedelta(days=tol_days):
            out.append(c)
    return pd.PeriodIndex(out, freq="M")


def _balance_integrity(df):
    """Number of rows whose running balance does NOT equal the neighbouring
    balance plus that row's amount (Rs 1 tolerance), in the file's own row
    order. Genuine exports chain perfectly; a hand-edited amount breaks it.
    Handles oldest-first and newest-first exports. None if no usable balance."""
    if "balance" not in df.columns or df["balance"].notna().sum() < 10:
        return None
    b, amt = df["balance"].reset_index(drop=True), df["signed_amount"].reset_index(drop=True)
    best = None
    for prev in (b.shift(1), b.shift(-1)):          # oldest-first, newest-first
        valid = b.notna() & prev.notna()
        if valid.sum():
            bad = int(((b - (prev + amt)).abs() > 1.0)[valid].sum())
            best = bad if best is None else min(best, bad)
    return best


def _compute_features(df, n_months, period=None):
    n_months = max(n_months, 1)  # avoid divide-by-zero on a single-month statement
    if period is None:
        period = _statement_period(df)
    start, end, period_source = period
    months_covered = max(((end - start).days + 1) / DAYS_PER_MONTH, 1 / DAYS_PER_MONTH)
    tol = 3   # a month missing at most 3 days at either end still counts as covered

    months = df["month"].unique()
    debits = df[df["signed_amount"] < 0]["signed_amount"].abs()

    # --- Income: classify every credit, then decide what counts ---
    df = classify_inflows(df, n_months)

    # Income is summed per INCOME CYCLE: calendar months shifted so the regular
    # credit (salary) falls mid-cycle. A salary paid on the 31st one month and the
    # 1st the next then still counts once per cycle, and only cycles the
    # statement fully covers are averaged (partial first/last months would
    # otherwise look like months with no salary).
    regular_rows = df["inflow_type"].isin(["salary", "regular"])
    counted_rows = df["inflow_type"].isin(["salary", "regular", "other"])

    def per_cycle(mask, shift):
        cycles = _complete_cycles(start, end, shift, tol)
        if len(cycles) < 2:   # under two complete cycles: one pooled average over the period covered
            return pd.Series([df.loc[mask, "signed_amount"].sum() / months_covered]), 0
        cyc = (df["date"] + pd.Timedelta(days=shift)).dt.to_period("M")
        return df[mask].groupby(cyc[mask])["signed_amount"].sum().reindex(cycles, fill_value=0.0), len(cycles)

    salary_shift = (15 - _salary_day(df.loc[regular_rows, "date"])) if regular_rows.any() else 0
    regular_m, n_cycles = per_cycle(regular_rows, salary_shift)
    counted_m, _ = per_cycle(counted_rows, salary_shift)
    excluded = {label: round(float(per_cycle(df["inflow_type"] == "excluded:" + label, 0)[0].mean()), 2)
                for label in EXCLUDED_INFLOWS}
    excluded = {k: v for k, v in excluded.items() if v > 0}

    # Lender practice: when regular income (salary or a steady recurring payer)
    # is the main source, only it is relied on; otherwise (self-employed, gig)
    # all genuine inflows count, per calendar month. Non-income inflows are never counted.
    if regular_m.mean() > 0 and regular_m.mean() >= 0.5 * counted_m.mean():
        income_m, income_basis = regular_m, "regular income"
    else:
        income_m, n_cycles = per_cycle(counted_rows, 0)
        income_basis = "all business / other inflows"
    avg_monthly_income = float(income_m.mean()) if len(income_m) else 0.0
    counted = df[df["inflow_type"].isin(["salary", "regular", "other"])]
    payer_ids = counted["cp"].where(counted["cp"].notna(), counted["description"].map(_counterparty))
    income_payers = int(payer_ids[payer_ids.astype(bool)].nunique())
    income_volatility = (income_m.std() / avg_monthly_income) if avg_monthly_income > 0 and len(income_m) > 1 else 0.0

    def cv(series):
        m = float(series.mean()) if len(series) else 0.0
        return round(float(series.std() / m), 4) if m > 0 and len(series) > 1 else 0.0
    all_m, _ = per_cycle(counted_rows, 0)
    # A salary that only STARTED during the statement (a new job): its stability and level are measured
    # from the first salary month on; the short salary history is flagged separately.
    lead = int(np.argmax(regular_m.values > 0)) if (regular_m > 0).any() else 0
    since = regular_m.iloc[lead:] if 0 < lead and len(regular_m) - lead >= 2 and (regular_m.iloc[lead:] > 0).all() \
        else None
    # both views are kept: the scoring picks the one that matches the declared employment
    income_views = {"regular_income_series": [round(float(v), 2) for v in regular_m],
                    "all_inflows_series": [round(float(v), 2) for v in all_m],
                    "regular_income_volatility": cv(regular_m),
                    # share of fully covered months with a salary / regular credit (salary-like = ~every month)
                    "regular_income_months_share": round(float((regular_m > 0).mean()), 3) if len(regular_m) else 0.0,
                    "avg_monthly_all_inflows": round(float(all_m.mean()), 2), "all_inflows_volatility": cv(all_m),
                    "salary_started_after_months": lead if since is not None else 0,
                    "regular_income_since_start": round(float(since.mean()), 2) if since is not None else None,
                    "regular_income_volatility_since_start": cv(since) if since is not None else None}

    # --- Spending (moving money to one's own savings/FD is not spending) ---
    investing = (df["signed_amount"] < 0) & df["description"].str.contains(INVESTMENT_PATTERN, na=False)
    own_debit = (df["signed_amount"] < 0) & (df["description"].str.contains(OWN_ACCOUNT_DEBIT, na=False)
                                             | df.get("own_account", False))
    spend_rows = (df["signed_amount"] < 0) & ~own_debit & ~investing & ~df["round_trip_out"]
    full_months = _complete_cycles(start, end, 0, tol)
    spend_months = full_months if len(full_months) >= 2 else months
    monthly_spend = df[spend_rows].groupby("month")["signed_amount"].sum().abs().reindex(spend_months, fill_value=0.0)
    avg_monthly_spend = float(df.loc[spend_rows, "signed_amount"].abs().sum() / months_covered)
    avg_monthly_invest = float(df.loc[investing, "signed_amount"].abs().sum() / months_covered)
    spending_volatility = (monthly_spend.std() / monthly_spend.mean()) \
        if len(monthly_spend) > 1 and monthly_spend.mean() > 0 else 0.0

    # --- Recurring debit detection (drives both bill-payment consistency and
    # the "existing credit" signal used for the consistency-flag check) ---
    debit_rows = df[df["signed_amount"] < 0].copy()
    debit_rows["amount_bucket"] = (debit_rows["signed_amount"].abs() / 10).round() * 10
    recurring_groups = debit_rows.groupby(["amount_bucket"])["month"].nunique()
    recurring_groups = recurring_groups[recurring_groups >= max(3, int(n_months * 0.6))]
    bill_payment_consistency = min(1.0, len(recurring_groups) / max(1, n_months / 2))

    credit_like_debits = debit_rows[is_credit_product_payment(debit_rows["description"])]
    detected_existing_credit = False
    detected_credit_count = 0
    if len(credit_like_debits):
        credit_recurrence = credit_like_debits.groupby("amount_bucket")["month"].nunique()
        qualifying = credit_recurrence[credit_recurrence >= min(3, n_months)]
        detected_existing_credit = len(qualifying) > 0
        detected_credit_count = len(qualifying)
        home = credit_like_debits[credit_like_debits["amount_bucket"].isin(qualifying.index)
                                  & credit_like_debits["description"].str.contains(HOME_LOAN_PATTERN, na=False)]
        detected_property_loans = int(home["amount_bucket"].nunique())
    else:
        detected_property_loans = 0

    # --- Overdraft ---
    if "balance" in df.columns and df["balance"].notna().any():
        overdraft_frequency = (df["balance"] < 0).mean()
        avg_balance = df["balance"].mean()
        min_balance = df["balance"].min()
        max_balance = df["balance"].max()
    else:
        overdraft_frequency, avg_balance, min_balance, max_balance = None, None, None, None

    # --- Betting / essential spend ratios (best-effort from description keywords) ---
    total_debit = debits.sum() if len(debits) else 0
    betting_debits = debit_rows[debit_rows["description"].str.contains(BETTING_PATTERN, na=False)]
    betting_ratio = (betting_debits["signed_amount"].abs().sum() / total_debit) if total_debit > 0 else 0.0

    essential_debits = debit_rows[debit_rows["description"].str.contains(ESSENTIAL_PATTERN, na=False)]
    essential_spend_ratio = (essential_debits["signed_amount"].abs().sum() / total_debit) if total_debit > 0 else None

    # --- Household bills: in how many of the (complete) months was a bill paid? ---
    bill = (debit_rows["description"].str.contains(BILL_PATTERN, na=False, regex=True)
            & ~is_credit_product_payment(debit_rows["description"]).fillna(False))
    bill_month_set = set(debit_rows.loc[bill, "month"])
    bill_months = sum(1 for m in spend_months if m in bill_month_set)
    bill_months_share = bill_months / len(spend_months) if len(spend_months) else 0.0

    total_in = float(df.loc[df["signed_amount"] > 0, "signed_amount"].sum())
    own_in = float(df.loc[(df["signed_amount"] > 0) & (df["inflow_type"] == "excluded:Transfers from own accounts"),
                          "signed_amount"].sum())
    round_trip_in = float(df.loc[(df["signed_amount"] > 0) & (df["inflow_type"] == "excluded:Round-trip transfers"),
                                 "signed_amount"].sum())
    features = {
        # whole months of history actually covered (14 May - 4 Oct = 4.7 -> 5), not calendar months touched
        "statement_months": int(round(months_covered)),
        "months_covered": round(months_covered, 2),
        "period_start": start.date().isoformat(), "period_end": end.date().isoformat(),
        "period_source": period_source,
        "income_cycles_averaged": n_cycles,
        "avg_monthly_income": round(avg_monthly_income, 2),
        "income_basis": income_basis,
        "avg_monthly_regular_income": round(float(regular_m.mean()), 2),
        "avg_monthly_counted_inflows": income_views["avg_monthly_all_inflows"],
        **income_views,
        "excluded_inflows": excluded,
        "income_payers": income_payers,
        "own_transfers_in_share": round(own_in / max(total_in, 1.0), 4),
        # money received from people the applicant sent most of it back to, as a share of all
        # money received from others (a pass-through / accommodation-entry account)
        "round_trip_in_share": round(round_trip_in / max(total_in - own_in, 1.0), 4),
        "bill_months": int(bill_months), "bill_months_of": int(len(spend_months)),
        "bill_months_share": round(float(bill_months_share), 3),
        "income_volatility": round(income_volatility, 4),
        "avg_monthly_spend": round(avg_monthly_spend, 2),
        "avg_monthly_investments": round(avg_monthly_invest, 2),
        "spending_volatility": round(spending_volatility, 4),
        "transaction_frequency": round(len(df) / n_months, 2),
        "bill_payment_consistency": round(bill_payment_consistency, 4),
        "detected_existing_credit": detected_existing_credit,
        "detected_credit_signal_count": detected_credit_count,
        "detected_property_loans": detected_property_loans,
        "overdraft_frequency": round(overdraft_frequency, 4) if overdraft_frequency is not None else None,
        "avg_balance": round(avg_balance, 2) if avg_balance is not None else None,
        "min_balance": round(min_balance, 2) if min_balance is not None else None,
        "max_balance": round(max_balance, 2) if max_balance is not None else None,
        "betting_ratio": round(betting_ratio, 4),
        "essential_spend_ratio": round(essential_spend_ratio, 4) if essential_spend_ratio is not None else None,
    }
    return features


if __name__ == "__main__":
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample_statement.csv"
    df, features, warnings = parse_statement(filepath)
    print("Parsed", len(df), "transactions across", features["statement_months"], "months\n")
    for k, v in features.items():
        print(f"{k}: {v}")
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(" -", w)

# ---------------------------------------------------------------------------
# Several statements (all of an applicant's accounts) in one assessment
# ---------------------------------------------------------------------------
ORIGIN_RANK = {"edited": 3, "re-saved": 2, "unknown": 1, "bank-generated": 0}


def match_internal_transfers(df, days=3):
    """A debit in one account and a credit of the same amount in another account of
    the same applicant within a few days is money moving between their own accounts."""
    df = df.copy()
    internal = pd.Series(False, index=df.index)
    deb = df[(df["signed_amount"] < 0)].copy()
    cre = df[(df["signed_amount"] > 0)].copy()
    used = set()
    for i, d in deb.iterrows():
        cand = cre[(cre["account"] != d["account"]) & ((cre["signed_amount"] + d["signed_amount"]).abs() <= 1.0)
                   & ((cre["date"] - d["date"]).dt.days.abs() <= days)]
        cand = cand[~cand.index.isin(used)]
        if len(cand):
            j = (cand["date"] - d["date"]).abs().idxmin()
            used.add(j)
            internal[i] = internal[j] = True
    df["own_account"] = df["own_account"] | internal
    return df, int(internal.sum() // 2)


def combine_statements(parsed):
    """`parsed`: list of (df, features, warnings) from parse_statement, one per account.
    Returns one combined (df, features, warnings): transfers between the accounts are
    neither income nor spending, balances are added up day by day, and every figure is
    taken over the period that ALL statements cover."""
    if len(parsed) == 1:
        return parsed[0]
    warnings = [w for _, _, ws in parsed for w in ws]
    # the same account uploaded twice (e.g. as a PDF and as Account Aggregator data) would count twice
    sigs = [set(zip(d["date"].dt.date, d["signed_amount"].round(2), d["balance"].round(2)))
            if "balance" in d else set() for d, _, _ in parsed]
    for i in range(len(sigs)):
        for j in range(i + 1, len(sigs)):
            if sigs[i] and sigs[j] and len(sigs[i] & sigs[j]) >= 0.8 * min(len(sigs[i]), len(sigs[j])):
                raise ValueError("Two of the uploaded statements are the same account: upload each account once.")
    keys = [set(st.get("holder_keys") or []) for _, st, _ in parsed]
    known = [k for k in keys if k]
    different_holders = len(known) >= 2 and not set.intersection(*known)
    starts = [pd.Timestamp(st["period_start"]) for _, st, _ in parsed]
    ends = [pd.Timestamp(st["period_end"]) for _, st, _ in parsed]
    start, end = max(starts), min(ends)
    if end <= start:
        raise ValueError("These statements do not cover a common period, so they cannot be assessed together.")
    if (start - min(starts)).days > 31 or (max(ends) - end).days > 31:
        warnings.append(f"The statements cover different periods; figures are taken over the common period "
                        f"{start:%d %b %Y} - {end:%d %b %Y}.")
    # same 3-day tolerance as for a single statement (the first / last transaction is
    # rarely on the first / last day of the printed period)
    start = max(start - pd.Timedelta(days=3), min(starts))
    end = min(end + pd.Timedelta(days=3), max(ends))
    df = pd.concat([d for d, _, _ in parsed], ignore_index=True)
    df = df.sort_values(["date"], kind="stable").reset_index(drop=True)
    df, n_internal = match_internal_transfers(df)
    window = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
    n_months = window["month"].nunique()
    features = _compute_features(window, n_months, (start, end, "combined"))
    infos = [st.get("read_info") or {} for _, st, _ in parsed]
    breaks = [st.get("balance_breaks") for _, st, _ in parsed]
    features["balance_breaks"] = None if any(b is None for b in breaks) else int(sum(breaks))
    features["transactions"] = int(len(window))
    recs = [i.get("reconciled") for i in infos if i.get("reconciled") is not None]
    origins = [(i.get("file_origin") or {}) for i in infos]
    worst = max(origins, key=lambda o: ORIGIN_RANK.get(o.get("category"), 1)) if origins else {}
    features["read_info"] = {"method": " + ".join(str(i.get("method")) for i in infos),
                             "reconciled": min(recs) if recs else None,
                             "checked_rows": sum(i.get("checked_rows") or 0 for i in infos),
                             "file_origin": worst, "printed_period": None,
                             "tamper": {"font_outliers": sum((i.get("tamper") or {}).get("font_outliers", 0) for i in infos),
                                        "font_outlier_rows": [r for i in infos for r in
                                                              (i.get("tamper") or {}).get("font_outlier_rows", [])][:10],
                                        "dominant_font_share": min([(i.get("tamper") or {}).get("dominant_font_share")
                                                                    for i in infos
                                                                    if (i.get("tamper") or {}).get("dominant_font_share")
                                                                    is not None] or [None], key=lambda v: v or 1),
                                        "summary": [x for i in infos for x in ((i.get("tamper") or {}).get("summary")
                                                                               or [])] or None}}
    features["accounts"] = len(parsed)
    features["internal_transfers"] = n_internal
    features["different_holders"] = different_holders
    features["account_summaries"] = [
        {"account": d["account"].iloc[0], "period": [st["period_start"], st["period_end"]],
         "transactions": st["transactions"], "reconciled": (st.get("read_info") or {}).get("reconciled"),
         "origin": ((st.get("read_info") or {}).get("file_origin") or {}).get("category")}
        for d, st, _ in parsed]
    features["holder_keys"] = sorted(set().union(*keys)) if keys else []
    from banking_analysis import analyse
    features["banking"] = analyse(window, features)
    return window, features, warnings
