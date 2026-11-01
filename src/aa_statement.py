"""
Account Aggregator (RBI AA framework) bank-account data as input.

Under India's Account Aggregator framework a customer consents in their AA app
and the bank (the FIP) sends the account data to the lender (the FIU) in the
ReBIT "deposit" schema -- digitally signed by the bank, so it cannot be edited
the way a PDF can, and no reading of layouts is needed. This module turns that
data (JSON, or the equivalent XML) into the same transaction table the PDF
reader produces, so it is scored by exactly the same rules.

Accepted shapes: the decrypted FI data of one or more deposit accounts --
{"Account": {...}}, a list of those, or an AA data response that nests them
(e.g. {"FI": [{"data": [{"decryptedFI": {"account": {...}}}]}]}). Field names
are matched case-insensitively (the schema writes Account / account,
Transactions / transactions, currentBalance / currentbalance ...).

The holder's name (Profile > Holders > Holder > name) is used only in memory to
recognise transfers between the applicant's own accounts; it is never stored.
"""
import json
import re
import xml.etree.ElementTree as ET

import pandas as pd


class AAFormatError(ValueError):
    """Raised with a message that can be shown to the user as-is."""


def _ci(d, *names):
    """Case-insensitive dict lookup (first match)."""
    if not isinstance(d, dict):
        return None
    low = {k.lower(): v for k, v in d.items()}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def _as_list(x):
    return x if isinstance(x, list) else [] if x is None else [x]


def _xml_to_dict(el):
    """ReBIT XML: attributes become keys, child elements become (lists of) dicts."""
    d = dict(el.attrib)
    for child in el:
        tag = child.tag.split("}")[-1]
        val = _xml_to_dict(child)
        if tag in d:
            d[tag] = _as_list(d[tag]) + [val]
        else:
            d[tag] = val
    return d


def _find_accounts(obj):
    """Every deposit-account object anywhere in the document."""
    found = []
    if isinstance(obj, dict):
        acc = _ci(obj, "Account")
        if isinstance(acc, dict) and _ci(acc, "Transactions") is not None:
            found.append(acc)
        elif _ci(obj, "Transactions") is not None and _ci(obj, "Summary", "Profile") is not None:
            found.append(obj)
        else:
            for v in obj.values():
                found += _find_accounts(v)
    elif isinstance(obj, list):
        for v in obj:
            found += _find_accounts(v)
    return found


def load(raw_bytes):
    """bytes of a JSON or XML AA document -> list of account dicts."""
    text = raw_bytes.decode("utf-8-sig", errors="replace").strip()
    if text[:1] not in "{[" and re.search(r"<!(?:DOCTYPE|ENTITY)", text[:4096], re.I):
        # ReBIT AA data never needs a DTD; refusing one closes entity-expansion attacks (billion laughs, XXE)
        raise AAFormatError("This Account Aggregator file could not be read (DTD / entities are not allowed).")
    try:
        doc = json.loads(text) if text[:1] in "{[" else {"Account": _xml_to_dict(ET.fromstring(text))}
        accounts = _find_accounts(doc)
    except (json.JSONDecodeError, ET.ParseError, RecursionError, ValueError) as exc:
        # RecursionError: absurdly deep nesting (a hostile file); a real AA document is a few levels deep
        raise AAFormatError(f"This Account Aggregator file could not be read ({exc.__class__.__name__}).") from exc
    if not accounts:
        raise AAFormatError("No deposit account with transactions was found in this Account Aggregator file.")
    return accounts


def _num(v):
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None


def to_transactions(account):
    """One AA deposit account -> (DataFrame[date, description, signed_amount, balance], info, holder_names)."""
    tx_block = _ci(account, "Transactions") or {}
    txs = _as_list(_ci(tx_block, "Transaction"))
    rows = []
    for t in txs:
        amount = _num(_ci(t, "amount"))
        kind = str(_ci(t, "type") or "").upper()
        if amount is None or kind not in ("CREDIT", "DEBIT"):
            continue
        when = _ci(t, "transactionTimestamp", "valueDate")
        mode = str(_ci(t, "mode") or "").upper()
        narration = str(_ci(t, "narration") or "")
        if mode == "CASH" and "cash" not in narration.lower():
            narration = f"CASH {'DEPOSIT' if kind == 'CREDIT' else 'WITHDRAWAL'} {narration}".strip()
        rows.append({"date": pd.to_datetime(str(when)[:10], errors="coerce"), "description": narration.lower(),
                     "signed_amount": abs(amount) if kind == "CREDIT" else -abs(amount),
                     "balance": _num(_ci(t, "currentBalance", "balance")), "_ts": str(when)})
    if len(rows) < 3:
        raise AAFormatError("The Account Aggregator data has fewer than 3 transactions.")
    df = pd.DataFrame(rows).dropna(subset=["date"]).sort_values("_ts", kind="stable").drop(columns="_ts")
    start, end = _ci(tx_block, "startDate"), _ci(tx_block, "endDate")
    holders = _as_list(_ci(_ci(_ci(account, "Profile") or {}, "Holders") or {}, "Holder"))
    names = []
    for h in holders:
        n = _ci(h, "name")
        if n:
            toks = [t for t in re.findall(r"[a-z]+", n.lower()) if len(t) >= 2]
            if len(toks) >= 2:
                names.append(toks)
    info = {"method": "Account Aggregator (bank-signed data)",
            "printed_period": [str(start)[:10], str(end)[:10]] if start and end else None,
            "file_origin": {"category": "bank-generated", "producer": "Account Aggregator FIP"},
            "masked_account": _ci(account, "maskedAccNumber")}
    return df.reset_index(drop=True), info, names


def parse_aa(raw_bytes):
    """-> list of (df, features, warnings), one per account, scored by the standard pipeline."""
    from statement_parser import features_from_transactions
    out = []
    for i, acc in enumerate(load(raw_bytes)):
        df, info, names = to_transactions(acc)
        if df["balance"].notna().sum() >= 2:
            b, amt = df["balance"].reset_index(drop=True), df["signed_amount"].reset_index(drop=True)
            ok = ((b - (b.shift(1) + amt)).abs() <= 1.0)[b.shift(1).notna()]
            info["reconciled"], info["checked_rows"] = round(float(ok.mean()), 4), int(len(ok))
        out.append(features_from_transactions(df, info, names, account_label=f"AA{i + 1}"))
        del names
    return out
