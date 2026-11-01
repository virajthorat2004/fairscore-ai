"""
Reads the transactions out of an Indian bank-statement PDF (text e-statements
from net banking; password-protected ones supported).

Real statements differ a lot between banks (ruled or unruled tables, one or two
date columns, separate Debit/Credit columns or one Amount with Dr/Cr, "INR"
prefixes, "-" for empty cells, narrations wrapped over several lines, summary
pages before the table, totals after it, newest-first ordering ...). Column
headings are not reliable either. So the reader does not depend on them:

 1. Three independent extraction strategies produce candidate tables:
      ruled   - pdfplumber's cell grid (ruled tables)
      layout  - words placed into columns by the heading positions (unruled tables)
      lines   - every text line that starts with a date; amounts are the
                numbers at the end of the line (works without any headings)
 2. Column roles are inferred from CONTENT: the date column is the one full of
    dates, money columns are the ones full of amounts, the narration is the
    longest text column.
 3. Which money column is debit, credit or balance (and the sign of a single
    Amount column) is decided by the RUNNING BALANCE: only the right assignment
    makes every row satisfy  balance = previous balance + credit - debit.
 4. The candidate that reconciles best wins. Its reconciliation rate is
    reported, so a statement the system could not read reliably is never
    silently scored.
"""
import itertools
import re

import numpy as np
import pandas as pd

_D_NUM = r"(?P<d1>\d{1,2})(?P<s1>[/\-.])(?P<m1>\d{1,2})(?P=s1)(?P<y1>\d{4}|\d{2})"        # 01/03/2026
_D_NAME = r"(?P<d2>\d{1,2})[ \-.](?P<m2>[A-Za-z]{3,9})[ \-.,]{1,2}(?P<y2>\d{4}|\d{2})"     # 14 May 2026
_D_ISO = r"(?P<y3>\d{4})-(?P<m3>\d{1,2})-(?P<d3>\d{1,2})"                                 # 2026-03-01
_D_US = r"(?P<m4>[A-Za-z]{3,9})[ .\-](?P<d4>\d{1,2}),?[ \-](?P<y4>\d{4})"                   # Apr 01, 2026
DATE_RE = re.compile(rf"^(?:{_D_NUM}|{_D_NAME}|{_D_ISO}|{_D_US})(?=\s|$|T\d)")
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
                                      "nov", "dec"], 1)}
PLACEHOLDER = {"-", "--", "—", "–", "na", "n/a", "nil", "0.00-"}
CURRENCY = re.compile(r"(?i)(\binr\b|\brs\b\.?|₹)")
MARKER_DR = re.compile(r"(?i)(\(\s*dr\.?\s*\)|\bdr\.?)\s*$")
MARKER_CR = re.compile(r"(?i)(\(\s*cr\.?\s*\)|\bcr\.?)\s*$")
NUMBER = re.compile(r"^-?\d+(\.\d{1,2})?-?$")
OPENING = re.compile(r"(?i)opening balance|b/f|brought forward|balance forward|balance b/f")
LINE_TOL = 3.0
ALL_HEADINGS = set()
# Printed statement period, e.g. "For period: 01 May 2026 - 12 Oct 2026",
# "Statement Period 01-04-2026 to 30-09-2026", "from 01/04/2026 to 30/09/2026".
_D_ANY = (r"\d{1,2}[/\-. ](?:\d{1,2}|[A-Za-z]{3,9})[/\-. ,]{1,2}\d{2,4}|\d{4}-\d{1,2}-\d{1,2}"
          r"|[A-Za-z]{3,9}\.? \d{1,2},? \d{4}")                       # 'Sep 01, 2026' (Bank of Baroda)
PERIOD = re.compile(rf"(?i)(?:period|from|between)\s*(?:of)?\s*[:\-]?\s*({_D_ANY})\s*(?:to|till|upto|-|–|and)\s*[:\-]?\s*({_D_ANY})")
# a bare range on its own in the page header, e.g. Kotak's "12 Apr 2026 - 12 Oct 2026"
BARE_PERIOD = re.compile(rf"^\s*({_D_ANY})\s*(?:to|-|–)\s*({_D_ANY})\s*$")

# Account-holder name in the page-1 header. Used ONLY in memory to recognise
# transfers between the applicant's own accounts (not income, not spending);
# never stored or shown.
NAME_LABELS = re.compile(r"(?i)^(account holder name|account holder|customer name|account name|a/c name|name)\s*:?\s*")
HONORIFICS = {"mr", "mrs", "ms", "miss", "shri", "smt", "kumari", "km", "dr", "m/s", "master"}
NOT_NAME = set("""road rd marg nagar lane street st west east north south near opp opposite behind building bldg
tower floor flat room plot sector society chs apartment apartments apt colony village taluka district dist city
mumbai delhi pune bangalore bengaluru chennai kolkata hyderabad thane navi india maharashtra karnataka gujarat
state bank branch account statement savings saving current summary details detail customer address mobile email
phone ifsc micr nominee period date currency inr rupee rupees indian page balance transaction transactions joint
holder type status active scheme product limited ltd private pvt cooperative your our the and of for to from as on
in with relationship assets liabilities total opening closing union kotak mahindra baroda hdfc icici axis sbi canara
punjab national federal idfc indusind central overseas bandhan rbl standard chartered citi first yes uco cif crn
generated report registered nomination available ckyc kyc customer service care helpline welcome dear tel fax
wadi naka chawl pada gaon peth galli gali bazar chowk nr my information accounts term deposits
cheque cash deposit withdrawal debit credit amount value valid upto charges interest rate
chq no particulars init br txn ref narration description remarks loan loans home emi dt sl sr serial code mode
main instr post val wdl dep bal sanctioned outstanding overdraft limit drawing power od cc""".split())


def _holder_names(page_words):
    """Candidate account-holder names (lists of lower-case words) from page 1 above the table."""
    lines = _lines(page_words[0]) if page_words else []
    out = []
    from statement_parser import COLUMN_SYNONYMS, _clean_header
    headings = set(sum(COLUMN_SYNONYMS.values(), []))
    for ln in lines[:45]:
        if _is_date(ln["text"]) and any(parse_money(w["text"])[2] == "money" for w in ln["words"]):
            break                                                    # the transactions have started
        if sum(_clean_header(c["text"]) in headings for c in _header_cells(ln["words"])) >= 2:
            break                                                    # the table heading: names come before it
        for seg in _header_cells(ln["words"], gap=14.0):
            text = NAME_LABELS.sub("", seg["text"].strip())
            toks = [t.strip(".,").lower() for t in text.replace(",", " ").split()]
            toks = [t for t in toks if t not in HONORIFICS]
            if 2 <= len(toks) <= 5 and all(t.isalpha() and len(t) >= 2 for t in toks) \
                    and not any(t in NOT_NAME for t in toks) and toks not in out:
                out.append(toks)
    return out[:4]


class StatementPDFError(ValueError):
    """Raised with a message that can be shown to the user as-is."""


class PasswordRequired(StatementPDFError):
    pass


# ---------------------------------------------------------------------------
# Opening the file
# ---------------------------------------------------------------------------
def _open(path, password):
    import pdfplumber
    try:
        pdf = pdfplumber.open(path, password=password or "")
        pdf.pages[0].extract_words()            # forces decryption
        return pdf
    except Exception as exc:                    # pdfminer raises several types here
        text = f"{type(exc).__name__} {exc!r}".lower()
        if "password" in text or "encrypt" in text:
            raise PasswordRequired(
                "This PDF is password-protected. Enter its password (banks often use your date of birth, "
                "or part of your name / PAN / customer ID) and try again." if not password else
                "The PDF password is incorrect. Please check it and try again.") from exc
        raise StatementPDFError(f"This file could not be opened as a PDF ({type(exc).__name__}).") from exc


# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------
def _clean(c):
    return " ".join(str(c or "").split())


def parse_money(cell):
    """'INR 1,500.00' / '1,500.00(Dr)' / '2,000.00 Cr' / '(500.00)' / '500.00-' ->
    (value, marker, kind) where kind is 'money', 'empty' (a '-' placeholder) or 'text'."""
    s = _clean(cell)
    if not s or s.lower() in PLACEHOLDER:
        return None, None, "empty"
    marker = "dr" if MARKER_DR.search(s) else "cr" if MARKER_CR.search(s) else None
    t = MARKER_CR.sub("", MARKER_DR.sub("", s))
    t = CURRENCY.sub("", t).replace(",", "").replace(" ", "")
    neg = False
    if t.startswith("(") and t.endswith(")"):
        t, neg = t[1:-1], True
    t = t.strip("+")
    if t.endswith("-"):
        t, neg = t[:-1], True
    if not NUMBER.match(t) or "." not in t:
        return None, None, "text"
    v = float(t)
    if neg or marker == "dr":
        v = -abs(v)
    return v, marker, "money"


def _is_date(cell):
    """A complete, plausible calendar date at the start of the cell."""
    m = DATE_RE.match(_clean(cell))
    if not m:
        return False
    g = m.groupdict()
    day = g["d1"] or g["d2"] or g["d3"] or g["d4"]
    mon = g["m1"] or g["m2"] or g["m3"] or g["m4"]
    month = int(mon) if mon.isdigit() else MONTHS.get(mon[:3].lower(), 0)
    return 1 <= int(day) <= 31 and 1 <= month <= 12



# ---------------------------------------------------------------------------
# Wrapped narration lines: which transaction does a text-only line belong to?
# ---------------------------------------------------------------------------
BOILERPLATE = re.compile(r"(?i)\bpage\s*\d+\s*(of|/|\|)\s*\d+|https?://|www\.|customer care|helpline|toll[- ]free"
                         r"|computer generated|system generated|does not require (a )?signature|statement of transactions"
                         r"|generated on|registered office|\bcin\b|continued|statement\s*summary|\bdr\.?\s*count\b")
MARKERS = {"cr", "dr", "cr.", "dr.", "(cr)", "(dr)"}


def _has_text(cells):
    """True if a row has words other than dates, amounts and Cr/Dr markers."""
    for c in cells:
        for tok in str(c).split():
            if tok.lower() in MARKERS or parse_money(tok)[2] in ("money", "empty") or _is_date(tok):
                continue
            if re.search(r"[A-Za-z@/]", tok):
                return True
    return False


def _place_wrapped(pages, narration_cells):
    """`pages`: per page, a list of entries {top, kind: tx|text|other, row} in reading order.
    Returns the rows in order with every text line placed straight after the transaction
    it belongs to, and page headers / footers / stray lines dropped.

    Two layouts exist. Most banks print the date and amounts on the FIRST line of a
    narration and wrap below it. Some (e.g. Bank of Baroda) centre the date and amounts
    vertically, so part of the narration sits ABOVE the date line and the date line has
    no narration of its own. That layout is recognised when most date lines carry no
    narration; each text line then goes to the nearest date line."""
    txs = [e for pg in pages for e in pg if e["kind"] == "tx"]
    if not txs:
        return []
    centred = sum(not _has_text(narration_cells(e["row"])) for e in txs) >= 0.5 * len(txs) and \
        any(e["kind"] == "text" for pg in pages for e in pg)
    out = []
    for pg in pages:
        tops = sorted(e["top"] for e in pg)
        gaps = sorted(b - a for a, b in zip(tops, tops[1:]) if b - a > 0.5)
        pitch = gaps[len(gaps) // 2] if gaps else 12.0
        owner = {}
        tx_idx = [i for i, e in enumerate(pg) if e["kind"] == "tx"]
        last_kept = None
        for i, e in enumerate(pg):
            if e["kind"] != "text":
                if e["kind"] == "tx":
                    last_kept = e["top"]
                continue
            text = " ".join(str(c) for c in e["row"] if c)
            if BOILERPLATE.search(text):
                continue
            if centred:
                near = min(tx_idx, key=lambda j: (abs(pg[j]["top"] - e["top"]), j > i), default=None)
                if near is not None and abs(pg[near]["top"] - e["top"]) <= 1.6 * pitch:
                    owner.setdefault(near, []).append(i)
            else:
                prev = max((j for j in tx_idx if j < i), default=None)
                if prev is not None and last_kept is not None and e["top"] - last_kept <= 2.2 * pitch:
                    owner.setdefault(prev, []).append(i)
                    last_kept = e["top"]
        for i, e in enumerate(pg):
            if e["kind"] == "text":
                continue
            out.append(e["row"])
            for j in sorted(owner.get(i, [])):
                out.append(pg[j]["row"])
    return out

# ---------------------------------------------------------------------------
# Word spacing
# ---------------------------------------------------------------------------
def _x_tolerance(pdf):
    """Some statement generators (HDFC's, for one) print no space characters at all: words are
    separated only by a small gap (about 2 pt), which pdfplumber's default tolerance of 3 pt merges
    ('INTERESTPAIDTILL', 'MRFIRSTLAST'). For such a document the tolerance is set below the typical
    gap between words, so the words come apart again. Documents with real spaces keep the default.
    Decided once per document (some pages hold few multi-word narrations)."""
    cached = getattr(pdf, "_fairscore_xtol", None)
    if cached is not None:
        return cached
    tol, gaps = 3, []
    for page in pdf.pages:
        chars = page.chars
        if any(c["text"] == " " for c in chars):
            gaps = None
            break
        gaps += [b["x0"] - a["x1"] for a, b in zip(chars, chars[1:])
                 if abs(a["top"] - b["top"]) < 1 and 0.3 < b["x0"] - a["x1"] < 3.0]
    if gaps and len(gaps) >= 10:
        tol = float(min(max(0.6 * float(np.median(gaps)), 0.5), 2.5))
    try:
        pdf._fairscore_xtol = tol
    except AttributeError:
        pass
    return tol


def _page_words(page, pdf):
    return page.extract_words(x_tolerance=_x_tolerance(pdf))


# ---------------------------------------------------------------------------
# Strategy 1: ruled tables
# ---------------------------------------------------------------------------
def _grid_ruled(pdf):
    by_width = {}
    for page in pdf.pages:
        for table in page.extract_tables({"text_x_tolerance": _x_tolerance(pdf)}):
            for raw in table:
                cells = [_clean(c) for c in raw]
                if any(cells):
                    by_width.setdefault(len(cells), []).append(cells)
    if not by_width:
        return None
    # the transaction table is the width with the most date-led rows
    best = max(by_width.values(), key=lambda rows: sum(any(_is_date(c) for c in r[:3]) for r in rows))
    return best if sum(any(_is_date(c) for c in r[:3]) for r in best) >= 3 else None


# ---------------------------------------------------------------------------
# Strategy 2: unruled table, columns from the heading positions
# ---------------------------------------------------------------------------
def _lines(words):
    lines = []
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        if lines and abs(lines[-1]["top"] - w["top"]) <= LINE_TOL:
            lines[-1]["words"].append(w)
        else:
            lines.append({"top": w["top"], "words": [w]})
    for ln in lines:
        ln["words"].sort(key=lambda w: w["x0"])
        ln["text"] = " ".join(w["text"] for w in ln["words"])
    return lines


def _header_cells(words, gap=6.0):
    cells = []
    for w in sorted(words, key=lambda w: w["x0"]):
        part = (round(w["top"]), w["x0"], w["text"])
        if cells and w["x0"] - cells[-1]["x1"] <= gap:
            cells[-1]["parts"].append(part)
            cells[-1]["x1"] = max(cells[-1]["x1"], w["x1"])
        else:
            cells.append({"parts": [part], "x0": w["x0"], "x1": w["x1"]})
    for c in cells:
        c["text"] = " ".join(t for _, _, t in sorted(c["parts"]))
    return cells


def _header_cells_by_names(words, known, clean, gap=6.0):
    """Headings printed too close together to split by spacing ('Cheque No.Dr Amount'):
    cut the line at known heading phrases (longest first), merge the rest by spacing."""
    ws = []
    firsts = {k.split()[0] for k in known if len(k.split()[0]) >= 5}
    for w in sorted(words, key=lambda w: w["x0"]):
        low = w["text"].lower()
        cut = next((f for f in sorted(firsts, key=len, reverse=True)
                    if low.endswith(f) and len(low) > len(f) and clean(w["text"]) not in known), None)
        if cut:                                   # 'NoWithdrawal': two headings printed touching
            k = len(w["text"]) - len(cut)
            xm = w["x0"] + (w["x1"] - w["x0"]) * k / len(w["text"])
            ws += [dict(w, text=w["text"][:k], x1=xm), dict(w, text=w["text"][k:], x0=xm + 0.01)]
        else:
            ws.append(w)
    cells, i = [], 0
    while i < len(ws):
        for k in (4, 3, 2, 1):
            if i + k <= len(ws) and clean(" ".join(w["text"] for w in ws[i:i + k])) in known:
                part = ws[i:i + k]
                cells.append({"parts": [(round(w["top"]), w["x0"], w["text"]) for w in part],
                              "x0": part[0]["x0"], "x1": part[-1]["x1"], "known": True})
                i += k
                break
        else:
            w = ws[i]
            if cells and not cells[-1].get("known") and w["x0"] - cells[-1]["x1"] <= gap:
                cells[-1]["parts"].append((round(w["top"]), w["x0"], w["text"]))
                cells[-1]["x1"] = max(cells[-1]["x1"], w["x1"])
            else:
                cells.append({"parts": [(round(w["top"]), w["x0"], w["text"])], "x0": w["x0"], "x1": w["x1"]})
            i += 1
    for c in cells:
        c["text"] = " ".join(t for _, _, t in sorted(c["parts"], key=lambda p: p[1]))
    return cells


def _find_header(lines, synonyms, clean):
    date_names = set(synonyms["date"])
    money_names = set(synonyms["debit"] + synonyms["credit"] + synonyms["amount"] + synonyms["balance"])
    known = set(sum(synonyms.values(), []))
    def ok(cells):
        names = {clean(c["text"]) for c in cells}
        return names & date_names and len(names & money_names) >= 2

    def recognised(cells):
        return sum(clean(c["text"]) in known for c in cells)

    for i, ln in enumerate(lines):
        # Candidates: the line alone, the line cut at known headings, and the line together
        # with the next one (headings split over two lines, 'Withdrawal / Amt.'). The one
        # with the most recognised headings wins; on a tie the single line, so a narration
        # line just below the heading is never swallowed into it.
        options = [(i, _header_cells(ln["words"])), (i, _header_cells_by_names(ln["words"], known, clean))]
        if i + 1 < len(lines) and lines[i + 1]["top"] - ln["top"] < 14 and not _is_date(lines[i + 1]["text"]):
            options.append((i + 1, _header_cells(ln["words"] + lines[i + 1]["words"])))
        good = [(recognised(c), -n, idx, c) for n, (idx, c) in enumerate(options) if ok(c)]
        if good:
            _, _, idx, cells = max(good, key=lambda g: (g[0], g[1]))
            return idx, cells
    return None, None


def _grid_layout(page_words, synonyms, clean):
    pages, cells = [], None
    for words in page_words:
        lines = _lines(words)
        h_idx, found = _find_header(lines, synonyms, clean)
        if found:
            cells = cells or found
            pages.append(lines[h_idx + 1:])
        elif cells is not None:
            pages.append(lines)
    if cells is None:
        return None
    names = [clean(c["text"]) for c in cells]
    money_names = set(synonyms["debit"] + synonyms["credit"] + synonyms["amount"] + synonyms["balance"])
    money_cols = [i for i, n in enumerate(names) if n in money_names]
    text_cols = [i for i in range(len(cells)) if i not in money_cols]
    money_start = min(cells[i]["x0"] for i in money_cols) - 25 if money_cols else 1e9

    def is_money_token(w):
        # an amount printed INSIDE the narration ('BONUS 2,500.00 PAID') is narration text
        return w["x0"] >= money_start and (parse_money(w["text"])[2] == "money" or
                                           w["text"].lower() in PLACEHOLDER or CURRENCY.fullmatch(w["text"]))

    edges = sorted(w["x1"] for body in pages for ln in body for w in ln["words"]
                   if parse_money(w["text"])[2] == "money" and w["x0"] >= money_start)
    clusters = []
    for x in edges:
        if clusters and x - clusters[-1][-1] <= 4:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    centers = [sum(c) / len(c) for c in clusters if len(c) >= 3]

    def money_column(w):
        if len(centers) == len(money_cols) and w["x0"] >= money_start - 1:
            return money_cols[min(range(len(centers)), key=lambda k: abs(centers[k] - w["x1"]))]
        return min(money_cols, key=lambda i: min(abs(cells[i]["x1"] - w["x1"]), abs(cells[i]["x0"] - w["x0"])))

    def text_column(w):
        starts = [i for i in text_cols if cells[i]["x0"] <= w["x0"] + 3]
        return max(starts, key=lambda i: cells[i]["x0"]) if starts else text_cols[0]

    entries_by_page = []
    for body in pages:
        entries = []
        for ln in body:
            vals = [""] * len(cells)
            last_money = None
            lw = ln["words"]
            for wi, w in enumerate(lw):
                nxt = lw[wi + 1]["text"] if wi + 1 < len(lw) else ""
                if CURRENCY.fullmatch(w["text"]) and (w["x0"] >= money_start or parse_money(nxt)[2] == "money"):
                    continue                                   # 'INR' before an amount
                if w["text"].lower() in MARKERS and w["x0"] >= money_start and last_money is not None:
                    vals[last_money] += " " + w["text"]        # '117648.61 Cr' printed as two words
                    continue
                if w["text"].lower() in PLACEHOLDER and w["x0"] >= money_start:
                    k = money_column(w)                        # '-' = empty money cell
                    vals[k] = vals[k] or "-"
                    continue
                if is_money_token(w) and money_cols:
                    k = last_money = money_column(w)
                else:
                    k = text_column(w)
                vals[k] = (vals[k] + " " + w["text"]).strip()
            has_money = any(parse_money(v)[2] == "money" for v in vals)
            kind = "tx" if any(_is_date(v) for v in vals[:3]) else "other" if has_money else "text"
            entries.append({"top": ln["top"], "kind": kind, "row": vals})
        entries_by_page.append(entries)

    def narration_cells(row):
        return [row[i] for i in text_cols]
    return [[c["text"] for c in cells]] + _place_wrapped(entries_by_page, narration_cells)


# ---------------------------------------------------------------------------
# Strategy 3: header-free, line by line
# ---------------------------------------------------------------------------
def _grid_lines(page_words):
    """[date, narration, amount, balance] from every line that starts with a date
    and ends with at least two amounts; other text lines continue the narration."""
    pages = []
    for words in page_words:
        entries = []
        for ln in _lines(words):
            toks = [w["text"] for w in ln["words"]]
            text = ln["text"]
            if re.match(r"^\d{1,5}\s", text) and _is_date(text.split(" ", 1)[1]):
                text = text.split(" ", 1)[1]                       # leading serial number
            m = DATE_RE.match(text) if _is_date(text) else None
            if m:
                rest = text[m.end():].split()
                while rest and _is_date(" ".join(rest[:3])):      # a second (value) date
                    mm = DATE_RE.match(" ".join(rest))
                    rest = " ".join(rest)[mm.end():].split()
                for _ in range(2):                                 # value date / branch code printed LAST
                    for k in (3, 2, 1):
                        if len(rest) > k and _is_date(" ".join(rest[-k:])) and \
                                DATE_RE.match(" ".join(rest[-k:])).end() == len(" ".join(rest[-k:])):
                            rest = rest[:-k]
                            break
                    if rest and re.fullmatch(r"\d{1,6}", rest[-1]):
                        rest = rest[:-1]
                amounts, j = [], len(rest)
                while j > 0:
                    tok = rest[j - 1]
                    if j >= 2 and tok.lower() in MARKERS:
                        tok = rest[j - 2] + " " + tok                  # '14,560.00 CR' printed as two words
                        j -= 1
                    kind = parse_money(tok)[2]
                    if kind == "money":
                        amounts.insert(0, tok)
                    elif kind == "empty" or CURRENCY.fullmatch(tok):
                        pass
                    else:
                        break
                    j -= 1
                if len(amounts) >= 2:
                    # with separate Debit / Credit columns the empty one may print as 0.00
                    pair = [x for x in amounts[-3:-1] if parse_money(x)[0] not in (0, 0.0, None)]
                    amount = pair[-1] if pair else amounts[-2]
                    entries.append({"top": ln["top"], "kind": "tx",
                                    "row": [m.group(0), " ".join(rest[:j]), amount, amounts[-1]]})
                    continue
            if toks and not any(parse_money(t)[2] == "money" for t in toks) and not OPENING.search(ln["text"]):
                entries.append({"top": ln["top"], "kind": "text", "row": ["", ln["text"], "", ""]})
            else:
                entries.append({"top": ln["top"], "kind": "other", "row": None})
        pages.append(entries)
    placed = _place_wrapped(pages, lambda row: [row[1]])
    rows = []
    for r in placed:
        if r is None:
            continue
        if r[0]:
            rows.append(list(r))
        elif rows:
            rows[-1][1] = _join_wrapped([rows[-1][1], r[1]])
    return [["Date", "Narration", "Amount", "Balance"]] + rows if len(rows) >= 3 else None


# ---------------------------------------------------------------------------
# Interpreting a grid: roles from content, verified by the running balance
# ---------------------------------------------------------------------------
def _chain_score(amounts, balances):
    """Share of rows where balance = previous balance + amount, best of the two
    orderings (oldest-first / newest-first). Returns (score, checked_rows)."""
    best = (0.0, 0)
    for step in (1, -1):
        ok = n = 0
        for i in range(len(balances)):
            j = i - step
            if 0 <= j < len(balances) and balances[i] is not None and balances[j] is not None \
                    and amounts[i] is not None:
                n += 1
                ok += abs(balances[i] - balances[j] - amounts[i]) <= 1.0
        if n and ok / n > best[0]:
            best = (ok / n, n)
    return best


def _fill_balances(amounts, balances):
    """A balance lost at a page break is recovered exactly from its neighbour:
    previous balance + this row's amount (or next balance - next amount)."""
    for _ in range(2):
        for i in range(len(balances)):
            if balances[i] is not None or amounts[i] is None:
                continue
            if i > 0 and balances[i - 1] is not None:
                balances[i] = round(balances[i - 1] + amounts[i], 2)
            elif i + 1 < len(balances) and balances[i + 1] is not None and amounts[i + 1] is not None:
                balances[i] = round(balances[i + 1] - amounts[i + 1], 2)
    return balances


def _infer_signs(abs_amounts, balances, desc, newest_first=False, opening=None):
    """Single unsigned amount column: the sign comes from the balance movement against
    the previous transaction in time (the row above, or below for newest-first
    statements). The very first transaction uses the printed opening balance."""
    signed, n = [], len(abs_amounts)
    for i, a in enumerate(abs_amounts):
        if a is None:
            signed.append(None)
            continue
        j = i + 1 if newest_first else i - 1
        prev = balances[j] if 0 <= j < n else opening
        if prev is not None and balances[i] is not None and abs(abs(balances[i] - prev) - a) <= 1.0:
            signed.append(a if balances[i] > prev else -a)
        else:
            d = desc[i].lower()
            credit_hint = re.search(r"/cr/|\bcr\b|credit|deposit|\bdep\b|received|transfer from|by transfer"
                                    r"|salary|neft cr|imps cr", d)
            signed.append(a if credit_hint else -a)
    return signed


def _newest_first(dates):
    parsed = pd.to_datetime(pd.Series(dates), dayfirst=True, format="mixed", errors="coerce").dropna()
    return len(parsed) > 1 and parsed.iloc[0] > parsed.iloc[-1]


def _join_wrapped(parts):
    """Join a narration wrapped over several lines. Lines that end or start with '/'
    or '-' were broken inside a token (UPI refs / IDs), so they join without a space."""
    out = ""
    for p in (x.strip() for x in parts if x and x.strip()):
        if out and not (out.endswith(("/", "-")) or p.startswith(("/", "-", "@"))):
            out += " "
        out += p
    return out


def _interpret(grid, synonyms, clean, method, opening=None):
    global ALL_HEADINGS
    ALL_HEADINGS = set(sum(synonyms.values(), [])) | {"transaction id", "remarks", "chq no", "ref no",
                                                      "cheque no", "chq ref no", "sl no", "s no"}
    width = max(len(r) for r in grid)
    grid = [r + [""] * (width - len(r)) for r in grid]
    header = None
    date_counts = [sum(_is_date(r[j]) for r in grid) for j in range(width)]
    # A date printed in words ('Mar 02, 2026') can straddle two columns when the heading
    # positions are tight: merge the neighbouring pair that turns fragments into dates.
    for j in range(width - 1):
        merged = sum(_is_date((r[j] + " " + r[j + 1]).strip()) and not _is_date(r[j]) for r in grid)
        if merged > max(3, 2 * max(date_counts)):
            grid = [r[:j] + [(r[j] + " " + r[j + 1]).strip()] + r[j + 2:] for r in grid]
            width -= 1
            date_counts = [sum(_is_date(r[k]) for r in grid) for k in range(width)]
            break
    date_col = max(range(width), key=lambda j: (date_counts[j], -j))
    if date_counts[date_col] < 3:
        return None
    # header row = the last row before the first date row whose cells look like headings
    first_date = next(i for i, r in enumerate(grid) if _is_date(r[date_col]))
    for r in grid[:first_date]:
        if sum(clean(c) in sum(synonyms.values(), []) for c in r) >= 2:
            header = r

    # transactions: date-led rows; text-only rows continue the narration
    tx, extra = [], []
    for r in grid[first_date:]:
        if _is_date(r[date_col]):
            tx.append(list(r))
            extra.append([])
        elif tx and any(r) and not any(parse_money(c)[2] == "money" for c in r):
            # a wrapped narration line, wherever its words landed -- but not a heading
            # repeated at the top of a page, nor a page footer
            text = " ".join(c for c in r if c and parse_money(c)[2] != "empty")
            if sum(clean(c) in ALL_HEADINGS for c in r) >= 2 or BOILERPLATE.search(text):
                continue
            extra[-1].append(text)

    kinds = [[parse_money(r[j])[2] for r in tx] for j in range(width)]
    money_cols = [j for j in range(width) if j != date_col
                  and sum(k == "money" for k in kinds[j]) >= max(1, 0.8 * sum(k != "empty" for k in kinds[j]))
                  and any(k == "money" for k in kinds[j])
                  and not all(_is_date(r[j]) for r in tx if r[j])]
    other = [j for j in range(width) if j not in money_cols and j != date_col
             and not sum(_is_date(r[j]) for r in tx) > 0.5 * len(tx)]
    desc_col = max(other, key=lambda j: sum(len(r[j]) for r in tx)) if other else None
    val = {j: [parse_money(r[j])[0] for r in tx] for j in money_cols}
    dates, desc = [], []
    for r in tx:                       # the date cell keeps only the date; spill-over text is narration
        m = DATE_RE.match(r[date_col])
        dates.append(m.group(0))
        spill = r[date_col][m.end():].strip()
        d = r[desc_col] if desc_col is not None else ""
        d = (spill + " " + d).strip() if spill and not _is_date(spill) else d
        desc.append(_join_wrapped([d] + extra[len(desc)]))

    newest_first = _newest_first(dates)
    best = None                                       # (score, n, amounts, balances, roles)
    for bal in money_cols:
        rest = [j for j in money_cols if j != bal]
        balances = val[bal]
        options = []
        if len(rest) >= 2:
            for deb, cre in itertools.permutations(rest, 2):
                amts = [None if (val[deb][i] is None and val[cre][i] is None) else
                        abs(val[cre][i] or 0) - abs(val[deb][i] or 0) for i in range(len(tx))]
                options.append((amts, f"debit={deb},credit={cre}"))
        if len(rest) >= 1:
            for a in rest:
                has_marker = any(parse_money(r[a])[1] for r in tx)
                raw = val[a]
                if has_marker or any(v is not None and v < 0 for v in raw):
                    options.append((raw, f"signed={a}"))
                options.append((_infer_signs([None if v is None else abs(v) for v in raw], balances, desc,
                                             newest_first, opening), f"unsigned={a}"))
        for amts, roles in options:
            score, n = _chain_score(amts, balances)
            if best is None or (score, n) > (best[0], best[1]):
                best = (score, n, amts, balances, f"balance={bal},{roles}")
    if best is None:
        return None
    score, n, amts, balances, roles = best
    balances = _fill_balances(amts, list(balances))
    out = pd.DataFrame({"Date": dates, "Description": desc,
                        "Amount": amts, "Balance": balances})
    opening = out["Amount"].isna() | ((out["Amount"] == 0) & out["Description"].str.contains(OPENING, na=False))
    out = out[~opening].reset_index(drop=True)
    return {"df": out, "score": score, "checked": n, "method": method, "roles": roles,
            "explicit": "unsigned" not in roles, "header": header}


# ---------------------------------------------------------------------------
# Where did the file come from? (document forensics, metadata only)
# ---------------------------------------------------------------------------
# Banks' statement systems write their own producer names (OpenPDF, iText, Jasper,
# BIRT, Crystal, PDFlib ...). Files re-saved afterwards carry the tool's name.
EDITING_TOOLS = re.compile(r"(?i)microsoft[\s\S]{0,20}word|\bexcel\b|libreoffice|openoffice|pdfescape|foxit phantom"
                           r"|foxit pdf editor|pdf-xchange editor|pdfelement|wondershare|canva|photoshop|nitro pro"
                           r"|sejda|inkscape|google docs|wps office")
RESAVE_TOOLS = re.compile(r"(?i)ilovepdf|smallpdf|pdf2go|pdf24|qpdf|ghostscript|skia/pdf|microsoft: print to pdf"
                          r"|print to pdf|chrome|quartz pdfcontext|pdfsam|sodapdf|adobe acrobat pro|pikepdf|pypdf")


def _file_origin(pdf, path):
    """Category of the software that last wrote the file. Only the category and the
    tool name are kept; author / title fields (which can hold names) are never read."""
    meta = pdf.metadata or {}
    tools = " | ".join(str(meta.get(k, "")) for k in ("Producer", "Creator"))[:200]
    try:
        with open(path, "rb") as f:
            raw = f.read()
        updates = raw.count(b"%%EOF")
    except OSError:
        updates = None
    m = EDITING_TOOLS.search(tools)
    if m:
        return {"category": "edited", "tool": m.group(0), "saves": updates}
    m = RESAVE_TOOLS.search(tools)
    if m:
        return {"category": "re-saved", "tool": m.group(0), "saves": updates}
    return {"category": "bank-generated" if tools.strip(" |") else "unknown", "tool": None, "saves": updates}


# ---------------------------------------------------------------------------
# Signs of editing that a consistent forgery can still leave behind
# ---------------------------------------------------------------------------
_AMT = r"(-?[\d,]+\.\d{2})\s*(\(?(?:cr|dr)\.?\)?)?"
SUMMARY_PATTERNS = {
    # the number must FOLLOW the label directly (a label row with values on the next line is skipped,
    # so an 'Opening Closing' header over two values can never be mismatched)
    "closing": re.compile(rf"(?i)(?:closing|ending|clear)\s+balance\s*[:\-]?\s*(?:inr|rs\.?|₹)?\s*{_AMT}"),
    "total_credits": re.compile(rf"(?i)total\s+(?:credits?|deposits?)\s*[:+\-]?\s*(?:inr|rs\.?|₹)?\s*{_AMT}"),
    "total_debits": re.compile(rf"(?i)total\s+(?:debits?|withdrawals?)\s*[:+\-]?\s*(?:inr|rs\.?|₹)?\s*{_AMT}"),
}


SUMMARY_COLUMNS = [("opening", r"opening\s*bal\w*"), ("dr_count", r"dr\.?\s*count"), ("cr_count", r"cr\.?\s*count"),
                   ("total_debits", r"(?:total\s*)?(?:debits?|withdrawals?)(?!\s*count)"),
                   ("total_credits", r"(?:total\s*)?(?:credits?|deposits?)(?!\s*count)"),
                   ("closing", r"closing\s*bal\w*")]


def _summary_table(lines):
    """'Opening Balance  Dr Count  Cr Count  Debits  Credits  Closing Bal' with the figures on the
    next line (HDFC and others print the summary this way). Only used when the label row names
    the closing balance and both totals and the next line has exactly one figure per label."""
    texts = [ln["text"] for ln in lines]
    for i, t in enumerate(texts[:-1]):
        low = t.lower()
        if not ("closing" in low and re.search(r"debit|withdraw", low) and re.search(r"credit|deposit", low)):
            continue
        found = []
        for key, rx in SUMMARY_COLUMNS:
            m = re.search(rf"(?i){rx}", t)
            if m:
                found.append((m.start(), key))
        keys = [k for _, k in sorted(found)]
        vals = texts[i + 1].split()
        if len(keys) >= 3 and len(vals) == len(keys):
            out = {}
            for k, v in zip(keys, vals):
                if k in ("closing", "total_debits", "total_credits"):
                    num = parse_money(v)[0]
                    if num is None:
                        return {}
                    out[k] = num
            return out
    return {}


def _printed_summary(page_words):
    """Closing balance and totals as the statement prints them (first two and last two pages)."""
    pages = page_words[:2] + (page_words[-2:] if len(page_words) > 2 else [])
    lines = [ln for words in pages for ln in _lines(words)]
    text = "\n".join(ln["text"] for ln in lines)
    out = _summary_table(lines)
    if out:
        return out
    for key, rx in SUMMARY_PATTERNS.items():
        hits = list(rx.finditer(text))
        if hits:
            m = hits[-1]
            v = parse_money(m.group(1))[0]
            if v is not None and m.group(2) and "dr" in m.group(2).lower():
                v = -abs(v)
            out[key] = v
    return out


def _summary_check(printed, df):
    """Compare the printed summary with what the transactions add up to."""
    if not printed or not len(df):
        return None
    d = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
    newest_first = len(d.dropna()) > 1 and d.dropna().iloc[0] > d.dropna().iloc[-1]
    bal = df["Balance"].dropna()
    computed = {"closing": (bal.iloc[0] if newest_first else bal.iloc[-1]) if len(bal) else None,
                "total_credits": float(df.loc[df["Amount"] > 0, "Amount"].sum()),
                "total_debits": float(-df.loc[df["Amount"] < 0, "Amount"].sum())}
    rows = []
    for k, v in printed.items():
        c = computed.get(k)
        if c is None or v is None:
            continue
        if k == "total_debits":
            v = abs(v)
        rows.append({"item": k, "printed": round(v, 2), "computed": round(c, 2), "match": abs(v - c) <= 1.0})
    return rows or None


def _font_outliers(pdf, page_words):
    """Amounts printed in a font / size that the rest of the statement's amounts never use.
    A forger who retypes a figure (even with every balance recalculated) usually leaves it in a
    different font. Returns (outlier rows as dates, share of amounts in the dominant font)."""
    from collections import Counter
    fonts, where = [], []
    for page, words in zip(pdf.pages, page_words):
        chars = [c for c in page.chars if c["text"].isdigit()]
        if not chars:
            continue
        by_line = {}
        for c in chars:
            by_line.setdefault(round(c["top"]), []).append(c)
        for ln in _lines(words):
            t = ln["text"]
            rest = t.split(" ", 1)[-1]
            if _is_date(t):
                when = DATE_RE.match(t).group(0)
            elif re.match(r"^\S{1,12}\s", t) and _is_date(rest):   # a serial / transaction number first
                when = DATE_RE.match(rest).group(0)
            else:
                continue                                  # transaction lines only (they start with a date)
            for w in ln["words"]:
                if parse_money(w["text"])[2] != "money":
                    continue
                cs = [c for t in (round(w["top"]) - 1, round(w["top"]), round(w["top"]) + 1)
                      for c in by_line.get(t, []) if w["x0"] - 0.5 <= c["x0"] <= w["x1"] + 0.5]
                if cs:
                    key = Counter((c.get("fontname", ""), round(c.get("size", 0), 1)) for c in cs).most_common(1)[0][0]
                    fonts.append(key)
                    where.append(when)
    if len(fonts) < 20:
        return [], None
    counts = Counter(fonts)
    top, n_top = counts.most_common(1)[0]
    rare = {f for f, n in counts.items() if n < max(2, 0.02 * len(fonts))}
    out = sorted({d for f, d in zip(fonts, where) if f in rare})
    return out, round(n_top / len(fonts), 3)


def _tamper_signals(pdf, page_words, df):
    try:
        rows, dominant = _font_outliers(pdf, page_words)
    except Exception:                       # never let a diagnostic stop the reading
        rows, dominant = [], None
    try:
        summary = _summary_check(_printed_summary(page_words), df)
    except Exception:
        summary = None
    return {"font_outlier_rows": rows[:10], "font_outliers": len(rows), "dominant_font_share": dominant,
            "summary": summary}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def read_pdf_statement(path, password=None, progress=None):
    from statement_parser import COLUMN_SYNONYMS, _clean_header
    pdf = _open(path, password)
    try:
        page_words = []                                          # extracted once, shared by all strategies
        for i, p in enumerate(pdf.pages):
            page_words.append(_page_words(p, pdf))
            if progress:
                progress(i + 1, len(pdf.pages))
        if sum(len(w["text"]) for words in page_words for w in words) < 40:
            raise StatementPDFError("This PDF has no readable text (it looks like a scanned image). Please "
                                    "download the e-statement PDF from net banking instead.")
        text = " ".join(" ".join(ln["text"] for ln in _lines(words)) for words in page_words[:3])
        m = re.search(r"(?i)opening balance[^0-9\-]{0,20}(-?[\d,]+\.\d{2})", text)
        opening = parse_money(m.group(1))[0] if m else None
        pm = PERIOD.search(text)
        if not pm:
            for ln in _lines(page_words[0])[:30]:
                pm = BARE_PERIOD.match(ln["text"])
                if pm:
                    break
        period = [pm.group(1), pm.group(2)] if pm else None
        candidates = []
        for method, builder in (("ruled", lambda: _grid_ruled(pdf)),
                                ("layout", lambda: _grid_layout(page_words, COLUMN_SYNONYMS, _clean_header)),
                                ("lines", lambda: _grid_lines(page_words))):
            try:
                grid = builder()
                res = _interpret(grid, COLUMN_SYNONYMS, _clean_header, method, opening) if grid else None
            except Exception:                     # one strategy failing must not stop the others
                res = None
            if res and len(res["df"]) >= 3:
                candidates.append(res)
        if not candidates:
            raise StatementPDFError("No transactions could be found in this PDF. Please upload the account "
                                    "statement PDF downloaded from net banking (not a summary or a scan).")
        # Most transactions first (a strategy that misses page-break rows never wins), then how well
        # the balances reconcile, then explicit debit/credit columns over inferred signs, then the
        # most structured method (ruled > lines > layout).
        rank = {"ruled": 2, "lines": 1, "layout": 0}
        best = max(candidates, key=lambda c: (len(c["df"]), round(c["score"], 3), c["explicit"], rank[c["method"]]))
        df = best["df"]
        df.attrs["pdf_read"] = {"method": best["method"], "reconciled": round(best["score"], 4),
                                "checked_rows": best["checked"], "roles": best["roles"],
                                "transactions": len(df), "printed_period": period}
        df.attrs["holder_names"] = _holder_names(page_words)       # in memory only; never stored
        df.attrs["pdf_read"]["file_origin"] = _file_origin(pdf, path)
        df.attrs["pdf_read"]["tamper"] = _tamper_signals(pdf, page_words, df)
        return df
    finally:
        pdf.close()
