"""
Audit trail and fairness monitoring (SQLite, standard library only).

One row per assessment: what was decided, why, under which policy version and
which model versions, how the statement was read, and -- for referred cases --
what the credit officer finally decided and why (human in the loop).

Data minimisation: the uploaded statement is never stored. Only its SHA-256
fingerprint (to recognise the same file again), how it was read, and the
applicant's form answers are kept. No names, account numbers or narrations.
"""
import json
import sqlite3
from contextlib import closing
from datetime import datetime

import pandas as pd

from config import PROJECT_ROOT

DB_PATH = PROJECT_ROOT / "data" / "assessments" / "fairscore.db"
LEGACY_CSV = PROJECT_ROOT / "data" / "assessments" / "history.csv"

AGE_GROUPS = [(25, "Under 25"), (35, "25-34"), (50, "35-49"), (200, "50 and over")]
MIN_GROUP_SIZE = 20          # below this a group's rate is shown but not judged
FOUR_FIFTHS = 0.80           # disparate-impact rule of thumb

SCHEMA = """
CREATE TABLE IF NOT EXISTS assessments (
    reference         TEXT PRIMARY KEY,
    assessed_at       TEXT NOT NULL,
    channel           TEXT,
    policy_version    TEXT,
    model_versions    TEXT,
    statement_sha256  TEXT,
    statement_read    TEXT,
    applicant         TEXT,
    age_group         TEXT,
    amount            REAL,
    term_months       INTEGER,
    purpose           TEXT,
    verified_income   REAL,
    foir              REAL,
    risk_level        TEXT,
    risk_percentile   REAL,
    confidence        REAL,
    confidence_label  TEXT,
    recommendation    TEXT NOT NULL,
    reasons           TEXT,
    checks            TEXT,
    age_flip_passed   INTEGER,
    officer_decision  TEXT,
    officer_reason    TEXT,
    officer_name      TEXT,
    officer_at        TEXT,
    fairscore         INTEGER,
    fairscore_band    TEXT,
    score_factors     TEXT,
    eligible_amount   REAL,
    accounts          INTEGER,
    outcome           TEXT,
    outcome_at        TEXT
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def age_group(age):
    if age is None or pd.isna(age):
        return None
    return next(label for cut, label in AGE_GROUPS if float(age) < cut)


NEW_COLUMNS = {"fairscore": "INTEGER", "fairscore_band": "TEXT", "score_factors": "TEXT", "eligible_amount": "REAL",
               "accounts": "INTEGER", "outcome": "TEXT", "outcome_at": "TEXT", "model_inputs": "TEXT",
               # FAIRScore 3.0: learning loop and starter ladder
               "ai_inputs": "TEXT", "ai_pd": "REAL", "decided_by": "TEXT", "starter": "TEXT", "propensity": "REAL"}
OUTCOMES = ("repaid", "repaying", "late 30+ days", "defaulted 90+ days", "written off", "not disbursed")


def _connect(path=None):
    p = path or DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p, timeout=10)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    have = {r[1] for r in con.execute("PRAGMA table_info(assessments)")}
    for col, typ in NEW_COLUMNS.items():          # databases created by earlier versions
        if col not in have:
            con.execute(f"ALTER TABLE assessments ADD COLUMN {col} {typ}")
    return con


def _import_legacy_csv(con):
    """One-time import of the earlier CSV history so nothing is lost."""
    done = con.execute("SELECT value FROM meta WHERE key='legacy_csv_imported'").fetchone()
    if done or not LEGACY_CSV.exists():
        return
    try:
        old = pd.read_csv(LEGACY_CSV)
    except Exception:          # an unreadable old file must not stop the app
        old = pd.DataFrame()
    for _, r in old.iterrows():
        con.execute(
            "INSERT OR IGNORE INTO assessments (reference, assessed_at, channel, amount, term_months, purpose, "
            "verified_income, foir, risk_level, risk_percentile, recommendation, reasons) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (r.get("reference"), str(r.get("assessed_at")), "imported from history.csv", r.get("amount"),
             r.get("term_months"), r.get("purpose"), r.get("verified_income"), r.get("emis_vs_income"),
             r.get("risk_level") if isinstance(r.get("risk_level"), str) else None,
             r.get("risk_percentile"), r.get("recommendation"),
             json.dumps(str(r.get("reasons", "")).split(" | ")) if isinstance(r.get("reasons"), str) else "[]"))
    con.execute("INSERT OR REPLACE INTO meta VALUES ('legacy_csv_imported', ?)", (datetime.now().isoformat(),))
    con.commit()


def _clean(v):
    if isinstance(v, float) and (pd.isna(v) or v in (float("inf"), float("-inf"))):
        return None
    return v


def log_assessment(reference, applicant, result, stmt=None, statement_sha256=None, channel="dashboard", path=None):
    pol = result["policy"]
    info = (stmt or {}).get("read_info") or {}
    read = {"method": info.get("method"), "reconciled": info.get("reconciled"),
            "transactions": (stmt or {}).get("transactions"), "months_covered": (stmt or {}).get("months_covered"),
            "period": [(stmt or {}).get("period_start"), (stmt or {}).get("period_end")]}
    conf = result.get("decision_confidence") or {}
    with closing(_connect(path)) as con:
        _import_legacy_csv(con)
        con.execute(
            "INSERT INTO assessments (reference, assessed_at, channel, policy_version, model_versions, "
            "statement_sha256, statement_read, applicant, age_group, amount, term_months, purpose, verified_income, "
            "foir, risk_level, risk_percentile, confidence, confidence_label, recommendation, reasons, checks, "
            "age_flip_passed, fairscore, fairscore_band, score_factors, eligible_amount, accounts, model_inputs, "
            "ai_inputs, ai_pd, decided_by, starter, propensity) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (reference, datetime.now().isoformat(timespec="seconds"), channel, result.get("policy_version"),
             json.dumps(result.get("model_versions")), statement_sha256, json.dumps(read),
             json.dumps(applicant, default=str), age_group(applicant.get("age")),
             applicant["requested_amount"], applicant["requested_duration_months"], applicant["purpose"],
             _clean(round(pol["income_used"] or 0)), _clean(round(pol["foir"], 4)),
             result.get("risk_level"), result.get("consensus_percentile"), conf.get("score"), conf.get("label"),
             result["verdict"], json.dumps(result["verdict_reasons"]),
             json.dumps([{k: c[k] for k in ("key", "status", "detail")} for c in pol["checks"]]),
             None if result.get("age_flip") is None else int(result["age_flip"]["passed"]),
             (result.get("fairscore") or {}).get("score"), (result.get("fairscore") or {}).get("band"),
             json.dumps((result.get("fairscore") or {}).get("fractions")),
             (result.get("eligibility") or {}).get("eligible_amount"), (stmt or {}).get("accounts", 1),
             # scale-free cash-flow model inputs (ratios, no amounts or names): for drift monitoring
             json.dumps(result.get("model_inputs")),
             json.dumps(result.get("ai_inputs")) if result.get("ai_inputs") else None,
             (result.get("ai") or {}).get("pd"), result.get("decided_by"),
             json.dumps({k: v for k, v in result["starter"].items() if k != "rungs"}, default=float)
             if result.get("starter") else None,
             # probability that THIS decision path was taken (1 = deterministic; a learning offer's share otherwise):
             # retraining weights outcomes by 1 / propensity (inverse-propensity weighting)
             (result.get("starter") or {}).get("propensity", 1.0)))
        con.commit()


def record_officer_decision(reference, decision, reason, officer=None, path=None):
    """Final human decision on an assessment (required for referred cases)."""
    if decision not in ("approve", "decline"):
        raise ValueError("decision must be 'approve' or 'decline'")
    if not reason or len(reason.strip()) < 5:
        raise ValueError("Please record the reason for the decision")
    with closing(_connect(path)) as con:
        cur = con.execute("UPDATE assessments SET officer_decision=?, officer_reason=?, officer_name=?, officer_at=? "
                          "WHERE reference=?", (decision, reason.strip(), officer,
                                                datetime.now().isoformat(timespec="seconds"), reference))
        con.commit()
        if cur.rowcount == 0:
            raise KeyError(reference)


def record_outcome(reference, outcome, path=None):
    """How the loan actually turned out (closes the loop: these outcomes are what the
    scorecard is recalibrated on, see src/train_from_outcomes.py)."""
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of: {', '.join(OUTCOMES)}")
    with closing(_connect(path)) as con:
        cur = con.execute("UPDATE assessments SET outcome=?, outcome_at=? WHERE reference=?",
                          (outcome, datetime.now().isoformat(timespec="seconds"), reference))
        con.commit()
        if cur.rowcount == 0:
            raise KeyError(reference)


def get_assessment(reference, path=None):
    with closing(_connect(path)) as con:
        row = con.execute("SELECT * FROM assessments WHERE reference=?", (reference,)).fetchone()
    if row is None:
        return None
    out = dict(row)
    for k in ("model_versions", "statement_read", "applicant", "reasons", "checks", "score_factors"):
        if out.get(k):
            out[k] = json.loads(out[k])
    return out


def history(path=None):
    with closing(_connect(path)) as con:
        _import_legacy_csv(con)
        df = pd.read_sql_query("SELECT * FROM assessments ORDER BY assessed_at DESC", con)
    df["final_decision"] = df["officer_decision"].fillna(df["recommendation"].where(df["recommendation"] != "refer"))
    return df


def fairness_summary(df=None, path=None):
    """Outcomes by age group on REAL assessments (synopsis: measure fairness
    between demographic groups). Approval rate = recommended approve; final
    approval counts the officer's decision on referred cases. Disparate-impact
    ratio = group rate / best group's rate; the four-fifths rule flags < 0.80.
    Groups with fewer than MIN_GROUP_SIZE assessments are reported, not judged."""
    df = history(path) if df is None else df
    df = df[df["age_group"].notna()]
    if not len(df):
        return None, []
    order = [label for _, label in AGE_GROUPS]
    g = df.groupby("age_group").agg(
        assessments=("reference", "size"),
        recommended_approve=("recommendation", lambda s: (s == "approve").mean()),
        referred=("recommendation", lambda s: (s == "refer").mean()),
        recommended_decline=("recommendation", lambda s: (s == "decline").mean()),
        final_approve=("final_decision", lambda s: (s == "approve").mean()),
        age_flip_failures=("age_flip_passed", lambda s: int((s == 0).sum())),
        avg_fairscore=("fairscore", lambda s: round(float(s.mean()), 0) if s.notna().any() else None),
    ).reindex([o for o in order if o in set(df["age_group"])])
    judged = g[g["assessments"] >= MIN_GROUP_SIZE]
    alerts = []
    best = judged["recommended_approve"].max() if len(judged) else 0
    g["disparate_impact"] = g["recommended_approve"] / best if best else None
    for grp, row in judged.iterrows():
        if best and row["recommended_approve"] / best < FOUR_FIFTHS:
            alerts.append(f"{grp}: approval rate {row['recommended_approve']:.0%} is "
                          f"{row['recommended_approve'] / best:.2f}x the best group's (below the four-fifths rule)")
    if int(g["age_flip_failures"].sum()):
        alerts.append(f"{int(g['age_flip_failures'].sum())} assessment(s) changed verdict when only age changed")
    return g.reset_index(), alerts
