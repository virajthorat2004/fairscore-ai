"""
Service layer: the one place an application is read, validated, assessed and
recorded. The Streamlit dashboard and the REST API (api/main.py) both call
it, so they can never disagree on a decision.
"""
import hashlib
import logging
import os
import secrets
import tempfile
from datetime import datetime
from pathlib import Path

import audit_store
import live_scoring as ls
from config import PROJECT_ROOT
from pdf_statement import StatementPDFError
from statement_parser import parse_statement

LOG_DIR = PROJECT_ROOT / "logs"
MAX_PDF_BYTES = 15 * 1024 * 1024

log = logging.getLogger("fairscore")
if not log.handlers:
    try:
        LOG_DIR.mkdir(exist_ok=True)
        h = logging.FileHandler(LOG_DIR / "fairscore.log", encoding="utf-8")
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    except OSError:          # read-only location: logging is best effort
        pass


class ApplicationError(ValueError):
    """Problems with the application itself, safe to show to the user."""


def new_reference():
    return f"FS-{datetime.now():%y%m%d-%H%M%S}-{secrets.token_hex(2).upper()}"


def normalise_applicant(raw):
    """Types from a form or JSON: numbers as numbers, blanks as None."""
    a = {k: raw.get(k) for k in ls.REQUIRED_FIELDS + ls.OPTIONAL_FIELDS}
    for k, v in a.items():
        if isinstance(v, str) and not v.strip():
            a[k] = None
    for k in ("age", "dependents", "requested_duration_months", "active_credit_lines", "property_loans",
              "bureau_score", "bureau_max_dpd_12m", "bureau_enquiries_6m"):
        if a[k] is not None:
            try:
                a[k] = int(float(a[k]))
            except (TypeError, ValueError, OverflowError):
                pass
    for k in ("employment_years", "total_savings", "requested_amount", "monthly_income", "asset_price", "bureau_emi"):
        if a[k] is not None:
            try:
                a[k] = float(a[k])
            except (TypeError, ValueError, OverflowError):
                pass
    for k in ("owns_car", "bureau_severe_24m"):
        if isinstance(a[k], str):
            a[k] = {"true": True, "yes": True, "1": True, "false": False, "no": False, "0": False}.get(
                a[k].strip().lower(), a[k])
    return a


def read_statement(pdf_bytes, password=None, progress=None):
    """Returns (transactions, statement features, warnings, sha256). The PDF is
    written to a private temporary file only while it is read, then deleted."""
    if not pdf_bytes:
        raise ApplicationError("Upload the applicant's bank statement (PDF).")
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise ApplicationError("The statement is larger than 15 MB. Please upload the e-statement PDF from net banking.")
    sha = hashlib.sha256(pdf_bytes).hexdigest()
    head = pdf_bytes[:1024].lstrip(b"\xef\xbb\xbf \r\n\t")
    if head[:1] in (b"{", b"[", b"<"):
        # Account Aggregator data (ReBIT deposit schema, JSON or XML): bank-signed, no layout to read
        from aa_statement import AAFormatError, parse_aa
        from statement_parser import combine_statements
        try:
            parsed = parse_aa(pdf_bytes)
            df, stmt, warnings = combine_statements(parsed) if len(parsed) > 1 else parsed[0]
        except (AAFormatError, ValueError) as exc:
            raise ApplicationError(str(exc)) from exc
        return df, stmt, warnings, sha
    if not head.startswith(b"%PDF"):
        raise ApplicationError("This file is not a PDF. Please upload the e-statement PDF from net banking "
                               "(or the Account Aggregator data file).")
    fd, tmp = tempfile.mkstemp(suffix=".pdf")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(pdf_bytes)
        df, stmt, warnings = parse_statement(tmp, password=password or None, progress=progress)
    except StatementPDFError as exc:
        raise ApplicationError(str(exc)) from exc
    finally:
        Path(tmp).unlink(missing_ok=True)
    return df, stmt, warnings, sha


def read_statements(files, passwords=None, progress=None):
    """Several statements (one per bank account) -> one combined statement.
    `files`: list of (name, bytes). `passwords`: candidate passwords, tried on each
    protected file in turn (different banks use different passwords)."""
    from pdf_statement import PasswordRequired
    from statement_parser import combine_statements
    if not files:
        raise ApplicationError("Upload the applicant's bank statement (PDF).")
    if len(files) > 4:
        raise ApplicationError("Upload at most 4 statements (one per bank account).")
    pw = [p for p in (passwords or []) if p] or [None]
    parsed, shas = [], []
    for k, (name, data) in enumerate(files):
        last = None
        # unprotected first, then each password given; report the most informative failure
        # ("incorrect password" beats "password needed" once a password was tried)
        for p in [None] + [x for x in pw if x]:
            try:
                df, st, w, sha = read_statement(data, p, progress)
                break
            except ApplicationError as exc:
                if "password" not in str(exc).lower():
                    raise ApplicationError(f"{name}: {exc}") from exc
                if last is None or p is not None:
                    last = exc
        else:
            raise ApplicationError(f"{name}: {last}")
        if sha in shas:
            raise ApplicationError(f"{name}: this file was uploaded twice.")
        df["account"] = f"A{k + 1}"
        parsed.append((df, st, w))
        shas.append(sha)
    try:
        df, st, warnings = combine_statements(parsed)
    except ValueError as exc:
        raise ApplicationError(str(exc)) from exc
    return df, st, warnings, ",".join(shas)


def assess_application(applicant, statement, channel="dashboard", record=True):
    """`statement` = (df, stmt, warnings, sha256) from read_statement().
    Returns {"reference", "result", "warnings", "applicant"}."""
    a = normalise_applicant(applicant)
    problems = ls.validate_applicant(a)
    if problems:
        raise ApplicationError("; ".join(problems))
    df, stmt, warnings, sha = statement
    result = ls.assess(a, stmt, df)
    ref = new_reference()
    if record:
        try:
            # the applicant's name is only compared with the statement, never stored
            logged = {k: v for k, v in a.items() if k != "account_holder_name"}
            audit_store.log_assessment(ref, logged, result, stmt, sha, channel)
        except Exception as exc:     # the decision must not be lost because the log failed
            log.error("audit log failed for %s: %r", ref, exc)
            warnings = list(warnings) + ["This assessment could not be saved to the history log."]
    log.info("assessed %s channel=%s recommendation=%s policy=%s confidence=%s", ref, channel,
             result["verdict"], result["policy_version"], result["decision_confidence"]["label"])
    return {"reference": ref, "result": result, "warnings": warnings, "applicant": a}


def public_summary(out):
    """JSON-safe summary of an assessment for the API (no transactions, no model internals)."""
    r, a = out["result"], out["applicant"]
    pol = r["policy"]

    def num(x):
        return None if x is None or x != x or x in (float("inf"), float("-inf")) else round(float(x), 4)

    return {
        "reference": out["reference"],
        "recommendation": r["verdict"],
        "reasons": r["verdict_reasons"],
        "decision_confidence": r["decision_confidence"],
        "checks": [{k: c[k] for k in ("key", "name", "status", "detail")} for c in pol["checks"]],
        "affordability": {"verified_monthly_income": num(pol["income_used"]), "new_emi": num(pol["new_emi"]),
                          "existing_emis": num(pol["existing_emi"]), "foir": num(pol["foir"]),
                          "interest_rate_assumed": ls.POLICY["annual_interest_rate"]},
        "risk": {"level": r["risk_level"], "percentile": r["consensus_percentile"],
                 "models_voting": r["voters"], "role": "can refer a case for review; never approves or declines"},
        "reasons_in_favour": [f["label"] for f in (r["explanation"] or {}).get("risk_down", [])],
        "reasons_against": [f["label"] for f in (r["explanation"] or {}).get("risk_up", [])],
        "what_would_help": [x["change"] for x in r["recourse"]],
        "statement": r["statement_insights"],
        "fairscore": {k: r["fairscore"][k] for k in ("score", "band")} | {
            "factors": [{k: f[k] for k in ("label", "points", "max", "reason")} for f in r["fairscore"]["factors"]]},
        "eligibility": {"eligible_amount": r["eligibility"]["eligible_amount"], "binding_limit": r["eligibility"]["binding"],
                        "term_months": r["eligibility"]["term"], "fits_at_term": r["eligibility"]["fits_at_term"]},
        "counter_offer": r["offer"],
        "documents_to_collect": r["documents"],
        "banking": {k: (r.get("banking") or {}).get(k) for k in
                    ("abb", "bounce_events", "penalty_charges", "lenders", "cash_withdrawal_share", "cash_deposit_share",
                     "low_balance_days_share", "salary_latest_vs_usual", "large_credits", "monthly")},
        "fairness_check": {"age_flip_passed": r["age_flip"]["passed"], "ages_tested": r["age_flip"]["ages_tested"]},
        "policy_version": r["policy_version"],
        "model_versions": {k: v["hash"] for k, v in r["model_versions"].items()},
        "warnings": out["warnings"],
        "requested": {"amount": a["requested_amount"], "term_months": a["requested_duration_months"],
                      "purpose": a["purpose"]},
    }
