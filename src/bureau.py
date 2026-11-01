"""
Credit-bureau layer (FAIRScore 3.0).

FAIRScore is built for applicants with NO bureau record: their bank statement is the evidence. Many "thin-file"
applicants do have something, though (a credit card, a BNPL line, a small consumer loan), and when they do a bank
must use it. This module adds the bureau report as a separate layer of hard stops and cautions; it never replaces
the statement checks and is not a score blend (no joint outcome data exists to learn one honestly).

Inputs (all optional; the officer copies them from the CIBIL / Experian / CRIF / Equifax report)
    bureau_record        "None (new to credit)" | "Thin file" | "Established file"
    bureau_score         300-900 (or the bureau's new-to-credit score, if that is what the report gives)
    bureau_max_dpd_12m   worst days-past-due in the last 12 months: 0, 30, 60 or 90
    bureau_severe_24m    true if a 90+ DPD, write-off or settlement in the last 24 months
    bureau_enquiries_6m  hard enquiries in the last 6 months
    bureau_emi           total monthly EMIs of all open loans on the report (Rs)

What it does
    * severe delinquency in 24 months -> decline (hard stop)
    * established file below `min_score` -> decline; thin file below it -> officer review
    * score below `caution_score`, any 30+ DPD in 12 months, or credit hunger (enquiries) -> officer review
    * bureau EMIs higher than the EMIs seen in the statement -> the higher figure is used for affordability (FOIR),
      as lenders do, and a gap is flagged: loans are being paid from another account
"""
from policy_config import POLICY

RECORDS = ["None (new to credit)", "Thin file", "Established file"]
DOCUMENTS = ["Credit bureau report (CIBIL / Experian / CRIF / Equifax), pulled with the applicant's consent"]


def _check(key, name, status, value, detail):
    return {"key": key, "name": name, "status": status, "value": value, "detail": detail}


def provided(a):
    return a.get("bureau_record") not in (None, "", "None")


def assess(a, statement_emi):
    """Returns None when no bureau information was given, else
    {'check': {...}, 'emi_for_affordability': float, 'record': str, 'score': int or None, 'gap': float}."""
    if not provided(a):
        return None
    B = POLICY["bureau"]
    rec = a["bureau_record"]
    if rec == RECORDS[0]:
        return {"check": _check("bureau", "Credit bureau record", "info", None,
                                "No bureau record (new to credit): assessed on the bank statement's cash flow alone"),
                "emi_for_affordability": float(statement_emi), "record": rec, "score": None, "gap": 0.0}
    score = a.get("bureau_score")
    score = int(score) if score not in (None, "") else None
    dpd = int(a.get("bureau_max_dpd_12m") or 0)
    severe = bool(a.get("bureau_severe_24m"))
    enq = int(a.get("bureau_enquiries_6m") or 0)
    b_emi = float(a.get("bureau_emi") or 0.0)
    gap = b_emi - float(statement_emi)
    fails, cautions, notes = [], [], []
    if severe and B["dpd90_fail_24m"]:
        fails.append("a 90+ days-past-due, write-off or settlement in the last 24 months")
    if score is not None:
        if score < B["min_score"]:
            (fails if rec == "Established file" else cautions).append(f"bureau score {score} is below {B['min_score']}")
        elif score < B["caution_score"]:
            cautions.append(f"bureau score {score} is below {B['caution_score']}")
        else:
            notes.append(f"bureau score {score}")
    if dpd >= 30 and B["dpd30_caution_12m"]:
        cautions.append(f"{dpd}+ days past due in the last 12 months")
    if enq > B["max_enquiries_6m"]:
        cautions.append(f"{enq} hard enquiries in 6 months (more than {B['max_enquiries_6m']}: possible credit hunger)")
    if gap > max(1000.0, 0.2 * float(statement_emi)):
        cautions.append(f"EMIs on the bureau report (₹{b_emi:,.0f}/month) exceed those seen in the statement "
                        f"(₹{statement_emi:,.0f}): some loans are paid from another account; the bureau figure is used "
                        f"for affordability")
    status = "fail" if fails else "caution" if cautions else "pass"
    parts = fails + cautions + (notes if status == "pass" else [])
    detail = f"{rec}: " + ("; ".join(parts) if parts else "no adverse information")
    return {"check": _check("bureau", "Credit bureau record", status, score, detail),
            "emi_for_affordability": max(float(statement_emi), b_emi), "record": rec, "score": score, "gap": gap}
