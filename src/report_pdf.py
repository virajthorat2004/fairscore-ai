"""Credit Appraisal Memo (CAM) for an assessment, in the order a credit officer
writes one: recommendation, the request, banking analysis, policy checks,
eligibility, score, risk models, documents, and a sign-off box (ReportLab)."""
import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

VERDICT_TEXT = {
    "approve": ("Recommend approve", colors.HexColor("#2E6B4F")),
    "refer": ("Refer to credit officer", colors.HexColor("#B8863B")),
    "decline": ("Recommend decline", colors.HexColor("#9E3A2E")),
}
STATUS_COLOR = {"pass": "#2E6B4F", "caution": "#B8863B", "fail": "#9E3A2E", "info": "#16324A"}
STATUS_WORD = {"pass": "OK", "caution": "REVIEW", "fail": "FAIL", "info": "NOTE"}
NAVY, LINE, GREY = colors.HexColor("#16324A"), colors.HexColor("#DCD7CC"), colors.HexColor("#5B6270")


def _rupees(x):
    # ReportLab's built-in fonts have no rupee glyph; "Rs" keeps the PDF portable.
    if x is None:
        return "n/a"
    return f"-Rs {abs(x):,.0f}" if x < 0 else f"Rs {x:,.0f}"


def _pct(x):
    return "n/a" if x is None else f"{x:.0%}"


def build_report(applicant, result, applicant_ref="Applicant"):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=13 * mm, bottomMargin=13 * mm, title="FAIRScore AI credit appraisal memo")
    ss = getSampleStyleSheet()
    h = ParagraphStyle("h", parent=ss["Heading2"], fontSize=10.5, spaceBefore=7, spaceAfter=3, textColor=NAVY)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=8.6, leading=11.2)
    small = ParagraphStyle("s", parent=body, fontSize=7.3, leading=9.4, textColor=GREY)

    def clean(text):
        return str(text).replace("₹", "Rs ").replace("−", "-").replace("&", "&amp;").replace("<", "&lt;")

    def table(rows, widths, header=True, size=8.2):
        style = [("FONTSIZE", (0, 0), (-1, -1), size), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                 ("LINEBELOW", (0, 1 if header else 0), (-1, -1), 0.25, LINE),
                 ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5), ("TOPPADDING", (0, 0), (-1, -1), 2.5)]
        if header:
            style += [("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("LINEBELOW", (0, 0), (-1, 0), 0.6, NAVY)]
        return Table(rows, colWidths=[w * mm for w in widths], style=TableStyle(style), repeatRows=1 if header else 0)

    def kv(pairs, widths=(36, 54, 36, 54)):
        rows = []
        for i in range(0, len(pairs), 2):
            row = []
            for k, v in pairs[i:i + 2]:
                row += [Paragraph(f'<font color="#5B6270">{clean(k)}</font>', body), Paragraph(clean(v), body)]
            rows.append(row + [""] * (4 - len(row)))
        return table(rows, widths, header=False)

    a, pol = applicant, result["policy"]
    stmt_b = result.get("banking") or {}
    verdict, color = VERDICT_TEXT[result["verdict"]]
    story = [
        Paragraph("<b>FAIRScore AI</b> &middot; Credit Appraisal Memo",
                  ParagraphStyle("t", parent=ss["Title"], fontSize=15, spaceAfter=1, alignment=0)),
        Paragraph(f"Reference {clean(applicant_ref)} &middot; prepared {datetime.now():%d %b %Y, %H:%M} &middot; "
                  f"decision support for a credit officer, not an automatic decision", small),
        Spacer(1, 5),
    ]

    # 1. Recommendation
    reasons = result["verdict_reasons"] or ["All lending-policy checks passed; FAIRScore and the risk models "
                                            "did not flag elevated risk."]
    conf, flip, fs = result.get("decision_confidence") or {}, result.get("age_flip") or {}, result.get("fairscore") or {}
    rec = [Paragraph(f'<font color="{color.hexval()}" size="13"><b>{verdict}</b></font>', body)]
    rec += [Paragraph("&bull; " + clean(r), body) for r in reasons]
    line = []
    ai = result.get("ai")
    if result.get("decided_by"):
        line.append(f"decided by <b>{clean(result['decided_by'])}</b>")
    if ai:
        line.append(f"<b>AI default risk {ai['pd_pct']:.1f}%</b>")
    if fs:
        line.append(f"<b>FAIRScore {fs['score']}</b> ({fs['band']})")
    if conf:
        line.append(f"decision confidence <b>{conf['label']}</b>")
    if flip.get("ages_tested"):
        line.append("fairness check: same recommendation at ages " + ", ".join(map(str, flip["ages_tested"]))
                    if flip["passed"] else "<b>fairness check: recommendation changes with age alone - review</b>")
    rec.append(Paragraph(" &middot; ".join(line), small))
    story.append(Table([[rec]], colWidths=[180 * mm], style=TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.8, color), ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)])))

    if result.get("conditions") and result["verdict"] == "approve":
        story.append(Paragraph("<b>Approved subject to:</b> " + "; ".join(clean(c) for c in result["conditions"]), small))

    # AI engine (version 2)
    if ai:
        cert = {"repays": "confident the loan will be repaid", "defaults": "confident the loan would not be repaid",
                "uncertain": "cannot rule out either outcome"}[ai["outcome_set"]]
        story += [Paragraph("AI risk assessment (decides within the lending rules)", h), kv([
            ("Probability of default", f"{ai['pd_pct']:.1f}% (calibrated)"), ("Certainty", cert),
            ("Auto-approve limit", f"{100 * ai['approve_max_pd']:.1f}%"),
            ("Conformal p-values", f"repays {ai['p_repays']:.2f} / defaults {ai['p_defaults']:.2f} (alpha {ai['alpha']:.2f})"),
            ("Profile typicality", f"{ai['typicality']:.0%} of training cases more unusual"),
            ("Risk-based rate", f"{100 * ai['risk_based_rate']:.1f}% a year"),
            ("Expected loss", f"{_rupees(ai['expected_loss'])} (PD x LGD {ai['lgd']:.0%} x amount)"),
            ("Version 1 (rules only)", clean(result.get("v1_verdict", "")).capitalize()),
            ("Security", (f"{pol['security']['type']}, {_rupees(pol['security']['value'])}: covers "
                          f"{pol['security']['coverage']:.0%}, LGD {pol['security']['lgd']:.0%}")
             if pol.get("security") else "Unsecured"),
            ("Co-applicant income", _rupees(pol["co_applicant_income"]) + " / month (to verify)"
             if pol.get("co_applicant_income") else "None")])]
        rows = [["What the AI weighed", "Effect", "Evidence"]]
        for g in ai["explanation"]["groups"][:6]:
            rows.append([clean(g["group"]), f"{'+' if g['impact'] > 0 else ''}{g['impact']:.2f} ({g['direction']})",
                         Paragraph(clean(g["evidence"]), body)])
        story.append(table(rows, [60, 45, 75]))
        rc = ai.get("recourse") or {}
        opts = []
        if rc.get("loan"):
            L = rc["loan"]
            opts.append(f"Borrow {_rupees(L['amount'])} over {L['term']} months (EMI {_rupees(L['emi'])}): AI approves")
        for x in (rc.get("habits") or {}).get("single") or []:
            opts.append(f"{x['change']}: risk falls to {100 * x['new_pd']:.1f}%")
        if opts:
            story += [Paragraph("<b>What would change the AI's decision (re-checked by the model):</b>", body)] + \
                     [Paragraph("&bull; " + clean(o), body) for o in opts]
        if result.get("ai_note"):
            story.append(Paragraph("Note: " + clean(result["ai_note"]), small))

    # 2. Request and applicant (no name / identity data in this memo)
    total = pol["new_emi"] * a["requested_duration_months"]
    story += [Paragraph("1. Loan request and applicant", h), kv([
        ("Amount", _rupees(a["requested_amount"])), ("Term", f"{a['requested_duration_months']} months"),
        ("Purpose", a["purpose"]), ("New EMI", f"{_rupees(pol['new_emi'])} at {result.get('interest_rate', 0.14):.0%} p.a. "
                                               f"(total repayable {_rupees(total)})"),
        ("Employment", f"{a['employment_type']}, {float(a.get('employment_years') or 0):g} years"),
        ("Housing", a["housing"]), ("Dependents", str(a.get("dependents"))),
        ("Vehicle", "yes" if a.get("owns_car") else "no"),
        ("Declared savings", _rupees(a.get("total_savings"))),
        ("Declared income", _rupees(a.get("monthly_income")) if a.get("monthly_income") else "not declared"),
    ] + ([("Price of the item", f"{_rupees(a['asset_price'])} (loan = "
                                  f"{float(a['requested_amount']) / float(a['asset_price']):.0%} of the price)")]
         if a.get("asset_price") else []))]

    # 3. Banking analysis
    inc = pol.get("income_used")
    story += [Paragraph("2. Banking analysis (from the statement)", h),
              Paragraph(clean(result.get("statement_read", "")), small), kv([
                  ("Verified income", f"{_rupees(inc)} / month" if inc else "none verified"),
                  ("Income basis", result.get("income_basis") or "n/a"),
                  ("Existing EMIs", f"{_rupees(pol.get('existing_emi'))} / month"),
                  ("All EMIs vs income", _pct(pol.get("foir")) if inc else "n/a"),
                  ("Average bank balance", _rupees(stmt_b.get("abb"))),
                  ("Lowest balance", _rupees(stmt_b.get("min_balance") if stmt_b.get("min_balance") is not None else
                                             min((m["min_balance"] for m in stmt_b.get("monthly") or []
                                                  if m.get("min_balance") is not None), default=None))),
                  ("Payments returned", str(stmt_b.get("bounce_events", 0))),
                  ("Penalty charges", str(stmt_b.get("penalty_charges", 0))),
                  ("Other lenders / apps", str(stmt_b.get("lenders_count", 0))),
                  ("Cash share of money in", _pct(stmt_b.get("cash_deposit_share"))),
              ])]
    monthly = stmt_b.get("monthly") or []
    if monthly:
        rows = [["Month", "Income counted", "Credits", "Debits", "EMIs", "Avg balance", "Lowest", "Bounces"]]
        for m in monthly:
            rows.append([m["month"] + (" *" if m.get("partial") else ""), _rupees(m.get("income_counted")),
                         _rupees(m.get("credits")), _rupees(m.get("debits")), _rupees(m.get("emis")),
                         _rupees(m.get("abb")), _rupees(m.get("min_balance")), str(m.get("bounces", 0))])
        story += [Spacer(1, 3), table(rows, [18, 24, 24, 24, 20, 24, 24, 14], size=7.4),
                  Paragraph("* partial month (shown, not used for averages). Average balance = mean of day-end balances.",
                            small)]
    if result.get("statement_insights"):
        story.append(Paragraph(clean(" · ".join(result["statement_insights"])), small))

    # 4. Policy checks
    rows = [["Check", "Result", "Detail"]]
    for c in pol["checks"]:
        rows.append([Paragraph(clean(c["name"]), body),
                     Paragraph(f'<font color="{STATUS_COLOR[c["status"]]}"><b>{STATUS_WORD[c["status"]]}</b></font>', body),
                     Paragraph(clean(c["detail"]), body)])
    story += [Paragraph("3. Lending-policy checks (hard stops; other cautions are conditions)", h), table(rows, [42, 17, 121])]

    # 5. Eligibility and offer
    elig, offer = result.get("eligibility") or {}, result.get("offer")
    if elig:
        lines = [f"<b>Eligible amount {_rupees(elig.get('eligible_amount'))}</b> over {elig.get('term')} months; "
                 f"binding limit: {clean(elig.get('binding', ''))}."]
        lim = elig.get("limits") or []
        if lim:
            lines.append("Limits: " + "; ".join(f"{clean(x['limit'])} {_rupees(x['amount'])}" for x in lim))
        if offer:
            o = offer[0]
            lines.append(f"<b>Counter-offer:</b> approvable at {_rupees(o['amount'])} over {o['term']} months "
                         f"(EMI {_rupees(o['emi'])}).")
        st_ = result.get("starter")
        if st_:
            lines.append(f"<b>Starter-loan ladder ({clean(st_['kind'])}):</b> {_rupees(st_['amount'])} over {st_['term']} "
                         f"months, EMI {_rupees(st_['emi'])} at {100 * st_['rate']:.1f}%, AI default risk at this size "
                         f"{100 * st_['pd']:.1f}%, expected loss {_rupees(st_['expected_loss'])}"
                         + (f", propensity {st_['propensity']:.2f} (learning offer, logged for inverse-propensity "
                            f"weighting)" if st_["kind"] == "learning" else "") + ". Next rungs: "
                         + "; ".join(f"{_rupees(x['limit'])} after {x['after_emis']} on-time EMIs" for x in st_["rungs"][1:])
                         + ".")
        story += [Paragraph("4. Eligibility", h)] + [Paragraph(x, body) for x in lines]

    # 6. FAIRScore
    if fs:
        rows = [["Factor", "Points", "Reason"]]
        for f in fs["factors"]:
            rows.append([Paragraph(clean(f["label"]), body), f"{f['points']}/{f['max']}", Paragraph(clean(f["reason"]), body)])
        story += [Paragraph(f"5. FAIRScore {fs['score']} ({fs['band']}): banking-behaviour scorecard, 300-900", h),
                  table(rows, [42, 17, 121])]

    # 7. Risk models
    story.append(Paragraph("6. Public-data benchmark models (research comparison; do not decide)", h))
    if result.get("consensus_percentile") is not None:
        story.append(Paragraph(
            f"<b>{result.get('risk_level', '')}</b>: riskier than {result['consensus_percentile']:.0f}% of past borrowers "
            f"with similar information.", body))
        rows = [["Model", "Risk percentile", "Vote weight"]]
        for name in result.get("voters", []):
            m = result["models"][name]
            rows.append([clean(m["label"]), f"{m['percentile']:.0f}", f"{m['weight']:.2f}"])
        story.append(table(rows, [100, 40, 40]))
    exp = result.get("explanation")
    if exp:
        favour = ", ".join(clean(f["label"]) for f in exp["risk_down"]) or "nothing notable"
        against = ", ".join(clean(f["label"]) for f in exp["risk_up"]) or "nothing notable"
        story += [Paragraph(f"<b>In favour:</b> {favour}", body), Paragraph(f"<b>Against:</b> {against}", body)]

    # 8. What would help / documents
    if result.get("recourse"):
        story += [Paragraph("7. What could change the outcome", h)] + \
                 [Paragraph("&bull; " + clean(r["change"]), body) for r in result["recourse"]]
    if result.get("documents"):
        story += [Paragraph("8. Documents to collect before sanction", h)] + \
                 [Paragraph("&#9744; " + clean(d), body) for d in result["documents"]]

    # 9. Sign-off
    sign = Table([[Paragraph("<b>Credit officer's decision</b> &nbsp;&nbsp; &#9744; Approve &nbsp; &#9744; Approve "
                             "with conditions &nbsp; &#9744; Decline", body)],
                  [Paragraph("Reason / conditions: ______________________________________________________________"
                             "______________________________", body)],
                  [Paragraph("Name / ID: ____________________ &nbsp;&nbsp; Signature: ____________________ "
                             "&nbsp;&nbsp; Date: ____________", body)]],
                 colWidths=[180 * mm], style=TableStyle([("BOX", (0, 0), (-1, -1), 0.6, NAVY),
                                                         ("TOPPADDING", (0, 0), (-1, -1), 5),
                                                         ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story += [Spacer(1, 8), KeepTogether([sign])]

    trace = [f"Policy {result.get('policy_version', 'n/a')}"]
    if result.get("model_versions"):
        trace.append("models " + ", ".join(f"{k} {v['hash']}" for k, v in result["model_versions"].items()))
    story += [Spacer(1, 5), Paragraph(
        "FAIRScore uses no credit-bureau data and no protected attribute (age only for the eligibility rule). The "
        "cash-flow model was trained on real loan outcomes from bank-account histories (Berka, Czech bank, 1993-98); "
        "the other models on public non-Indian credit data, so their percentiles are relative to those populations. "
        "The lender's Key Fact Statement with the actual rate, fees and APR must be given to the borrower before "
        "signing. Traceability: " + clean(" · ".join(trace)), small)]
    doc.build(story)
    return buf.getvalue()
