"""
Decision letter for the applicant, in plain language.

RBI's Fair Practices Code asks lenders to tell applicants, in writing, the main
reasons a loan is not sanctioned. Credit-officer wording ("FOIR 48%, take-home
check caution") is translated into what the applicant can understand and act
on: the outcome, the main reasons, what would help, and how the assessment was
made -- including what was NOT used (no credit-bureau data, no gender, religion
or caste; age only for the legal / product age limit) and the right to ask a
credit officer to review it.
"""
import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

PLAIN = {
    "affordability": "The new EMI together with your existing EMIs would take too large a share of the income we "
                     "could verify from your statement.",
    "take_home": "After paying all EMIs, too little of your income would remain for household expenses.",
    "loan_size": "The amount requested is large compared with the income we could verify.",
    "income_source": "We could not see your regular salary or business income being received in this account.",
    "stability": "Your monthly income varied a lot over the statement period.",
    "conduct": "Some payments from your account were returned unpaid, or the account often went below zero.",
    "evidence": "We need a bank statement covering the last 6 months.",
    "recency": "We need your bank statement for the most recent months.",
    "integrity": "We need to verify your statement directly with your bank.",
    "tamper": "We need to verify your statement directly with your bank.",
    "file_origin": "We need to verify your statement directly with your bank.",
    "gambling": "Your statement shows spending on betting or gaming apps.",
    "lenders": "Your statement shows several recent loans from other lenders or loan apps.",
    "eligibility": "The loan, as requested, is outside this product's eligibility rules (for example the term "
                   "would run past the maximum age for this loan).",
    "tenure": "You have been in your current job or business for a short time.",
    "consistency": "The loans you told us about differ from the EMI payments on your statement.",
    "income_support": "The income on your statement is lower than the income you declared.",
    "holder_match": "We need to confirm that the bank account is in your name.",
    "holders": "The statements you gave us appear to belong to different people.",
    "source_of_funds": "A large recent deposit in your account needs proof of where it came from.",
    "salary_trend": "Your most recent salary was lower than usual, or did not arrive.",
    "balance_cover": "The balance kept in your account is low compared with the EMI.",
    "own_contribution": "The loan would cover too large a share of the item's price; a larger down payment would help.",
    "profile_match": "The income on your statement looks like a full-time salary, while you told us you are a "
                     "student; we need to confirm which applies.",
    "student_limits": "For students we keep loans smaller and shorter, or ask for a co-applicant with income.",
    "pass_through": "Money received in your account is often sent back to the same people, so we need to "
                    "understand these transfers.",
}
OUTCOME = {
    "approve": ("Recommended for approval",
                "Your application has been recommended for approval, subject to verification of your documents "
                "and the lender's final sanction."),
    "refer": ("Under review by a credit officer",
              "Your application needs a closer look by our credit officer before a decision is made. You may be "
              "contacted for the documents listed below."),
    "decline": ("Not approved as requested",
                "We are unable to approve the loan as requested at this time."),
}


AI_PLAIN = {
    "Balance kept in the account": "The balance kept in your account is often very low compared with your income "
                                   "and the EMI.",
    "Affordability of the EMI": "After your current spending, the new EMI would leave little money each month.",
    "Income regularity": "Your income varies a lot from month to month, or is not regular.",
    "Repayment conduct": "Some payments from your account were returned unpaid or attracted penalty charges.",
    "Other borrowing and gambling": "Your statement shows borrowing from other lenders or loan apps, or betting.",
    "Saving and bill habits": "We saw little regular saving or regular payment of household bills.",
    "Statement completeness": "The statement does not cover enough months to judge your banking pattern well.",
}


def plain_reasons(result):
    ai = result.get("ai")
    if ai and result.get("decided_by") == "AI" and result["verdict"] != "approve":
        out = [AI_PLAIN[g["group"]] for g in ai["explanation"]["groups"] if g["impact"] > 0.05 and g["group"] in AI_PLAIN]
        if out:
            return out[:3]
    reasons, seen = [], set()
    for c in result["policy"]["checks"]:
        if c["status"] in ("fail", "caution"):
            text = PLAIN.get(c["key"])
            if text and text not in seen:
                reasons.append((0 if c["status"] == "fail" else 1, text))
                seen.add(text)
    reasons.sort(key=lambda r: r[0])
    out = [t for _, t in reasons]
    if result["verdict"] != "approve" and not out:
        out.append("Our overall assessment of the banking pattern in your statement needs a closer review by a "
                   "credit officer.")
    return out[:4]


def build_letter(applicant, result, reference):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=22 * mm, rightMargin=22 * mm, topMargin=20 * mm,
                            bottomMargin=18 * mm, title="Loan application decision")
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=10, leading=14)
    h = ParagraphStyle("h", parent=ss["Heading3"], fontSize=11, spaceBefore=8, spaceAfter=3,
                       textColor=colors.HexColor("#16324A"))
    small = ParagraphStyle("s", parent=body, fontSize=8, leading=11, textColor=colors.HexColor("#5B6270"))

    def clean(t):
        return str(t).replace("₹", "Rs ").replace("−", "-").replace("&", "&amp;").replace("<", "&lt;")

    a = applicant
    title, opening = OUTCOME[result["verdict"]]
    story = [Paragraph(f"Reference {clean(reference)} &nbsp;&middot;&nbsp; {datetime.now():%d %B %Y}", small),
             Spacer(1, 8), Paragraph("Dear Applicant,", body), Spacer(1, 4),
             Paragraph(f"Thank you for applying for a loan of <b>Rs {a['requested_amount']:,.0f}</b> over "
                       f"<b>{a['requested_duration_months']} months</b> ({clean(a['purpose'])}).", body),
             Paragraph(f"<b>Outcome: {title}.</b> {opening}", body)]
    offer = result.get("offer")
    if offer and result["verdict"] != "approve":
        o = offer[0]
        story.append(Paragraph(f"We could consider a loan of <b>Rs {o['amount']:,.0f}</b> over <b>{o['term']} months</b> "
                               f"(EMI about Rs {o['emi']:,.0f}). Please let us know if you would like to proceed "
                               f"with this.", body))
    st_ = result.get("starter")
    if st_ and not offer and st_.get("kind") != "suggested":
        steps = "; ".join(f"Rs {x['limit']:,.0f} after {x['after_emis']} on-time EMIs" for x in st_["rungs"][1:])
        story.append(Paragraph(f"<b>Starter loan available today:</b> Rs {st_['amount']:,.0f} over {st_['term']} months "
                               f"(EMI about Rs {st_['emi']:,.0f}). Repaying it on time builds your credit history with "
                               f"the credit bureaus" + (f", and opens larger limits: {steps}." if steps else ".") , body))
    reasons = plain_reasons(result)
    if reasons and result["verdict"] != "approve":
        story.append(Paragraph("Main reasons", h))
        story += [Paragraph("&bull; " + clean(r), body) for r in reasons]
    help_ = [r["change"] for r in result.get("recourse") or []]
    rc = (result.get("ai") or {}).get("recourse") or {}
    if rc.get("loan"):
        L = rc["loan"]
        help_.insert(0, f"A loan of Rs {L['amount']:,.0f} over {L['term']} months (EMI about Rs {L['emi']:,.0f}) "
                        f"would meet our criteria today.")
    help_ += [x["change"] for x in (rc.get("habits") or {}).get("single") or []]
    if help_ and result["verdict"] != "approve":
        story.append(Paragraph("What could help", h))
        story += [Paragraph("&bull; " + clean(x), body) for x in help_[:4]]
    if result.get("documents"):
        story.append(Paragraph("Documents we may ask for", h))
        story += [Paragraph("&bull; " + clean(d), body) for d in result["documents"]]
    story += [Paragraph("How your application was assessed", h), Paragraph(
        "We read the bank statement(s) you provided: the income received, existing EMIs, balances, returned "
        "payments and spending, together with the details on your application. <b>We did not use any credit "
        "bureau score or credit history</b>, and we did not use your gender, religion or caste. Your age was used "
        "only to check the product's age limits. Every assessment is also re-checked to confirm that the "
        "recommendation would be the same at a different age.", body),
        Paragraph("Your rights", h), Paragraph(
        "You can ask for this assessment to be reviewed by a credit officer, and to correct any information that "
        "is wrong, by quoting the reference above. Your statement file itself is not stored; only a summary of "
        "this assessment is kept for our records.", body),
        Spacer(1, 10), Paragraph("Yours sincerely,<br/>Credit team", body), Spacer(1, 14),
        Paragraph("This letter was prepared with FAIRScore AI, a decision-support system. The final decision on "
                  "any loan is taken by an authorised credit officer of the lender.", small)]
    doc.build(story)
    return buf.getvalue()
