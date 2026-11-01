"""
FAIRScore banking-behaviour score (300-900).

An expert (judgemental) scorecard of the kind banks use for new-to-credit
customers before they have their own repayment data: every factor is read
from the bank statement or the application, earns points up to a maximum set
in config/policy.toml, and every point is explained. It uses no age, gender
or other protected attribute and no credit-bureau data.

Each factor's points rise (or stay level) as behaviour improves, so the score
can never reward a worse habit. Once repayment outcomes are recorded,
src/train_from_outcomes.py re-estimates the factor weights from them.
"""
import numpy as np

from policy_config import POLICY

SALARIED = ("Salaried (private sector)", "Government employee", "Pensioner")
SELF_EMPLOYED = ("Self-employed / business owner", "Freelancer / gig worker", "Farmer / agriculture")


def _lin(x, points):
    """Piecewise-linear interpolation through (x, value) points sorted by x."""
    xs, ys = zip(*points)
    return float(np.interp(x, xs, ys))


def _inr(x):
    return f"−₹{abs(x):,.0f}" if x < 0 else f"₹{x:,.0f}"


def factors(a, stmt, pol):
    """{factor: (fraction 0-1, reason)} for one application."""
    stmt = stmt or {}
    b = stmt.get("banking") or {}
    emp = a.get("employment_type") or ""
    income = pol.get("income_used") or 0
    all_emi = (pol.get("existing_emi") or 0) + (pol.get("new_emi") or 0)
    out = {}

    bounces, pen = b.get("bounce_events", 0), b.get("penalty_charges", 0)
    f = {0: 1.0, 1: 0.45, 2: 0.2}.get(bounces, 0.0)
    if bounces == 0 and pen:
        f = 0.8 if pen <= 2 else 0.6
    out["conduct"] = (f, "No payment returned unpaid" if not bounces and not pen else
                      f"{bounces} payment(s) returned unpaid, {pen} penalty charge(s)")

    foir = pol.get("foir", float("inf"))
    f = 0.0 if not income or foir == float("inf") else _lin(foir, [(0.2, 1.0), (0.3, 0.9), (0.45, 0.6), (0.55, 0.2),
                                                                   (0.7, 0.0)])
    out["affordability"] = (f, "No verified income" if not income else f"All EMIs would be {foir:.0%} of income")

    cv = stmt.get("income_volatility", 1.0) if stmt else 1.0
    f = _lin(cv, [(0.05, 1.0), (0.25, 0.6), (0.5, 0.2), (1.0, 0.05)]) if income else 0.0
    if emp in SALARIED and income:
        share = stmt.get("regular_income_months_share", 0.0)
        f *= max(share, 0.2) if str(stmt.get("income_basis", "")).startswith("regular") else 0.4
        if b.get("salary_stopped"):
            f *= 0.3
    out["income_stability"] = (f, "No verified income" if not income else f"Monthly income varies by {cv:.0%}")

    abb = b.get("abb")
    if abb is None:
        out["balance_cover"] = (0.3, "No running balance to measure")
    else:
        cover = abb / max(all_emi, 1.0)
        out["balance_cover"] = (_lin(cover, [(0.0, 0.0), (0.5, 0.1), (1.0, 0.35), (1.5, 0.6), (3.0, 1.0)]),
                                f"Average balance {_inr(abb)} = {cover:.1f}x the monthly EMIs")

    low, neg = b.get("low_balance_days_share"), b.get("negative_days_share")
    if low is None:
        out["liquidity"] = (0.4, "No running balance to measure")
    else:
        f = 1.0 - 0.6 * min(1.0, low * 1.2) - 0.4 * min(1.0, (neg or 0) * 5)
        out["liquidity"] = (max(f, 0.0), f"Balance under ₹1,000 on {low:.0%} of days, negative on {neg or 0:.0%}")

    spend = pol.get("monthly_spend")
    if income and spend is not None:
        r = (income - spend - (pol.get("new_emi") or 0)) / income
        out["residual"] = (_lin(r, [(-0.2, 0.0), (0.0, 0.15), (0.1, 0.5), (0.3, 1.0)]),
                           f"{r:.0%} of income left after spending and the new EMI")
    else:
        out["residual"] = (0.0, "No verified income")

    n_l = b.get("lenders_count", 0)
    f = {0: 1.0, 1: 0.8, 2: 0.55, 3: 0.3}.get(n_l, 0.0)
    if b.get("new_loans_in_period", 0) >= 2:
        f = max(0.0, f - 0.2)
    out["credit_appetite"] = (f, "No NBFC / loan-app activity" if not n_l else f"{n_l} lender(s) / loan app(s) used")

    bet = stmt.get("betting_ratio", 0.0) if stmt else 0.0
    out["gambling"] = (_lin(bet, [(0.0, 1.0), (0.005, 0.8), (0.02, 0.6), (0.1, 0.2), (0.2, 0.0)]),
                       "No gambling / betting" if not bet else f"Gambling is {bet:.1%} of spending")

    inv = stmt.get("avg_monthly_investments", 0.0) if stmt else 0.0
    if income and inv > 0:
        f = _lin(inv / income, [(0.0, 0.6), (0.1, 1.0)])
        reason = f"Invests {_inr(inv)} a month (SIP / insurance / deposits)"
    else:
        trend = (b.get("abb_last_3_months") or 0) >= (b.get("abb") or 0) if abb is not None else False
        f, reason = (0.5, "Balance held up over the last 3 months") if trend else (0.3, "No regular saving seen")
    out["savings"] = (f, reason)

    months = (stmt or {}).get("months_covered", 0) or 0
    info = (stmt or {}).get("read_info") or {}
    rec = info.get("reconciled")
    origin = (info.get("file_origin") or {}).get("category")
    f = 0.5 * min(months / 6.0, 1.0)
    f += 0.4 if rec is not None and rec >= 0.999 else 0.2 if rec is not None and rec >= 0.98 else 0.0
    f += {"bank-generated": 0.1, "re-saved": 0.05}.get(origin, 0.0)
    out["evidence"] = (min(f, 1.0), f"{months:.1f} months, " + (f"{rec:.0%} of balances reconciled" if rec is not None
                                                                   else "balances not verifiable"))

    years = float(a.get("employment_years") or 0)
    if emp == "Unemployed":
        f = 0.0
    elif emp in SELF_EMPLOYED:
        f = _lin(years, [(0, 0.1), (2, 0.55), (3, 0.75), (5, 1.0)])
    else:
        f = _lin(years, [(0, 0.15), (0.5, 0.35), (1, 0.55), (2, 0.75), (5, 1.0)])
    out["tenure"] = (f, f"{years:g} years " + ("farming" if emp.startswith("Farmer") else "in current business / work"
                                                if emp in SELF_EMPLOYED else "in current work"))

    dep, wd = b.get("cash_deposit_share", 0.0) or 0.0, b.get("cash_withdrawal_share", 0.0) or 0.0
    if emp in SELF_EMPLOYED:
        f = _lin(dep, [(0.5, 1.0), (0.8, 0.6)])
    else:
        f = _lin(dep, [(0.1, 1.0), (0.3, 0.6), (0.6, 0.2)])
    if wd > 0.5:
        f *= 0.7
    out["cash"] = (f, f"Cash deposits {dep:.0%} of money in, cash withdrawals {wd:.0%} of money out")

    # Household bills paid month after month (alternative data for new-to-credit applicants).
    # Bonus only: an applicant whose bills are paid by family or by card is neutral (half
    # marks), never penalised, so people living with parents are not marked down.
    n, of = (stmt or {}).get("bill_months"), (stmt or {}).get("bill_months_of")
    if n and of:
        share = min(n / of, 1.0)
        out["bills"] = (0.5 + 0.5 * share, f"Household bills (electricity, phone, rent, insurance) paid in {n} of "
                                           f"{of} months")
    else:
        out["bills"] = (0.5, "No household bills paid from this account (neutral: they may be paid by family or "
                             "by card)")

    # Own contribution to a purchase (loan-to-value). Only scored when the price is known; otherwise the
    # factor is left out of the total, so loans without an item to buy are not affected. The steps follow
    # first-EMI default rates on 116,950 Indian new-to-credit vehicle loans (results/india_ntc_ltv.csv).
    try:
        price = float(a.get("asset_price") or 0)
    except (TypeError, ValueError):
        price = 0.0
    # ---- applicant profile (form facts) -------------------------------------------------------------
    out["employment"] = ({"Government employee": 1.0, "Pensioner": 0.95, "Salaried (private sector)": 0.8,
                          "Self-employed / business owner": 0.7, "Farmer / agriculture": 0.65,
                          "Freelancer / gig worker": 0.6, "Student": 0.5, "Unemployed": 0.2}.get(emp, 0.6),
                         f"{emp or 'Not stated'}")
    housing = a.get("housing") or ""
    f = {"Own home": 1.0, "Living with family (no rent)": 0.65, "Rented home": 0.6}.get(housing, 0.6)
    f = min(1.0, f + (0.15 if a.get("owns_car") else 0.0))
    out["home_assets"] = (f, f"{housing or 'Housing not stated'}" + (", owns a vehicle" if a.get("owns_car") else ""))
    dep = int(a.get("dependents") or 0)
    out["dependents"] = ({0: 1.0, 1: 0.9, 2: 0.8, 3: 0.65}.get(dep, 0.5), f"{dep} dependent(s) on this income")
    sav = float(a.get("total_savings") or 0)
    ratio = sav / max(float(a.get("requested_amount") or 1), 1.0)
    out["savings_buffer"] = (_lin(ratio, [(0.0, 0.3), (0.1, 0.55), (0.25, 0.8), (0.5, 1.0)]),
                             f"Declared savings {_inr(sav)} = {ratio:.0%} of the loan (to be verified)")

    if price > 0:
        ltv = float(a["requested_amount"]) / price
        out["own_contribution"] = (_lin(ltv, [(0.6, 1.0), (0.7, 0.8), (0.8, 0.55), (0.85, 0.35), (0.9, 0.2),
                                              (1.0, 0.0)]),
                                   f"Loan is {ltv:.0%} of the price; the borrower pays {max(0.0, 1 - ltv):.0%}")
    else:
        out["own_contribution"] = (None, "No purchase price entered (not scored)")
    return out


LABELS = {"conduct": "Account conduct", "affordability": "Affordability", "income_stability": "Income regularity",
          "balance_cover": "Balance cover", "liquidity": "Liquidity", "residual": "Money left over",
          "credit_appetite": "Other borrowing", "gambling": "Gambling", "savings": "Saving habit",
          "evidence": "Evidence quality", "tenure": "Work / business tenure", "cash": "Cash dependence",
          "bills": "Bill payments", "own_contribution": "Own contribution", "employment": "Type of work",
          "home_assets": "Home and assets", "dependents": "Dependents", "savings_buffer": "Savings vs loan"}


def band(score):
    for name, cut in POLICY["score_bands"].items():
        if score >= cut:
            return name
    return "Poor"


def fairscore(a, stmt, pol):
    """{'score', 'band', 'factors': [...], 'fractions': {...}}."""
    fs = factors(a, stmt, pol)
    # a factor that does not apply to this loan (share None) is left out of the total
    weights = {k: w for k, w in POLICY["score_points"].items() if fs.get(k, (None,))[0] is not None}
    total_max = sum(weights.values())
    points = {k: weights[k] * fs[k][0] for k in weights}
    q = sum(points.values()) / total_max
    score = int(round(300 + 600 * q ** POLICY["score_curve"]))
    rows = [{"key": k, "label": LABELS[k], "points": round(points[k]), "max": weights[k],
             "share": round(fs[k][0], 3), "reason": fs[k][1]} for k in weights]
    rows.sort(key=lambda r: r["points"] - r["max"])          # weakest factors first
    return {"score": score, "band": band(score), "factors": rows,
            "fractions": {k: round(fs[k][0], 4) for k in weights}}
