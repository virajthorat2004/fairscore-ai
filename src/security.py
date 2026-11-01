"""
Collateral (security) for a loan: gold, a fixed deposit, or property.

How a bank uses it, and how FAIRScore does:
  * Collateral does NOT show whether a person CAN repay -- that is the bank statement's job (probability of
    default, PD). It shows how much the bank LOSES if they do not (loss given default, LGD). Expected loss =
    PD x LGD x amount, so good collateral lowers the loss, the interest rate and lets the bank accept a somewhat
    higher PD -- but never a loan the borrower cannot afford: the affordability hard stops still apply.
  * The loan may not exceed the collateral's value times a loan-to-value (LTV) cap. For gold the caps follow the
    RBI (Lending Against Gold and Silver Collateral) Directions, 2025: 85% up to Rs 2.5 lakh, 80% up to Rs 5 lakh,
    75% above; those Directions also require a repayment-capacity assessment for gold loans above Rs 2.5 lakh.
    Fixed-deposit and property caps are lender settings (config/policy.toml, [security]).
"""
from policy_config import POLICY

SECURITY_TYPES = ["None", "Gold", "Fixed deposit", "Property (loan against property)"]
_KEY = {"Gold": "gold", "Fixed deposit": "deposit", "Property (loan against property)": "property"}


def ltv_cap(kind, amount, S=None):
    S = S or POLICY["security"]
    if kind == "Gold":
        for upto, cap in S["gold_ltv"]:
            if amount <= upto:
                return float(cap)
        return float(S["gold_ltv"][-1][1])
    return float(S[f"{_KEY[kind]}_ltv"]) if kind in _KEY else 0.0


def assess(a, lgd_unsecured=None):
    """None for an unsecured loan; otherwise the cover, the effective LGD and the check."""
    S = POLICY["security"]
    kind = a.get("security_type") or "None"
    try:
        value = float(a.get("security_value") or 0)
    except (TypeError, ValueError):
        value = 0.0
    if kind not in _KEY or value <= 0:
        return None
    lgd_u = POLICY["ai"]["lgd"] if lgd_unsecured is None else lgd_unsecured
    amount = float(a["requested_amount"])
    cap = ltv_cap(kind, amount)
    max_amount = value * cap
    coverage = min(1.0, max_amount / amount) if amount > 0 else 0.0
    lgd_s = float(S["lgd"][_KEY[kind]])
    lgd = coverage * lgd_s + (1 - coverage) * lgd_u
    ltv = amount / value
    ok = amount <= max_amount + 1
    note = (" Small gold loan: RBI does not require a credit appraisal up to Rs 2.5 lakh; FAIRScore still checks "
            "affordability." if kind == "Gold" and amount <= S["gold_no_appraisal_upto"] else "")
    check = {"key": "security", "name": "Security cover", "status": "pass" if ok else "caution",
             "value": round(ltv, 3),
             "detail": (f"{kind} worth Rs {value:,.0f}: the loan is {ltv:.0%} of its value (limit {cap:.0%}, so at most "
                        f"Rs {max_amount:,.0f}). Loss if the borrower defaults falls from {lgd_u:.0%} to {lgd:.0%} of "
                        f"the amount." + note) if ok else
                       (f"{kind} worth Rs {value:,.0f} allows at most Rs {max_amount:,.0f} ({cap:.0%} of its value); "
                        f"the request is {ltv:.0%} of its value")}
    return {"type": kind, "value": value, "ltv": ltv, "ltv_cap": cap, "max_amount": max_amount,
            "coverage": coverage, "lgd": lgd, "lgd_secured": lgd_s, "lgd_unsecured": lgd_u, "check": check}


DOCUMENTS = {"Gold": "Gold appraisal (purity and weight) by the lender's appraiser; ornaments kept in safe custody",
             "Fixed deposit": "Fixed deposit receipt and lien marked in the lender's favour",
             "Property (loan against property)": "Title deed, encumbrance certificate and an independent valuation report"}
