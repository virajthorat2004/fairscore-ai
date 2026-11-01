"""
FAIRScore 3.0 starter-loan ladder: what the bank can say YES to today when the AI is unsure about the full loan.

Why
    New-to-credit applicants are exactly the people a model is least sure about, and "refer" is often a dead end:
    an officer, seeing no credit history, declines. The ladder turns uncertainty into a small, priced, graduated
    exposure instead:

        rung 1  a starter loan the bank can afford to lose: at most `max_amount` and `income_multiple` months of
                verified income, short term, re-checked by the WHOLE decision (every policy rule, the AI and its
                conformal test at the starter size); the AI's default risk at that size must be <= `max_pd`
        rung 2  after N on-time EMIs, a larger limit, re-assessed with a fresh statement and the repayment record
        rung 3  the requested amount

    Each repaid rung is reported to the credit bureaus, so the applicant leaves new-to-credit: the ladder builds the
    credit history the bureau-first system demands.

Learning offers (exploration)
    A model retrained only on loans it approved never learns about the people it turned away (selection bias,
    src/exploration_study.py). A small, RANDOM, logged share (`explore_share`) of the cases slightly above `max_pd`
    (never a confident "defaults", never past a policy hard stop) also receives a starter loan. The propensity is
    recorded with the decision, so retraining can weight these outcomes by 1 / propensity (inverse-propensity
    weighting) and see the whole population. The randomisation is a deterministic hash of the application, so
    the same application always gets the same answer and the decision is reproducible for audit.

The requested loan's verdict does not change ("refer" stays "refer"); the starter offer is returned next to it.
"""
import hashlib
import json

import numpy as np

import ai_engine
from policy_config import POLICY


def _amount(a, income):
    S = POLICY["starter"]
    cap = min(S["max_amount"], S["income_multiple"] * float(income), float(a["requested_amount"]))
    return float(np.floor(cap / 1000) * 1000)


def _u(a, stmt):
    """Uniform(0, 1) number fixed for this application (customer id if the lender passes one)."""
    key = {"customer": a.get("customer_id"), "age": a.get("age"), "emp": a.get("employment_type"),
           "amount": a.get("requested_amount"), "term": a.get("requested_duration_months"),
           "period": [(stmt or {}).get("period_start"), (stmt or {}).get("period_end")],
           "txns": (stmt or {}).get("transactions"), "sha": (stmt or {}).get("sha256")}
    h = hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()
    return int(h[:15], 16) / 16 ** 15


def rungs(first_amount, requested):
    S = POLICY["starter"]
    out = [{"step": 1, "after_emis": 0, "limit": first_amount}]
    for i, (n, mult) in enumerate(S["rungs"], start=2):
        lim = float(min(np.floor(first_amount * mult / 1000) * 1000, requested))
        if lim > out[-1]["limit"]:
            out.append({"step": i, "after_emis": n, "limit": lim})
    if out[-1]["limit"] < requested:
        last_n = S["rungs"][-1][0] if S["rungs"] else 12
        # the full amount comes one review (6 more EMIs) after the last rung that was actually reached
        after = max(last_n, out[-1]["after_emis"] + 6) if out[-1]["after_emis"] else last_n
        out.append({"step": len(out) + 1, "after_emis": after, "limit": float(requested),
                    "note": "the requested amount, re-assessed"})
    return out


def offer(a, stmt, stmt_df, result, quick_decision):
    """Starter offer for an application whose requested loan was referred, or None.
    `quick_decision` is live_scoring.quick_decision (passed in to avoid a circular import)."""
    S = POLICY["starter"]
    by = result.get("decided_by")
    if not S["enabled"] or result.get("verdict") != "refer" or by not in ("AI", "policy (AI advisory)"):
        return None
    if by == "policy (AI advisory)" and not S["advisory_segments"]:
        return None
    income = (result.get("policy") or {}).get("income_used")
    if not income:
        return None
    amount = _amount(a, income)
    if amount < POLICY["min_amount"]:
        return None
    settings = POLICY["ai"]
    u = _u(a, stmt)
    terms = [t for t in S["terms"] if POLICY["min_term_months"] <= t <= max(int(a["requested_duration_months"]), min(S["terms"]))]
    for term in terms:
        q = quick_decision({**a, "requested_amount": amount, "requested_duration_months": term}, stmt, stmt_df)
        ai = q.get("ai")
        statuses = [c["status"] for c in q["policy"]["checks"]]
        forced = [c["key"] for c in q["policy"]["checks"] if c["status"] == "caution" and c["key"] in settings["force_refer"]]
        if "fail" in statuses or forced or ai is None:
            continue
        kind, propensity = None, 1.0
        if q["verdict"] == "approve":
            kind = "approved"                      # the small loan passes the normal rules outright
        elif q.get("decided_by") == "AI" and ai["outcome_set"] != "defaults":
            if ai["pd"] <= S["max_pd"]:
                kind = "starter"
            elif ai["pd"] <= S["explore_max_pd"] and u < S["explore_share"]:
                kind, propensity = "learning", S["explore_share"]
        elif (q.get("decided_by") == "policy (AI advisory)" and ai["outcome_set"] != "defaults"
              and ai["pd"] <= S["max_pd"]):
            kind = "suggested"                     # AI advisory for this applicant type: an officer confirms
        if kind is None:
            continue
        lgd = ai.get("lgd", settings["lgd"])
        rate, _ = ai_engine.risk_based_rate(ai["pd"], term, {**settings, "lgd": lgd})
        from live_scoring import emi
        return {"kind": kind, "amount": amount, "term": term, "emi": emi(amount, term, rate), "rate": rate,
                "pd": ai["pd"], "outcome_set": ai["outcome_set"], "expected_loss": ai["pd"] * lgd * amount,
                "propensity": propensity, "rungs": rungs(amount, float(a["requested_amount"])),
                "auto": kind in ("approved", "starter", "learning"),
                "label": LABELS[kind]}
    return None


LABELS = {"approved": "Starter loan approved: it passes every rule at this size",
          "starter": "Starter loan approved: the AI is unsure about the full amount, but the risk at this size is small",
          "learning": "Starter loan approved as a learning offer (random, logged): the bank learns about applicants "
                      "like this one",
          "suggested": "Starter loan suggested: the AI is advisory for this applicant type, an officer confirms"}


# --------------------------------------------------------------------------------------------- Berka evidence
def evidence():
    """How the ladder would have changed the outcomes of the 679 Berka loans (nested out-of-fold decisions,
    results/ai_engine_oof.csv). Conservative: a starter loan is assumed to default exactly when the full loan did
    (smaller loans are, if anything, safer: the model is monotone in loan size)."""
    import pandas as pd
    from config import PROJECT_ROOT
    S = POLICY["starter"]
    lgd = POLICY["ai"]["lgd"]
    o = pd.read_csv(PROJECT_ROOT / "results" / "ai_engine_oof.csv")
    d = pd.read_csv(PROJECT_ROOT / "results" / "berka_backtest_loans.csv")
    loans = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "berka" / "loan.csv", sep=";")[["loan_id", "amount"]]
    d = d.merge(loans, on="loan_id", how="left")
    amount = d["amount"].values[o["i"]].astype(float)
    income_m = amount / (12 * d["loan_to_annual_income"].values[o["i"]].clip(min=1e-6))
    start = np.minimum(amount, S["income_multiple"] * income_m)     # first rung (statement currency)
    rng = np.random.default_rng(0)
    u = rng.random(len(o))
    referred_ai = (o["v2"] == "refer") & (o["by"] == "ai") & (o["set"] != "defaults")
    starter = referred_ai & (o["pd"] <= S["max_pd"])
    learning = referred_ai & ~starter & (o["pd"] <= S["explore_max_pd"]) & (u < S["explore_share"])
    approve = o["v2"] == "approve"
    y = o["y"].values
    A = POLICY["ai"]
    term_full = d["term_months"].values[o["i"]].astype(float)
    term_start = float(S["terms"][0])
    rate_full = np.full(len(o), POLICY["annual_interest_rate"])
    rate_start = np.array([ai_engine.risk_based_rate(p, term_start, A)[0] for p in o["pd"].values])
    rows = []

    def row(name, m, exposure, rate, term):
        # indicative profit, as in src/train_ai_engine.py: a repaid loan earns the margin over funding and operating
        # cost on ~55% average outstanding over its life; a default loses LGD of the amount
        gain = (rate - A["cost_of_funds"] - A["operating_cost"]) * 0.55 * term / 12.0
        prof = exposure * np.where(y == 1, -lgd, gain)
        rows.append({"group": name, "share_of_applicants": float(m.mean()),
                     "default_rate": float(y[m].mean()) if m.any() else np.nan,
                     "exposure_per_applicant": float(exposure[m].sum() / len(m)),
                     "loss_per_100_of_exposure": float(100 * lgd * (exposure * y)[m].sum() / max(exposure[m].sum(), 1e-9)),
                     "loss_per_applicant": float(lgd * (exposure * y)[m].sum() / len(m)),
                     "profit_per_applicant": float(prof[m].sum() / len(m)),
                     "mean_rate": float(np.mean(rate[m])) if m.any() else np.nan})

    row("approved by the AI (full amount)", approve.values, amount, rate_full, term_full)
    row("referred by the AI, not confident of default", referred_ai.values, amount, rate_full, term_full)
    row("  ... of which starter loan (PD <= max_pd)", starter.values, start, rate_start, np.full(len(o), term_start))
    row("  ... of which learning offer (random 5% above max_pd)", learning.values, start, rate_start,
        np.full(len(o), term_start))
    row("  ... the same starter cases if lent the FULL amount", starter.values, amount, rate_full, term_full)
    credit = (approve | starter | learning).values
    row("offered credit today: approval or starter (version 3)", credit, np.where(approve, amount, start),
        np.where(approve, rate_full, rate_start), np.where(approve, term_full, term_start))
    row("offered credit today: approval only (version 2)", approve.values, amount, rate_full, term_full)
    # stress test: the AI is UNSURE about these applicants, and in a new market its PDs can be wrong. What if the
    # starter group's true default rate were k times what Berka shows? Expected profit per applicant of the group:
    m = starter.values
    for k in (1.0, 1.5, 2.0, 3.0):
        dr = min(1.0, k * float(y[m].mean()))
        for name, exp_, rate, term in (("starter loan", start, rate_start, np.full(len(o), term_start)),
                                       ("full amount", amount, rate_full, term_full)):
            gain = (rate - A["cost_of_funds"] - A["operating_cost"]) * 0.55 * term / 12.0
            prof = float(np.sum(exp_[m] * ((1 - dr) * gain[m] - dr * lgd)) / len(m))
            rows.append({"group": f"stress: default rate x{k:g} ({dr:.1%}), {name}", "share_of_applicants": float(m.mean()),
                         "default_rate": dr, "profit_per_applicant": prof,
                         "loss_per_applicant": float(np.sum(exp_[m] * dr * lgd) / len(m))})
    out = pd.DataFrame(rows)
    out["starter_size_vs_requested"] = float(np.median(start[starter] / amount[starter])) if starter.any() else np.nan
    out.to_csv(PROJECT_ROOT / "results" / "starter_ladder_evidence.csv", index=False)
    return out


if __name__ == "__main__":
    import pandas as pd
    pd.set_option("display.width", 200)
    print(evidence().round(4).to_string(index=False))
