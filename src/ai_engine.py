"""
FAIRScore AI decision engine (2.0; 3.0 adds the learning-loop adaptation, see src/learning_loop.py).

What it does for one application
    1. Probability of default (PD): a monotone gradient-boosted model (CatBoost) on two kinds of input
       read from the bank statement -- 12 cash-flow ratios and the 12 behaviour scores of the expert
       FAIRScore scorecard ("knowledge-guided" machine learning: expert knowledge as model inputs).
       Calibrated (Platt scaling on out-of-fold predictions) so 0.04 means "about 4 in 100 such
       borrowers defaulted".
    2. Uncertainty with a guarantee: Mondrian (class-conditional) conformal prediction. The engine
       returns the set of outcomes it cannot rule out at significance alpha. Only when the set is
       {repays} is the case decided automatically; then, for exchangeable data, at most alpha of the
       borrowers who would default can ever be auto-approved. Otherwise the case goes to an officer:
       "Refer" means the AI is genuinely unsure, not that a rule tripped.
    3. Knows what it does not know: an Isolation Forest trained on the training applicants measures
       how unusual an applicant is. Outside the AI's experience the engine defers to the expert
       policy path instead of guessing.
    4. Explanations: exact SHAP contributions (CatBoost TreeSHAP) per applicant, in plain language.
    5. Pricing: expected loss (PD x LGD x EAD) and a risk-based interest rate.

Trained by src/train_ai_engine.py on 679 real loans with their pre-loan bank statements (Berka).
"""
import pickle
from functools import lru_cache

import numpy as np

import cashflow_model as cm
from config import PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "models" / "ai_engine.pkl"

# the expert scorecard's statement-based behaviour scores (0-1, higher = better) used as model inputs;
# the profile factors (type of work, home, dependents, savings, tenure) are NOT inputs: they are
# self-declared and could proxy for protected attributes.
EXPERT_FACTORS = ["conduct", "affordability", "income_stability", "balance_cover", "liquidity", "residual",
                  "credit_appetite", "gambling", "savings", "evidence", "cash", "bills"]
FEATURES = list(cm.FEATURES) + [f"sc_{k}" for k in EXPERT_FACTORS]
MONOTONE = [cm.MONOTONE.get(f, 0) for f in cm.FEATURES] + [-1] * len(EXPERT_FACTORS)
CATBOOST_PARAMS = dict(cm.CATBOOST_PARAMS)

LABELS = {
    "foir": "All EMIs vs income", "emi_to_income": "New EMI vs income", "loan_to_annual_income": "Loan vs yearly income",
    "term_months": "Loan length", "income_volatility": "Month-to-month income change",
    "regular_income_share": "Share of income that is regular", "overdraft_share": "Times balance went below zero",
    "penalty_charges": "Bounce / penalty charges", "low_balance_days": "Days with balance under Rs 1,000",
    "negative_days": "Days with negative balance", "abb_to_emi": "Average balance vs EMI",
    "min_balance_to_income": "Lowest balance vs income",
    "sc_conduct": "Payments returned unpaid", "sc_affordability": "Affordability score",
    "sc_income_stability": "Income regularity score", "sc_balance_cover": "Balance cover score",
    "sc_liquidity": "Liquidity score", "sc_residual": "Money left after spending",
    "sc_credit_appetite": "Borrowing from other lenders", "sc_gambling": "Gambling / betting",
    "sc_savings": "Saving habit", "sc_evidence": "Statement completeness", "sc_cash": "Cash dependence",
    "sc_bills": "Bills paid regularly"}

# human-sized groups for the "why" panel (several model inputs describe the same habit)
GROUPS = {
    "Balance kept in the account": ["abb_to_emi", "min_balance_to_income", "sc_balance_cover", "sc_liquidity",
                                     "low_balance_days", "negative_days", "overdraft_share"],
    "Affordability of the EMI": ["foir", "emi_to_income", "loan_to_annual_income", "sc_affordability",
                                 "sc_residual", "term_months"],
    "Income regularity": ["income_volatility", "regular_income_share", "sc_income_stability"],
    "Repayment conduct": ["penalty_charges", "sc_conduct"],
    "Other borrowing and gambling": ["sc_credit_appetite", "sc_gambling"],
    "Saving and bill habits": ["sc_savings", "sc_bills", "sc_cash"],
    "Statement completeness": ["sc_evidence"],
}


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def feature_vector(a, stmt, fair_fractions, new_emi, existing_emi):
    """The engine's 24 inputs for one application (same code path as the backtest)."""
    v, tags = cm.features(a, stmt, new_emi, existing_emi)
    row = [float(v[f]) for f in cm.FEATURES]
    row += [float(fair_fractions.get(k, 0.5)) for k in EXPERT_FACTORS]
    imputed = [f for f in cm.FEATURES if tags.get(f) == "imputed"]
    return np.array([row]), imputed


@lru_cache(maxsize=1)
def load():
    if not MODEL_PATH.exists():
        return None
    import warnings
    with warnings.catch_warnings():          # trained with scikit-learn 1.9.x; patch versions load identically
        warnings.simplefilter("ignore")
        with open(MODEL_PATH, "rb") as fh:
            return pickle.load(fh)


def available():
    return load() is not None


ADAPTATION_PATH = PROJECT_ROOT / "models" / "ai_engine_adaptation.pkl"


@lru_cache(maxsize=1)
def adaptation():
    """FAIRScore 3.0: the lender's approved adaptation (src/learning_loop.py), or None. Used only when the
    model-risk team has switched it on in config/policy.toml [learning]."""
    from policy_config import POLICY
    if not POLICY["learning"]["use_adaptation"] or not ADAPTATION_PATH.exists():
        return None
    with open(ADAPTATION_PATH, "rb") as fh:
        return pickle.load(fh)


def adapt_pd(pd_):
    ad = adaptation()
    if ad is None:
        return pd_
    import learning_loop
    return learning_loop.adapted_pd(pd_, ad)


def conformal_shift_aware(bundle, pd_, X):
    """Conformal p-values; weighted Mondrian when the approved adaptation detected a covariate shift.
    (Recalibrating PDs with an increasing map does not change conformal p-values, so the source PD is used.)"""
    from policy_config import POLICY
    ad = adaptation()
    dens = (ad or {}).get("density")
    if dens is None or not POLICY["learning"]["shift_aware_conformal"] or "cal_X" not in bundle:
        return conformal(bundle, pd_)
    import learning_loop
    p0, p1 = learning_loop.shift_aware_pvalues(bundle["oof"]["pd"], bundle["oof"]["y"], np.array([pd_]), X, dens)
    return float(p0[0]), float(p1[0])


def predict_pd(bundle, X):
    raw = bundle["model"].predict_proba(X)[:, 1]
    return bundle["calibrator"].predict_proba(logit(raw)[:, None])[:, 1]


def conformal(bundle, pd_):
    """Mondrian conformal p-values for 'repays' (0) and 'defaults' (1)."""
    s0, s1 = bundle["conformal"]["s0"], bundle["conformal"]["s1"]
    p0 = (np.sum(s0 >= pd_) + 1) / (len(s0) + 1)
    p1 = (np.sum(s1 >= 1 - pd_) + 1) / (len(s1) + 1)
    return float(p0), float(p1)


def typicality(bundle, X):
    """Share of training applicants that look MORE unusual than this one (0 = never seen anything like it)."""
    score = bundle["ood"]["model"].score_samples(X)[0]
    ref = bundle["ood"]["train_scores"]
    return float(np.mean(ref <= score))


def shap_contributions(bundle, X):
    from catboost import Pool
    sv = bundle["model"].get_feature_importance(Pool(X), type="ShapValues")[0]
    return dict(zip(FEATURES, sv[:-1])), float(sv[-1])


def _fmt(f, x):
    if f in ("foir", "emi_to_income", "income_volatility", "regular_income_share", "overdraft_share",
             "low_balance_days", "negative_days"):
        return f"{x:.0%}"
    if f in ("abb_to_emi", "loan_to_annual_income", "min_balance_to_income"):
        return f"{x:.1f}x"
    if f == "term_months":
        return f"{x:.0f} months"
    if f == "penalty_charges":
        return f"{x:.0f}"
    return f"{x:.2f} of 1"


def explain(bundle, X):
    """Grouped SHAP drivers: which habits raised or lowered the default risk, and by how much."""
    contrib, base = shap_contributions(bundle, X)
    vals = dict(zip(FEATURES, X[0]))
    groups = []
    for g, feats in GROUPS.items():
        total = sum(contrib[f] for f in feats)
        top = max(feats, key=lambda f: abs(contrib[f]))
        groups.append({"group": g, "impact": float(total), "direction": "raises risk" if total > 0 else "lowers risk",
                       "evidence": f"{LABELS[top]}: {_fmt(top, vals[top])}",
                       "features": {LABELS[f]: round(float(contrib[f]), 4) for f in feats}})
    groups.sort(key=lambda r: -abs(r["impact"]))
    return {"groups": groups, "base_logodds": base,
            "features": sorted(({"feature": LABELS[f], "value": _fmt(f, vals[f]), "impact": round(float(c), 4)}
                                for f, c in contrib.items()), key=lambda r: -abs(r["impact"]))}


def risk_based_rate(pd_life, term_months, settings):
    """Interest rate that covers funding, operating cost, a margin and the expected loss.
    PD over the loan's life is converted to a yearly PD before pricing."""
    years = max(term_months / 12.0, 0.5)
    pd_year = 1 - (1 - min(pd_life, 0.99)) ** (1 / years)
    el_year = pd_year * settings["lgd"]
    rate = (settings["cost_of_funds"] + settings["operating_cost"] + settings["margin"] + el_year) / (1 - pd_year)
    return float(min(max(rate, settings["min_rate"]), settings["max_rate"])), float(pd_year)


def assess(a, stmt, fair_fractions, new_emi, existing_emi, settings, lgd=None, coverage=0.0):
    """Everything the AI says about one application, or None if no engine is installed."""
    bundle = load()
    if bundle is None:
        return None
    X, imputed = feature_vector(a, stmt, fair_fractions, new_emi, existing_emi)
    pd_src = float(predict_pd(bundle, X)[0])
    p0, p1 = conformal_shift_aware(bundle, pd_src, X)
    pd_ = float(adapt_pd(np.array([pd_src]))[0])
    alpha = settings["alpha"]
    in0, in1 = p0 > alpha, p1 > alpha
    if in0 and not in1:
        outcome_set, certainty = "repays", "confident it will repay"
    elif in1 and not in0:
        outcome_set, certainty = "defaults", "confident it will default"
    else:
        outcome_set, certainty = "uncertain", "cannot rule out either outcome"
    typ = typicality(bundle, X)
    in_experience = typ >= settings["min_typicality"] and len(imputed) <= 2
    amount = float(a["requested_amount"])
    lgd_eff = settings["lgd"] if lgd is None else float(lgd)
    rate, pd_year = risk_based_rate(pd_, float(a["requested_duration_months"]), {**settings, "lgd": lgd_eff})
    exp = explain(bundle, X)
    return {
        "pd": pd_, "pd_pct": round(100 * pd_, 1), "pd_source": pd_src, "adapted": adaptation() is not None,
        "p_repays": p0, "p_defaults": p1, "alpha": alpha,
        "outcome_set": outcome_set, "certainty": certainty,
        "typicality": typ, "in_experience": in_experience, "imputed_inputs": imputed,
        "approve_max_pd": bundle["thresholds"]["approve_max_pd"],
        "decline_min_pd": settings["decline_min_pd"],
        "expected_loss": pd_ * lgd_eff * amount, "lgd": lgd_eff, "lgd_unsecured": settings["lgd"],
        "secured_coverage": float(coverage),
        "risk_based_rate": rate, "pd_year": pd_year,
        "explanation": exp, "X": X, "version": bundle["version"],
        "metrics": bundle.get("metrics", {}),
    }


def _lin(x, pts):
    xs, ys = zip(*pts)
    return float(np.interp(x, xs, ys))


# habits an applicant can change over the next months; each sets model inputs to the values that habit
# would produce (the expert-score inputs move with the raw ratios they summarise)
def _levers(v):
    return [
        ("Keep an average balance of at least 3 times the EMI",
         {"abb_to_emi": max(v["abb_to_emi"], 3.0), "sc_balance_cover": max(v["sc_balance_cover"], 1.0)}),
        ("Never let the balance fall below a quarter of a month's income",
         {"min_balance_to_income": max(v["min_balance_to_income"], 0.25)}),
        ("Keep the balance above Rs 1,000 every day (no overdraft)",
         {"low_balance_days": 0.0, "negative_days": 0.0, "overdraft_share": 0.0, "sc_liquidity": 1.0}),
        ("No bounced payments or penalty charges", {"penalty_charges": 0.0, "sc_conduct": 1.0}),
        ("Save regularly (SIP / recurring deposit of about 10% of income)", {"sc_savings": 1.0}),
        ("Spend less: keep at least 30% of income after the EMI", {"sc_residual": 1.0}),
        ("Stop borrowing from other lenders / loan apps", {"sc_credit_appetite": 1.0}),
    ]


def counterfactuals(X, settings, approve_max_pd):
    """Model-checked recourse: which changes of habit would make the AI approve. Each candidate is re-scored
    by the same calibrated model and conformal test; only changes that improve the applicant are tried.
    Returns {'single': [...], 'combined': {...} or None}."""
    bundle = load()
    idx = {f: i for i, f in enumerate(FEATURES)}
    base_v = dict(zip(FEATURES, X[0]))

    def check(changes):
        Z = X.copy()
        for f, val in changes.items():
            Z[0, idx[f]] = val
        p_src = float(predict_pd(bundle, Z)[0])
        p0, p1 = conformal_shift_aware(bundle, p_src, Z)
        p = float(adapt_pd(np.array([p_src]))[0])
        ok = p0 > settings["alpha"] and not p1 > settings["alpha"] and p <= approve_max_pd
        return p, ok

    base_pd, _ = check({})
    single, tried = [], []
    for name, ch in _levers(base_v):
        ch = {f: val for f, val in ch.items() if val != base_v[f]}
        if not ch:
            continue
        p, ok = check(ch)
        tried.append((name, ch, p))
        if ok:
            single.append({"change": name, "new_pd": p})
    single.sort(key=lambda r: r["new_pd"])
    combined = None
    if not single and tried:
        chosen, merged, p = [], {}, base_pd
        for name, ch, _ in sorted(tried, key=lambda t: t[2]):
            merged.update(ch)
            chosen.append(name)
            p, ok = check(merged)
            if ok:
                combined = {"changes": chosen, "new_pd": p}
                break
            if len(chosen) >= 4:
                break
    closest = [{"change": n, "new_pd": p} for n, _, p in sorted(tried, key=lambda t: t[2])[:2] if p < base_pd]
    return {"base_pd": base_pd, "single": single[:3], "combined": combined,
            "closest": closest if not single and not combined else []}


def pd_only(a, stmt, fair_fractions, new_emi, existing_emi):
    bundle = load()
    X, _ = feature_vector(a, stmt, fair_fractions, new_emi, existing_emi)
    return float(adapt_pd(predict_pd(bundle, X))[0])


def ordinal_risk(pd_):
    return ("Very low" if pd_ < 0.02 else "Low" if pd_ < 0.05 else "Moderate" if pd_ < 0.12 else
            "High" if pd_ < 0.30 else "Very high")


def odds_text(pd_):
    n = max(1, int(round(pd_ * 100)))
    return f"about {n} in 100" if pd_ >= 0.01 else "fewer than 1 in 100"


def approve_max(bundle, settings):
    v = settings.get("approve_max_pd")
    return float(v) if v is not None else float(bundle["thresholds"]["approve_max_pd"])


def decide(statuses, ai, settings, approve_max_pd):
    """FAIRScore 2.0 decision. `statuses` = {check key: pass/caution/fail/info}.
    Returns (verdict, decided_by, codes): decided_by is 'policy' (a hard stop), 'ai', or 'defer'
    (the AI hands the case to the version-1 expert policy path because the applicant is outside its
    experience or no engine is installed)."""
    fails = [k for k, s in statuses.items() if s == "fail"]
    if fails:
        return "decline", "policy", fails
    forced = [k for k, s in statuses.items() if s == "caution" and k in settings["force_refer"]]
    if forced:
        return "refer", "policy", forced
    if ai is None or not ai["in_experience"]:
        return None, "defer", []
    if ai["outcome_set"] == "repays" and ai["pd"] <= approve_max_pd:
        return "approve", "ai", []
    # secured loan: collateral lowers the loss if the borrower defaults, so the bank can accept a higher PD as long
    # as the EXPECTED LOSS stays within the appetite for an unsecured loan -- never a confident 'defaults', never
    # above the secured PD ceiling, and the affordability hard stops have already applied
    sec = settings.get("_security") or {}
    if (ai.get("secured_coverage", 0) >= sec.get("min_coverage", 1.1) and ai["outcome_set"] != "defaults"
            and ai["pd"] <= sec.get("secured_max_pd", 0) and ai["pd"] * ai["lgd"] <= approve_max_pd * ai["lgd_unsecured"]):
        return "approve", "ai", ["secured"]
    if ai["outcome_set"] == "defaults" and ai["pd"] >= settings["decline_min_pd"]:
        return "decline", "ai", []
    return "refer", "ai", []
