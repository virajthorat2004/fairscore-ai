"""
FAIRScore AI -- live applicant scoring.

Inputs: the intake form + a bank statement. NO credit-bureau / credit-history
data (no score, no past delinquencies, no card utilisation) and no protected
attributes are used: the system is built for applicants a bureau cannot score.

Pipeline for one applicant:
  1. Lending-policy checks (config/policy.toml) on verified statement facts:
     eligibility, affordability (FOIR), income source / stability, conduct,
     statement integrity, surplus, gambling, declaration consistency.
  2. Three deployable, AGE-BLIND risk models (German Credit, GMSC, Home Credit)
     each build a feature row tagged real / real_approx / imputed. A model votes
     only if >= 60% of its SHAP-weighted importance is evidenced; its score is
     turned into a percentile of its own test set and combined, weighted by
     evidence x training-range coverage x skill (Gini).
  3. Decision (policy-first): any failed check -> decline; any caution -> refer;
     the models (riskiest fifth) or a low decision confidence can only send an
     otherwise clean case to an officer. They never approve or decline alone.
  4. Explanations: SHAP reasons in plain language (LIME on request, for
     comparison), recourse that re-scores realistic changes, and an age-flip
     counterfactual fairness check of the individual decision.

The research pipeline (ensemble_predictor.py etc.) is left untouched.
"""
import pickle
from functools import lru_cache

import numpy as np
import pandas as pd

from config import PROJECT_ROOT, APPLICABILITY_THRESHOLD, TAG_WEIGHTS
from deployable_features import GMSC_DEBT_RATIO_CAP
from policy_config import POLICY, POLICY_VERSION  # noqa: F401  (lender settings, config/policy.toml)
from scorecard import fairscore

DEPLOY_DIR = PROJECT_ROOT / "models" / "deployable"
# The cash-flow model is trained on REAL loan outcomes from bank-account histories (Berka,
# see cashflow_model.py); the other three on public credit datasets restricted to inputs an
# applicant can provide. Each votes in proportion to evidence x training-range coverage x skill.
VOTING_MODELS = ["cashflow", "german_credit", "gmsc", "homecredit"]

DATASET_LABELS = {
    "cashflow": "Cash-flow model (real repayment outcomes)",
    "german_credit": "German Credit",
    "gmsc": "Give Me Some Credit",
    "homecredit": "Home Credit",
    "africa": "Africa Synth (behavioural)",
}

# Model-only bands on the consensus risk percentile (0 = safest, 100 = riskiest).
# Used for model diagnostics and recourse; the recommendation itself is policy-first (see decide()).
APPROVE_BELOW = 50.0     # safer than half of past applicants
DECLINE_FROM = POLICY["refer_from_percentile"]   # riskiest fifth: sends a clean case for review
MAX_AGREEMENT_SPREAD = 40.0  # percentile points between voting models

# ---------------------------------------------------------------------------
# Intake options (shown in the form) and their dataset encodings
# ---------------------------------------------------------------------------
PURPOSE_OPTIONS = {   # German Credit purpose codes (UCI documentation)
    "New vehicle": "A40", "Used vehicle": "A41",
    "Furniture / home equipment": "A42", "Electronics (TV, phone, laptop)": "A43",
    "Home appliances": "A44", "Home repairs / renovation": "A45",
    "Education": "A46", "Skill training / course": "A48",
    "Business": "A49", "Other": "A410",
}
HOUSING_OPTIONS = {   # -> (German housing, Home Credit NAME_HOUSING_TYPE)
    "Own home": ("own", "House___apartment"),
    "Rented home": ("rent", "Rented_apartment"),
    "Living with family (no rent)": (None, "With_parents"),   # German baseline = "free"
}
EMPLOYMENT_TYPES = {  # -> Home Credit NAME_INCOME_TYPE
    "Salaried (private sector)": "Working",
    "Government employee": "State_servant",
    "Self-employed / business owner": "Commercial_associate",
    "Freelancer / gig worker": "Commercial_associate",
    "Farmer / agriculture": "Working",
    "Pensioner": "Pensioner",
    "Student": "Student",
    "Unemployed": "Unemployed",
}
EDUCATION_LEVELS = {  # -> Home Credit NAME_EDUCATION_TYPE (None = baseline "Academic degree")
    "Postgraduate / doctorate": None,
    "Graduate": "Higher_education",
    "Diploma / incomplete degree": "Incomplete_higher",
    "12th pass": "Secondary___secondary_special",
    "10th pass or below": "Lower_secondary",
}


# Intake contract. A bank statement is mandatory (it supplies verified income,
# EMIs, balances and spending). REQUIRED fields are facts a statement cannot
# show. OPTIONAL fields may be None: they are then taken from the statement
# where possible, or treated as unknown exactly as each training set coded
# missing values. When given, they are cross-checked against the statement.
REQUIRED_FIELDS = ["age", "employment_type", "employment_years", "housing", "dependents",
                   "owns_car", "total_savings",
                   "requested_amount", "requested_duration_months", "purpose"]
OPTIONAL_FIELDS = ["monthly_income", "active_credit_lines", "property_loans", "education", "account_holder_name",
                   "asset_price", "security_type", "security_value", "co_applicant_income",
                   "bureau_record", "bureau_score", "bureau_max_dpd_12m", "bureau_severe_24m", "bureau_enquiries_6m",
                   "bureau_emi", "customer_id"]


def missing_required(a):
    return [k for k in REQUIRED_FIELDS if a.get(k) in (None, "")]


def credit_lines(a, stmt):
    """(count, tag): declared count if given, else EMIs detected in the statement."""
    if a.get("active_credit_lines") is not None:
        return int(a["active_credit_lines"]), "real"
    if _has_statement(stmt):
        return int(stmt.get("detected_credit_signal_count", 0)), "real_approx"
    return None, "imputed"


# ---------------------------------------------------------------------------
# Cached artefacts
# ---------------------------------------------------------------------------
def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=None)
def load_model(name):
    return _load(DEPLOY_DIR / name / "CatBoost.pkl")


@lru_cache(maxsize=None)
def load_reference(name):
    return _load(DEPLOY_DIR / f"{name}_reference.pkl")


@lru_cache(maxsize=None)
def score_reference():
    return _load(DEPLOY_DIR / "score_reference.pkl")


@lru_cache(maxsize=None)
def gate_weights(name):
    raw = _load(DEPLOY_DIR / "shap_weights.pkl")[name]
    cols = load_reference(name)["all_columns"]
    w = {c: abs(float(raw.get(c, 0.0))) for c in cols}
    total = sum(w.values()) or 1.0
    return {c: v / total for c, v in w.items()}


@lru_cache(maxsize=None)
def explainer(name):
    import shap
    return shap.TreeExplainer(load_model(name))


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------
def _onehot(features, tags, cols, prefix, chosen, tag):
    """Set every column with this prefix to 0 except `chosen` (None = baseline)."""
    for c in cols:
        if c.startswith(prefix):
            features[c] = 1 if c == chosen else 0
            tags[c] = tag


def _statement_income(stmt):
    v = (stmt or {}).get("avg_monthly_income")
    return v if v and v > 0 else None


SALARIED_TYPES = ("Salaried (private sector)", "Government employee", "Pensioner")
# Income from their own work, not a salary: judged on all genuine inflows
SELF_EMPLOYED_TYPES = ("Self-employed / business owner", "Freelancer / gig worker", "Farmer / agriculture")


def employment_view(stmt, a):
    """Which statement income to rely on, by declared employment (lender practice):
    salaried / government / pensioners -> the regular salary or pension credits only;
    everyone else -> all genuine inflows (business income arrives from many payers).
    Without any salary in the account, a salaried applicant's income falls back to all
    inflows and the income-source check refers the case for the salary account."""
    if not _has_statement(stmt) or "avg_monthly_all_inflows" not in stmt:
        return stmt
    view = dict(stmt)
    regular = stmt.get("avg_monthly_regular_income") or 0
    if a.get("employment_type") in SALARIED_TYPES + ("Student",) and regular > 0:
        view.update(avg_monthly_income=regular, income_basis="regular income",
                    income_volatility=stmt["regular_income_volatility"])
        if stmt.get("salary_started_after_months") and stmt.get("regular_income_since_start"):
            # new job: rely on the salary actually being paid now; the short history is a separate check
            view.update(avg_monthly_income=stmt["regular_income_since_start"],
                        avg_monthly_regular_income=stmt["regular_income_since_start"],
                        income_volatility=stmt["regular_income_volatility_since_start"],
                        regular_income_months_share=1.0)
    else:
        view.update(avg_monthly_income=stmt["avg_monthly_all_inflows"],
                    income_basis="all business / other inflows" if a.get("employment_type") not in SALARIED_TYPES
                    else "all inflows (no salary found)",
                    income_volatility=stmt["all_inflows_volatility"])
    return view


def _has_statement(stmt):
    return bool(stmt) and stmt.get("statement_months", 0) >= 1


def new_loan_emi(a):
    """The lender's quoted EMI when given, else the standard reducing-balance EMI."""
    return float(a["proposed_emi"]) if a.get("proposed_emi") else emi(a["requested_amount"],
                                                                      a["requested_duration_months"])


def prepare_statement(stmt, a, stmt_df=None):
    """The statement as the decision sees it: income chosen by declared employment, plus the
    existing loan EMIs (needed by the cash-flow model before the policy checks run)."""
    view = employment_view(stmt, a)
    if _has_statement(view) and stmt_df is not None and "_existing_emi" not in view:
        view = dict(view, _existing_emi=existing_obligations(stmt_df, view))
    return view


def build_features(name, a, stmt):
    """Returns (X one-row DataFrame, tags dict). `a` is the applicant dict,
    `stmt` the parsed statement feature dict (or None)."""
    ref = load_reference(name)
    cols = ref["all_columns"]
    if name == "cashflow":
        if not _has_statement(stmt):
            return pd.DataFrame([[ref["median"].get(c, 0) for c in cols]], columns=cols), {c: "imputed" for c in cols}
        import cashflow_model
        v, tags = cashflow_model.features(a, stmt, new_loan_emi(a), stmt.get("_existing_emi", 0.0))
        for c in cols:
            if tags.get(c) == "imputed":
                v[c] = ref["median"].get(c, 0)
        return pd.DataFrame([[v[c] for c in cols]], columns=cols), tags
    f, t = {}, {}
    years = float(a.get("employment_years") or 0)
    emp_type = a["employment_type"]
    n_credit, credit_tag = credit_lines(a, stmt)

    if name == "german_credit":
        f["duration"], t["duration"] = a["requested_duration_months"], "real"
        if n_credit is not None:
            f["existing_credits"], t["existing_credits"] = int(np.clip(n_credit, 1, 4)), "real_approx"

        if emp_type == "Unemployed":
            bucket = "employment_since_unemployed"
        elif years < 1:
            bucket = "employment_since_lt1_year"
        elif years < 4:
            bucket = None                     # baseline "1-4 years"
        elif years < 7:
            bucket = "employment_since_4-7_years"
        else:
            bucket = "employment_since_gt7_years"
        _onehot(f, t, cols, "employment_since_", bucket, "real")

        code = PURPOSE_OPTIONS[a["purpose"]]
        _onehot(f, t, cols, "purpose_", None if code == "A40" else f"purpose_{code}", "real")

        g_house = HOUSING_OPTIONS[a["housing"]][0]
        _onehot(f, t, cols, "housing_", f"housing_{g_house}" if g_house else None, "real")

        # Other instalment plans: A141 bank (baseline), A142 stores, A143 none.
        if n_credit is not None:
            _onehot(f, t, cols, "other_installment_plans_",
                    "other_installment_plans_A143" if n_credit == 0 else None, "real_approx")

        # Savings buckets are in Deutsche Marks: rupee thresholds are a heuristic.
        s = a.get("total_savings")
        if s is not None:
            col = ("savings_account_lt100_DM" if s < 3000 else None if s < 15000
                   else "savings_account_500-1000_DM" if s < 30000 else "savings_account_gt1000_DM")
        else:
            col = "savings_account_unknown_none"   # German Credit's own code for "unknown"
        _onehot(f, t, cols, "savings_account_", col, "real_approx")

        if _has_statement(stmt) and stmt.get("avg_balance") is not None:
            bal = stmt["avg_balance"]
            # Holding an account is directly evidenced by the statement itself.
            f["checking_account_no_checking_account"] = 0
            t["checking_account_no_checking_account"] = "real"
            f["checking_account_lt0_DM"] = int(bal < 0)
            f["checking_account_gt200_DM"] = int(bal > 20000)
            t["checking_account_lt0_DM"] = t["checking_account_gt200_DM"] = "real_approx"

    elif name == "gmsc":
        if a.get("dependents") is not None:
            f["NumberOfDependents"], t["NumberOfDependents"] = a["dependents"], "real"
            f["dependents_was_missing"], t["dependents_was_missing"] = 0, "real"
        else:   # coded exactly as GMSC coded unknown dependents: flag + training median
            f["NumberOfDependents"] = ref["median"]["NumberOfDependents"]
            f["dependents_was_missing"] = 1
            t["NumberOfDependents"] = t["dependents_was_missing"] = "real"
        if a.get("property_loans") is not None:
            f["NumberRealEstateLoansOrLines"] = int(a["property_loans"])
            t["NumberRealEstateLoansOrLines"] = "real"
        elif _has_statement(stmt) and stmt.get("detected_property_loans") is not None:
            # not declared: home-loan EMIs found in the statement (may miss loans paid elsewhere)
            f["NumberRealEstateLoansOrLines"] = int(stmt["detected_property_loans"])
            t["NumberRealEstateLoansOrLines"] = "real_approx"
        f["monthly_income_was_missing"], t["monthly_income_was_missing"] = 0, "real"
        inc = _statement_income(stmt)
        if _has_statement(stmt) and inc and stmt.get("avg_monthly_spend") is not None:
            f["DebtRatio"] = round(min(stmt["avg_monthly_spend"] / inc, GMSC_DEBT_RATIO_CAP), 4)
            t["DebtRatio"] = "real_approx"   # all spending, not only debt + living costs

    elif name == "homecredit":
        if a.get("dependents") is not None:
            f["CNT_CHILDREN"], t["CNT_CHILDREN"] = a["dependents"], "real_approx"
        income_type = EMPLOYMENT_TYPES[emp_type]
        if income_type in ("Pensioner", "Unemployed"):
            # Home Credit's own convention: these rows carry the 365243 sentinel,
            # which preprocessing turned into anomaly=1 + median DAYS_EMPLOYED.
            f["days_employed_anomaly"] = 1
            f["DAYS_EMPLOYED"] = ref["median"]["DAYS_EMPLOYED"]
        else:
            f["days_employed_anomaly"] = 0
            f["DAYS_EMPLOYED"] = -years * 365
        t["days_employed_anomaly"] = t["DAYS_EMPLOYED"] = "real"
        _onehot(f, t, cols, "NAME_INCOME_TYPE_", f"NAME_INCOME_TYPE_{income_type}", "real")
        if a.get("education"):
            edu = EDUCATION_LEVELS[a["education"]]
            _onehot(f, t, cols, "NAME_EDUCATION_TYPE_", f"NAME_EDUCATION_TYPE_{edu}" if edu else None,
                    "real_approx")
        hc_house = HOUSING_OPTIONS[a["housing"]][1]
        _onehot(f, t, cols, "NAME_HOUSING_TYPE_", f"NAME_HOUSING_TYPE_{hc_house}", "real_approx")
        if a.get("owns_car") is not None:
            f["FLAG_OWN_CAR_Y"], t["FLAG_OWN_CAR_Y"] = int(bool(a["owns_car"])), "real"
        f["FLAG_OWN_REALTY_Y"] = int(a["housing"] == "Own home")
        t["FLAG_OWN_REALTY_Y"] = "real_approx"
        income = _statement_income(stmt) or a.get("monthly_income")
        if income:
            f["credit_to_income"] = a["requested_amount"] / (income * 12)
            t["credit_to_income"] = "real"

    else:
        raise ValueError(name)

    for c in cols:   # anything not evidenced -> training median / mode, tagged imputed
        if c not in f:
            f[c] = ref["mode"].get(c, 0) if c in ref["binary_columns"] else ref["median"].get(c, 0)
            t[c] = "imputed"
    X = pd.DataFrame([[f[c] for c in cols]], columns=cols)
    return X, t


def applicability(name, tags):
    return sum(w * TAG_WEIGHTS.get(tags.get(c, "imputed"), 0.0) for c, w in gate_weights(name).items())


def out_of_range(name, X, tags):
    """Evidenced numeric inputs outside the model's training range (1st-99th
    percentile). The model's output there is extrapolation, so the share of its
    importance resting on such inputs is removed from its vote weight."""
    ref = load_reference(name)
    out = []
    for c, lo in ref.get("q01", {}).items():
        if tags.get(c) == "imputed":
            continue
        v, hi = float(X.iloc[0][c]), ref["q99"][c]
        if v < lo or v > hi:
            out.append({"feature": c, "value": v, "lo": lo, "hi": hi})
    coverage = 1.0 - sum(gate_weights(name).get(o["feature"], 0.0) for o in out)
    return out, max(coverage, 0.0)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _percentile(name, prob):
    ref = score_reference()[name]
    scores = ref["test_scores_sorted"]
    pct = 100.0 * np.searchsorted(scores, prob, side="right") / len(scores)
    decile = next((d for d in ref["deciles"] if prob <= d["hi"]), ref["deciles"][-1])
    return float(pct), decile["default_rate"], ref["base_rate"]


def check_consistency(a, stmt):
    """Declared credit vs statement evidence."""
    if not _has_statement(stmt):
        return None, None
    detected = stmt.get("detected_existing_credit", False)
    if a.get("active_credit_lines") is None:
        return None, None
    declared = int(a["active_credit_lines"])
    if detected and declared == 0:
        return "conflict", (f"Applicant declared no active loans or cards, but the statement shows "
                            f"{stmt.get('detected_credit_signal_count', 1)} recurring EMI/card payment(s).")
    if declared > 0 and not detected and stmt.get("statement_months", 0) >= 3:
        return "note", ("Applicant declared active loans/cards, but no recurring EMI or card payment "
                        "appears in this statement (they may be paid from another account).")
    return None, None


def _band(pct):
    if pct < APPROVE_BELOW:
        return "approve"
    if pct >= DECLINE_FROM:
        return "decline"
    return "refer"


def _confidence(voters, consensus_pct):
    if not voters:
        return {"label": "Low", "score": 0.0, "detail": "No model had enough evidence to vote."}
    pcts = [v["percentile"] for v in voters.values()]
    spread = max(pcts) - min(pcts)
    margin = min(abs(consensus_pct - APPROVE_BELOW), abs(consensus_pct - DECLINE_FROM))
    margin_score = min(margin / 20.0, 1.0)
    if len(pcts) > 1:
        agreement = max(0.0, 1 - spread / (1.5 * MAX_AGREEMENT_SPREAD))
        score = 0.5 * agreement + 0.5 * margin_score
        detail = (f"{len(pcts)} models voted; their risk percentiles span {spread:.0f} points; "
                  f"the result sits {margin:.0f} points from the nearest band boundary.")
    else:
        score = 0.6 * margin_score
        detail = "Only one model had enough evidence to vote, so confidence is capped."
    label = "High" if score >= 0.65 else "Medium" if score >= 0.4 else "Low"
    return {"label": label, "score": round(score, 3), "spread": round(spread, 1), "detail": detail}


def score(a, stmt, exclude_models=()):
    """Core scoring without explanations (used directly by recourse)."""
    models = {}
    for name in VOTING_MODELS:
        if name in exclude_models:
            continue
        X, tags = build_features(name, a, stmt)
        app = applicability(name, tags)
        entry = {"label": DATASET_LABELS[name], "applicability": round(app, 3),
                 "qualifies": app >= APPLICABILITY_THRESHOLD, "tags": tags, "X": X}
        if entry["qualifies"]:
            prob = float(load_model(name).predict_proba(X)[0][1])
            pct, decile_rate, base = _percentile(name, prob)
            skill = max(2 * score_reference()[name]["auc"] - 1, 0.0)
            ood, coverage = out_of_range(name, X, tags)
            entry.update({"probability": prob, "percentile": pct,
                          "decile_default_rate": decile_rate, "base_rate": base,
                          "skill": skill, "out_of_range": ood, "coverage": round(coverage, 3),
                          "weight": app * coverage * skill})
        else:
            entry["reason"] = (f"Only {app:.0%} of this model's decision-relevant information could be "
                               f"evidenced (needs {APPLICABILITY_THRESHOLD:.0%}).")
        models[name] = entry

    voters = {k: v for k, v in models.items() if v["qualifies"]}
    total_w = sum(v["weight"] for v in voters.values())
    consensus = (sum(v["percentile"] * v["weight"] for v in voters.values()) / total_w
                 if voters and total_w > 0 else None)

    consistency, consistency_msg = check_consistency(a, stmt)
    confidence = _confidence(voters, consensus if consensus is not None else 50.0)

    return {
        "model_band": _band(consensus) if consensus is not None else None,
        "consensus_percentile": round(consensus, 1) if consensus is not None else None,
        "models": models, "voters": list(voters),
        "confidence": confidence,
        "consistency": consistency, "consistency_message": consistency_msg,
    }


# ---------------------------------------------------------------------------
# Lending-policy checks (transparent rules on the statement + form)
# ---------------------------------------------------------------------------
# Thresholds are lender settings loaded from config/policy.toml (see policy_config.py).
# Fixed obligations for FOIR = loan EMIs. Credit-card bill payments are not
# counted as EMIs (a card paid in full each month is spending, not a fixed
# obligation); they still show up in the spending ratio.
import ai_engine  # noqa: E402
import starter_ladder  # noqa: E402
import security  # noqa: E402
import bureau  # noqa: E402
from statement_parser import is_loan_payment  # noqa: E402  (shared whole-word loan/EMI rules)


def emi(principal, months, annual_rate=None):
    r = (POLICY["annual_interest_rate"] if annual_rate is None else annual_rate) / 12
    if r == 0:
        return principal / months
    return principal * r * (1 + r) ** months / ((1 + r) ** months - 1)


def max_affordable_amount(income, existing_emi, months, foir=None):
    """Largest loan whose EMI keeps total obligations within the FOIR limit."""
    room = (POLICY["foir_pass"] if foir is None else foir) * income - existing_emi
    if room <= 0:
        return 0.0
    return room / emi(1.0, months)


def existing_obligations(stmt_df, stmt=None):
    """Median monthly total of loan-EMI debits, over the calendar months the
    statement fully covers (a partial first/last month would look EMI-free)."""
    if stmt_df is None or not len(stmt_df):
        return 0.0
    debits = stmt_df[stmt_df["signed_amount"] < 0]
    mask = is_loan_payment(debits["description"])
    if not mask.any():
        return 0.0
    months = pd.PeriodIndex(stmt_df["month"].unique(), freq="M")
    if stmt and stmt.get("period_start"):
        start, end = pd.Timestamp(stmt["period_start"]), pd.Timestamp(stmt["period_end"])
        tol = pd.Timedelta(days=3)
        full = [m for m in months if m.start_time >= start - tol and m.end_time.normalize() <= end + tol]
        if len(full) >= 2:
            months = pd.PeriodIndex(full, freq="M")
    monthly = debits[mask].groupby("month")["signed_amount"].sum().abs().reindex(months, fill_value=0)
    return float(monthly.median())


def _lower_first(t):
    """'Take-home for ...' -> 'take-home for ...', but 'EMI-to-income' stays as it is."""
    return t if len(t) > 1 and t[1].isupper() else t[:1].lower() + t[1:]


def _inr(x):
    return f"−₹{abs(x):,.0f}" if x < 0 else f"₹{x:,.0f}"


def _check(key, name, status, value, detail):
    return {"key": key, "name": name, "status": status, "value": value, "detail": detail}


def _grade(value, pass_max, fail_above):
    return "pass" if value <= pass_max else ("fail" if value > fail_above else "caution")


def age_at_maturity(a):
    return float(a["age"]) + float(a["requested_duration_months"]) / 12.0


def loan_to_value(a):
    """Requested amount / price of the item bought (purchase loans), or None if no price was given."""
    try:
        price = float(a.get("asset_price") or 0)
    except (TypeError, ValueError):
        return None
    return float(a["requested_amount"]) / price if price > 0 else None


def eligibility_check(a, ignore_age=False, income=None):
    """The ONLY place age is used: legal/product eligibility (adult; loan ends
    before the maximum age for the employment type). Employment-type rules too --
    but a student or an unemployed applicant whose statement PROVES an income of their
    own is assessed on that income, not turned away for the label."""
    P = POLICY
    emp = a.get("employment_type") or ""
    max_age = P["max_age_at_maturity"].get(emp, 60)
    end_age = age_at_maturity(a)
    own = (income or 0) >= P["own_income_min"]
    co = _co_income(a)
    co_ok = co >= P["own_income_min"]
    if emp in P["not_eligible"] and not own and not co_ok:
        return _check("eligibility", "Eligibility", "fail", emp,
                      f"{emp}, and no income of their own seen in the statement (under ₹{P['own_income_min']:,.0f} a "
                      f"month) to repay from: a co-applicant with income could apply")
    if a["age"] < P["min_age"] and not ignore_age:
        return {**_check("eligibility", "Eligibility", "fail", a["age"],
                         f"Minimum age for this loan is {P['min_age']}"), "age_rule": True}
    if end_age > max_age and not ignore_age:
        return {**_check("eligibility", "Eligibility", "fail", round(end_age, 1),
                         f"The loan would end at age {end_age:.0f}, beyond the limit of {max_age} for "
                         f"{emp.lower()} applicants: a shorter term may be possible"), "age_rule": True}
    if emp in P["not_eligible"] + P["needs_co_applicant"] and not own and co_ok:
        status = "caution" if emp in P["not_eligible"] else "pass"
        return _check("eligibility", "Eligibility", status, round(co),
                      f"{emp} without an income of their own: assessed with the co-applicant's income of "
                      f"₹{co:,.0f} a month (to be verified with the co-applicant's statement or salary slips)")
    if emp in P["needs_co_applicant"] and not own:
        return _check("eligibility", "Eligibility", "caution", emp,
                      f"{emp} without an income of their own in the statement (under ₹{P['own_income_min']:,.0f} a "
                      f"month): needs a co-applicant or guarantor with income")
    if emp in P["not_eligible"] + P["needs_co_applicant"]:
        note = ("declared unemployed, but" if emp in P["not_eligible"] else "a student with")
        status = "caution" if emp in P["not_eligible"] else "pass"
        return _check("eligibility", "Eligibility", status, round(end_age, 1),
                      f"{emp}: {note} an income of ₹{income:,.0f} a month verified from the statement, so assessed "
                      f"on that income" + (": confirm what this income is" if status == "caution" else
                                           " (stipend / part-time / freelance: confirm with a letter or contract)"))
    return _check("eligibility", "Eligibility", "pass", round(end_age, 1),
                  f"Loan ends at age {end_age:.0f} (limit {max_age} for {emp.lower()} applicants)")


def tenure_check(a):
    emp = a.get("employment_type") or ""
    need = POLICY["min_work_years"].get(emp)
    if need is None:
        return None
    years = float(a.get("employment_years") or 0)
    what = ("years farming" if emp.startswith("Farmer") else "business / freelance work" if emp in SELF_EMPLOYED_TYPES
            else "in current work")
    if years >= need:
        return _check("tenure", "Work / business tenure", "pass", years, f"{years:g} years {what}")
    return _check("tenure", "Work / business tenure", "caution", years,
                  f"{years:g} years {what}; the usual minimum is {need:g}")


def _co_income(a):
    try:
        return float(a.get("co_applicant_income") or 0) if POLICY["co_applicant_counts"] else 0.0
    except (TypeError, ValueError):
        return 0.0


def student_limits(a, income):
    """(largest amount, longest term) a lender lends unsecured to a student with this verified income. With a
    co-applicant who has an income the Rs 5 lakh cap does not apply and the multiple uses both incomes."""
    P = POLICY
    co = _co_income(a)
    if co >= P["own_income_min"]:
        return P["student_income_multiple"] * ((income or 0) + co), P["student_max_term_months"]
    return (min(P["student_income_multiple"] * (income or 0), P["student_max_amount"]), P["student_max_term_months"])


def student_checks(a, stmt, income):
    """Two checks a credit officer applies to a declared student with an income of their own:
    1. does the statement look like a student's (stipend / part-time), or like a full-time salary?
    2. is the loan kept to a student's size and term (policy settings)?"""
    P = POLICY
    if (a.get("employment_type") or "") != "Student" or not income or income < P["own_income_min"]:
        return []
    out = []
    share = (stmt or {}).get("regular_income_months_share") or 0.0
    regular = str((stmt or {}).get("income_basis", "")).startswith("regular")
    if regular and share >= 0.8 and income >= P["student_salary_query"]:
        out.append(_check("profile_match", "Declared profile vs statement", "caution", round(income),
                          f"Declared a student, but a regular credit of {_inr(income)} arrives almost every month: "
                          f"the size of a full-time salary. Confirm whether the applicant studies or works; if "
                          f"employed full-time, re-assess as salaried with the employer's salary slips"))
    else:
        out.append(_check("profile_match", "Declared profile vs statement", "pass", round(income),
                          f"A student's income of {_inr(income)} a month (stipend / part-time / freelance) fits the "
                          f"declared profile"))
    cap, term_cap = student_limits(a, income)
    amt, term = float(a["requested_amount"]), int(a["requested_duration_months"])
    over = []
    if amt > cap:
        over.append(f"the amount is above {_inr(cap)} (the lower of {P['student_income_multiple']:g}x verified income "
                    f"and {_inr(P['student_max_amount'])} without a co-applicant)")
    if term > term_cap:
        over.append(f"the term is above {term_cap} months")
    out.append(_check("student_limits", "Loan size for a student", "caution" if over else "pass", round(amt),
                      ("For a student " + " and ".join(over) + ": a smaller loan, or a co-applicant / guarantor "
                       "with income") if over else
                      f"Within a student's limits ({_inr(cap)}, up to {term_cap} months)"))
    return out


def policy_checks(a, stmt, stmt_df):
    P = POLICY
    _verified = _statement_income(stmt) if _has_statement(stmt) else None
    checks = [eligibility_check(a, income=_verified or a.get("monthly_income") or 0)]
    t = tenure_check(a)
    if t:
        checks.append(t)
    checks += student_checks(a, stmt, _verified)
    has_stmt = _has_statement(stmt)
    months = stmt.get("months_covered", stmt.get("statement_months", 0)) if has_stmt else 0
    declared = a.get("monthly_income") or 0
    verified = _statement_income(stmt)
    income = verified or declared
    co = _co_income(a)
    if co > 0:
        checks.append(_check("co_applicant", "Co-applicant income", "caution", round(co),
                             f"₹{co:,.0f} a month declared for a co-applicant, added to the income used for affordability "
                             f"(₹{(income or 0):,.0f} + ₹{co:,.0f}); verify with the co-applicant's statement or salary slips"))
        income = (income or 0) + co
    sec = security.assess(a)
    if sec:
        checks.append(sec["check"])

    # 1. Evidence: enough statement history (period actually covered, ~1 week tolerance)
    # Too short a history is a missing document, not a sign of bad credit: refer and ask for it.
    status = "pass" if months >= P["min_months_pass"] - 0.25 else "caution"
    period = (f" ({pd.Timestamp(stmt['period_start']):%d %b %Y} – {pd.Timestamp(stmt['period_end']):%d %b %Y})"
              if has_stmt and stmt.get("period_start") else "")
    checks.append(_check("evidence", "Statement history", status, round(months, 1),
                         f"{months:.1f} months of transactions{period}; {P['min_months_pass']} needed"
                         if has_stmt else "No bank statement provided: income and conduct are unverified"))
    # 1b. Recency: underwriting needs the LATEST months, not an old statement
    if has_stmt and stmt.get("period_end"):
        as_of = pd.Timestamp(a.get("application_date") or pd.Timestamp.now().normalize())
        age_days = int((as_of - pd.Timestamp(stmt["period_end"])).days)
        if age_days > P["max_statement_age_days"]:
            checks.append(_check("recency", "Statement is recent", "caution", age_days,
                                 f"The statement ends {age_days} days before the application "
                                 f"(limit {P['max_statement_age_days']}): ask for the latest months"))
        elif age_days < -3:
            checks.append(_check("recency", "Statement is recent", "caution", age_days,
                                 "The statement period ends after the application date: check the dates"))
        else:
            checks.append(_check("recency", "Statement is recent", "pass", age_days,
                                 f"Ends {max(age_days, 0)} day(s) before the application"))

    # 2. Affordability (FOIR) with the new loan
    obligations = existing_obligations(stmt_df, stmt) if has_stmt else 0.0
    bur = bureau.assess(a, obligations)          # FAIRScore 3.0: bureau layer, when the applicant has a record
    if bur:
        checks.append(bur["check"])
        obligations = bur["emi_for_affordability"]
    # the lender's own quoted EMI when known (e.g. a loan-origination system), else the standard formula
    new_emi = new_loan_emi(a)
    foir = (obligations + new_emi) / income if income else float("inf")
    if income:
        checks.append(_check("affordability", "Affordability (FOIR)",
                             _grade(foir, P["foir_pass"], P["foir_fail"]), round(foir, 3),
                             f"Existing EMIs ₹{obligations:,.0f} + new EMI ₹{new_emi:,.0f} "
                             f"= {foir:.0%} of {'verified' if verified else 'declared'} income "
                             f"(limit {P['foir_pass']:.0%}, hard stop {P['foir_fail']:.0%})"))
    else:
        # No income found is missing evidence, not proof the loan is unaffordable: refer.
        checks.append(_check("affordability", "Affordability (FOIR)", "caution", None,
                             f"No income could be verified from this statement, so the new EMI of "
                             f"₹{new_emi:,.0f} cannot be checked against income"))

    if has_stmt:
        # 3. Declared income supported by the statement
        if declared and verified is not None:
            ratio = verified / declared
            checks.append(_check("income_support", "Declared income verified",
                                 "pass" if ratio >= P["income_support_min"] else "caution",
                                 round(ratio, 2),
                                 f"Statement shows ₹{verified:,.0f}/month vs ₹{declared:,.0f} declared"))
        # 4. Income source matches the declared employment
        regular = stmt.get("avg_monthly_regular_income") or 0
        payers = stmt.get("income_payers") or 0
        emp = a.get("employment_type") or ""
        if emp in SALARIED_TYPES:
            share = stmt.get("regular_income_months_share", 1.0 if regular > 0 else 0.0)
            if regular > 0 and share >= 0.8:
                status, detail = "pass", f"Regular salary / pension credits of ₹{regular:,.0f} a month found"
            elif regular > 0:
                status, detail = "caution", (f"Salary-like credits (₹{regular:,.0f} a month on average) appear in only "
                                             f"{share:.0%} of the months, so the declared salary is not confirmed")
            else:
                status, detail = "caution", ("No regular salary or pension credits found, so the declared employment "
                                             "could not be verified from this statement")
            checks.append(_check("income_source", "Income source", status, round(regular), detail))
        elif emp in SELF_EMPLOYED_TYPES:
            # a farmer sells to one or two buyers (mandi / dairy); a freelancer has a few clients
            need = {"Farmer / agriculture": 1, "Freelancer / gig worker": 2}.get(emp, P["self_employed_min_payers"])
            checks.append(_check("income_source", "Income source", "pass" if payers >= need else "caution", payers,
                                 f"Business inflows from {payers} different payers" if payers >= need else
                                 f"Inflows come from only {payers} payer(s), so business income could not be verified"))
        own = stmt.get("own_transfers_in_share") or 0
        if own >= 0.5:
            for c in checks:
                if c["key"] == "income_source":
                    c["status"] = "caution"
                    c["detail"] += (f". {own:.0%} of the money coming into this account is from the applicant's own "
                                    f"other accounts (not income): ask for the statement of the account where "
                                    f"salary / business income is received")
        # 4b. Income stability
        vol = stmt.get("income_volatility", 0.0) or 0.0
        status = _grade(vol, P["volatility_pass"], P["volatility_fail"])
        detail = f"Monthly income varies by {vol:.0%} (coefficient of variation)"
        if emp.startswith("Farmer"):
            # Farm income comes with the harvest: large month-to-month swings are normal.
            months_with = sum(1 for v in (stmt.get("all_inflows_series") or []) if v > 0)
            if vol <= P["farm_volatility_caution"]:
                status, detail = "pass", detail + ": seasonal variation is normal for farm income"
            else:
                status = "caution"
                detail += (": very uneven even for farm income; verify with land records (7/12 extract / RoR) and "
                           "crop-sale receipts")
            if months_with and months_with < 3:
                status = "caution"
                detail += f"; money came in during only {months_with} month(s)"
        elif emp in SELF_EMPLOYED_TYPES and status == "fail":
            # Business income is lumpy: lenders refer these for ITR / GST review rather than decline.
            status = "caution"
            detail += ": irregular business / freelance income, needs ITR / GST returns to assess"
        elif emp in SALARIED_TYPES and status == "fail" and \
                (not regular or stmt.get("regular_income_months_share", 1.0) < 0.8):
            # No salary in this account: the money measured here is not the declared salary,
            # so its variation says nothing about the salary. Refer for the salary account.
            status = "caution"
            detail += ": no steady monthly salary in this account to measure; refer for the salary account"
        late = stmt.get("salary_started_after_months") or 0
        if emp in SALARIED_TYPES and late and stmt.get("regular_income_since_start"):
            n = len(stmt.get("regular_income_series") or []) - late
            status = "caution"
            detail = (f"Salary credited only in the last {n} months of the statement (none in the first {late}): "
                      f"a new job, so too short a salary history; it has varied by {vol:.0%} since it started. "
                      f"Confirm employment (appointment letter, latest salary slips)")
        if verified:      # nothing to measure when no income was found (income source already says so)
            checks.append(_check("stability", "Income stability", status, round(vol, 3), detail))
        # 5. Account conduct: overdrafts and the applicant's OWN payments returned unpaid
        #    (an outward return -- a cheque they deposited that bounced -- is not their default)
        bank = stmt.get("banking") or {}
        od = stmt.get("overdraft_frequency")
        bounces = bank.get("bounce_events", 0)
        penalties = bank.get("penalty_charges", 0)
        if od is not None or bank:
            status = _grade(od or 0.0, P["overdraft_pass"], P["overdraft_fail"])
            if bounces > P["max_penalty_charges"]:
                status = "fail"
            elif (bounces or penalties >= 3) and status == "pass":
                status = "caution"
            detail = (f"Balance below zero on {(od or 0):.0%} of transactions; {bounces} payment(s) returned "
                      f"unpaid (EMI / cheque / mandate)")
            if penalties:
                detail += f"; {penalties} penalty charge(s) (minimum balance, overdue)"
            if bank.get("outward_returns"):
                detail += f"; {bank['outward_returns']} deposited cheque(s) returned (not counted against the applicant)"
            checks.append(_check("conduct", "Account conduct", status, bounces, detail))
        # 5b. Loan stacking: lenders / loan apps seen in the statement
        if bank:
            n_l = bank.get("lenders_count", 0)
            names = ", ".join(sorted(bank.get("lenders", {})))
            status = "fail" if n_l >= P["lenders_fail"] else "caution" if n_l >= P["lenders_caution"] else "pass"
            checks.append(_check("lenders", "Other lenders / loan apps", status, n_l,
                                 "No NBFC or loan-app activity" if not n_l else
                                 f"{n_l} lender(s) seen: {names}" + (" (several lenders in 6 months suggests "
                                                                     "loan stacking)" if status != "pass" else "")))
        # 5c. Latest salary (salaried): stopped or fallen?
        if emp in SALARIED_TYPES and regular > 0 and bank.get("salary_latest_vs_usual") is not None:
            ratio = bank["salary_latest_vs_usual"]
            if bank.get("salary_stopped"):
                checks.append(_check("salary_trend", "Latest salary", "caution", 0.0,
                                     "No salary credited in the latest month: confirm the applicant is still employed"))
            elif ratio < P["salary_drop_caution"]:
                checks.append(_check("salary_trend", "Latest salary", "caution", ratio,
                                     f"The latest salary is {ratio:.0%} of the usual amount: ask the reason "
                                     f"(job change, unpaid leave, pay cut)"))
            else:
                checks.append(_check("salary_trend", "Latest salary", "pass", ratio,
                                     "Latest salary in line with earlier months"))
        # 5c2. Pass-through account: money received from people it is sent straight back to
        #      (accommodation entries that inflate turnover). Already left out of income; refer.
        rt_share = stmt.get("round_trip_in_share") or 0
        rt_month = (stmt.get("excluded_inflows") or {}).get("Round-trip transfers", 0)
        if rt_share >= P["pass_through_caution"] and rt_month >= 10000:
            checks.append(_check("pass_through", "Money passing through", "caution", round(rt_share, 3),
                                 f"{rt_share:.0%} of the money received (₹{rt_month:,.0f} a month) came from people "
                                 f"the applicant sent most of it back to. It is not counted as income: ask what "
                                 f"these transfers are"))
        # 5d. One-off large credits still in the account at the end (window dressing)
        if bank.get("window_dressing"):
            big = max((c for c in bank["large_credits"] if c["last_30_days"]), key=lambda c: c["amount"])
            checks.append(_check("source_of_funds", "Large recent credit", "caution", big["amount"],
                                 f"₹{big['amount']:,.0f} came in on {pd.Timestamp(big['date']):%d %b %Y} (not salary or "
                                 f"business income) shortly before the statement end and is still in the account: "
                                 f"ask for its source"))
        # 6. Statement integrity (running-balance chain)
        breaks = stmt.get("balance_breaks")
        n_tx = stmt.get("transactions") or 0
        if breaks is None:
            checks.append(_check("integrity", "Statement integrity", "caution", None,
                                 "No running balance in the file, so it cannot be checked for edits"))
        elif n_tx and breaks > P["max_breaks_share"] * n_tx:
            checks.append(_check("integrity", "Statement integrity", "caution", breaks,
                                 f"Only {1 - breaks / n_tx:.0%} of rows reconcile with the running balance: this "
                                 f"statement layout could not be read reliably. Please verify the figures manually"))
        else:
            checks.append(_check("integrity", "Statement integrity", "pass" if breaks == 0 else "caution", breaks,
                                 f"{n_tx} transactions read; the running balance reconciles on every row"
                                 if breaks == 0 else
                                 f"{breaks} row(s) where the balance does not add up: the file may have been edited"))
        # 5b'. Where the data came from: bank-signed Account Aggregator data is the strongest evidence (FAIRScore 3.0
        # prefers it); a PDF is checked for editing below
        method = (stmt.get("read_info") or {}).get("method") or ""
        if method.startswith("Account Aggregator"):
            checks.append(_check("source", "Data source", "info", "AA",
                                 "Account Aggregator: bank-signed data fetched with the customer's consent; it cannot be "
                                 "edited, so the document-editing checks are not needed"))
        else:
            checks.append(_check("source", "Data source", "info", "PDF",
                                 "PDF statement: checked for editing (balance chain, fonts, printed totals, file "
                                 "origin). An Account Aggregator fetch removes the risk of an edited document"))
        # 5c. Signs of a consistent edit (balances recalculated so the chain still adds up)
        tamper = (stmt.get("read_info") or {}).get("tamper") or {}
        mism = [x for x in (tamper.get("summary") or []) if not x["match"]]
        names = {"closing": "closing balance", "total_credits": "total credits", "total_debits": "total debits"}
        if tamper.get("font_outliers") or mism:
            parts = []
            if tamper.get("font_outliers"):
                parts.append(f"{tamper['font_outliers']} amount(s) printed in a different font from every other "
                             f"amount (on {', '.join(tamper['font_outlier_rows'][:3])})")
            for x in mism:
                parts.append(f"the printed {names.get(x['item'], x['item'])} {_inr(x['printed'])} does not match the "
                             f"transactions ({_inr(x['computed'])})")
            checks.append(_check("tamper", "Signs of editing", "caution", len(parts),
                                 "; ".join(parts) + ": ask for the original e-statement or an Account Aggregator fetch"))
        elif tamper.get("dominant_font_share") is not None or tamper.get("summary"):
            done = (["fonts of all amounts consistent"] if tamper.get("dominant_font_share") is not None else []) + \
                   ([f"printed {', '.join(names.get(x['item'], x['item']) for x in tamper['summary'])} match the "
                     f"transactions"] if tamper.get("summary") else [])
            checks.append(_check("tamper", "Signs of editing", "pass", 0, "None found: " + "; ".join(done)))
        # 6. Is the statement the applicant's own? (name on the statement vs the application;
        # compared as hashes, the name is never stored)
        if a.get("account_holder_name") and stmt.get("holder_keys"):
            from statement_parser import name_keys
            match = bool(name_keys(a["account_holder_name"]) & set(stmt["holder_keys"]))
            checks.append(_check("holder_match", "Statement belongs to the applicant", "pass" if match else "caution",
                                 match, "The account holder's name on the statement matches the application" if match
                                 else "The name on the statement does not match the applicant's name: confirm the "
                                      "account is the applicant's own (joint account, name change, spelling)"))
        # 6a. Several statements: do they belong to the same person?
        if stmt.get("accounts", 1) > 1:
            checks.append(_check("holders", "Same account holder", "caution" if stmt.get("different_holders") else "pass",
                                 stmt["accounts"],
                                 "The statements are in the names of different people: ask whose accounts they are"
                                 if stmt.get("different_holders") else
                                 f"{stmt['accounts']} accounts assessed together; {stmt.get('internal_transfers', 0)} "
                                 f"transfer(s) between them are not counted as income or spending"))
        # 6b. Where the file came from (metadata): edited after the bank produced it?
        origin = ((stmt.get("read_info") or {}).get("file_origin") or {})
        if origin:
            cat, tool = origin.get("category"), origin.get("tool")
            if cat == "edited":
                checks.append(_check("file_origin", "Document origin", "caution", tool,
                                     f"The PDF was last saved with {tool}, a document editor, after the bank produced "
                                     f"it: ask for the original e-statement or an Account Aggregator fetch"))
            elif cat == "re-saved":
                checks.append(_check("file_origin", "Document origin", "pass", tool,
                                     f"Re-saved with {tool} (often done to remove the password); the balances still "
                                     f"reconcile, so no figures were altered"
                                     if not stmt.get("balance_breaks") else
                                     f"Re-saved with {tool}; combined with balances that do not reconcile, ask for "
                                     f"the original e-statement"))
            else:
                checks.append(_check("file_origin", "Document origin", "pass", cat,
                                     "Produced by the bank's statement system and not re-saved since"
                                     if cat == "bank-generated" else "Producer not recorded in the file"))
        # 7. Money left each month after current spending and the new EMI: shown for the officer,
        # but it does NOT refer the case. Backtested on 679 real loans (results/berka_backtest_*), it
        # flagged 85% of applicants -- in a real account money out is close to money in, and people
        # adjust discretionary spending to a new EMI -- while adding little: 11.4% defaults when
        # flagged vs 6.9% when not. The affordability that matters is tested by FOIR and the
        # take-home-for-the-household rule, which both separate defaults far better.
        spend = stmt.get("avg_monthly_spend")
        if spend is not None and income:
            surplus = income - spend - new_emi
            checks.append(_check("surplus", "Money left each month", "info", round(surplus),
                                 f"After current spending (₹{spend:,.0f}, incl. rent and EMIs; SIP / investments not counted) "
                                 f"and the new EMI, {_inr(surplus)} a month would remain at today's spending"))
        # 8. Gambling
        bet = stmt.get("betting_ratio", 0.0) or 0.0
        checks.append(_check("gambling", "Gambling / betting spend",
                             _grade(bet, P["betting_pass"], P["betting_fail"]), round(bet, 3),
                             f"{bet:.1%} of all spending"))

    # 10. Net take-home after all EMIs (family size) / loan size vs income / balance cover
    if income:
        need = P["min_take_home"] + P["take_home_per_dependent"] * int(a.get("dependents") or 0)
        left = income - obligations - new_emi
        checks.append(_check("take_home", "Take-home after EMIs", "pass" if left >= need else "caution", round(left),
                             f"After all EMIs {_inr(left)} a month remains; at least {_inr(need)} is needed for a "
                             f"household with {int(a.get('dependents') or 0)} dependent(s)"))
        multiple = a["requested_amount"] / income
        checks.append(_check("loan_size", "Loan size vs income",
                             "pass" if multiple <= P["max_income_multiple"] else "caution", round(multiple, 1),
                             f"The loan is {multiple:.1f}x monthly income (limit {P['max_income_multiple']}x)"))
    bank = (stmt or {}).get("banking") or {}
    if has_stmt and a.get("employment_type") in P["abb_cover_applies_to"] and bank.get("abb") is not None:
        cover = bank["abb"] / max(obligations + new_emi, 1.0)
        checks.append(_check("balance_cover", "Average balance cover", "pass" if cover >= P["abb_cover_min"]
                             else "caution", round(cover, 2),
                             f"Average daily balance {_inr(bank['abb'])} is {cover:.1f}x the monthly EMIs "
                             f"(at least {P['abb_cover_min']}x expected for business owners)"))

    # 9a. Own contribution (loan-to-value) for a purchase loan whose price is known
    ltv = loan_to_value(a)
    if ltv is not None:
        status = "fail" if ltv > P["ltv_fail"] else "caution" if ltv > P["ltv_caution"] else "pass"
        detail = (f"The loan is {ltv:.0%} of the price (₹{float(a['asset_price']):,.0f}); the borrower pays "
                  f"{max(0.0, 1 - ltv):.0%}")
        if status == "fail":
            detail += ": the loan is larger than the price of the item"
        elif status == "caution":
            detail += (f" (limit {P['ltv_caution']:.0%}: on 116,950 Indian new-to-credit vehicle loans, borrowers "
                       f"financing more than 85% of the price missed their first EMI 30% of the time, against 22% "
                       f"for the rest)")
        checks.append(_check("own_contribution", "Own contribution (loan-to-value)", status, round(ltv, 3), detail))

    # 9. Declared credit vs statement
    consistency, msg = check_consistency(a, stmt)
    if consistency == "conflict":
        checks.append(_check("consistency", "Declaration consistency", "caution", None, msg))
    elif consistency == "note":
        checks.append(_check("consistency", "Declaration consistency", "pass", None, msg))
    elif has_stmt and a.get("active_credit_lines") is None:
        n = stmt.get("detected_credit_signal_count", 0)
        checks.append(_check("consistency", "Existing loans", "pass", n,
                             f"Not declared; {n} recurring EMI(s) found in the statement and used"))
    elif has_stmt:
        checks.append(_check("consistency", "Declaration consistency", "pass", None,
                             "Declared loans/cards match the statement"))

    return {"checks": checks, "foir": foir, "new_emi": new_emi, "existing_emi": obligations,
            "income_used": income, "income_verified": verified is not None, "security": sec, "bureau": bur,
            "co_applicant_income": co,
            "monthly_spend": stmt.get("avg_monthly_spend") if has_stmt else None}


RISK_LEVELS = [(POLICY["level_cuts"]["Low"], "Low"), (POLICY["level_cuts"]["Moderate"], "Moderate"),
               (POLICY["level_cuts"]["Elevated"], "Elevated"), (101, "High")]


def risk_level(pct):
    if pct is None:
        return None
    return next(label for cut, label in RISK_LEVELS if pct < cut)


def _ordinal(n):
    n = int(round(n))
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def decision_confidence(policy, ml, stmt):
    """How sure the recommendation is, from the things that make it fragile:
    how the statement was read, how close the case sits to a policy limit, what
    kind of income it rests on, and whether the risk models agree. Low
    confidence sends an otherwise approvable case to an officer (manual review)."""
    P = POLICY
    score_, notes = 1.0, []
    info = (stmt or {}).get("read_info") or {}
    if not _has_statement(stmt):
        return {"score": 0.0, "label": "Low", "notes": ["No bank statement: nothing is verified"]}
    rec = info.get("reconciled")
    if rec is not None and rec < 1.0:
        score_ -= 0.15 if rec >= P["min_reconciled"] else 0.4
        notes.append(f"{rec:.0%} of statement rows reconcile with the running balance")
    if stmt.get("balance_breaks") is None:
        score_ -= 0.3
        notes.append("No running balance to verify the figures")
    tamper = info.get("tamper") or {}
    if tamper.get("font_outliers") or any(not x["match"] for x in (tamper.get("summary") or [])):
        score_ -= 0.4
        notes.append("Signs of editing in the statement")
    if not str(stmt.get("income_basis", "")).startswith("regular"):
        score_ -= 0.2
        notes.append("Income is business / farm / other inflows, not a salary (ITR / GST returns, or land records "
                     "for farmers, would confirm it)")
    foir = policy.get("foir", 0)
    if P["foir_pass"] - 0.05 < foir <= P["foir_pass"]:
        score_ -= 0.25
        notes.append(f"EMIs at {foir:.0%} of income, within 5 points of the {P['foir_pass']:.0%} limit")
    pct = ml.get("consensus_percentile")
    agreement = ml.get("confidence", {})
    if pct is not None and POLICY["level_cuts"]["Moderate"] <= pct < P["refer_from_percentile"]:
        score_ -= 0.2
        notes.append(f"Risk models place the applicant at the {_ordinal(pct)} percentile (elevated)")
    if agreement.get("spread", 0) >= 50 and pct is not None and pct >= 50:
        score_ -= 0.15
        notes.append(f"Risk models disagree by {agreement['spread']:.0f} percentile points")
    score_ = max(0.0, round(score_, 2))
    label = "High" if score_ >= 0.75 else "Medium" if score_ >= P["min_confidence_to_approve"] else "Low"
    return {"score": score_, "label": label, "notes": notes}


def decide(policy, ml, confidence=None, fair=None):
    """Policy decides; the FAIRScore, the risk models and low decision confidence can
    only send an otherwise clean case to an officer (refer), never approve or decline."""
    fails = [c for c in policy["checks"] if c["status"] == "fail"]
    cautions = [c for c in policy["checks"] if c["status"] == "caution"]
    reasons = []
    if fails:
        return "decline", [f"{c['name']}: {c['detail']}" for c in fails]
    if cautions:
        reasons += [f"{c['name']}: {c['detail']}" for c in cautions]
    if ml["consensus_percentile"] is None:
        reasons.append("Risk models: not enough evidence for any model to vote")
    elif ml["consensus_percentile"] >= POLICY["refer_from_percentile"]:
        reasons.append(f"Risk models: applicant is riskier than {ml['consensus_percentile']:.0f}% "
                       f"of past applicants")
    if fair and fair["score"] < POLICY["score_gate"]:
        weakest = [f["label"].lower() for f in fair["factors"][:2]]
        reasons.append(f"FAIRScore {fair['score']} ({fair['band']}) is below {POLICY['score_gate']}: weakest on "
                       f"{' and '.join(weakest)}")
    if not reasons and confidence and confidence["score"] < POLICY["min_confidence_to_approve"]:
        reasons.append("Low decision confidence: " + "; ".join(confidence["notes"]))
    return ("refer", reasons) if reasons else ("approve", [])


def ai_recourse(a, stmt, stmt_df, ai):
    """What would make the AI approve, checked by re-running the model: (1) a smaller loan or longer term,
    re-assessed through the WHOLE decision; (2) changes of habit, re-scored by the AI engine."""
    settings = POLICY["ai"]
    amax = ai_engine.approve_max(ai_engine.load(), settings)
    out = {"loan": None, "habits": ai_engine.counterfactuals(ai["X"], settings, amax)}
    amount, term = float(a["requested_amount"]), int(a["requested_duration_months"])
    longest = min(POLICY["max_term_months"], int((POLICY["max_age_at_maturity"].get(a.get("employment_type"), 60)
                                                  - float(a["age"])) * 12))
    for frac in (0.8, 0.6, 0.45, 0.3):
        amt = float(np.floor(amount * frac / 1000) * 1000)
        if amt < POLICY["min_amount"]:
            break
        for t in sorted({term, min(longest, term + 12)}):
            if t < POLICY["min_term_months"]:
                continue
            q = quick_decision({**a, "requested_amount": amt, "requested_duration_months": t}, stmt, stmt_df)
            if q["verdict"] == "approve":
                out["loan"] = {"amount": amt, "term": t, "emi": emi(amt, t),
                               "pd": (q.get("ai") or {}).get("pd")}
                return out
    return out


def final_decision(a, stmt, policy, ml, conf, fair):
    """FAIRScore 2.0: the AI engine decides within the policy's hard stops (see ai_engine.decide);
    version 1 (policy-first) is kept for comparison and as the path the AI defers to.
    Returns (verdict, reasons, extra) where extra = {decided_by, v1_verdict, v1_reasons, ai, conditions}."""
    v1, v1_reasons = decide(policy, ml, conf, fair)
    settings = POLICY["ai"]
    extra = {"decided_by": "policy (version 1)", "v1_verdict": v1, "v1_reasons": v1_reasons, "ai": None,
             "conditions": []}
    if settings["mode"] != "ai_first" or not ai_engine.available() or not _has_statement(stmt):
        return v1, v1_reasons, extra
    try:
        sec = policy.get("security")
        ai = ai_engine.assess(a, stmt, fair["fractions"], policy.get("new_emi") or 0.0,
                              policy.get("existing_emi") or 0.0, settings,
                              lgd=sec["lgd"] if sec else None, coverage=sec["coverage"] if sec else 0.0)
    except Exception as exc:          # the AI must never block a decision
        extra["ai_error"] = str(exc)
        return v1, v1_reasons, extra
    statuses = {}
    for c in policy["checks"]:
        rank = {"fail": 3, "caution": 2, "info": 1, "pass": 0}
        if rank.get(c["status"], 0) >= rank.get(statuses.get(c["key"]), -1):
            statuses[c["key"]] = c["status"]
    amax = ai_engine.approve_max(ai_engine.load(), settings)
    verdict, by, codes = ai_engine.decide(statuses, ai, {**settings, "_security": POLICY["security"]}, amax)
    extra["ai"] = ai
    by_key = {c["key"]: c for c in policy["checks"]}
    in_scope = (a.get("employment_type") or "") in settings["ai_decides_for"]
    if by == "ai" and not in_scope:
        by = "defer"
    if by == "defer":
        extra["decided_by"] = "policy (AI advisory)"
        if not ai["in_experience"]:
            note = "the AI deferred: this statement looks unlike any it learned from"
        elif not in_scope:
            note = (f"the AI is advisory for {(a.get('employment_type') or 'this kind of').lower()} applicants "
                    f"(none in its training data); the lending policy decided")
        else:
            note = "the AI could not assess this case"
        extra["ai_note"] = note
        if v1 == "approve" and ai["outcome_set"] == "defaults" and ai["pd"] >= settings["decline_min_pd"]:
            return "refer", [f"AI warning: default risk {ai['pd_pct']:.1f}%; an officer should review ({note})"], extra
        return v1, v1_reasons, extra
    if by == "policy":
        extra["decided_by"] = "policy (hard stop)"
        status = "fail" if verdict == "decline" else "caution"
        reasons = [f"{c['name']}: {c['detail']}" for c in policy["checks"] if c["key"] in codes and c["status"] == status]
        return verdict, reasons, extra
    extra["decided_by"] = "AI"
    extra["conditions"] = [f"{c['name']}: {c['detail']}" for c in policy["checks"] if c["status"] == "caution"]
    drivers = [g for g in ai["explanation"]["groups"] if g["impact"] > 0][:3]
    if verdict == "approve":
        return verdict, [], extra
    if verdict == "decline":
        if POLICY["learning"]["ai_decline_needs_local_validation"] and not ai.get("adapted"):
            # not yet validated on local outcomes: the AI's "confident default" is a strong warning for an officer,
            # not a decline (see [learning] in policy.toml)
            extra["ai_decline_held"] = True
            return "refer", [f"AI warning: default risk {ai['pd_pct']:.1f}% ({ai_engine.odds_text(ai['pd'])} similar "
                             f"borrowers in its training data defaulted). The AI is not yet validated on local repayment "
                             f"outcomes, so it does not decline on its own: an officer decides"] + \
                [f"Main reason: {g['group']} ({g['evidence']})" for g in drivers], extra
        return verdict, [f"AI: default risk {ai['pd_pct']:.1f}% ({ai_engine.odds_text(ai['pd'])} similar borrowers "
                         f"defaulted); it is confident this loan would not be repaid"] + \
            [f"Main reason: {g['group']} ({g['evidence']})" for g in drivers], extra
    if ai["outcome_set"] == "repays":
        why = (f"default risk {ai['pd_pct']:.1f}% is above the bank's automatic-approval limit of "
               f"{100 * amax:.1f}%")
    elif ai["outcome_set"] == "defaults":
        why = (f"signs of high risk (default risk {ai['pd_pct']:.1f}%), but below the {100 * settings['decline_min_pd']:.0f}% "
               f"needed to recommend decline")
    else:
        why = f"cannot rule out either outcome (default risk {ai['pd_pct']:.1f}%)"
    return verdict, [f"AI not certain: {why}"] + [f"Weighs against: {g['group']} ({g['evidence']})"
                                                  for g in drivers[:2]], extra


AFFORDABILITY_KEYS = {"affordability", "take_home", "loan_size", "balance_cover", "own_contribution",
                      "student_limits"}


def eligibility(a, stmt, pol):
    """Eligible loan amount, computed the way a credit officer does: the LOWEST of the
    affordability limits, each rounded down to the nearest Rs 1,000, plus the shortest
    term (within the age limit) at which the requested amount would fit."""
    P = POLICY
    inc, ex, spend = pol["income_used"], pol["existing_emi"], pol.get("monthly_spend")
    emp = a.get("employment_type")
    abb = ((stmt or {}).get("banking") or {}).get("abb")
    max_age = P["max_age_at_maturity"].get(emp, 60)
    longest = max(0, min(P["max_term_months"], int((max_age - float(a["age"])) * 12)))
    if emp == "Student":
        longest = min(longest, P["student_max_term_months"])

    def limits(term):
        unit = emi(1.0, term)
        out = []
        if inc:
            out.append(("EMI-to-income limit (FOIR)", (P["foir_pass"] * inc - ex) / unit))
            need = P["min_take_home"] + P["take_home_per_dependent"] * int(a.get("dependents") or 0)
            out.append(("Take-home for the household", (inc - ex - need) / unit))
            out.append((f"{P['max_income_multiple']}x monthly income", P["max_income_multiple"] * inc))
            if emp in P["abb_cover_applies_to"] and abb is not None:
                out.append(("Average balance cover", (abb / P["abb_cover_min"] - ex) / unit))
        if loan_to_value(a) is not None:
            out.append((f"Own contribution (at most {P['ltv_caution']:.0%} of the price)",
                        P["ltv_caution"] * float(a["asset_price"])))
        out.append(("Maximum loan amount", P["max_amount"]))
        sec = pol.get("security")
        if sec:
            out.append((f"{sec['type']} security ({sec['ltv_cap']:.0%} of its value)", sec["max_amount"]))
        if emp == "Student" and inc:
            out.append(("Student limit (income multiple / no co-applicant)", student_limits(a, inc)[0]))
        return [{"limit": n, "amount": float(max(0.0, np.floor(v / 1000) * 1000))} for n, v in out]

    term = a["requested_duration_months"]
    if emp == "Student" and 0 < longest < term:
        term = longest                                  # a student's longest term (policy)
    if not inc:
        return {"eligible_amount": 0.0, "term": term, "binding": "No verified income", "limits": [],
                "longest_term": longest, "fits_at_term": None}
    lim = limits(term)
    best = min(lim, key=lambda x: x["amount"])
    fits_at = None
    for t in range(term + 6, longest + 1, 6):
        if min(x["amount"] for x in limits(t)) >= a["requested_amount"]:
            fits_at = t
            break
    return {"eligible_amount": best["amount"], "term": term, "binding": best["limit"], "limits": lim,
            "emi": emi(best["amount"], term) if best["amount"] else 0.0, "longest_term": longest,
            "fits_at_term": fits_at}


def counter_offer(a, stmt, stmt_df, verdict, policy, elig):
    """If the request fails only on size, the amount (or term) at which it would be
    approved -- confirmed by re-running the whole decision, not just the arithmetic."""
    if verdict == "approve" or elig["eligible_amount"] < POLICY["min_amount"]:
        return None
    others = [c for c in policy["checks"] if c["status"] in ("caution", "fail") and c["key"] not in AFFORDABILITY_KEYS]
    if others or a["requested_amount"] <= elig["eligible_amount"]:
        return None
    options = [{"amount": elig["eligible_amount"], "term": elig.get("term") or a["requested_duration_months"]}]
    if elig.get("fits_at_term"):
        options.append({"amount": float(a["requested_amount"]), "term": elig["fits_at_term"]})
    for o in options:
        q = quick_decision({**a, "requested_amount": o["amount"], "requested_duration_months": o["term"]}, stmt, stmt_df)
        o["verdict"] = q["verdict"]
        o["emi"] = emi(o["amount"], o["term"])
    ok = [o for o in options if o["verdict"] == "approve"]
    return ok or None


def cash_flow(stmt_df, stmt=None):
    """Month-by-month income counted / other money in / money out / closing balance."""
    if stmt_df is None or not len(stmt_df):
        return None
    df = stmt_df.copy()
    basis_all = (stmt or {}).get("income_basis", "").startswith("all")
    counted_types = ["salary", "regular", "other"] if basis_all else ["salary", "regular"]
    if "inflow_type" not in df.columns:
        df["inflow_type"] = "other"
    credit = df["signed_amount"] > 0
    df["income"] = df["signed_amount"].where(credit & df["inflow_type"].isin(counted_types), 0.0)
    df["other_in"] = df["signed_amount"].where(credit & ~df["inflow_type"].isin(counted_types), 0.0)
    df["out"] = (-df["signed_amount"]).where(df["signed_amount"] < 0, 0.0)
    g = df.groupby("month").agg(income=("income", "sum"), other_in=("other_in", "sum"), money_out=("out", "sum"))
    if "balance" in df.columns and df["balance"].notna().any():
        g["closing_balance"] = df.groupby("month")["balance"].last()
    g.index = g.index.strftime("%b %Y")
    return g.reset_index().rename(columns={"month": "Month"})


def _quick_one(a, stmt, stmt_df, exclude_models=(), ignore_age_rule=False):
    """Verdict only (no explanations), fast enough for interactive what-if. With ignore_age_rule the
    legal / product age limit is set aside (used by the age-flip test, which asks whether age matters
    for anything ELSE)."""
    stmt = prepare_statement(stmt, a, stmt_df)
    ml = score(a, stmt, exclude_models)
    policy = policy_checks(a, stmt, stmt_df)
    if ignore_age_rule:
        policy["checks"] = [eligibility_check(a, ignore_age=True, income=policy["income_used"])
                            if c["key"] == "eligibility" else c for c in policy["checks"]]
    conf = decision_confidence(policy, ml, stmt)
    fair = fairscore(a, stmt, policy)
    verdict, reasons, extra = final_decision(a, stmt, policy, ml, conf, fair)
    return {"verdict": verdict, "reasons": reasons, "policy": policy, "decision_confidence": conf, "ai": extra["ai"],
            "decided_by": extra["decided_by"], "fairscore": fair, "risk_level": risk_level(ml["consensus_percentile"]),
            "consensus_percentile": ml["consensus_percentile"]}


def age_flip_test(a, stmt, stmt_df, ages=(23, 30, 40, 50, 58)):
    """Counterfactual fairness check on THIS application: re-decide it with only
    the age changed. The legal / product age limit (age-based by design, and shown
    as its own check) is set aside on both sides, so an applicant above the age limit
    is compared fairly too. Any change of verdict means age is influencing the
    decision through something else, and is reported."""
    base = quick_decision(a, stmt, stmt_df, ignore_age_rule=True)["verdict"]
    tried, changed = [], []
    for age in ages:
        if age == a["age"] or age < POLICY["min_age"]:
            continue
        v = quick_decision({**a, "age": age}, stmt, stmt_df, ignore_age_rule=True)["verdict"]
        tried.append(age)
        if v != base:
            changed.append({"age": age, "verdict": v})
    return {"verdict": base, "ages_tested": tried, "changed": changed, "passed": not changed}


def validate_applicant(a):
    """Range / value checks on the intake (API and dashboard). Returns a list of problems."""
    P = POLICY
    problems = [f"Missing: {k}" for k in missing_required(a)]
    if problems:
        return problems

    def num(k, lo, hi):
        try:
            v = float(a[k])
        except (TypeError, ValueError, OverflowError):
            problems.append(f"{k} must be a number")
            return
        if v != v:
            problems.append(f"{k} must be a number")
            return
        if not lo <= v <= hi:
            problems.append(f"{k} must be between {lo:,} and {hi:,} (got {v:,.0f})")

    num("age", 18, 100)
    num("employment_years", 0, 60)
    num("dependents", 0, 20)
    num("total_savings", 0, 1e10)
    num("requested_amount", P["min_amount"], P["max_amount"])
    num("requested_duration_months", P["min_term_months"], P["max_term_months"])
    for k, opts in (("employment_type", EMPLOYMENT_TYPES), ("housing", HOUSING_OPTIONS),
                    ("purpose", PURPOSE_OPTIONS)):
        if not isinstance(a[k], str) or a[k] not in opts:
            problems.append(f"{k} must be one of: {', '.join(opts)}")
    if a.get("education") not in (None, "") and (not isinstance(a["education"], str)
                                                 or a["education"] not in EDUCATION_LEVELS):
        problems.append(f"education must be one of: {', '.join(EDUCATION_LEVELS)}")
    if not isinstance(a["owns_car"], bool):
        problems.append("owns_car must be true or false")
    for k in ("monthly_income", "active_credit_lines", "property_loans"):
        if a.get(k) is not None:
            num(k, 0, 1e9)
    if a.get("asset_price") not in (None, ""):
        num("asset_price", 1000, 1e9)
    if a.get("security_type") not in (None, "", "None"):
        if not isinstance(a["security_type"], str) or a["security_type"] not in security.SECURITY_TYPES:
            problems.append(f"security_type must be one of: {', '.join(security.SECURITY_TYPES)}")
        elif a.get("security_value") in (None, ""):
            problems.append("security_value is needed when a security is offered")
    if a.get("security_value") not in (None, ""):
        num("security_value", 1000, 1e10)
    if a.get("co_applicant_income") not in (None, ""):
        num("co_applicant_income", 0, 1e8)
    if a.get("bureau_record") not in (None, "", "None"):
        if a["bureau_record"] not in bureau.RECORDS:
            problems.append(f"bureau_record must be one of: {', '.join(bureau.RECORDS)}")
        if a.get("bureau_score") not in (None, ""):
            num("bureau_score", 1, 999)
        if a.get("bureau_max_dpd_12m") not in (None, "") and a["bureau_max_dpd_12m"] not in (0, 30, 60, 90):
            problems.append("bureau_max_dpd_12m must be 0, 30, 60 or 90")
        if a.get("bureau_severe_24m") not in (None, "") and not isinstance(a["bureau_severe_24m"], bool):
            problems.append("bureau_severe_24m must be true or false")
        for k, hi in (("bureau_enquiries_6m", 100), ("bureau_emi", 1e8)):
            if a.get(k) not in (None, ""):
                num(k, 0, hi)
    if a.get("account_holder_name") not in (None, "") and (not isinstance(a["account_holder_name"], str)
                                                           or len(a["account_holder_name"]) > 120):
        problems.append("account_holder_name must be text (up to 120 characters)")
    return problems


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------
CASHFLOW_FACTORS = {"foir": "emi_burden", "emi_to_income": "emi_burden", "loan_to_annual_income": "loan_to_income",
                    "term_months": "loan_duration", "income_volatility": "income_regularity",
                    "regular_income_share": "income_regularity", "overdraft_share": "low_balances",
                    "low_balance_days": "low_balances", "negative_days": "low_balances",
                    "min_balance_to_income": "low_balances", "penalty_charges": "penalties",
                    "abb_to_emi": "balance_cover", "abb_trend": "balance_trend", "cash_deposit_share": "cash_use",
                    "cash_withdrawal_share": "cash_use", "txn_per_month": "activity"}


def _factor(feature):
    if feature in CASHFLOW_FACTORS:
        return CASHFLOW_FACTORS[feature]
    if feature == "duration":
        return "loan_duration"
    if feature == "age":
        return "age"
    if feature in ("existing_credits", "NumberOfOpenCreditLinesAndLoans") or \
            feature.startswith("other_installment_plans_"):
        return "existing_credit"
    if feature.startswith("checking_account_"):
        return "account_balance"
    if feature.startswith("purpose_"):
        return "purpose"
    if feature.startswith("savings_account_"):
        return "savings"
    if feature.startswith(("employment_since_", "NAME_INCOME_TYPE_")) or \
            feature in ("DAYS_EMPLOYED", "days_employed_anomaly"):
        return "employment"
    if feature.startswith(("housing_", "NAME_HOUSING_TYPE_")) or feature == "FLAG_OWN_REALTY_Y":
        return "housing"
    if feature == "DebtRatio":
        return "spend_ratio"
    if feature == "NumberRealEstateLoansOrLines":
        return "property_loans"
    if feature in ("NumberOfDependents", "CNT_CHILDREN"):
        return "dependents"
    if feature in ("monthly_income_was_missing", "dependents_was_missing"):
        return "completeness"
    if feature == "FLAG_OWN_CAR_Y":
        return "vehicle"
    if feature.startswith("NAME_EDUCATION_TYPE_"):
        return "education"
    if feature == "credit_to_income":
        return "loan_to_income"
    return feature


def _factor_label(key, a, stmt):
    inc = _statement_income(stmt) or a.get("monthly_income") or 0
    spend = (stmt or {}).get("avg_monthly_spend")
    b = (stmt or {}).get("banking") or {}
    new = new_loan_emi(a)
    all_emi = new + (stmt or {}).get("_existing_emi", 0.0)
    abb = b.get("abb")
    labels = {
        "emi_burden": (f"All EMIs would be {all_emi / inc:.0%} of income" if inc else "No verified income for the EMI"),
        "income_regularity": (f"Monthly income varies by {(stmt or {}).get('income_volatility', 0):.0%}"
                              if inc else "No regular income found"),
        "low_balances": (f"Balance under ₹1,000 on {b.get('low_balance_days_share', 0):.0%} of days, lowest "
                         f"{_inr((stmt or {}).get('min_balance') or 0)}" if b.get("low_balance_days_share") is not None
                         else "Balances not available"),
        "penalties": f"{int(b.get('penalty_charges', 0) or 0)} penalty / bounce charge(s)",
        "balance_cover": (f"Average balance {_inr(abb)} = {abb / max(new, 1):.1f}x the new EMI" if abb is not None
                          else "Average balance not available"),
        "balance_trend": ("Balance rising over the last 3 months" if (b.get("abb_last_3_months") or 0) >= (abb or 0)
                          else "Balance falling over the last 3 months"),
        "cash_use": (f"Cash: {b.get('cash_deposit_share', 0):.0%} of money in, "
                     f"{b.get('cash_withdrawal_share', 0):.0%} of money out"),
        "activity": f"{(stmt or {}).get('transaction_frequency', 0):.0f} transactions a month",
        "loan_duration": f"Loan length: {a['requested_duration_months']} months",
        "age": f"Age: {a['age']}",
        "existing_credit": f"Existing loans / cards: {credit_lines(a, stmt)[0] or 0}",
        "account_balance": (f"Average account balance: ₹{stmt['avg_balance']:,.0f}"
                            if _has_statement(stmt) and stmt.get("avg_balance") is not None
                            else "Account balance (no statement)"),
        "purpose": f"Loan purpose: {a['purpose']}",
        "savings": (f"Declared savings: ₹{a['total_savings']:,.0f}" if a.get("total_savings") is not None
                    else "Savings: not declared"),
        "employment": f"Employment: {a['employment_type']}, {float(a.get('employment_years') or 0):g} yrs",
        "housing": f"Housing: {a['housing']}",
        "spend_ratio": (f"Monthly spending is {spend / inc:.0%} of income"
                        if spend is not None and inc else "Spending vs income (no statement)"),
        "property_loans": (f"Home / property loans: {int(a['property_loans'])}"
                           if a.get("property_loans") is not None else
                           f"Home-loan EMIs in the statement: {int((stmt or {}).get('detected_property_loans') or 0)}"
                           if _has_statement(stmt) else "Home / property loans: not declared"),
        "dependents": (f"Dependents: {a['dependents']}" if a.get("dependents") is not None
                       else "Dependents: not declared"),
        "completeness": ("Dependents not declared" if a.get("dependents") is None
                         else "Income and dependents known"),
        "vehicle": f"Owns a vehicle: {'yes' if a.get('owns_car') else 'no'}",
        "education": f"Education: {a.get('education') or 'not declared'}",
        "loan_to_income": (f"Loan is {a['requested_amount'] / (inc * 12):.1f}x annual income"
                           if inc else "Loan vs income"),
    }
    return labels.get(key, key)


def explain(result, a, stmt, top_n=4):
    """Adds per-model SHAP detail and combined plain-language reasons."""
    combined, per_model = {}, {}
    total_w = sum(result["models"][n]["weight"] for n in result["voters"]) or 1.0
    for name in result["voters"]:
        m = result["models"][name]
        sv = explainer(name).shap_values(m["X"])
        sv = np.asarray(sv[1] if isinstance(sv, list) else sv)
        sv = sv[0] if sv.ndim == 2 else sv
        rows = [{"feature": c, "value": m["X"].iloc[0][c], "shap": float(v),
                 "source": m["tags"].get(c, "imputed")} for c, v in zip(m["X"].columns, sv)]
        rows.sort(key=lambda r: abs(r["shap"]), reverse=True)
        per_model[name] = rows

        by_factor = {}
        for r in rows:
            if r["source"] == "imputed":   # unknown inputs are not shown to users as reasons
                continue
            k = _factor(r["feature"])
            by_factor[k] = by_factor.get(k, 0.0) + r["shap"]
        norm = sum(abs(v) for v in by_factor.values()) or 1.0
        share = m["weight"] / total_w
        for k, v in by_factor.items():
            entry = combined.setdefault(k, {"impact": 0.0, "models": []})
            entry["impact"] += share * v / norm
            if abs(v) / norm >= 0.05:
                entry["models"].append(DATASET_LABELS[name])

    factors = [{"key": k, "label": _factor_label(k, a, stmt), "impact": round(v["impact"], 4),
                "direction": "raises risk" if v["impact"] > 0 else "lowers risk",
                "models": v["models"]}
               for k, v in combined.items() if abs(v["impact"]) >= 0.01]
    up = sorted([f for f in factors if f["impact"] > 0], key=lambda f: -f["impact"])[:top_n]
    down = sorted([f for f in factors if f["impact"] < 0], key=lambda f: f["impact"])[:top_n]
    return {"risk_up": up, "risk_down": down, "per_model": per_model}


@lru_cache(maxsize=None)
def _lime_explainer(name):
    from lime.lime_tabular import LimeTabularExplainer
    bg = pd.read_pickle(DEPLOY_DIR / f"{name}_background.pkl")
    binary = set(load_reference(name)["binary_columns"])
    cat_idx = [i for i, c in enumerate(bg.columns) if c in binary]
    return LimeTabularExplainer(bg.values.astype(float), feature_names=list(bg.columns),
                                categorical_features=cat_idx, class_names=["repaid", "default"],
                                mode="classification", discretize_continuous=True, random_state=42)


def lime_explanation(result, a, stmt, num_samples=2000, top_n=5):
    """LIME (Ribeiro et al., 2016) for each voting model, next to SHAP.

    LIME fits a local linear surrogate on perturbed copies of the applicant;
    SHAP attributes the model's own trees exactly. Where the two agree on the
    main factors and their direction, the explanation is robust; where they
    differ, it should be read with care. Features that were imputed (not
    evidenced) are left out, as in the SHAP reasons."""
    out = {}
    exp = result.get("explanation") or explain(result, a, stmt)
    for name in result["voters"]:
        m = result["models"][name]
        model = load_model(name)
        lime_exp = _lime_explainer(name).explain_instance(
            m["X"].values[0].astype(float), lambda Z: model.predict_proba(pd.DataFrame(Z, columns=m["X"].columns)),
            num_features=len(m["X"].columns), num_samples=num_samples)
        names = list(m["X"].columns)
        lime_f = {}
        for idx, w in lime_exp.as_map()[1]:
            c = names[idx]
            if m["tags"].get(c, "imputed") == "imputed":
                continue
            k = _factor(c)
            lime_f[k] = lime_f.get(k, 0.0) + float(w)
        shap_f = {}
        for r in exp["per_model"][name]:
            if r["source"] != "imputed":
                k = _factor(r["feature"])
                shap_f[k] = shap_f.get(k, 0.0) + r["shap"]
        top_lime = sorted(lime_f, key=lambda k: -abs(lime_f[k]))[:top_n]
        top_shap = sorted(shap_f, key=lambda k: -abs(shap_f[k]))[:top_n]
        shared = [k for k in top_lime if k in top_shap]
        same_sign = [k for k in shared if np.sign(lime_f[k]) == np.sign(shap_f[k])]
        out[name] = {
            "label": DATASET_LABELS[name],
            "factors": [{"factor": _factor_label(k, a, stmt), "lime": round(lime_f.get(k, 0.0), 4),
                         "shap": round(shap_f.get(k, 0.0), 4)} for k in dict.fromkeys(top_shap + top_lime)],
            "top_overlap": len(shared) / max(len(top_shap), 1),
            "direction_agreement": len(same_sign) / max(len(shared), 1),
            "local_fit_r2": round(float(lime_exp.score), 3),
        }
    return out


@lru_cache(maxsize=None)
def model_versions():
    """Short content hash of every deployed model + its training date, recorded
    with each decision so it can be traced to the exact model that made it."""
    import hashlib
    out = {}
    for name in VOTING_MODELS:
        h = hashlib.sha256((DEPLOY_DIR / name / "CatBoost.pkl").read_bytes()).hexdigest()[:10]
        ref = score_reference()[name]
        out[name] = {"hash": h, "auc": round(ref["auc"], 4), "trained_at": ref.get("trained_at"),
                     "n_train": ref.get("n_train")}
    return out


def statement_insights(stmt):
    """Plain-language facts read directly from the statement (no model)."""
    if not _has_statement(stmt):
        return []
    months = stmt.get("months_covered", stmt["statement_months"])
    out = [f"{months:.1f} months of transactions analysed"
           + (f" ({pd.Timestamp(stmt['period_start']):%d %b} – {pd.Timestamp(stmt['period_end']):%d %b %Y})"
              if stmt.get("period_start") else "")]
    inc, spend = stmt.get("avg_monthly_income"), stmt.get("avg_monthly_spend")
    if inc:
        basis = stmt.get("income_basis", "income")
        out.append(f"Income counted ₹{inc:,.0f}/month from {basis} "
                   f"({'steady' if stmt.get('income_volatility', 0) < 0.15 else 'irregular'})")
    if (stmt.get("own_transfers_in_share") or 0) >= 0.3:
        out.append(f"{stmt['own_transfers_in_share']:.0%} of money in comes from the applicant's own other accounts")
    for label, amount in (stmt.get("excluded_inflows") or {}).items():
        out.append(f"Not counted as income: {label.lower()} (₹{amount:,.0f}/month)")
    other = (stmt.get("avg_monthly_counted_inflows") or 0) - (inc or 0)
    if inc and other > 0.05 * inc:
        out.append(f"Other irregular inflows of ₹{other:,.0f}/month not relied on")
    if inc and spend is not None:
        out.append(f"Spends {spend / inc:.0%} of income on average")
    if stmt.get("overdraft_frequency") is not None:
        od = stmt["overdraft_frequency"]
        out.append("No overdrafts" if od == 0 else f"Balance negative on {od:.0%} of transactions")
    if stmt.get("min_balance") is not None:
        out.append(f"Lowest balance ₹{stmt['min_balance']:,.0f}")
    if (stmt.get("avg_monthly_investments") or 0) > 0:
        out.append(f"Invests ₹{stmt['avg_monthly_investments']:,.0f}/month (SIP / mutual funds / insurance), "
                   f"not counted as spending")
    if stmt.get("detected_existing_credit"):
        out.append(f"{stmt.get('detected_credit_signal_count', 1)} recurring EMI/card payment(s) detected")
    if stmt.get("betting_ratio", 0) > 0:
        out.append(f"Gambling / betting is {stmt['betting_ratio']:.1%} of spending")
    if stmt.get("bill_months_of"):
        n, of = stmt.get("bill_months", 0), stmt["bill_months_of"]
        out.append(f"Household bills (electricity, phone, rent, insurance) paid in {n} of {of} months" if n else
                   "No household bills paid from this account")
    return out


def africa_advisory(a, stmt, stmt_df):
    """Behavioural model (n=101, synthetic, mixed currencies): shown for
    insight only, never counted in the decision."""
    if not _has_statement(stmt) or stmt_df is None:
        return None
    from feature_adapters import ADAPTERS, load_reference as load_ref_old
    core = {"age": a["age"], "monthly_income": a.get("monthly_income"),
            "requested_amount": a["requested_amount"],
            "requested_duration_months": a["requested_duration_months"],
            "total_savings": a.get("total_savings"), "dependents": a.get("dependents"),
            "employment_days": float(a.get("employment_years") or 0) * 365}
    feats, tags = ADAPTERS["africa"](core, stmt, stmt_df, "Yes")
    cols = load_ref_old("africa")["all_columns"]
    X = pd.DataFrame([[feats.get(c, 0) for c in cols]], columns=cols)
    model = _load(PROJECT_ROOT / "models" / "africa" / "CatBoost.pkl")
    return {"probability": float(model.predict_proba(X)[0][1]),
            "caveat": ("Trained on 101 synthetic records in four currencies; its near-perfect test "
                       "score indicates overfitting. Shown for insight only, never counted.")}


# ---------------------------------------------------------------------------
# Recourse
# ---------------------------------------------------------------------------
REMEDIES = {
    "conduct": "Keep the account out of overdraft for the next 6 months, then re-apply",
    "gambling": "Stop betting / gaming spends for 6 months, then re-apply",
    "stability": "Provide income proof (ITR / GST returns), or re-apply with 6 months of regular income credited here",
    "evidence": "Upload the last 6 months of bank statement",
    "recency": "Upload the statement for the latest 6 months (up to last month)",
    "tamper": "Provide the original e-statement from net banking, or fetch it through an Account Aggregator",
    "holder_match": "Provide KYC proof that the account belongs to the applicant (passbook / bank letter)",
    "income_support": "Provide salary slips or the account where income is credited",
    "consistency": "Clarify the EMI / card payments seen on the statement",
    "integrity": "Provide the statement directly from the bank (net-banking export or Account Aggregator)",
    "surplus": "Lower the EMI (smaller amount or longer term) or reduce regular spending",
    "income_source": "Provide salary slips / Form 16 (salaried) or GST returns / ITR (self-employed), or the "
                     "statement of the account where income is received",
    "tenure": "Provide proof of total work experience / business vintage (offer letters, GST registration)",
    "lenders": "Close or consolidate the loans from other lenders / loan apps before re-applying",
    "salary_trend": "Provide the latest salary slip and an employer letter confirming employment",
    "source_of_funds": "Explain the large recent credit with proof of its source (sale deed, gift letter, maturity advice)",
    "file_origin": "Upload the original e-statement from net banking (with its password) instead of an edited copy",
    "take_home": "Lower the EMI (smaller amount or longer term) so enough is left for the household",
    "loan_size": "Ask for a smaller amount, in line with income",
    "balance_cover": "Keep a higher average balance in the account, or lower the EMI",
    "pass_through": "Explain the money received and sent back to the same people (e.g. family, business partner), "
                    "with supporting documents",
    "profile_match": "Confirm student status (college ID / bonafide certificate) or, if employed full-time, the "
                     "employer's salary slips, and re-assess under the right employment type",
    "student_limits": "Ask for a smaller amount or a shorter term, or add a co-applicant / guarantor with income",
    "own_contribution": f"Make a larger down payment, so that the loan is at most {POLICY['ltv_caution']:.0%} of the "
                        f"price",
}


def documents_to_collect(a, stmt, result):
    """What the officer should ask for before sanction, given this case."""
    st = {c["key"]: c["status"] for c in result["policy"]["checks"]}
    emp = a.get("employment_type") or ""
    docs = ["PAN and Aadhaar (KYC)"]
    sec = result["policy"].get("security")
    if sec:
        docs.append(security.DOCUMENTS[sec["type"]])
    if result["policy"].get("co_applicant_income"):
        docs.append("Co-applicant's KYC, consent, and last 6 months' bank statement or salary slips")
    if result["policy"].get("bureau"):
        docs += bureau.DOCUMENTS
    if emp in SALARIED_TYPES:
        docs.append("Last 3 salary slips" + (" and Form 16" if st.get("income_source") != "pass" else ""))
    elif emp.startswith("Farmer"):
        docs.append("Land records (7/12 extract / RoR), crop-sale receipts (APMC / mandi / dairy) or Kisan Credit "
                    "Card statement")
    elif emp.startswith("Freelancer"):
        docs.append("ITR for the last 2 years, and client contracts or platform payout statements")
    elif emp in SELF_EMPLOYED_TYPES:
        docs.append("ITR with computation for the last 2 years, and GST returns (if registered)")
    elif emp == "Student" and st.get("eligibility") == "pass":
        docs.append("Proof of the student's income (stipend letter, internship / part-time contract)")
    elif emp == "Student":
        docs.append("Co-applicant's KYC and income proof")
    if st.get("income_source") == "caution" or (stmt or {}).get("own_transfers_in_share", 0) >= 0.3:
        docs.append("Statement of the account where the salary / business income is received")
    if st.get("evidence") == "caution":
        docs.append("Bank statement for the full last 6 months")
    if st.get("source_of_funds") == "caution":
        docs.append("Proof of source for the large recent credit")
    if a.get("asset_price") not in (None, ""):
        docs.append("Proforma invoice / quotation for the item, to confirm the price")
    if st.get("pass_through") == "caution":
        docs.append("Explanation of the money received and sent back to the same people")
    if st.get("file_origin") == "caution" or st.get("integrity") == "caution" or st.get("tamper") == "caution":
        docs.append("Original e-statement from net banking (or an Account Aggregator fetch)")
    if st.get("consistency") == "caution":
        docs.append("Loan statements / sanction letters for the EMIs seen on the statement")
    if st.get("salary_trend") == "caution":
        docs.append("Employer letter confirming current employment")
    if st.get("profile_match") == "caution":
        docs.append("College ID / bonafide certificate, or the employer's last 3 salary slips if working full-time")
    if "renovation" in str(a.get("purpose", "")).lower() or "repair" in str(a.get("purpose", "")).lower():
        docs.append("Proof of ownership of the property being repaired" + ("" if a.get("housing") == "Own home" else
                    " (or the owner's written consent, e.g. a parent's)") + ", and the contractor's estimate")
    if a.get("housing") == "Own home":
        docs.append("Proof of address (property tax receipt or utility bill)")
    return docs


def recourse(a, stmt, stmt_df, verdict, policy, ml, elig=None):
    """What would change the outcome: affordability first (exact, computed),
    then conduct remedies, then ML-driven loan-term changes."""
    if verdict == "approve":
        return []
    out = []
    status = {c["key"]: c["status"] for c in policy["checks"]}
    if status.get("eligibility") == "fail" and a.get("employment_type") not in POLICY["not_eligible"]:
        max_age = POLICY["max_age_at_maturity"].get(a.get("employment_type"), 60)
        months = int((max_age - float(a["age"])) * 12) // 6 * 6
        if months >= POLICY["min_term_months"] and a["age"] >= POLICY["min_age"]:
            out.append({"change": f"A term of up to {months} months keeps the loan within the age limit "
                                  f"(EMI ₹{emi(a['requested_amount'], months):,.0f})", "kind": "policy"})

    if elig and policy["income_used"] and status.keys() & AFFORDABILITY_KEYS and \
            any(status.get(k) in ("fail", "caution") for k in AFFORDABILITY_KEYS):
        d = elig.get("term") or a["requested_duration_months"]
        cap = elig["eligible_amount"]
        if cap >= POLICY["min_amount"]:
            out.append({"change": f"Borrow up to ₹{cap:,.0f} over {d} months (EMI ₹{emi(cap, d):,.0f}; limited by "
                                  f"the {_lower_first(elig['binding'])})", "kind": "policy"})
        if elig.get("fits_at_term"):
            t = elig["fits_at_term"]
            out.append({"change": f"Same amount over {t} months (EMI ₹{emi(a['requested_amount'], t):,.0f})",
                        "kind": "policy"})
    for key, text in REMEDIES.items():
        if status.get(key) in ("fail", "caution"):
            out.append({"change": text, "kind": "conduct"})

    if verdict == "refer" and ml["consensus_percentile"] is not None \
            and ml["consensus_percentile"] >= DECLINE_FROM:
        for s in ml_recourse(a, stmt, ml):
            out.append({"change": f"{s['change']} (risk percentile {s['new_percentile']:.0f})",
                        "kind": "model"})
    return out[:5]


def ml_recourse(a, stmt, base):
    """Loan-term changes re-scored through the full ML consensus."""
    variants = []
    for pct in (0.2, 0.35, 0.5):
        variants.append((f"Borrow {pct:.0%} less (₹{a['requested_amount'] * (1 - pct):,.0f})",
                         {"requested_amount": a["requested_amount"] * (1 - pct)}))
    d = a["requested_duration_months"]
    for months in sorted({max(6, int(d * 0.75)), max(6, int(d * 0.5))}, reverse=True):
        if months < d:
            variants.append((f"Shorter term: {months} months", {"requested_duration_months": months}))
    if len(variants) >= 2:
        variants.append((f"Borrow 35% less over {max(6, int(d * 0.75))} months",
                         {"requested_amount": a["requested_amount"] * 0.65,
                          "requested_duration_months": max(6, int(d * 0.75))}))

    out = []
    for label, change in variants:
        r = score({**a, **change, "proposed_emi": None}, stmt)    # a changed loan has a new EMI
        if r["consensus_percentile"] is None:
            continue
        drop = base["consensus_percentile"] - r["consensus_percentile"]
        if drop >= 3 or r["model_band"] != base["model_band"]:
            out.append({"change": label, "new_percentile": r["consensus_percentile"],
                        "drop": round(drop, 1), "new_band": r["model_band"]})
    out.sort(key=lambda s: -s["drop"])
    return out[:3]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def _assess_one(applicant, stmt=None, stmt_df=None):
    stmt = prepare_statement(stmt, applicant, stmt_df)
    result = score(applicant, stmt)
    result["policy"] = policy_checks(applicant, stmt, stmt_df)
    result["decision_confidence"] = decision_confidence(result["policy"], result, stmt)
    result["fairscore"] = fairscore(applicant, stmt, result["policy"])
    result["verdict"], result["verdict_reasons"], extra = final_decision(
        applicant, stmt, result["policy"], result, result["decision_confidence"], result["fairscore"])
    ai = extra.pop("ai")
    result.update(extra)
    result["ai"] = {k: v for k, v in ai.items() if k != "X"} if ai else None
    # the engine's 24 scale-free inputs (ratios and 0-1 scores, no amounts or names): drift monitoring and the
    # learning loop (src/learning_loop.py) re-score recorded outcomes from these
    result["ai_inputs"] = [round(float(v), 6) for v in ai["X"][0]] if ai else None
    if ai and result["decided_by"] == "AI" and result["verdict"] != "approve":
        try:
            result["ai"]["recourse"] = ai_recourse(applicant, stmt, stmt_df, ai)
        except Exception as exc:      # advisory: never block the decision
            result["ai"]["recourse"] = {"error": str(exc)}
    result["eligibility"] = eligibility(applicant, stmt, result["policy"])
    result["offer"] = counter_offer(applicant, stmt, stmt_df, result["verdict"], result["policy"],
                                    result["eligibility"])
    result["starter"] = None
    if not result["offer"]:
        try:                          # FAIRScore 3.0 starter-loan ladder (advisory to the requested loan's verdict)
            result["starter"] = starter_ladder.offer(applicant, stmt, stmt_df, result, quick_decision)
        except Exception as exc:      # never block the decision
            result["starter_error"] = str(exc)
    result["risk_level"] = risk_level(result["consensus_percentile"])
    result["policy_version"] = POLICY_VERSION
    result["model_versions"] = model_versions()
    cf = result["models"].get("cashflow")
    result["model_inputs"] = ({k: round(float(v), 4) for k, v in cf["X"].iloc[0].items()
                               if cf["tags"].get(k) != "imputed"} if cf else None)
    result["interest_rate"] = POLICY["annual_interest_rate"]
    info = (stmt or {}).get("read_info") or {}
    result["statement_read"] = (
        f"statement {stmt.get('period_start')} to {stmt.get('period_end')}, {stmt.get('transactions')} transactions"
        + (f", {info['reconciled']:.1%} of balances reconciled ({info.get('method')} layout)"
           if info.get("reconciled") is not None else "") if _has_statement(stmt) else "no statement")
    result["explanation"] = explain(result, applicant, stmt) if result["voters"] else None
    result["recourse"] = recourse(applicant, stmt, stmt_df, result["verdict"], result["policy"], result,
                                  result["eligibility"])
    result["documents"] = documents_to_collect(applicant, stmt, result)
    result["banking"] = (stmt or {}).get("banking")
    result["statement_insights"] = statement_insights(stmt)
    result["income_basis"] = (stmt or {}).get("income_basis")
    result["age_flip"] = age_flip_test(applicant, stmt, stmt_df)
    try:   # advisory only: a failure here must never block the verdict
        result["africa_advisory"] = africa_advisory(applicant, stmt, stmt_df)
    except Exception as exc:
        result["africa_advisory"] = {"probability": None, "caveat": f"Unavailable: {exc}"}
    return result


_VERDICT_RANK = {"decline": 0, "refer": 1, "approve": 2}


def _cf_is_worse(main, cf):
    """True when the assessment WITHOUT the unexplained credit is worse for the applicant (lower verdict, or the
    same verdict at a higher AI default risk)."""
    rm, rc = _VERDICT_RANK[main["verdict"]], _VERDICT_RANK[cf["verdict"]]
    if rc != rm:
        return rc < rm
    return ((cf.get("ai") or {}).get("pd") or 0) > ((main.get("ai") or {}).get("pd") or 0) + 1e-9


def _unexplained(stmt):
    cf = (stmt or {}).get("_without_unexplained_credits")
    if not cf or not _has_statement(cf):
        return None, []
    big = [c for c in ((stmt.get("banking") or {}).get("large_credits") or []) if c["last_30_days"]]
    return cf, big


def _cf_reason(big):
    c = max(big, key=lambda c: c["amount"])
    return (f"Assessed without the unexplained Rs {c['amount']:,.0f} credit of {pd.Timestamp(c['date']):%d %b %Y} that is "
            f"still in the account: money that arrived just before the application cannot improve the decision. "
            f"Ask for its source")


def quick_decision(a, stmt, stmt_df, exclude_models=(), ignore_age_rule=False):
    """Verdict only (no explanations). Anti-gaming: with a recent unexplained large credit, the worse of the
    assessments with and without it stands (see statement_parser.without_recent_large_credits)."""
    q = _quick_one(a, stmt, stmt_df, exclude_models, ignore_age_rule)
    cf, big = _unexplained(stmt)
    if cf is not None:
        q2 = _quick_one(a, cf, cf.get("_df"), exclude_models, ignore_age_rule)
        if _cf_is_worse(q, q2):
            q2["reasons"] = [_cf_reason(big)] + list(q2.get("reasons") or [])
            q2["unexplained_credit_removed"] = True
            return q2
    return q


def assess(applicant, stmt=None, stmt_df=None):
    """Public entry point. Anti-gaming: when the statement shows a large unexplained credit in the last 30 days that
    is still in the account (window dressing), the WHOLE assessment -- rules, FAIRScore, AI, eligible amount -- is
    also run as if that money had never arrived, and the worse of the two stands. The officer sees the credit, the
    source-of-funds caution and why the figures exclude it."""
    result = _assess_one(applicant, stmt, stmt_df)
    cf, big = _unexplained(stmt)
    if cf is None:
        return result
    r2 = _assess_one(applicant, cf, cf.get("_df"))
    if not _cf_is_worse(result, r2):
        return result
    keep = [c for c in result["policy"]["checks"] if c["key"] == "source_of_funds"]
    r2["policy"]["checks"] = [c for c in r2["policy"]["checks"] if c["key"] != "source_of_funds"] + keep
    r2["verdict_reasons"] = [_cf_reason(big)] + list(r2.get("verdict_reasons") or [])
    r2["unexplained_credit_removed"] = {"amount": max(c["amount"] for c in big),
                                        "with_credit": {"verdict": result["verdict"],
                                                        "pd": (result.get("ai") or {}).get("pd")}}
    r2["banking"] = result["banking"]                 # the officer sees the statement as it is, credit included
    r2["statement_read"] = result["statement_read"]
    r2["documents"] = list(dict.fromkeys(list(r2.get("documents") or []) + list(result.get("documents") or [])))
    return r2


if __name__ == "__main__":
    from statement_parser import parse_statement

    stmt_df, stmt, _ = parse_statement(str(PROJECT_ROOT / "data" / "raw" / "sample_statement.csv"))
    applicant = {
        "age": 29, "monthly_income": 42000, "requested_amount": 200000,
        "requested_duration_months": 24, "total_savings": 25000, "dependents": 1,
        "employment_years": 2.2, "employment_type": "Salaried (private sector)",
        "education": "Graduate", "housing": "Rented home", "owns_car": False,
        "purpose": "Electronics (TV, phone, laptop)",
        "active_credit_lines": 1, "property_loans": 0,
    }
    r = assess(applicant, stmt, stmt_df)
    print(f"Verdict: {r['verdict'].upper()}  {r['verdict_reasons']}")
    print(f"ML consensus risk percentile: {r['consensus_percentile']}")
    for c in r["policy"]["checks"]:
        print(f"  [{c['status']:7s}] {c['name']}: {c['detail']}")
    print(f"Confidence: {r['confidence']['label']} -- {r['confidence']['detail']}")
    for n, m in r["models"].items():
        line = f"  {m['label']:22s} applicability={m['applicability']:.2f}"
        if m["qualifies"]:
            line += (f"  prob={m['probability']:.3f}  percentile={m['percentile']:.0f}  "
                     f"coverage={m['coverage']:.2f}  weight={m['weight']:.3f}  "
                     f"out_of_range={[o['feature'] for o in m['out_of_range']]}")
        else:
            line += "  (did not vote)"
        print(line)
    print("Raises risk:", [f["label"] for f in r["explanation"]["risk_up"]])
    print("Lowers risk:", [f["label"] for f in r["explanation"]["risk_down"]])
    print("Recourse:", r["recourse"])
    print("Statement:", r["statement_insights"])
    print("Africa advisory:", r["africa_advisory"] and round(r["africa_advisory"]["probability"], 4))
