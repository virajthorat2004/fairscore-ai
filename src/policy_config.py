"""
Loads the lending policy from config/policy.toml (lender settings, editable
without touching code) and validates it, so a typo in the file stops the app
with a clear message instead of silently changing decisions.

Every assessment records POLICY_VERSION (declared version + a hash of the
file), so any past decision can be reproduced under the exact rules it used.
"""
import hashlib
import tomllib

from config import PROJECT_ROOT

POLICY_PATH = PROJECT_ROOT / "config" / "policy.toml"


class PolicyError(ValueError):
    pass


def _flatten(cfg):
    """The flat dict the scoring code uses (names kept from earlier versions)."""
    L, A, E, I, C = cfg["loan"], cfg["affordability"], cfg["evidence"], cfg["income"], cfg["conduct"]
    G, R, V = cfg["eligibility"], cfg["risk_models"], cfg["review"]
    return {
        "annual_interest_rate": L["annual_interest_rate"],
        "min_amount": L["min_amount"], "max_amount": L["max_amount"],
        "min_term_months": L["min_term_months"], "max_term_months": L["max_term_months"],
        "foir_pass": A["foir_pass"], "foir_fail": A["foir_fail"], "surplus_min_share": A["surplus_min_share"],
        "min_months_pass": E["min_months_pass"], "min_months_caution": E["min_months_caution"],
        "min_reconciled": E["min_reconciled"], "max_breaks_share": E["max_breaks_share"],
        "max_statement_age_days": E.get("max_statement_age_days", 45),
        "income_support_min": I["support_min"],
        "volatility_pass": I["volatility_pass"], "volatility_fail": I["volatility_fail"],
        "self_employed_min_payers": I["self_employed_min_payers"],
        "overdraft_pass": C["overdraft_pass"], "overdraft_fail": C["overdraft_fail"],
        "max_penalty_charges": C["max_penalty_charges"],
        "betting_pass": C["betting_pass"], "betting_fail": C["betting_fail"],
        "lenders_caution": C["lenders_caution"], "lenders_fail": C["lenders_fail"],
        "salary_drop_caution": C["salary_drop_caution"],
        "pass_through_caution": C.get("pass_through_caution", 0.30),
        "ltv_caution": cfg.get("own_contribution", {}).get("ltv_caution", 0.85),
        "ltv_fail": cfg.get("own_contribution", {}).get("ltv_fail", 1.00),
        "max_income_multiple": cfg["affordability_limits"]["max_income_multiple"],
        "min_take_home": cfg["affordability_limits"]["min_take_home"],
        "take_home_per_dependent": cfg["affordability_limits"]["take_home_per_dependent"],
        "abb_cover_min": cfg["affordability_limits"]["abb_cover_min"],
        "abb_cover_applies_to": list(cfg["affordability_limits"]["abb_cover_applies_to"]),
        "score_gate": cfg["scorecard"]["gate_min_score"], "score_bands": dict(cfg["scorecard"]["bands"]),
        "score_points": dict(cfg["scorecard"]["max_points"]), "score_curve": float(cfg["scorecard"]["curve"]),
        "min_age": G["min_age"], "max_age_at_maturity": dict(G["max_age_at_maturity"]),
        "min_work_years": dict(G.get("min_work_years", {})),
        "not_eligible": list(G.get("not_eligible", [])), "needs_co_applicant": list(G.get("needs_co_applicant", [])),
        "own_income_min": float(G.get("own_income_min", 15000)),
        "farm_volatility_caution": float(G.get("farm_volatility_caution", 1.0)),
        "student_max_term_months": int(G.get("student_max_term_months", 36)),
        "student_income_multiple": float(G.get("student_income_multiple", 8)),
        "student_max_amount": float(G.get("student_max_amount", 500000)),
        "student_salary_query": float(G.get("student_salary_query", 50000)),
        "refer_from_percentile": float(R["refer_from_percentile"]), "level_cuts": dict(R["level_cuts"]),
        "min_confidence_to_approve": V["min_confidence_to_approve"],
        "ai": _ai(cfg.get("ai", {})),
        "security": _security(cfg.get("security", {})),
        "co_applicant_counts": bool(cfg.get("co_applicant", {}).get("counts_for_affordability", True)),
        "starter": _starter(cfg.get("starter", {})),
        "learning": _learning(cfg.get("learning", {})),
        "bureau": _bureau(cfg.get("bureau", {})),
    }


def _security(c):
    return {"gold_ltv": [(float(u), float(v)) for u, v in c.get("gold_ltv", [[250000, 0.85], [500000, 0.80], [1e12, 0.75]])],
            "gold_no_appraisal_upto": float(c.get("gold_no_appraisal_upto", 250000)),
            "deposit_ltv": float(c.get("deposit_ltv", 0.90)), "property_ltv": float(c.get("property_ltv", 0.60)),
            "lgd": {k: float(v) for k, v in c.get("lgd", {"gold": 0.10, "deposit": 0.03, "property": 0.25}).items()},
            "min_coverage": float(c.get("min_coverage", 0.5)), "secured_max_pd": float(c.get("secured_max_pd", 0.15))}


def _starter(c):
    """FAIRScore 3.0 starter-loan ladder ([starter] in policy.toml)."""
    return {"enabled": bool(c.get("enabled", True)), "max_amount": float(c.get("max_amount", 50000)),
            "income_multiple": float(c.get("income_multiple", 1.0)),
            "terms": [int(t) for t in c.get("terms", [12, 6])],
            "max_pd": float(c.get("max_pd", 0.12)),
            "explore_share": float(c.get("explore_share", 0.05)),
            "explore_max_pd": float(c.get("explore_max_pd", 0.25)),
            "rungs": [(int(m), float(x)) for m, x in c.get("rungs", [[6, 2.0], [12, 4.0]])],
            "advisory_segments": bool(c.get("advisory_segments", True))}


def _learning(c):
    """FAIRScore 3.0 learning loop ([learning] in policy.toml)."""
    return {"use_adaptation": bool(c.get("use_adaptation", False)),
            "shift_aware_conformal": bool(c.get("shift_aware_conformal", True)),
            "min_live_applications": int(c.get("min_live_applications", 30)),
            "prior_sd": tuple(float(x) for x in c.get("prior_sd", [0.5, 0.25])),
            "challenger_min_outcomes": int(c.get("challenger_min_outcomes", 200)),
            "challenger_min_defaults": int(c.get("challenger_min_defaults", 30)),
            "ai_decline_needs_local_validation": bool(c.get("ai_decline_needs_local_validation", True))}


def _bureau(c):
    """Credit-bureau layer ([bureau] in policy.toml): used when the applicant HAS some bureau record."""
    return {"min_score": int(c.get("min_score", 650)), "caution_score": int(c.get("caution_score", 700)),
            "max_enquiries_6m": int(c.get("max_enquiries_6m", 6)),
            "dpd30_caution_12m": bool(c.get("dpd30_caution_12m", True)),
            "dpd90_fail_24m": bool(c.get("dpd90_fail_24m", True))}


def _ai(c):
    """FAIRScore 2.0 AI decision engine settings ([ai] in policy.toml)."""
    return {"mode": c.get("mode", "ai_first"), "alpha": float(c.get("alpha", 0.10)),
            "target_default_rate": float(c.get("target_default_rate", 0.03)),
            "approve_max_pd": c.get("approve_max_pd"),          # None: use the value learned at training
            "decline_min_pd": float(c.get("decline_min_pd", 0.30)),
            "min_typicality": float(c.get("min_typicality", 0.01)),
            "force_refer": list(c.get("force_refer", [])),
            "ai_decides_for": list(c.get("ai_decides_for", [])),
            "lgd": float(c.get("lgd", 0.65)), "cost_of_funds": float(c.get("cost_of_funds", 0.075)),
            "operating_cost": float(c.get("operating_cost", 0.025)), "margin": float(c.get("margin", 0.02)),
            "min_rate": float(c.get("min_rate", 0.105)), "max_rate": float(c.get("max_rate", 0.26))}


def _validate(p):
    problems = []

    def share(*keys):
        for k in keys:
            if not 0 <= p[k] <= 1:
                problems.append(f"{k} must be between 0 and 1 (got {p[k]})")

    share("foir_pass", "foir_fail", "surplus_min_share", "min_reconciled", "max_breaks_share",
          "income_support_min", "overdraft_pass", "overdraft_fail", "betting_pass", "betting_fail",
          "min_confidence_to_approve")
    for lo, hi in (("foir_pass", "foir_fail"), ("overdraft_pass", "overdraft_fail"),
                   ("betting_pass", "betting_fail"), ("volatility_pass", "volatility_fail"),
                   ("min_months_caution", "min_months_pass"), ("min_amount", "max_amount"),
                   ("min_term_months", "max_term_months")):
        if p[lo] > p[hi]:
            problems.append(f"{lo} ({p[lo]}) must not be above {hi} ({p[hi]})")
    if not 0 < p["annual_interest_rate"] < 1:
        problems.append("annual_interest_rate must be a fraction, e.g. 0.14 for 14%")
    if p["lenders_caution"] > p["lenders_fail"]:
        problems.append("lenders_caution must not be above lenders_fail")
    if any(v < 0 for v in p["score_points"].values()) or not sum(p["score_points"].values()):
        problems.append("scorecard.max_points must be positive")
    if not 0.5 <= p["score_curve"] <= 3:
        problems.append("scorecard.curve should be between 0.5 and 3")
    bands = list(p["score_bands"].values())
    if bands != sorted(bands, reverse=True) or not 300 <= min(bands) <= max(bands) <= 900:
        problems.append("scorecard.bands must fall from Excellent to Weak, within 300-900")
    cuts = list(p["level_cuts"].values())
    if cuts != sorted(cuts):
        problems.append("risk_models.level_cuts must increase (Low < Moderate < Elevated)")
    if problems:
        raise PolicyError(f"{POLICY_PATH.name}: " + "; ".join(problems))
    return p


def load_policy(path=POLICY_PATH):
    raw = path.read_bytes()
    try:
        cfg = tomllib.loads(raw.decode("utf-8"))
        policy = _validate(_flatten(cfg))
    except KeyError as exc:
        raise PolicyError(f"{path.name}: missing setting {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise PolicyError(f"{path.name}: {exc}") from exc
    version = f"{cfg.get('version', 'unversioned')}+{hashlib.sha256(raw).hexdigest()[:8]}"
    return policy, version


POLICY, POLICY_VERSION = load_policy()
