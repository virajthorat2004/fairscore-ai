import pickle

from config import MODELS_DIR


def load_reference(dataset_name):
    # Absolute path via config, so this works from any working directory.
    with open(MODELS_DIR / f"{dataset_name}_reference.pkl", "rb") as f:
        return pickle.load(f)


def _fill_remaining(features, tags, ref):
    """
    Generic, safe fallback for every column an adapter didn't explicitly set:
    binary/dummy columns fall back to the training set's most common value
    (usually 0 -- the baseline category), numeric columns fall back to the
    training median. Every one of these is tagged 'imputed'.
    """
    for col in ref["all_columns"]:
        if col in features:
            continue
        if col in ref["binary_columns"]:
            features[col] = ref["mode"].get(col, 0)
        else:
            features[col] = ref["median"].get(col, 0)
        tags[col] = "imputed"
    return features, tags


# ---------------------------------------------------------------------------
# German Credit
# ---------------------------------------------------------------------------
def adapt_to_german_credit(core, stmt_features, stmt_df, gate_answer):
    ref = load_reference("german_credit")
    features, tags = {}, {}

    features["age"] = core["age"]; tags["age"] = "real"
    features["duration"] = core["requested_duration_months"]; tags["duration"] = "real"
    features["credit_amount"] = core["requested_amount"]; tags["credit_amount"] = "real"

    if core.get("dependents") is not None:
        features["dependents"] = core["dependents"]; tags["dependents"] = "real"

    # Employment duration bucket -- unit-independent (days), fully derivable
    if core.get("employment_days") is not None:
        days = core["employment_days"]
        if days == 0:
            bucket = "employment_since_unemployed"
        elif days < 365:
            bucket = "employment_since_lt1_year"
        elif days < 365 * 4:
            bucket = None  # baseline "1-4 years"
        elif days < 365 * 7:
            bucket = "employment_since_4-7_years"
        else:
            bucket = "employment_since_gt7_years"
        for c in ref["all_columns"]:
            if c.startswith("employment_since_"):
                features[c] = 1 if c == bucket else 0
                tags[c] = "real"
                
    # existing_credits: only trustworthy if we actually have a statement to detect from
    if stmt_features.get("statement_months", 0) >= 3:
        features["existing_credits"] = max(1, stmt_features.get("detected_credit_signal_count", 0))
        tags["existing_credits"] = "real"

    # Savings bucket -- derivable from declared total savings ONLY as a rough
    # heuristic. German Credit's savings categories are denominated in
    # Deutsche Marks (currency discontinued 2002, dataset collected in the
    # 1990s) -- there is no economically valid way to convert a modern Rupee
    # amount into a DM bucket. Any threshold here is a heuristic approximation,
    # not a genuine derivation, and is tagged accordingly rather than as "real".
    savings = core.get("total_savings")
    if savings is not None:
        if savings < 3000:
            col = "savings_account_lt100_DM"
        elif savings < 15000:
            col = None  # falls into the dropped baseline bucket ("100-500 DM")
        elif savings < 30000:
            col = "savings_account_500-1000_DM"
        else:
            col = "savings_account_gt1000_DM"
        for c in ref["all_columns"]:
            if c.startswith("savings_account_"):
                features[c] = 1 if c == col else 0
                tags[c] = "real_approx"

    # Checking account bucket -- the SIGN of the balance (overdrawn or not) is
    # a scale-independent, genuinely meaningful signal regardless of currency.
    # The magnitude threshold below is still a heuristic, so also tagged
    # "real_approx" rather than "real".
    if stmt_features.get("avg_balance") is not None:
        bal = stmt_features["avg_balance"]
        col = "checking_account_lt0_DM" if bal < 0 else ("checking_account_gt200_DM" if bal > 20000 else None)
        for c in ref["all_columns"]:
            if c.startswith("checking_account_"):
                features[c] = 1 if c == col else 0
                tags[c] = "real_approx"

    return _fill_remaining(features, tags, ref)


# ---------------------------------------------------------------------------
# Give Me Some Credit
# ---------------------------------------------------------------------------
def adapt_to_gmsc(core, stmt_features, stmt_df, gate_answer):
    ref = load_reference("gmsc")
    features, tags = {}, {}

    features["age"] = core["age"]; tags["age"] = "real"

    income = stmt_features.get("avg_monthly_income") or core.get("monthly_income")
    if income is not None:
        features["MonthlyIncome"] = income
        tags["MonthlyIncome"] = "real"
        features["monthly_income_was_missing"] = 0
        tags["monthly_income_was_missing"] = "real"
    else:
        features["monthly_income_was_missing"] = 1
        tags["monthly_income_was_missing"] = "real"

    if core.get("dependents") is not None:
        features["NumberOfDependents"] = core["dependents"]
        tags["NumberOfDependents"] = "real"
        features["dependents_was_missing"] = 0
        tags["dependents_was_missing"] = "real"
    else:
        features["dependents_was_missing"] = 1
        tags["dependents_was_missing"] = "real"

    # DebtRatio -- a genuine, defensible proxy computable from the statement alone
    spend = stmt_features.get("avg_monthly_spend")
    if spend is not None and income:
        features["DebtRatio"] = round(spend / income, 4)
        tags["DebtRatio"] = "real"

    # RevolvingUtilization, past-due counts, open credit lines, real-estate
    # loans -- genuinely require bureau/credit-line data a statement can't
    # reliably reconstruct. Left to the generic imputed fallback.

    return _fill_remaining(features, tags, ref)


# ---------------------------------------------------------------------------
# Home Credit Default Risk
# ---------------------------------------------------------------------------
def adapt_to_home_credit(core, stmt_features, stmt_df, gate_answer):
    ref = load_reference("homecredit")
    features, tags = {}, {}

    features["age"] = core["age"]; tags["age"] = "real"
    features["AMT_CREDIT"] = core["requested_amount"]; tags["AMT_CREDIT"] = "real"
    features["AMT_ANNUITY"] = core["requested_amount"] / max(core["requested_duration_months"], 1)
    tags["AMT_ANNUITY"] = "real"
    # AMT_GOODS_PRICE approximated as the requested amount -- a reasonable
    # assumption for most personal-loan use cases, not a fabricated figure,
    # but noted as an approximation rather than a directly-stated value.
    features["AMT_GOODS_PRICE"] = core["requested_amount"]; tags["AMT_GOODS_PRICE"] = "real_approx"

    income = stmt_features.get("avg_monthly_income") or core.get("monthly_income")
    if income is not None:
        features["AMT_INCOME_TOTAL"] = income * 12
        tags["AMT_INCOME_TOTAL"] = "real"

    if core.get("employment_days") is not None:
        features["DAYS_EMPLOYED"] = -core["employment_days"]
        tags["DAYS_EMPLOYED"] = "real"

    if core.get("dependents") is not None:
        features["CNT_CHILDREN"] = core["dependents"]
        features["CNT_FAM_MEMBERS"] = core["dependents"] + 1
        tags["CNT_CHILDREN"] = "real"
        tags["CNT_FAM_MEMBERS"] = "real"

    # EXT_SOURCE_2 / EXT_SOURCE_3 -- bureau-derived scores, NEVER derivable
    # from a bank statement. Deliberately left imputed. Since these are Home
    # Credit's top two SHAP-important features, this is exactly what should
    # cause Home Credit to fail the applicability gate for most statement-only
    # applicants -- an honest gap, not a bug.

    return _fill_remaining(features, tags, ref)


# ---------------------------------------------------------------------------
# Africa Synth (Financial Inclusion)
# ---------------------------------------------------------------------------
def adapt_to_africa_synth(core, stmt_features, stmt_df, gate_answer):
    ref = load_reference("africa")
    features, tags = {}, {}

    direct_map = {
        "avg_monthly_income": "avg_monthly_income",
        "income_volatility": "income_volatility",
        "avg_monthly_spend": "avg_monthly_spend",
        "spending_volatility": "spending_volatility",
        "transaction_frequency": "transaction_frequency",
        "bill_payment_consistency": "bill_payment_consistency",
        "overdraft_frequency": "overdraft_frequency",
        "avg_balance": "avg_balance",
        "min_balance": "min_balance",
        "max_balance": "max_balance",
        "betting_ratio": "betting_ratio",
        "essential_spend_ratio": "essential_spend_ratio",
    }
    for target_col, src_key in direct_map.items():
        val = stmt_features.get(src_key)
        if val is not None:
            features[target_col] = val
            tags[target_col] = "real"

    if "essential_spend_ratio" in features:
        features["discretionary_spend_ratio"] = round(1 - features["essential_spend_ratio"], 4)
        tags["discretionary_spend_ratio"] = "real"

    # Transaction-level features computed directly from the applicant's own
    # uploaded statement -- the same category of feature engineering used to
    # build this dataset's enrichment features originally.
    if stmt_df is not None and len(stmt_df):
        features["txn_count"] = len(stmt_df)
        features["avg_txn_amount"] = stmt_df["signed_amount"].abs().mean()
        features["txn_amount_std"] = stmt_df["signed_amount"].abs().std() or 0.0
        tags["txn_count"] = "real"
        tags["avg_txn_amount"] = "real"
        tags["txn_amount_std"] = "real"
        if "balance" in stmt_df.columns and stmt_df["balance"].notna().any():
            features["min_balance_observed"] = stmt_df["balance"].min()
            features["overdraft_event_count"] = int((stmt_df["balance"] < 0).sum())
            tags["min_balance_observed"] = "real"
            tags["overdraft_event_count"] = "real"

    # cashflow_stability_score and risk_behavior_score are this dataset's own
    # proprietary composite scores -- not something we should approximate
    # with an invented formula and present as real. Left honestly imputed.
    # channel_diversity, category_diversity, region/currency/archetype
    # dummies -- not derivable from a generic bank statement. Imputed.

    return _fill_remaining(features, tags, ref)


ADAPTERS = {
    "german_credit": adapt_to_german_credit,
    "gmsc": adapt_to_gmsc,
    "homecredit": adapt_to_home_credit,
    "africa": adapt_to_africa_synth,
}