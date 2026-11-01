"""
Deployable feature sets: the subset of each dataset's features that can be
GENUINELY obtained for a live Indian applicant from the intake form + a bank
statement, with no protected attributes, no credit-bureau / credit-history data
and no currency-dependent amounts.

AGE IS NOT A MODEL INPUT. Age is the protected attribute this project audits,
so the live models are trained age-blind ("fairness through unawareness") and
age is used only for the legal/product eligibility rule (loan must end before
the maximum age) and for measuring outcomes by age group. Unawareness alone
does not remove proxies (e.g. years in work), which is why outcomes are still
monitored by age group: see src/age_ablation.py and the History tab.

Every exclusion has a stated reason, so the "cost of deployability" table in
the report can explain exactly what was given up and why.
"""

DEPLOYABLE_SPEC = {
    "german_credit": {
        "exact": ["duration", "existing_credits"],
        "prefixes": [
            "employment_since_",          # form: years employed
            "purpose_",                   # form: loan purpose
            "housing_",                   # form: rent / own / family
            "checking_account_",          # statement: has an account, balance sign
            "savings_account_",           # form: declared savings (DM buckets, approx)
            "other_installment_plans_",   # form: declared active loans
        ],
        "excluded": {
            "age": "Protected attribute audited by this project: used only for eligibility and "
                   "fairness monitoring, never as a model input (see src/age_ablation.py).",
            "credit_amount": "Denominated in Deutsche Marks (1973-75 data). A rupee amount "
                             "cannot be validly mapped onto it.",
            "personal_status_sex_*": "Encodes sex: a protected attribute.",
            "foreign_worker_*": "Encodes nationality status: a protected attribute.",
            "installment_rate": "Ordinal 1-4 code whose direction is disputed in the dataset's "
                                "own documentation (Groemping, 2019), so it cannot be derived safely.",
            "credit_history_*": "Credit-bureau history; not observable from 6 months of statement.",
            "dependents": "Coded 1-2 with disputed meaning (Groemping, 2019).",
            "property_*, job_*, other_debtors_*, telephone_*, residence_since":
                "Not collected at intake; kept out to keep the form short.",
        },
    },
    "gmsc": {
        "exact": [
            "DebtRatio", "NumberOfDependents", "NumberRealEstateLoansOrLines",
            "monthly_income_was_missing", "dependents_was_missing",
        ],
        "prefixes": [],
        "excluded": {
            "age": "Protected attribute audited by this project: used only for eligibility and "
                   "fairness monitoring, never as a model input (see src/age_ablation.py).",
            "MonthlyIncome": "US-dollar amount; a rupee income cannot be fed in. DebtRatio "
                             "(spend / income) carries the income information scale-free.",
            "RevolvingUtilizationOfUnsecuredLines": "Needs credit-card limits: bureau data.",
            "NumberOfOpenCreditLinesAndLoans": "Counts all US credit lines incl. cards (median 8). "
                                               "Indian borrowers hold far fewer cards, so the same "
                                               "number means something different: it put every "
                                               "Indian test applicant in GMSC's riskiest 2%.",
            "NumberOfTime30-59DaysPastDueNotWorse, NumberOfTime60-89DaysPastDueNotWorse, "
            "NumberOfTimes90DaysLate": "Delinquency history: bureau data.",
        },
    },
    "homecredit": {
        "exact": ["DAYS_EMPLOYED", "days_employed_anomaly", "CNT_CHILDREN",
                  "credit_to_income"],
        "prefixes": [
            "NAME_EDUCATION_TYPE_", "NAME_HOUSING_TYPE_", "NAME_INCOME_TYPE_",
            "FLAG_OWN_CAR_", "FLAG_OWN_REALTY_",
        ],
        "excluded": {
            "age": "Protected attribute audited by this project: used only for eligibility and "
                   "fairness monitoring, never as a model input (see src/age_ablation.py).",
            "AMT_CREDIT, AMT_INCOME_TOTAL, AMT_GOODS_PRICE": "Unknown currency; replaced by the "
                                                             "scale-free credit_to_income ratio.",
            "AMT_ANNUITY": "Payment period (monthly vs yearly) is undocumented, so a ratio built "
                           "on it cannot be reproduced reliably.",
            "EXT_SOURCE_*": "External bureau scores.",
            "CODE_GENDER_*": "Protected attribute.",
            "NAME_FAMILY_STATUS_*, CNT_FAM_MEMBERS": "Marital status: sensitive; not collected.",
            "DAYS_ID_PUBLISH, DAYS_REGISTRATION, DAYS_LAST_PHONE_CHANGE, FLAG_DOCUMENT_*, "
            "REGION_*, ORGANIZATION_TYPE_*, OCCUPATION_TYPE_*, *_SOCIAL_CIRCLE, "
            "AMT_REQ_CREDIT_BUREAU_*, application-process metadata":
                "Lender-side or bureau data, not obtainable from the applicant.",
        },
    },
}


# Risk direction each deployable model is REQUIRED to respect (+1: higher value
# can only raise risk). Imposed only where the direction is clear in the data
# itself and in lending practice, so explanations never contradict common
# sense ("spending 99% of income lowers risk").
MONOTONE = {
    "german_credit": {"duration": 1},
    "gmsc": {"DebtRatio": 1},
    "homecredit": {"DAYS_EMPLOYED": 1},   # stored as negative days: closer to 0 = shorter tenure
}

# GMSC's DebtRatio holds the raw debt AMOUNT, not a ratio, in a few percent of
# rows (a known data error); values above 5 are winsorised to 5. The live
# adapter applies the same cap.
GMSC_DEBT_RATIO_CAP = 5.0


def add_engineered(dataset_name, X):
    """Adds scale-free engineered columns. Must run before select_deployable."""
    if dataset_name == "homecredit":
        X = X.copy()
        X["credit_to_income"] = X["AMT_CREDIT"] / X["AMT_INCOME_TOTAL"].clip(lower=1)
    if dataset_name == "gmsc":
        X = X.copy()
        X["DebtRatio"] = X["DebtRatio"].clip(upper=GMSC_DEBT_RATIO_CAP)
    return X


def deployable_columns(dataset_name, all_columns):
    spec = DEPLOYABLE_SPEC[dataset_name]
    cols = [c for c in all_columns
            if c in spec["exact"] or any(c.startswith(p) for p in spec["prefixes"])]
    missing = [c for c in spec["exact"] if c not in all_columns]
    if missing:
        raise KeyError(f"[{dataset_name}] deployable columns not found: {missing}")
    return cols


def select_deployable(dataset_name, X, include_age=False):
    """include_age=True is used ONLY by age_ablation.py to measure what dropping age costs."""
    X = add_engineered(dataset_name, X)
    cols = deployable_columns(dataset_name, X.columns)
    if include_age:
        cols = ["age"] + cols
    return X[cols]
