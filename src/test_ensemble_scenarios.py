from statement_parser import parse_statement
from ensemble_predictor import run_ensemble


def print_result(label, result):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    print("Consensus risk probability:", result["consensus_risk_probability"])
    print("Models used:", result["models_used"])
    print("Models excluded:", result["models_excluded"])
    print("Manual review required:", result["manual_review_required"])
    if result["consistency_flagged"]:
        print("CONSISTENCY FLAG:", result["consistency_message"])
    for name, detail in result["details"].items():
        status = "QUALIFIES" if detail["qualifies"] else "excluded"
        print(f"  {name}: applicability={detail['applicability']}  [{status}]")


core_base = {
    "age": 29, "monthly_income": 42000, "requested_amount": 200000,
    "requested_duration_months": 24, "total_savings": 25000,
    "dependents": 1, "employment_days": 800,
}

stmt_df, stmt_features, warnings = parse_statement("data/raw/sample_statement.csv")

# Scenario 1: statement provided, gate truthfully says "Yes" (has credit) —
# baseline, already tested before.
result1 = run_ensemble(core_base, stmt_features, stmt_df, gate_answer="Yes")
print_result("Scenario 1: Statement provided, gate = Yes (consistent)", result1)

# Scenario 2: no statement uploaded at all, declares no existing credit.
# Expect: likely NO model qualifies -- an honest "insufficient data" outcome.
result2 = run_ensemble(core_base, {}, None, gate_answer="No")
print_result("Scenario 2: No statement uploaded, gate = No", result2)

# Scenario 3: statement provided (which HAS a detected recurring EMI debit),
# but applicant declares "No" existing credit -- should trigger the
# consistency flag.
result3 = run_ensemble(core_base, stmt_features, stmt_df, gate_answer="No")
print_result("Scenario 3: Statement provided, gate = No (INCONSISTENT)", result3)