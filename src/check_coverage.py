import sys
sys.path.insert(0, "src")
from statement_parser import parse_statement
from feature_adapters import ADAPTERS, load_reference

stmt_df, f, _ = parse_statement("data/raw/sample_statement.csv")
core = {"age": 29, "monthly_income": 42000, "requested_amount": 200000,
        "requested_duration_months": 24, "total_savings": 25000,
        "dependents": 1, "employment_days": 800}

for ds, fn in ADAPTERS.items():
    feats, tags = fn(core, f, stmt_df, "Yes")
    ref = load_reference(ds)["all_columns"]
    missing = [c for c in ref if c not in feats]
    print(f"{ds}: {len(ref)} cols, {len(missing)} not returned by adapter (become 0)")
    print("   ", missing[:30])