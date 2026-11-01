import pickle

for name in ["german_credit", "gmsc", "homecredit", "africa"]:
    ref = pickle.load(open(f"models/{name}_reference.pkl", "rb"))
    print(f"--- {name} ({len(ref['all_columns'])} cols) ---")
    print(ref["all_columns"])
    print()