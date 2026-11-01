import pandas as pd
import os

# Column names in the order they appear in german.data
COLUMN_NAMES = [
    "checking_account", "duration", "credit_history", "purpose",
    "credit_amount", "savings_account", "employment_since",
    "installment_rate", "personal_status_sex", "other_debtors",
    "residence_since", "property", "age", "other_installment_plans",
    "housing", "existing_credits", "job", "dependents",
    "telephone", "foreign_worker", "target"
]

def load_german_credit(raw_path="data/raw/german.data"):
    """Loads the raw German Credit file into a pandas DataFrame."""
    df = pd.read_csv(raw_path, sep=" ", header=None, names=COLUMN_NAMES)

    # In the raw file, target is 1 = Good credit, 2 = Bad credit.
    # We convert it to 0 = Good, 1 = Bad so it's ML-friendly (1 = the risk we're predicting).
    df["target"] = df["target"].map({1: 0, 2: 1})

    return df


if __name__ == "__main__":
    df = load_german_credit()
    print("Shape:", df.shape)
    print("\nFirst 5 rows:\n", df.head())
    print("\nTarget distribution:\n", df["target"].value_counts())
    print("\nMissing values:\n", df.isnull().sum().sum())