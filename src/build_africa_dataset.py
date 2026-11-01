import pandas as pd
import numpy as np

COUNTRIES = ["BR", "IN", "NG", "US"]
BASE_URL = "hf://datasets/electricsheepafrica/africa-synth-financial-inclusion-electric-sheep-credit-all"


def engineer_transaction_features(txn_df):
    """
    Computes richer per-user features from raw transaction logs than the
    dataset's own pre-aggregated profile table provides. Kept to a modest
    number of features deliberately -- with only ~120 users total, adding
    too many engineered columns risks overfitting/instability even for a
    qualitative SHAP demonstration.
    """
    grouped = txn_df.groupby("user_id")

    features = grouped.agg(
        txn_count=("transaction_id", "count"),
        avg_txn_amount=("amount", "mean"),
        txn_amount_std=("amount", "std"),
        pct_debit=("transaction_type", lambda x: (x == "debit").mean()),
        channel_diversity=("channel", "nunique"),
        category_diversity=("category", "nunique"),
        recurring_txn_ratio=("is_recurring", "mean"),
        min_balance_observed=("balance_estimate", "min"),
    ).reset_index()

    # Overdraft events: how often balance_estimate dipped below zero
    overdraft_counts = txn_df[txn_df["balance_estimate"] < 0].groupby("user_id").size()
    features["overdraft_event_count"] = features["user_id"].map(overdraft_counts).fillna(0)

    features["txn_amount_std"] = features["txn_amount_std"].fillna(0)

    return features


def build_dataset():
    all_profiles = []
    all_txn_features = []

    for country in COUNTRIES:
        print(f"Fetching {country}...")
        profiles = pd.read_parquet(f"{BASE_URL}/{country}/credit_profiles.parquet")
        profiles["country"] = country
        all_profiles.append(profiles)

        txns = pd.read_parquet(f"{BASE_URL}/transactions/{country}.parquet")
        txn_features = engineer_transaction_features(txns)
        all_txn_features.append(txn_features)

    profiles_df = pd.concat(all_profiles, ignore_index=True)
    txn_features_df = pd.concat(all_txn_features, ignore_index=True)

    # Merge engineered transaction features onto the profile table
    merged = profiles_df.merge(txn_features_df, on="user_id", how="left")

    merged.to_csv("data/raw/africa_synth_enriched.csv", index=False)
    print(f"\nSaved data/raw/africa_synth_enriched.csv — shape: {merged.shape}")
    print("\nCredit outcome distribution:\n", merged["credit_outcome"].value_counts())

    return merged


if __name__ == "__main__":
    build_dataset()