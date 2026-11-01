import pandas as pd
import pickle
import os

from data_loader import load_german_credit
from preprocessing import preprocess as preprocess_gc, get_train_test_split as split_gc

from data_loader_gmsc import load_gmsc
from preprocessing_gmsc import preprocess_gmsc, get_train_test_split_gmsc

from data_loader_homecredit import load_home_credit
from preprocessing_homecredit import preprocess_home_credit, get_train_test_split_homecredit

from data_loader_africa import load_africa_synth
from preprocessing_africa import preprocess_africa, get_train_test_split_africa


def save_reference(X_train, name):
    X_numeric = X_train.astype(float)

    # Automatically detect which columns are binary (0/1 dummy/flag columns,
    # e.g. one-hot encoded categories) vs genuinely continuous numeric columns.
    # This lets the adapters fall back sensibly on datasets with very different
    # shapes (German Credit: 48 cols, Home Credit: 174 cols) without hardcoding
    # column-by-column defaults for every dataset by hand.
    binary_columns = [
        c for c in X_train.columns
        if set(X_numeric[c].dropna().unique()).issubset({0.0, 1.0})
    ]

    reference = {
        "all_columns": X_train.columns.tolist(),
        "median": X_numeric.median().to_dict(),
        "mode": X_train.mode().iloc[0].to_dict(),
        "binary_columns": binary_columns,
    }
    os.makedirs("models", exist_ok=True)
    with open(f"models/{name}_reference.pkl", "wb") as f:
        pickle.dump(reference, f)
    print(f"Saved models/{name}_reference.pkl — {len(reference['all_columns'])} columns "
          f"({len(binary_columns)} binary/dummy, {len(reference['all_columns'])-len(binary_columns)} numeric)")


def build_german_credit_reference():
    df = load_german_credit()
    _, df_encoded = preprocess_gc(df)
    X_train, X_test, y_train, y_test, age_train, age_test = split_gc(df_encoded)
    save_reference(X_train, "german_credit")


def build_gmsc_reference():
    df = load_gmsc()
    df_encoded = preprocess_gmsc(df)
    X_train, X_test, y_train, y_test, age_train, age_test = get_train_test_split_gmsc(df_encoded)
    save_reference(X_train, "gmsc")


def build_homecredit_reference():
    df = load_home_credit()
    df_encoded = preprocess_home_credit(df)
    X_train, X_test, y_train, y_test, age_train, age_test = get_train_test_split_homecredit(df_encoded)
    save_reference(X_train, "homecredit")


def build_africa_reference():
    df = load_africa_synth()
    df_encoded = preprocess_africa(df)
    X_train, X_test, y_train, y_test, age_train, age_test = get_train_test_split_africa(df_encoded)
    save_reference(X_train, "africa")


if __name__ == "__main__":
    print("Building reference files for all 4 datasets...\n")
    build_german_credit_reference()
    build_gmsc_reference()
    build_homecredit_reference()
    build_africa_reference()
    print("\nAll reference files created.")