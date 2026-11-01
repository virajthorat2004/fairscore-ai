import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from data_loader_homecredit import load_home_credit

# Columns with >40% missing are dropped rather than imputed — imputing that much
# of a column invents more data than it preserves.
MISSING_THRESHOLD = 0.40


def preprocess_home_credit(df):
    df = df.copy()

    # --- Drop the ID column, keep everything else for now ---
    df = df.drop(columns=["SK_ID_CURR"])

    # --- Known Home Credit data quirk: DAYS_EMPLOYED uses 365243 as a sentinel
    # value meaning "not employed / retired", not a real number of days.
    # Left as-is, it would look like someone employed for 1000 years. We flag it
    # and replace it with NaN (to be imputed like normal missing data below). ---
    df["days_employed_anomaly"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # --- DAYS_BIRTH is negative (days before application date). Convert to a
    # normal positive age in years, same convention as our other datasets. ---
    df["age"] = (-df["DAYS_BIRTH"] / 365).astype(int)
    df = df.drop(columns=["DAYS_BIRTH"])

    # --- Create the protected attribute, same convention as German Credit / GMSC ---
    df["age_group"] = df["age"].apply(lambda x: "under_25" if x < 25 else "25_and_over")

    # --- Drop columns that are too sparse to impute reliably ---
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > MISSING_THRESHOLD].index.tolist()
    print(f"Dropping {len(cols_to_drop)} columns with >{int(MISSING_THRESHOLD*100)}% missing values")
    df = df.drop(columns=cols_to_drop)

    # --- Impute remaining missing values ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "target"]
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna("Unknown")

    # --- One-hot encode categoricals (excluding age_group, kept separate for fairness slicing) ---
    categorical_cols = [c for c in categorical_cols if c != "age_group"]
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # --- Clean column names for XGBoost/LightGBM compatibility ---
    df_encoded.columns = (
        df_encoded.columns
        .str.replace("<", "lt", regex=False)
        .str.replace(">", "gt", regex=False)
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace("=", "", regex=False)
        .str.replace(" ", "_", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace(":", "", regex=False)
        .str.replace("/", "_", regex=False)
    )

    return df_encoded


def get_train_test_split_homecredit(df_encoded, test_size=0.2, random_state=42):
    X = df_encoded.drop(columns=["target", "age_group"])
    y = df_encoded["target"]
    age_group = df_encoded["age_group"]

    X_train, X_test, y_train, y_test, age_train, age_test = train_test_split(
        X, y, age_group, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, age_train, age_test


if __name__ == "__main__":
    df = load_home_credit()
    df_encoded = preprocess_home_credit(df)

    print("\nEncoded shape:", df_encoded.shape)
    print("\nMissing values remaining:", df_encoded.isnull().sum().sum())
    print("\nAge group distribution:\n", df_encoded["age_group"].value_counts())
    print("\nDays employed anomaly count:", df_encoded["days_employed_anomaly"].sum())

    X_train, X_test, y_train, y_test, age_train, age_test = get_train_test_split_homecredit(df_encoded)
    print("\nTrain size:", X_train.shape, "Test size:", X_test.shape)