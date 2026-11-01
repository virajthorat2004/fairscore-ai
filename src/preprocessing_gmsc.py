import pandas as pd
from sklearn.model_selection import train_test_split
from data_loader_gmsc import load_gmsc

def preprocess_gmsc(df):
    df = df.copy()

    # --- Handle missing MonthlyIncome (~20% missing) ---
    # We add a flag column so the model can learn "this was originally unknown"
    # rather than silently treating imputed values as if they were real.
    df["monthly_income_was_missing"] = df["MonthlyIncome"].isnull().astype(int)
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())

    # --- Handle missing NumberOfDependents (~2.6% missing) ---
    df["dependents_was_missing"] = df["NumberOfDependents"].isnull().astype(int)
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(df["NumberOfDependents"].median())

    # --- Create the protected attribute for fairness analysis: same age_group logic as German Credit ---
    df["age_group"] = df["age"].apply(lambda x: "under_25" if x < 25 else "25_and_over")

    # GMSC has no categorical columns needing one-hot encoding — everything is already numeric.
    # But we do clean column names for XGBoost/LightGBM compatibility, same as before.
    df.columns = (
        df.columns
        .str.replace("<", "lt", regex=False)
        .str.replace(">", "gt", regex=False)
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace("=", "", regex=False)
        .str.replace(" ", "_", regex=False)
    )

    return df  # GMSC needs no separate "readable vs encoded" split — it's already numeric


def get_train_test_split_gmsc(df, test_size=0.2, random_state=42):
    X = df.drop(columns=["target", "age_group"])
    y = df["target"]
    age_group = df["age_group"]

    X_train, X_test, y_train, y_test, age_train, age_test = train_test_split(
        X, y, age_group, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, age_train, age_test


if __name__ == "__main__":
    df = load_gmsc()
    df_processed = preprocess_gmsc(df)

    print("Processed shape:", df_processed.shape)
    print("\nMissing values after imputation:\n", df_processed.isnull().sum().sum())
    print("\nAge group distribution:\n", df_processed["age_group"].value_counts())

    X_train, X_test, y_train, y_test, age_train, age_test = get_train_test_split_gmsc(df_processed)
    print("\nTrain size:", X_train.shape, "Test size:", X_test.shape)