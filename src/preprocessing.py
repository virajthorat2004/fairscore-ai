import pandas as pd
from sklearn.model_selection import train_test_split
from data_loader import load_german_credit

# UCI code → human-readable label mappings (from the dataset documentation)
CHECKING_MAP = {"A11": "<0 DM", "A12": "0-200 DM", "A13": ">=200 DM", "A14": "no checking account"}
CREDIT_HISTORY_MAP = {
    "A30": "no credits/all paid", "A31": "all paid at this bank",
    "A32": "existing paid duly", "A33": "delay in past", "A34": "critical account"
}
SAVINGS_MAP = {"A61": "<100 DM", "A62": "100-500 DM", "A63": "500-1000 DM",
               "A64": ">=1000 DM", "A65": "unknown/none"}
EMPLOYMENT_MAP = {"A71": "unemployed", "A72": "<1 year", "A73": "1-4 years",
                   "A74": "4-7 years", "A75": ">=7 years"}
HOUSING_MAP = {"A151": "rent", "A152": "own", "A153": "free"}
FOREIGN_WORKER_MAP = {"A201": "yes", "A202": "no"}

def preprocess(df):
    df = df.copy()

    # Decode key categorical codes into readable labels
    df["checking_account"] = df["checking_account"].map(CHECKING_MAP)
    df["credit_history"] = df["credit_history"].map(CREDIT_HISTORY_MAP)
    df["savings_account"] = df["savings_account"].map(SAVINGS_MAP)
    df["employment_since"] = df["employment_since"].map(EMPLOYMENT_MAP)
    df["housing"] = df["housing"].map(HOUSING_MAP)
    df["foreign_worker"] = df["foreign_worker"].map(FOREIGN_WORKER_MAP)

    # Create the protected attribute for fairness analysis: age group
    df["age_group"] = df["age"].apply(lambda x: "under_25" if x < 25 else "25_and_over")

    # One-hot encode all remaining categorical columns (for the ML models)
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    categorical_cols = [c for c in categorical_cols if c not in ["age_group"]]  # keep age_group separate, unencoded

    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Clean column names: XGBoost/LightGBM reject characters like <, >, [, ]
    df_encoded.columns = (
        df_encoded.columns
        .str.replace("<", "lt", regex=False)
        .str.replace(">", "gt", regex=False)
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace("=", "", regex=False)
        .str.replace(" ", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )

    return df, df_encoded


def get_train_test_split(df_encoded, test_size=0.2, random_state=42):
    X = df_encoded.drop(columns=["target", "age_group"])
    y = df_encoded["target"]
    age_group = df_encoded["age_group"]

    X_train, X_test, y_train, y_test, age_train, age_test = train_test_split(
        X, y, age_group, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, age_train, age_test


if __name__ == "__main__":
    df = load_german_credit()
    df_readable, df_encoded = preprocess(df)

    print("Readable columns sample:\n", df_readable[["checking_account", "age", "age_group"]].head())
    print("\nEncoded shape:", df_encoded.shape)
    print("\nAge group distribution:\n", df_readable["age_group"].value_counts())

    X_train, X_test, y_train, y_test, age_train, age_test = get_train_test_split(df_encoded)
    print("\nTrain size:", X_train.shape, "Test size:", X_test.shape)