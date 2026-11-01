import pandas as pd
from sklearn.model_selection import train_test_split
from data_loader_africa import load_africa_synth

# Columns not useful as model inputs (identifiers, or would leak the target)
DROP_COLS = ["user_id", "default_probability", "risk_bucket", "credit_outcome"]


def preprocess_africa(df):
    df = df.copy()

    # --- Protected attribute: derive from age_range bucket ---
    # Note: "18-25" includes 25-year-olds, unlike our numeric under_25 cutoff
    # used in the other 3 datasets. This is a real, stated inconsistency —
    # not silently smoothed over — because age here is only available as a
    # pre-existing bucket, not a raw value.
    df["age_group"] = df["age_range"].apply(lambda x: "18-25 (youngest bucket)" if x == "18-25" else "26+")

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # --- One-hot encode categoricals ---
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    categorical_cols = [c for c in categorical_cols if c not in ["age_group", "age_range"]]
    df = df.drop(columns=["age_range"])  # keep only the derived age_group

    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # --- Clean column names for CatBoost compatibility ---
    df_encoded.columns = (
        df_encoded.columns
        .str.replace("<", "lt", regex=False)
        .str.replace(">", "gt", regex=False)
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace("=", "", regex=False)
        .str.replace(" ", "_", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )

    return df_encoded


def get_train_test_split_africa(df_encoded, test_size=0.2, random_state=42):
    X = df_encoded.drop(columns=["target", "age_group"])
    y = df_encoded["target"]
    age_group = df_encoded["age_group"]

    # With only 101 rows, stratifying by target keeps the already-thin
    # test set from accidentally losing the minority class entirely.
    X_train, X_test, y_train, y_test, age_train, age_test = train_test_split(
        X, y, age_group, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, age_train, age_test


if __name__ == "__main__":
    df = load_africa_synth()
    df_encoded = preprocess_africa(df)

    print("Encoded shape:", df_encoded.shape)
    print("\nMissing values:", df_encoded.isnull().sum().sum())
    print("\nAge group distribution:\n", df_encoded["age_group"].value_counts())

    X_train, X_test, y_train, y_test, age_train, age_test = get_train_test_split_africa(df_encoded)
    print("\nTrain size:", X_train.shape, "Test size:", X_test.shape)