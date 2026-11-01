import pandas as pd

def load_home_credit(raw_path="data/raw/application_train.csv"):
    """Loads the main Home Credit application table."""
    df = pd.read_csv(raw_path)
    df = df.rename(columns={"TARGET": "target"})
    return df


if __name__ == "__main__":
    df = load_home_credit()
    print("Shape:", df.shape)
    print("\nTarget distribution:\n", df["target"].value_counts())
    print("\nDtypes summary:\n", df.dtypes.value_counts())