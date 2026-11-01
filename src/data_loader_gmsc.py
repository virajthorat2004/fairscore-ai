import pandas as pd

def load_gmsc(raw_path="data/raw/cs-training.csv"):
    """Loads the raw Give Me Some Credit file into a pandas DataFrame."""
    df = pd.read_csv(raw_path)

    # Drop the leftover index column from the CSV export
    df = df.drop(columns=["Unnamed: 0"])

    # Rename target to match our pipeline convention: 'target' (1 = bad/risky, 0 = good)
    df = df.rename(columns={"SeriousDlqin2yrs": "target"})

    return df


if __name__ == "__main__":
    df = load_gmsc()
    print("Shape:", df.shape)
    print("\nColumns:\n", df.columns.tolist())
    print("\nTarget distribution:\n", df["target"].value_counts())
    print("\nMissing values:\n", df.isnull().sum())