import pandas as pd

def load_africa_synth(raw_path="data/raw/africa_synth_enriched.csv"):
    """Loads the enriched Africa Synth Financial Inclusion dataset."""
    df = pd.read_csv(raw_path)

    # Drop indeterminate outcomes — not usable for binary classification
    df = df[df["credit_outcome"] != "indeterminate"].copy()

    # target: 1 = bad credit (risk), 0 = good — same convention as our other datasets
    df["target"] = df["credit_outcome"].map({"good": 0, "bad": 1})

    return df


if __name__ == "__main__":
    df = load_africa_synth()
    print("Shape after removing indeterminate:", df.shape)
    print("\nTarget distribution:\n", df["target"].value_counts())
    print("\nColumns:\n", df.columns.tolist())