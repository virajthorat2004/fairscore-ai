"""
FAIRScore 2.0 research study on AlfaBattle 2.0 (Alfa-Bank, 2021): card-transaction histories BEFORE a loan was
granted, and whether the borrower defaulted. 963,811 clients in the full data, 2.76% defaults; this study uses
the first N training files (about 23,000 clients each).

Question: how well does a model that reads only modern transaction behaviour predict default -- and does
SELF-SUPERVISED pre-training on unlabelled transactions help when labelled loans are scarce (as for a new lender
or a new-to-credit segment)?

Compared on the same held-out 30% of clients (bootstrap 95% intervals):
    A. gradient boosting (LightGBM) on 150+ hand-made aggregate features (the strong standard approach)
    B. GRU trained from scratch on the raw transaction sequence (supervised)
    C. GRU pre-trained without labels (contrastive, CoLES-style) on ALL clients' sequences, then a logistic
       regression on its 128-number embedding
    D. hybrid: LightGBM on aggregates + embedding
and a label-efficiency curve: each method trained on 1%, 5%, 20% and 100% of the labelled training clients.

    python src\\alfabattle.py prepare       (reads the parquet files -> data/raw/alfabattle/fairscore_cache.npz)
    python src\\alfabattle.py baseline      (A)
    python src\\alfabattle.py sequence      (B, C, D and the label-efficiency curve; needs torch; hours on a CPU)
Results: results/alfabattle_*.csv
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROJECT_ROOT

warnings.filterwarnings("ignore")
RAW = PROJECT_ROOT / "data" / "raw" / "alfabattle"
CACHE = RAW / "fairscore_cache.npz"
RESULTS = PROJECT_ROOT / "results"
CAT_COLS = ["currency", "operation_kind", "card_type", "operation_type", "operation_type_group", "ecommerce_flag",
            "payment_system", "income_flag", "mcc", "country", "city", "mcc_category", "day_of_week", "hour"]
NUM_COLS = ["amnt", "days_before", "hour_diff"]
SEQ_LEN = 150
SEED = 0


def _files(n_files):
    pats = sorted((RAW).rglob("train_transactions_contest/*.parquet"))
    if not pats:
        raise SystemExit(f"No AlfaBattle training files under {RAW}")
    return pats[:n_files]


def _read(f):
    df = pd.read_parquet(f, columns=["app_id", "transaction_number"] + CAT_COLS + NUM_COLS)
    for c in CAT_COLS + ["transaction_number"]:
        df[c] = df[c].astype(np.int16 if df[c].max() < 32000 else np.int32)
    df["amnt"] = df["amnt"].astype(np.float32)
    df["days_before"] = df["days_before"].astype(np.float32)
    df["hour_diff"] = df["hour_diff"].astype(np.float32)
    return df.sort_values(["app_id", "transaction_number"]).reset_index(drop=True)


AGG_CATS = ["operation_kind", "card_type", "operation_type", "operation_type_group", "ecommerce_flag", "payment_system",
            "income_flag", "mcc_category", "day_of_week", "currency", "country"]


def _aggregates(df, tops):
    g = df.groupby("app_id", sort=True)
    agg = pd.DataFrame({"n_tx": g.size()})
    for c in NUM_COLS:
        s = g[c]
        agg[f"{c}_mean"], agg[f"{c}_std"], agg[f"{c}_min"], agg[f"{c}_max"] = s.mean(), s.std(), s.min(), s.max()
    agg["amnt_sum"] = g["amnt"].sum()
    agg["active_days"] = g["days_before"].nunique()
    for c in AGG_CATS:
        col = df[c].where(df[c].isin(tops[c]), -1)
        share = pd.crosstab(df["app_id"], col, normalize="index").reindex(columns=list(tops[c]) + [-1], fill_value=0)
        share.columns = [f"{c}_share_{k}" for k in share.columns]
        col6 = df[c].where(df[c].isin(tops[c][:6]), -1)
        amt = df.pivot_table(index="app_id", columns=col6, values="amnt", aggfunc="mean").reindex(
            columns=list(tops[c][:6]) + [-1])
        amt.columns = [f"{c}_amnt_{k}" for k in amt.columns]
        agg = agg.join(share).join(amt)
    h = df["hour"]
    agg["night_share"] = ((h < 6) | (h >= 23)).groupby(df["app_id"]).mean()
    agg["n_mcc"] = g["mcc"].nunique()
    agg["n_city"] = g["city"].nunique()
    return agg


def prepare(n_files=6):
    """One file at a time (each holds whole clients), so memory stays small."""
    t0 = time.time()
    target = pd.read_csv(next(RAW.rglob("alfabattle2_train_target.csv"))).set_index("app_id")
    files = _files(n_files)
    first = _read(files[0])
    tops = {c: list(first[c].value_counts().index[:12]) for c in AGG_CATS}
    card = np.zeros(len(CAT_COLS), dtype=np.int64)
    aggs, C_all, N_all, L_all, ids_all = [], [], [], [], []
    for k, f in enumerate(files):
        df = first if k == 0 else _read(f)
        ids, start, counts = np.unique(df["app_id"].values, return_index=True, return_counts=True)
        aggs.append(_aggregates(df, tops))
        card = np.maximum(card, df[CAT_COLS].max().values.astype(np.int64) + 3)
        lens = np.minimum(counts, SEQ_LEN).astype(np.int32)
        C = (df[CAT_COLS].values.astype(np.int32) + 1).astype(np.int16)
        hd = df["hour_diff"].values
        N = np.c_[df["amnt"].values, np.log1p(np.clip(df["days_before"].values, 0, None)) / 6.0,
                  np.sign(hd) * np.log1p(np.abs(hd)) / 8.0].astype(np.float32)
        cats = np.zeros((len(ids), SEQ_LEN, len(CAT_COLS)), dtype=np.int16)
        nums = np.zeros((len(ids), SEQ_LEN, len(NUM_COLS)), dtype=np.float32)
        for i in range(len(ids)):
            s, n = start[i] + counts[i] - lens[i], lens[i]
            cats[i, :n], nums[i, :n] = C[s:s + n], N[s:s + n]
        C_all.append(cats), N_all.append(nums), L_all.append(lens), ids_all.append(ids)
        del df, C, N
        if k == 0:
            first = None
        print(f"  file {k + 1}/{len(files)}: {len(ids):,} clients ({time.time() - t0:.0f}s)", flush=True)
    agg = pd.concat(aggs).join(target[["product", "flag"]], how="left")
    ids = np.concatenate(ids_all)
    np.savez(CACHE, ids=ids, cats=np.concatenate(C_all), nums=np.concatenate(N_all), lens=np.concatenate(L_all),
             card=card)
    agg.loc[ids].to_parquet(RAW / "fairscore_aggregates.parquet")
    print(f"cached {len(ids):,} clients, {agg.shape[1]} aggregate features, {int(agg['flag'].sum()):,} defaults "
          f"({time.time() - t0:.0f}s)")


def split(y):
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    return train_test_split(idx, test_size=0.3, stratify=y, random_state=SEED)


def boot(y, p, n=500):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    v = [roc_auc_score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n)) if y[i].min() != y[i].max()]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def lgbm(X, y, tr, te, seed=SEED):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=100,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5, random_state=seed,
                           verbose=-1)
    m.fit(X[tr], y[tr])
    return m.predict_proba(X[te])[:, 1]


FRACTIONS = (0.01, 0.05, 0.2, 1.0)


def baseline():
    from sklearn.metrics import roc_auc_score
    agg = pd.read_parquet(RAW / "fairscore_aggregates.parquet")
    y = agg.pop("flag").values.astype(int)
    X = agg.astype(float).values
    tr, te = split(y)
    rows, rng = [], np.random.default_rng(SEED)
    for frac in FRACTIONS:
        sub = tr if frac == 1.0 else rng.choice(tr, int(frac * len(tr)), replace=False)
        p = lgbm(X, y, sub, te)
        auc = roc_auc_score(y[te], p)
        lo, hi = boot(y[te], p) if frac == 1.0 else (np.nan, np.nan)
        rows.append({"method": "A. LightGBM on aggregates", "train_fraction": frac, "train_clients": len(sub),
                     "train_defaults": int(y[sub].sum()), "auc": auc, "ci_low": lo, "ci_high": hi})
        print(rows[-1])
    pd.DataFrame(rows).to_csv(RESULTS / "alfabattle_baseline.csv", index=False)
    np.save(RAW / "fairscore_baseline_pred.npy", p)


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    if step == "prepare":
        prepare(int(sys.argv[2]) if len(sys.argv) > 2 else 6)
    elif step == "baseline":
        baseline()
    elif step == "sequence":
        import alfabattle_seq
        alfabattle_seq.main()
