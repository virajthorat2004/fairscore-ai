"""
FAIRScore 2.0 statement anomaly detector: learns what GENUINE bank statements look like and flags
statements whose money pattern looks manipulated -- without needing any labelled fraud.

Genuine statements
    * real ones: 6-month windows from the Berka bank data (every account, before any loan)
    * Indian-style ones: the 11 realistic statement profiles (salaried, shop owner, farmer, student, ...)
      regenerated with many random seeds
Manipulations injected into held-out genuine statements (balances recomputed so every row still adds up,
i.e. the forgeries pass the arithmetic check):
    window dressing      a large credit (3-6x monthly income) in the last 10 days
    inflated income      every income credit multiplied by 1.5-2
    round-tripping       4-8 transfers in and straight back out of the same amount
    deleted spending     the largest debits removed

Unsupervised Isolation Forest on 14 scale-free statement features (Benford first-digit fit, round amounts,
credit concentration, late-period inflows, balance jump at the end, matching in/out pairs, ...).

    python src\\statement_anomaly.py      writes models/statement_anomaly.pkl, results/statement_anomaly_metrics.csv
"""
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

from config import PROJECT_ROOT

warnings.filterwarnings("ignore")
MODEL = PROJECT_ROOT / "models" / "statement_anomaly.pkl"
RESULTS = PROJECT_ROOT / "results"
BENFORD = np.log10(1 + 1 / np.arange(1, 10))


def _first_digit(x):
    x = np.abs(np.asarray(x, dtype=float))
    x = x[x >= 10]
    if not len(x):
        return np.array([])
    return (x / 10 ** np.floor(np.log10(x))).astype(int)


def features(df):
    """df: date, signed_amount (+ balance). Scale-free features of the whole statement."""
    df = df.sort_values("date", kind="stable")
    t = pd.to_datetime(df["date"])
    amt = df["signed_amount"].astype(float).values
    cr, dr = amt[amt > 0], -amt[amt < 0]
    months = max((t.max() - t.min()).days / 30.4, 1.0)
    inc = max(cr.sum() / months, 1.0)
    fd = _first_digit(np.abs(amt))
    benford = (np.abs(np.bincount(fd, minlength=10)[1:] / max(len(fd), 1) - BENFORD).mean() if len(fd) >= 20 else 0.0)
    late = t >= t.max() - pd.Timedelta(days=10)
    bal = df["balance"].astype(float).values if "balance" in df and df["balance"].notna().any() else np.zeros(len(df))
    last_month = t >= t.max() - pd.Timedelta(days=30)
    top3 = np.sort(cr)[-3:].sum() / max(cr.sum(), 1.0) if len(cr) else 0.0
    # in/out pairs of the same amount within 3 days
    pairs = 0
    if len(cr) and len(dr):
        d_in = df[amt > 0][["date", "signed_amount"]]
        d_out = df[amt < 0][["date", "signed_amount"]].assign(signed_amount=lambda x: -x["signed_amount"])
        m = d_in.merge(d_out, on="signed_amount", suffixes=("_in", "_out"))
        if len(m):
            gap = (pd.to_datetime(m["date_out"]) - pd.to_datetime(m["date_in"])).dt.days
            pairs = int(((gap >= 0) & (gap <= 3) & (m["signed_amount"] >= 0.05 * inc)).sum())
    monthly_cr = pd.Series(np.where(amt > 0, amt, 0)).groupby(t.dt.to_period("M").values).sum()
    return {
        "benford_mad": benford,
        "round_credit_share": float(np.mean((cr % 1000) == 0)) if len(cr) else 0.0,
        "round_debit_share": float(np.mean((dr % 500) == 0)) if len(dr) else 0.0,
        "top3_credit_share": float(top3),
        "max_credit_to_income": float(cr.max() / inc) if len(cr) else 0.0,
        "late_credit_share": float(amt[late & (amt > 0)].sum() / max(cr.sum(), 1.0)),
        "late_debit_share": float(-amt[late & (amt < 0)].sum() / max(dr.sum(), 1.0)),
        "end_balance_jump": float(np.clip((bal[-1] - np.median(bal[~last_month]) if (~last_month).any() else 0) / inc, -10, 10)),
        "spend_to_income": float(np.clip(dr.sum() / max(cr.sum(), 1.0), 0, 5)),
        "inout_pairs_per_month": pairs / months,
        "credit_cv": float(monthly_cr.std() / monthly_cr.mean()) if len(monthly_cr) > 1 and monthly_cr.mean() > 0 else 0.0,
        "debits_per_credit": float(len(dr) / max(len(cr), 1)),
        "median_debit_to_income": float(np.median(dr) / inc) if len(dr) else 0.0,
        "median_credit_to_income": float(np.median(cr) / inc) if len(cr) else 0.0,
    }


FEATURES = list(features(pd.DataFrame({"date": pd.date_range("2020-01-01", periods=30),
                                       "signed_amount": np.r_[np.ones(15) * 100, -np.ones(15) * 50],
                                       "balance": np.arange(30)})).keys())


# ---------------------------------------------------------------- genuine corpora
def berka_windows(n_random=1500, seed=0):
    import berka_data
    loan, trans = berka_data.load_tables()
    trans = trans.sort_values(["account_id", "date", "trans_id"])
    cut = loan.set_index("account_id")["loan_date"]
    rng = np.random.default_rng(seed)
    out = []
    groups = list(trans.groupby("account_id"))
    for acc, t in [groups[i] for i in rng.choice(len(groups), size=min(n_random, len(groups)), replace=False)]:
        if acc in cut.index:
            t = t[t["date"] < cut[acc]]
        if len(t) < 40:
            continue
        end = t["date"].iloc[int(rng.integers(len(t) // 2, len(t)))]
        w = t[(t["date"] > end - pd.DateOffset(months=6)) & (t["date"] <= end)]
        if len(w) < 25:
            continue
        signed = np.where(w["type"] == "PRIJEM", w["amount"], -w["amount"])
        out.append(pd.DataFrame({"date": w["date"].values, "signed_amount": signed, "balance": w["balance"].values,
                                 "source": "berka"}))
    return out


def indian_windows(seeds_per_profile=40):
    sys.path.insert(0, str(PROJECT_ROOT / "tests"))
    import realistic_statements as R
    real_gen = R.Gen
    out = []
    for c in R.CASES:
        for s in range(seeds_per_profile):
            class G(real_gen):
                def __init__(self, seed, *a, **k):
                    super().__init__(seed * 1000 + s + 7, *a, **k)
            R.Gen = G
            try:
                df = c["build"]()
            finally:
                R.Gen = real_gen
            amt = pd.to_numeric(df["credit"].replace("", 0)).fillna(0) - pd.to_numeric(df["debit"].replace("", 0)).fillna(0)
            out.append(pd.DataFrame({"date": df["date"].values, "signed_amount": amt.values,
                                     "balance": df["balance"].values, "source": "indian"}))
    return out


# ---------------------------------------------------------------- manipulations (balances recomputed)
def _rebalance(df):
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    opening = float(df["balance"].iloc[0] - df["signed_amount"].iloc[0])
    df["balance"] = opening + df["signed_amount"].cumsum()
    return df


def manipulate(df, kind, rng):
    df = df.copy().reset_index(drop=True)
    t = pd.to_datetime(df["date"])
    amt = df["signed_amount"].astype(float)
    months = max((t.max() - t.min()).days / 30.4, 1.0)
    inc = max(amt[amt > 0].sum() / months, 1.0)
    if kind == "window dressing":
        d = t.max() - pd.Timedelta(days=int(rng.integers(0, 10)))
        big = round(inc * rng.uniform(3, 6), -2)
        df = pd.concat([df, pd.DataFrame({"date": [d], "signed_amount": [big], "balance": [0.0]})])
    elif kind == "inflated income":
        top = amt > 0
        biggest = amt[top].groupby(t[top].dt.to_period("M")).idxmax().values
        df.loc[biggest, "signed_amount"] = (df.loc[biggest, "signed_amount"] * rng.uniform(1.5, 2.0)).round(-1)
    elif kind == "round-tripping":
        rows = []
        for _ in range(int(rng.integers(4, 9))):
            d = t.min() + pd.Timedelta(days=int(rng.integers(0, max((t.max() - t.min()).days, 1))))
            v = round(inc * rng.uniform(0.2, 0.8), -3)
            rows += [{"date": d, "signed_amount": v, "balance": 0.0},
                     {"date": d + pd.Timedelta(days=int(rng.integers(0, 3))), "signed_amount": -v, "balance": 0.0}]
        df = pd.concat([df, pd.DataFrame(rows)])
    elif kind == "deleted spending":
        dr = amt[amt < 0].sort_values()
        df = df.drop(dr.index[:max(3, len(dr) // 10)])
    return _rebalance(df)


KINDS = ["window dressing", "inflated income", "round-tripping", "deleted spending"]


def main():
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    gen = berka_windows() + indian_windows()
    print(f"{len(gen)} genuine statements ({sum(g.source.iloc[0] == 'berka' for g in gen)} Berka, "
          f"{sum(g.source.iloc[0] == 'indian' for g in gen)} Indian-style)")
    idx = rng.permutation(len(gen))
    test_idx, train_idx = idx[: len(gen) // 4], idx[len(gen) // 4:]
    F = lambda dfs: pd.DataFrame([features(x) for x in dfs])[FEATURES].values
    Xtr = F([gen[i] for i in train_idx])
    iso = IsolationForest(n_estimators=400, random_state=0, contamination="auto").fit(Xtr)
    train_scores = np.sort(iso.score_samples(Xtr))
    genuine_test = [gen[i] for i in test_idx]
    s_gen = -iso.score_samples(F(genuine_test))
    rows = []
    for kind in KINDS:
        fakes = [manipulate(g, kind, rng) for g in genuine_test]
        s_fake = -iso.score_samples(F(fakes))
        for src in ("berka", "indian", "all"):
            sel = np.array([g.source.iloc[0] == src or src == "all" for g in genuine_test])
            y = np.r_[np.zeros(sel.sum()), np.ones(sel.sum())]
            sc = np.r_[s_gen[sel], s_fake[sel]]
            thr = np.quantile(s_gen, 0.95)          # flag the most unusual 5% of genuine statements
            rows.append({"manipulation": kind, "statements": src, "auc": roc_auc_score(y, sc),
                         "detected_at_5pct_false_alarms": float(np.mean(s_fake[sel] > thr))})
    met = pd.DataFrame(rows)
    met.to_csv(RESULTS / "statement_anomaly_metrics.csv", index=False)
    print(met.round(3).to_string(index=False))
    iso_all = IsolationForest(n_estimators=400, random_state=0).fit(F(gen))
    with open(MODEL, "wb") as fh:
        pickle.dump({"model": iso_all, "train_scores": np.sort(iso_all.score_samples(F(gen))), "features": FEATURES}, fh)
    print(f"saved {MODEL}")


_CACHE = {}


def assess(df):
    """Percentile of how unusual this statement is among genuine ones (low = unusual), with the main reasons."""
    if "m" not in _CACHE:
        if not MODEL.exists():
            return None
        with open(MODEL, "rb") as fh:
            _CACHE["m"] = pickle.load(fh)
    b = _CACHE["m"]
    f = features(df[["date", "signed_amount", "balance"]] if "balance" in df else df[["date", "signed_amount"]])
    x = np.array([[f[k] for k in b["features"]]])
    score = b["model"].score_samples(x)[0]
    return {"percentile": float(np.mean(b["train_scores"] <= score)), "features": f}


if __name__ == "__main__":
    main()
