"""
FAIRScore 2.0 transaction-sequence model: self-supervised pre-training + transfer to loan default.

    1. Pre-training (no labels needed): a GRU encoder reads transaction-event sequences (src/seq_events.py)
       from ALL Berka accounts (4,500; for accounts that took a loan, only the history BEFORE the loan, so no
       repayment behaviour can leak). Objective: CoLES-style contrastive learning -- two random slices of the
       same account's history should map close together, slices of different accounts apart. The encoder thus
       learns "how this customer manages money" without ever seeing a default label.
    2. Transfer: each loan's pre-loan 6-month statement is encoded into a 128-number "behaviour embedding".
    3. Nested evaluation on the 679 labelled loans (5 folds x 3 repeats), comparing
         a. logistic regression on the embedding alone
         b. the deployed tabular engine inputs (24 features, CatBoost)
         c. hybrid: tabular + embedding (8 principal components)
         d. the same GRU trained from scratch on the 679 labels only (no pre-training)

    python src\\train_sequence_model.py      (about 10-15 minutes on a laptop CPU; needs torch)

Writes models/seq_encoder.npz (numpy weights; live inference needs no torch), results/seq_model_metrics.csv
"""
import time
import warnings

import numpy as np
import pandas as pd

import ai_engine as ae
import berka_data
import seq_events as se
from config import PROJECT_ROOT

warnings.filterwarnings("ignore")
RESULTS = PROJECT_ROOT / "results"
OUT = PROJECT_ROOT / "models" / "seq_encoder.npz"
SEED = 0


def build_sequences():
    """Unlabelled pre-training histories per account and the labelled pre-loan windows."""
    loan, trans = berka_data.load_tables()
    trans = trans.sort_values(["account_id", "date", "trans_id"])
    cut = loan.set_index("account_id")["loan_date"]
    unl = []
    for acc, t in trans.groupby("account_id"):
        if acc in cut.index:
            t = t[t["date"] < cut[acc]]
        if len(t) < 20:
            continue
        signed = np.where(t["type"] == "PRIJEM", t["amount"], -t["amount"])
        df = pd.DataFrame({"date": t["date"].values, "description": [berka_data.narration(r).lower() for _, r in t.iterrows()],
                           "signed_amount": signed, "balance": t["balance"].values})
        unl.append(se.events(df, max_events=2000))
    lab = {}
    for ln, df, _ in berka_data.statements():
        lab[int(ln["loan_id"])] = se.events(df)
    return unl, lab


def main():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from catboost import CatBoostClassifier
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(2)
    t0 = time.time()
    unl, lab = build_sequences()
    print(f"{len(unl)} account histories for pre-training ({sum(len(c) for c, _ in unl):,} events); "
          f"{len(lab)} labelled loan windows ({time.time() - t0:.0f}s)")

    class Encoder(nn.Module):
        def __init__(self, n_cat=len(se.CATEGORIES), d_cat=12, d_in=32, hidden=64):
            super().__init__()
            self.emb = nn.Embedding(n_cat, d_cat, padding_idx=0)
            self.inp = nn.Linear(d_cat + se.N_NUM, d_in)
            self.gru = nn.GRU(d_in, hidden, batch_first=True)
            self.hidden = hidden

        def forward(self, cats, nums, lengths):
            x = torch.relu(self.inp(torch.cat([self.emb(cats), nums], -1)))
            out, _ = self.gru(x)
            mask = (torch.arange(cats.shape[1])[None, :] < lengths[:, None]).float()[..., None]
            mean = (out * mask).sum(1) / lengths[:, None].float()
            last = out[torch.arange(len(lengths)), lengths - 1]
            return torch.cat([mean, last], -1)

    def batchify(seqs):
        L = max(1, max(len(c) for c, _ in seqs))
        cats = np.zeros((len(seqs), L), dtype=np.int64)
        nums = np.zeros((len(seqs), L, se.N_NUM), dtype=np.float32)
        lens = np.ones(len(seqs), dtype=np.int64)
        for i, (c, n) in enumerate(seqs):
            if len(c):
                cats[i, :len(c)], nums[i, :len(c)], lens[i] = c, n, len(c)
        return torch.from_numpy(cats), torch.from_numpy(nums), torch.from_numpy(lens)

    def random_slice(c, n, rng):
        L = len(c)
        k = int(rng.integers(min(25, L), min(200, L) + 1))
        s = int(rng.integers(0, L - k + 1))
        return c[s:s + k], n[s:s + k]

    def encode(enc, seqs, bs=256):
        enc.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(seqs), bs):
                out.append(enc(*batchify(seqs[i:i + bs])).numpy())
        return np.vstack(out)

    # ---------------- 1. self-supervised pre-training (CoLES-style contrastive)
    enc = Encoder()
    head = nn.Linear(2 * enc.hidden, 64)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=2e-3)
    rng = np.random.default_rng(SEED)
    epochs, bs, temp = 25, 128, 0.1
    for ep in range(epochs):
        enc.train()
        order = rng.permutation(len(unl))
        tot = 0.0
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            a = [random_slice(*unl[j], rng) for j in idx]
            b = [random_slice(*unl[j], rng) for j in idx]
            za = F.normalize(head(enc(*batchify(a))), dim=-1)
            zb = F.normalize(head(enc(*batchify(b))), dim=-1)
            logits = za @ zb.T / temp
            target = torch.arange(len(idx))
            loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  pre-training epoch {ep + 1}/{epochs}: contrastive loss {tot / len(order):.3f} ({time.time() - t0:.0f}s)")

    # ---------------- 2. embeddings of the labelled pre-loan windows
    d = pd.read_csv(RESULTS / "berka_backtest_loans.csv")
    for k in ae.EXPERT_FACTORS:
        d[f"sc_{k}"] = d[f"sc_{k}"].fillna(0.5)
    d = d[d["loan_id"].isin(lab)].reset_index(drop=True)
    seqs = [lab[int(i)] for i in d["loan_id"]]
    E = encode(enc, seqs)
    y = d["bad"].values.astype(int)
    T = d[ae.FEATURES].astype(float).values

    # ---------------- 3. nested comparison
    def gru_from_scratch(tr, te):
        torch.manual_seed(SEED)
        e2 = Encoder()
        clf = nn.Linear(2 * e2.hidden, 1)
        o2 = torch.optim.Adam(list(e2.parameters()) + list(clf.parameters()), lr=2e-3, weight_decay=1e-4)
        pos_w = torch.tensor((1 - y[tr].mean()) / y[tr].mean())
        r2 = np.random.default_rng(SEED)
        for _ in range(15):
            e2.train()
            for i in range(0, len(tr), 64):
                idx = r2.permutation(tr)[:64] if i == 0 else tr[r2.integers(0, len(tr), 64)]
                logit = clf(e2(*batchify([seqs[j] for j in idx]))).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logit, torch.tensor(y[idx], dtype=torch.float32),
                                                          pos_weight=pos_w)
                o2.zero_grad()
                loss.backward()
                o2.step()
        e2.eval()
        with torch.no_grad():
            return torch.sigmoid(clf(e2(*batchify([seqs[j] for j in te]))).squeeze(-1)).numpy()

    res = {k: [] for k in ("a. embedding only (logistic)", "b. tabular engine inputs (CatBoost)",
                           "c. hybrid: tabular + embedding (CatBoost)", "d. GRU from scratch, labels only")}
    for rep in range(3):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=300 + rep).split(T, y):
            lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)).fit(E[tr], y[tr])
            res["a. embedding only (logistic)"].append(roc_auc_score(y[te], lr.predict_proba(E[te])[:, 1]))
            cb = CatBoostClassifier(**ae.CATBOOST_PARAMS, monotone_constraints=ae.MONOTONE).fit(T[tr], y[tr])
            res["b. tabular engine inputs (CatBoost)"].append(roc_auc_score(y[te], cb.predict_proba(T[te])[:, 1]))
            pca = make_pipeline(StandardScaler(), PCA(8, random_state=0)).fit(E[tr])
            H = np.c_[T, pca.transform(E)]
            hy = CatBoostClassifier(**ae.CATBOOST_PARAMS, monotone_constraints=ae.MONOTONE + [0] * 8).fit(H[tr], y[tr])
            res["c. hybrid: tabular + embedding (CatBoost)"].append(roc_auc_score(y[te], hy.predict_proba(H[te])[:, 1]))
            if rep == 0:
                res["d. GRU from scratch, labels only"].append(roc_auc_score(y[te], gru_from_scratch(tr, te)))
        print(f"  evaluation repeat {rep + 1}/3 done ({time.time() - t0:.0f}s)")
    met = pd.DataFrame([{"model": k, "auc": np.mean(v), "auc_sd": np.std(v), "folds": len(v)} for k, v in res.items()])
    met.to_csv(RESULTS / "seq_model_metrics.csv", index=False)
    print(met.round(3).to_string(index=False))

    # ---------------- 4. export numpy weights for torch-free inference
    sd = {k: v.detach().numpy() for k, v in enc.state_dict().items()}
    np.savez(OUT, emb=sd["emb.weight"], inp_w=sd["inp.weight"], inp_b=sd["inp.bias"],
             w_ih=sd["gru.weight_ih_l0"], w_hh=sd["gru.weight_hh_l0"], b_ih=sd["gru.bias_ih_l0"], b_hh=sd["gru.bias_hh_l0"])
    np.save(RESULTS / "seq_embeddings_berka.npy", E)
    print(f"Encoder saved to {OUT} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
