"""
AlfaBattle 2.0 sequence models (see src/alfabattle.py). Same 70/30 client split as the LightGBM baseline.
Pre-training uses ONLY the training clients' transactions and no labels.

    B. GRU from scratch (supervised)                        C. pre-trained GRU, logistic on the frozen embedding
    C2. pre-trained GRU, fine-tuned end to end              D. LightGBM on aggregates + embedding
Each at 1%, 5%, 20% and 100% of the labelled training clients.  -> results/alfabattle_sequence.csv
"""
import time
import warnings

import numpy as np
import pandas as pd

import alfabattle as ab

warnings.filterwarnings("ignore")


def main(pre_epochs=3, sup_epochs=2):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(ab.SEED)
    torch.set_num_threads(2)
    t0 = time.time()
    z = np.load(ab.CACHE)
    cats, nums, lens, card = z["cats"], z["nums"], z["lens"], z["card"]
    agg = pd.read_parquet(ab.RAW / "fairscore_aggregates.parquet")
    y = agg.pop("flag").values.astype(int)
    X_agg = agg.astype(float).values
    tr, te = ab.split(y)
    print(f"{len(y):,} clients, {y.sum():,} defaults; train {len(tr):,} / test {len(te):,} ({time.time() - t0:.0f}s)", flush=True)

    class Enc(nn.Module):
        def __init__(self, hidden=96):
            super().__init__()
            self.embs = nn.ModuleList([nn.Embedding(int(c), 4 if c < 20 else 12, padding_idx=0) for c in card])
            d = sum(e.embedding_dim for e in self.embs) + nums.shape[2]
            self.inp = nn.Linear(d, 64)
            self.gru = nn.GRU(64, hidden, batch_first=True)
            self.out_dim = 2 * hidden

        def forward(self, c, x, L):
            e = torch.cat([emb(c[..., i]) for i, emb in enumerate(self.embs)] + [x], -1)
            h, _ = self.gru(torch.relu(self.inp(e)))
            mask = (torch.arange(c.shape[1])[None, :] < L[:, None]).float()[..., None]
            return torch.cat([(h * mask).sum(1) / L[:, None].clamp(min=1).float(),
                              h[torch.arange(len(L)), (L - 1).clamp(min=0)]], -1)

    def batch(idx, crop=False, rng=None):
        c = cats[idx].astype(np.int64)
        x = nums[idx]
        L = lens[idx].copy()
        if crop:                                    # random contiguous slice of each sequence (a "view")
            c2, x2 = np.zeros_like(c), np.zeros_like(x)
            for j, l in enumerate(L):
                k = int(rng.integers(max(5, l // 4), l + 1)) if l > 5 else l
                s = int(rng.integers(0, l - k + 1))
                c2[j, :k], x2[j, :k] = c[j, s:s + k], x[j, s:s + k]
                L[j] = k
            c, x = c2, x2
        return torch.from_numpy(c), torch.from_numpy(x), torch.from_numpy(np.maximum(L, 1).astype(np.int64))

    def embed(enc, idx, bs=1024):
        enc.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), bs):
                out.append(enc(*batch(idx[i:i + bs])).numpy())
        return np.vstack(out)

    def supervised(enc, idx, epochs, lr=2e-3, eval_idx=None):
        head = nn.Linear(enc.out_dim, 1)
        opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=lr)
        pw = torch.tensor((1 - y[idx].mean()) / max(y[idx].mean(), 1e-3))
        rng = np.random.default_rng(1)
        for ep in range(epochs):
            enc.train()
            order = rng.permutation(idx)
            for i in range(0, len(order), 256):
                b = order[i:i + 256]
                logit = head(enc(*batch(b))).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logit, torch.tensor(y[b], dtype=torch.float32), pos_weight=pw)
                opt.zero_grad()
                loss.backward()
                opt.step()
        enc.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(eval_idx), 1024):
                out.append(torch.sigmoid(head(enc(*batch(eval_idx[i:i + 1024]))).squeeze(-1)).numpy())
        return np.concatenate(out)

    # ---------------- pre-training (no labels, training clients only)
    enc = Enc()
    proj = nn.Linear(enc.out_dim, 64)
    opt = torch.optim.Adam(list(enc.parameters()) + list(proj.parameters()), lr=2e-3)
    rng = np.random.default_rng(ab.SEED)
    for ep in range(pre_epochs):
        enc.train()
        order = rng.permutation(tr)
        tot = 0.0
        for i in range(0, len(order), 256):
            b = order[i:i + 256]
            za = F.normalize(proj(enc(*batch(b, True, rng))), dim=-1)
            zb = F.normalize(proj(enc(*batch(b, True, rng))), dim=-1)
            lg = za @ zb.T / 0.1
            t = torch.arange(len(b))
            loss = (F.cross_entropy(lg, t) + F.cross_entropy(lg.T, t)) / 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        print(f"  pre-training epoch {ep + 1}/{pre_epochs}: loss {tot / len(order):.3f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save(enc.state_dict(), ab.RAW / "fairscore_pretrained.pt")
    E = np.zeros((len(y), enc.out_dim), dtype=np.float32)
    E[tr], E[te] = embed(enc, tr), embed(enc, te)
    np.save(ab.RAW / "fairscore_embeddings.npy", E)
    print(f"  embeddings done ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    rng = np.random.default_rng(ab.SEED)            # same subsamples as the baseline
    subs = {f: (tr if f == 1.0 else rng.choice(tr, int(f * len(tr)), replace=False)) for f in ab.FRACTIONS}

    def add(method, frac, p):
        auc = roc_auc_score(y[te], p)
        lo, hi = ab.boot(y[te], p) if frac == 1.0 else (np.nan, np.nan)
        rows.append({"method": method, "train_fraction": frac, "train_clients": len(subs[frac]),
                     "train_defaults": int(y[subs[frac]].sum()), "auc": auc, "ci_low": lo, "ci_high": hi})
        print(f"  {method:55s} {frac:>5.0%}  AUC {auc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(ab.RESULTS / "alfabattle_sequence.csv", index=False)

    pca = make_pipeline(StandardScaler(), PCA(32, random_state=0)).fit(E[tr])
    H = np.c_[X_agg, pca.transform(E)]
    state = {k: v.clone() for k, v in enc.state_dict().items()}
    for frac in ab.FRACTIONS:
        s = subs[frac]
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)).fit(E[s], y[s])
        add("C. pre-trained GRU embedding + logistic", frac, lr.predict_proba(E[te])[:, 1])
        add("D. LightGBM on aggregates + embedding", frac, ab.lgbm(H, y, s, te))
        ep = sup_epochs if frac >= 0.2 else max(sup_epochs, int(2 / frac) // 10 + 2)
        torch.manual_seed(ab.SEED)
        add("B. GRU from scratch (supervised)", frac, supervised(Enc(), s, ep, eval_idx=te))
        ft = Enc()
        ft.load_state_dict(state)
        add("C2. pre-trained GRU fine-tuned", frac, supervised(ft, s, ep, lr=1e-3, eval_idx=te))
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
