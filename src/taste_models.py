"""Phase 3 - taste (preference) models.

All models expose  scores(user_idx_array) -> dense (len(users), n_items) float32.
Higher = the user is predicted to like the recipe more. Already-seen items are NOT
masked here; the evaluator / recommender masks them.

  Popularity : number of positive train interactions per recipe   (baseline)
  Content    : cosine(user profile, recipe TF-IDF)                (content-based)
  ALS        : implicit-feedback matrix factorisation             (collaborative)
  Hybrid     : w * z(ALS) + (1-w) * z(Content), z = per-user z-score
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.metrics import ranking_metrics, topk_indices

P = Path("data/processed")
M = Path("models")
LIKE = 4


def interaction_matrix(df, n_users, n_items, like=LIKE):
    d = df[df.rating >= like]
    return sp.csr_matrix((np.ones(len(d), dtype=np.float32), (d.user_idx, d.item_idx)),
                         shape=(n_users, n_items))


def zscore_rows(S):
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)


class Popularity:
    def fit(self, R, X=None):
        self.pop = np.asarray(R.sum(0)).ravel().astype(np.float32)
        return self

    def scores(self, users):
        return np.tile(self.pop, (len(users), 1))


class Content:
    """User profile = mean TF-IDF vector of the recipes they liked (train)."""
    def fit(self, R, X):
        self.R, self.X = R.tocsr(), X.tocsr()
        return self

    def scores(self, users):
        prof = self.R[users] @ self.X                      # (u, terms) sparse
        norm = np.sqrt(np.asarray(prof.multiply(prof).sum(1))).ravel() + 1e-9
        prof = sp.diags(1.0 / norm) @ prof
        return np.asarray((prof @ self.X.T).todense(), dtype=np.float32)


class ALS:
    """Implicit ALS (Hu, Koren & Volinsky 2008). Binary preference p=1 for liked items,
    confidence c = 1 + alpha. Each step solves a small ridge system per user/item."""
    def __init__(self, factors=64, reg=100.0, alpha=10.0, iters=10, seed=0):
        self.f, self.reg, self.alpha, self.iters = factors, reg, alpha, iters
        self.rng = np.random.default_rng(seed)

    def _solve(self, Rcsr, Y):
        f = self.f
        YtY = Y.T @ Y + self.reg * np.eye(f, dtype=np.float32)
        out = np.zeros((Rcsr.shape[0], f), dtype=np.float32)
        ind, ptr = Rcsr.indices, Rcsr.indptr
        for u in range(Rcsr.shape[0]):
            idx = ind[ptr[u]:ptr[u + 1]]
            if len(idx) == 0:
                continue
            Yu = Y[idx]
            A = YtY + self.alpha * (Yu.T @ Yu)
            b = (1.0 + self.alpha) * Yu.sum(0)
            out[u] = np.linalg.solve(A, b)
        return out

    def fit(self, R, X=None):
        R = R.tocsr(); Rt = R.T.tocsr()
        self.U = (0.1 * self.rng.standard_normal((R.shape[0], self.f))).astype(np.float32)
        self.V = (0.1 * self.rng.standard_normal((R.shape[1], self.f))).astype(np.float32)
        for it in range(self.iters):
            self.U = self._solve(R, self.V)
            self.V = self._solve(Rt, self.U)
        return self

    def scores(self, users):
        return (self.U[users] @ self.V.T).astype(np.float32)


class Hybrid:
    def __init__(self, cf, content, w=0.5):
        self.cf, self.content, self.w = cf, content, w

    def scores(self, users):
        return (self.w * zscore_rows(self.cf.scores(users)) +
                (1 - self.w) * zscore_rows(self.content.scores(users)))


def evaluate(model, seen: sp.csr_matrix, target_df, n_users, k=10, batch=1000):
    """Rank all unseen recipes for each user; compare with their held-out positives."""
    rel = target_df[target_df.rating >= LIKE].groupby("user_idx").item_idx.apply(set).to_dict()
    users = np.array(sorted(rel))
    tops, rels = [], []
    for s in range(0, len(users), batch):
        u = users[s:s + batch]
        S = model.scores(u)
        S[seen[u].nonzero()] = -np.inf          # never recommend what they already have
        tops.append(topk_indices(S, k)); rels += [rel[x] for x in u]
    return ranking_metrics(np.vstack(tops), rels, k)


def main():
    meta = json.loads((P / "meta.json").read_text())
    nu, ni = meta["n_users"], meta["n_items"]
    train, val = pd.read_parquet(P / "train.parquet"), pd.read_parquet(P / "val.parquet")
    X = sp.load_npz(P / "tfidf.npz")
    R = interaction_matrix(train, nu, ni)
    seen = interaction_matrix(train, nu, ni, like=0)       # every train interaction

    pop = Popularity().fit(R)
    con = Content().fit(R, X)
    t = time.time(); als = ALS().fit(R); print(f"ALS trained in {time.time()-t:.0f}s")

    rows = {"Popularity": evaluate(pop, seen, val, nu),
            "Content": evaluate(con, seen, val, nu),
            "ALS (CF)": evaluate(als, seen, val, nu)}
    best_w, best = 0.5, -1
    for w in (0.2, 0.4, 0.5, 0.6, 0.8):
        r = evaluate(Hybrid(als, con, w), seen, val, nu)
        rows[f"Hybrid w={w}"] = r
        if r["NDCG@10"] > best:
            best, best_w = r["NDCG@10"], w
    print(pd.DataFrame(rows).T.round(4).to_string())
    print("best hybrid weight (by NDCG@10 on validation):", best_w)

    M.mkdir(exist_ok=True)
    np.savez(M / "als.npz", U=als.U, V=als.V)
    (M / "hybrid.json").write_text(json.dumps({"w": best_w}))


if __name__ == "__main__":
    sys.exit(main())
