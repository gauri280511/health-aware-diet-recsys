"""Ranking metrics shared by Phase 3 (tuning) and Phase 6 (final evaluation)."""
import numpy as np


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Row-wise top-k item indices, best first. scores: (users, items)."""
    part = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, 1), axis=1)
    return np.take_along_axis(part, order, 1)


def ranking_metrics(top: np.ndarray, relevant: list, k: int) -> dict:
    """top: (users, k) recommended item ids. relevant: list of sets (one per user).
    Returns mean Precision@k, Recall@k, NDCG@k over users with >=1 relevant item."""
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    P, R, N = [], [], []
    for row, rel in zip(top, relevant):
        if not rel:
            continue
        hit = np.array([i in rel for i in row[:k]], dtype=float)
        P.append(hit.sum() / k)
        R.append(hit.sum() / len(rel))
        ideal = disc[: min(len(rel), k)].sum()
        N.append((hit * disc).sum() / ideal)
    return {f"P@{k}": np.mean(P), f"R@{k}": np.mean(R), f"NDCG@{k}": np.mean(N)}
