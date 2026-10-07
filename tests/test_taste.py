import numpy as np
import scipy.sparse as sp
from src.taste_models import ALS, Popularity, Content, zscore_rows
from src.metrics import ranking_metrics


def _R():
    # users 0,1 like items 0,1 ; users 2,3 like items 2,3
    d = np.zeros((4, 4), dtype=np.float32); d[:2, :2] = 1; d[2:, 2:] = 1
    d[0, 1] = 0                      # hide one item for user 0
    return sp.csr_matrix(d)


def test_als_recovers_block_structure():
    m = ALS(factors=4, iters=15).fit(_R())
    s = m.scores(np.array([0]))[0]
    assert s[1] > s[2] and s[1] > s[3]     # hidden item 1 outranks other-block items


def test_content_prefers_similar_item():
    X = sp.csr_matrix(np.array([[1, 0], [0.9, 0.1], [0, 1.0]], dtype=np.float32))
    R = sp.csr_matrix(np.array([[1, 0, 0]], dtype=np.float32))
    s = Content().fit(R, X).scores(np.array([0]))[0]
    assert s[1] > s[2]


def test_metrics_perfect_ranking():
    r = ranking_metrics(np.array([[5, 6, 7]]), [{5, 6}], 3)
    assert r["P@3"] == 2 / 3 and r["R@3"] == 1.0 and abs(r["NDCG@3"] - 1) < 1e-9
