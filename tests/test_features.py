import pandas as pd
from src.features import time_split, build_text


def _toy():
    rows = []
    for u in range(3):
        for k in range(10):
            rows.append((u, k, 5, pd.Timestamp("2020-01-01") + pd.Timedelta(days=k)))
    return pd.DataFrame(rows, columns=["user_idx", "item_idx", "rating", "date"])


def test_split_is_chronological_and_disjoint():
    tr, va, te = time_split(_toy())
    for u in range(3):
        a, b, c = (d[d.user_idx == u] for d in (tr, va, te))
        assert a.date.max() < b.date.min() <= b.date.max() < c.date.min()
        assert len(a) + len(b) + len(c) == 10
        assert len(c) == 2 and len(b) == 1       # ceil(20%)=2, round(10%)=1


def test_text_glues_multiword_ingredients():
    class R: ingredients = ["Olive Oil".lower(), "salt"]; tags = ["vegetarian", "time-to-make", "30-minutes-or-less"]
    assert build_text(R) == "olive_oil salt vegetarian"
