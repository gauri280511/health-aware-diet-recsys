"""Phase 2 - features and the train / validation / test split.

Reads : data/processed/recipes_gl.parquet, interactions.parquet
Writes: data/processed/{train,val,test}.parquet   (user_idx, item_idx, rating, date)
        data/processed/tfidf.npz                  (items x terms, L2-normalised, sparse)
        data/processed/nutrition.npy              (items x 8, standardised)
        data/processed/meta.json                  (index maps, scaler, sizes)

Row order of recipes_gl.parquet defines item_idx. Users are numbered in sorted order.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

P = Path("data/processed")
NUTR = ["calories", "fat_g", "sugar_g", "sodium_mg", "protein_g",
        "sat_fat_g", "carbs_g", "gl_est"]
LIKE_THRESHOLD = 4            # rating >= 4 counts as a positive
TEST_FRAC, VAL_FRAC = 0.20, 0.10

# Tags that describe dish type / cuisine / diet are useful content signals;
# 'time-to-make', '60-minutes-or-less' etc. are noise for taste.
STOP_TAGS = {"time-to-make", "course", "main-ingredient", "preparation", "occasion",
             "cuisine", "dietary", "equipment", "number-of-servings", "taste-mood"}


def build_text(row) -> str:
    """One 'document' per recipe: ingredient names + informative tags.
    Multi-word ingredients are glued with '_' so 'olive oil' stays one token."""
    ing = [i.strip().replace(" ", "_") for i in row.ingredients]
    tags = [t.replace(" ", "_") for t in row.tags
            if t not in STOP_TAGS and "minutes" not in t and "hours" not in t]
    return " ".join(ing + tags)


def build_tfidf(recipes: pd.DataFrame):
    docs = [build_text(r) for r in recipes.itertuples()]
    vec = TfidfVectorizer(token_pattern=r"\S+", min_df=5, max_df=0.5,
                          sublinear_tf=True, norm="l2")
    X = vec.fit_transform(docs)
    return X.tocsr(), vec


def build_nutrition(recipes: pd.DataFrame):
    """log1p tames the long right tail, then z-score. Keeps scaler for later use."""
    A = np.log1p(recipes[NUTR].to_numpy(dtype=float))
    mu, sd = A.mean(0), A.std(0) + 1e-9
    return ((A - mu) / sd).astype(np.float32), mu, sd


def time_split(it: pd.DataFrame, test_frac=TEST_FRAC, val_frac=VAL_FRAC):
    """Per-user chronological split: oldest -> train, newest -> test.
    Why not random? A random split lets the model 'see the future' of a user's
    taste, which inflates accuracy and is unrealistic for a live recommender."""
    it = it.sort_values(["user_idx", "date", "item_idx"]).copy()
    it["rank"] = it.groupby("user_idx").cumcount()
    n = it["user_idx"].map(it.groupby("user_idx").size())
    n_test = np.maximum(1, np.ceil(n * test_frac)).astype(int)
    n_val = np.maximum(1, np.round(n * val_frac)).astype(int)
    from_end = n - 1 - it["rank"]                 # 0 = most recent
    split = np.where(from_end < n_test, "test",
                     np.where(from_end < n_test + n_val, "val", "train"))
    it["split"] = split
    return (it[it.split == s].drop(columns=["rank", "split"]).reset_index(drop=True)
            for s in ("train", "val", "test"))


def main():
    rec = pd.read_parquet(P / "recipes_gl.parquet").reset_index(drop=True)
    it = pd.read_parquet(P / "interactions.parquet")

    item_of = {rid: k for k, rid in enumerate(rec.recipe_id)}
    users = np.sort(it.user_id.unique())
    user_of = {u: k for k, u in enumerate(users)}
    it = it.assign(user_idx=it.user_id.map(user_of), item_idx=it.recipe_id.map(item_of))
    it = it[["user_idx", "item_idx", "rating", "date"]]

    train, val, test = time_split(it)
    for name, d in (("train", train), ("val", val), ("test", test)):
        d.to_parquet(P / f"{name}.parquet")

    X, vec = build_tfidf(rec)
    sp.save_npz(P / "tfidf.npz", X)
    N, mu, sd = build_nutrition(rec)
    np.save(P / "nutrition.npy", N)

    meta = {"n_users": len(users), "n_items": len(rec), "n_terms": X.shape[1],
            "like_threshold": LIKE_THRESHOLD, "nutr_cols": NUTR,
            "nutr_mu": mu.tolist(), "nutr_sd": sd.tolist(),
            "user_ids": users.tolist(), "recipe_ids": rec.recipe_id.tolist()}
    (P / "meta.json").write_text(json.dumps(meta))

    print(f"users {len(users)}  items {len(rec)}  tfidf {X.shape}  nnz/item {X.nnz/X.shape[0]:.1f}")
    print(f"train {len(train)}  val {len(val)}  test {len(test)}")
    rel = test[test.rating >= LIKE_THRESHOLD]
    print(f"test users with >=1 relevant item: {rel.user_idx.nunique()} / {test.user_idx.nunique()}")
    cold = (~test.item_idx.isin(set(train.item_idx))).mean()
    print(f"share of test interactions on items unseen in train: {cold:.3f}")


if __name__ == "__main__":
    sys.exit(main())
