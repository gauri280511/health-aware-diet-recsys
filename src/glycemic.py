"""Phase 2a - glycemic index (GI) / glycemic load (GL) estimation.

Food.com gives ingredient NAMES but not quantities, so a per-recipe GL cannot be
computed exactly. We therefore estimate:

  recipe_GI = weighted mean of the GI of matched ingredients
              (weight = how much a typical portion of that food contributes carbs;
               see configs/gi_table.csv 'weight' column: staples 4 ... veg 1)
  GL        = recipe_GI * carbs_per_serving_g / 100

Carbs per serving come from the dataset (reliable); GI comes from the lookup table
(approximate). Recipes with no matched ingredient get the table-wide median GI and
are flagged gi_matched = False so they can be excluded or analysed separately.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

TABLE = Path("configs/gi_table.csv")


def load_gi_table(path=TABLE):
    t = pd.read_csv(path)
    # longest keyword first so "sweet potato" wins over "potato"
    t = t.assign(n=t.keyword.str.len()).sort_values("n", ascending=False)
    rows = []
    for r in t.itertuples():
        pat = re.compile(r"\b" + re.escape(r.keyword) + r"\b")
        rows.append((pat, r.gi, r.weight))   # gi/weight NaN => ignore entry
    return rows


def match_ingredient(name, rows):
    """Return (gi, weight) for the first (longest) keyword found, else None."""
    for pat, gi, w in rows:
        if pat.search(name):
            return None if np.isnan(gi) else (float(gi), float(w))
    return None


def recipe_gi(ingredients, rows):
    matched = [m for m in (match_ingredient(i.lower(), rows) for i in ingredients) if m]
    if not matched:
        return np.nan, 0
    gi = np.array([m[0] for m in matched]); w = np.array([m[1] for m in matched])
    return float((gi * w).sum() / w.sum()), len(matched)


def add_glycemic_features(recipes: pd.DataFrame) -> pd.DataFrame:
    rows = load_gi_table()
    res = [recipe_gi(ing, rows) for ing in recipes["ingredients"]]
    out = recipes.copy()
    out["gi_est"] = [r[0] for r in res]
    out["gi_n_matched"] = [r[1] for r in res]
    out["gi_matched"] = out["gi_n_matched"] > 0
    out["gi_est"] = out["gi_est"].fillna(out["gi_est"].median())
    out["gl_est"] = out["gi_est"] * out["carbs_g"] / 100.0
    return out


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/recipes.parquet")
    df = add_glycemic_features(df)
    print("share of recipes with >=1 GI-matched ingredient:", df.gi_matched.mean().round(3))
    print(df[["gi_est", "carbs_g", "gl_est"]].describe().round(1))
    # Standard GL bands per serving: low <=10, medium 11-19, high >=20
    bands = pd.cut(df.gl_est, [-1, 10, 19.999, 1e9], labels=["low", "medium", "high"])
    print(bands.value_counts(normalize=True).round(3))
    df.to_parquet("data/processed/recipes_gl.parquet")
