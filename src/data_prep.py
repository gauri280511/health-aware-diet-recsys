"""Phase 1 - load, clean and convert the raw Food.com data.

Reads : data/raw/RAW_recipes.csv, data/raw/RAW_interactions.csv
Writes: data/processed/recipes.parquet, data/processed/interactions.parquet

Food.com stores nutrition as a list:
  [calories, total_fat_PDV, sugar_PDV, sodium_PDV, protein_PDV, sat_fat_PDV, carbs_PDV]
PDV = % of daily value. We convert to grams (mg for sodium) with the daily values
below. These were checked against the 4/9/4 kcal-per-gram rule (see README).
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed")

# Daily values used to turn %DV into absolute amounts.
DV = {"fat_g": 65, "sugar_g": 50, "sodium_mg": 2400,
      "protein_g": 50, "sat_fat_g": 20, "carbs_g": 300}

NUTR_COLS = ["calories", "fat_g", "sugar_g", "sodium_mg", "protein_g", "sat_fat_g", "carbs_g"]

# Plausibility limits for ONE serving; recipes outside are data errors.
MAX_CAL, MAX_CARBS, MAX_SODIUM = 2500, 300, 6000


def load_recipes(path=RAW / "RAW_recipes.csv") -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["name", "id", "minutes", "tags", "nutrition",
                                    "n_ingredients", "ingredients"])
    n = np.array(df["nutrition"].map(ast.literal_eval).tolist(), dtype=float)
    df["calories"] = n[:, 0]
    df["fat_g"] = n[:, 1] / 100 * DV["fat_g"]
    df["sugar_g"] = n[:, 2] / 100 * DV["sugar_g"]
    df["sodium_mg"] = n[:, 3] / 100 * DV["sodium_mg"]
    df["protein_g"] = n[:, 4] / 100 * DV["protein_g"]
    df["sat_fat_g"] = n[:, 5] / 100 * DV["sat_fat_g"]
    df["carbs_g"] = n[:, 6] / 100 * DV["carbs_g"]
    df["ingredients"] = df["ingredients"].map(ast.literal_eval)
    df["tags"] = df["tags"].map(ast.literal_eval)
    df = df.drop(columns="nutrition").rename(columns={"id": "recipe_id"})
    return df


def clean_recipes(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.dropna(subset=["name"]).drop_duplicates("recipe_id")
    ok = ((df.calories > 20) & (df.calories <= MAX_CAL) &
          (df.carbs_g <= MAX_CARBS) & (df.sodium_mg <= MAX_SODIUM) &
          (df.n_ingredients > 0))
    df = df[ok].reset_index(drop=True)
    print(f"recipes: {n0} -> {len(df)} after cleaning")
    return df


def load_interactions(recipe_ids, path=RAW / "RAW_interactions.csv",
                      min_user=5, min_item=5) -> pd.DataFrame:
    it = pd.read_csv(path, usecols=["user_id", "recipe_id", "date", "rating"])
    it["date"] = pd.to_datetime(it["date"])
    it = it[it.recipe_id.isin(set(recipe_ids))]
    it = it[it.rating > 0]                      # rating 0 = "no rating given"
    it = it.drop_duplicates(["user_id", "recipe_id"], keep="last")
    # iterative k-core filter: keep users/items with enough interactions
    while True:
        u = it.user_id.value_counts(); i = it.recipe_id.value_counts()
        keep = it.user_id.map(u).ge(min_user) & it.recipe_id.map(i).ge(min_item)
        if keep.all():
            break
        it = it[keep]
    print(f"interactions: {len(it)}  users: {it.user_id.nunique()}  recipes: {it.recipe_id.nunique()}")
    return it.reset_index(drop=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rec = clean_recipes(load_recipes())
    it = load_interactions(rec.recipe_id)
    rec = rec[rec.recipe_id.isin(set(it.recipe_id))].reset_index(drop=True)
    rec.to_parquet(OUT / "recipes.parquet")
    it.to_parquet(OUT / "interactions.parquet")
    print("saved to", OUT)


if __name__ == "__main__":
    sys.exit(main())
