"""Phase 4 - condition-aware health scoring.

Input : recipes_gl.parquet (per-serving nutrients + gl_est) and a user Profile
Output: for every recipe  (a) pass  = meets ALL of the user's limits  (hard check)
                          (b) score = 0..1, how comfortably inside the limits (soft score)

Soft score per constraint:  s = clip(1 - v / (2*L), 0, 1)
    v = 0 -> 1.0,  v = L (exactly at the limit) -> 0.5,  v >= 2L -> 0
Overall score = weighted mean of the s values (weights from the YAML).
Several conditions can be combined: for a shared feature the STRICTEST limit wins.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CONFIG = Path("configs/conditions.yaml")


def load_conditions(path=CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text())


@dataclass
class Profile:
    conditions: list = field(default_factory=lambda: ["none"])
    calorie_target: float | None = None      # kcal/day; overrides weight_loss default
    overrides: dict = field(default_factory=dict)   # {feature: per-meal limit}


def health_features(rec: pd.DataFrame) -> pd.DataFrame:
    cal = rec["calories"].clip(lower=1)
    return pd.DataFrame({
        "calories": rec["calories"],
        "gl_est": rec["gl_est"],
        "sodium_mg": rec["sodium_mg"],
        "sugar_pct_kcal": rec["sugar_g"] * 4 / cal * 100,
        "sat_fat_pct_kcal": rec["sat_fat_g"] * 9 / cal * 100,
    }, index=rec.index)


def resolve_limits(profile: Profile, cfg: dict) -> dict:
    """-> {feature: (per_meal_limit, weight)}; strictest limit across conditions."""
    share = cfg["meal_share"]
    out = {}
    for name in profile.conditions:
        for feat, c in cfg["conditions"][name]["constraints"].items():
            lim = c["limit"]
            if feat == "calories" and profile.calorie_target:
                lim = profile.calorie_target
            lim = lim * share if c["scale"] == "daily" else lim
            if feat not in out or lim < out[feat][0]:
                out[feat] = (lim, c["weight"])
    for feat, lim in profile.overrides.items():
        out[feat] = (lim, out.get(feat, (0, 1.0))[1])
    return out


def score_recipes(feats: pd.DataFrame, profile: Profile, cfg: dict | None = None):
    """Returns (pass_mask: bool array, score: float array in [0,1])."""
    cfg = cfg or load_conditions()
    limits = resolve_limits(profile, cfg)
    n = len(feats)
    if not limits:
        return np.ones(n, bool), np.ones(n)
    ok = np.ones(n, bool); num = np.zeros(n); den = 0.0
    for feat, (lim, w) in limits.items():
        v = feats[feat].to_numpy(dtype=float)
        ok &= v <= lim
        num += w * np.clip(1 - v / (2 * lim), 0, 1)
        den += w
    return ok, num / den


def explain(feats_row: pd.Series, profile: Profile, cfg: dict | None = None) -> list:
    """Human-readable per-limit report for one recipe (used by the UI)."""
    cfg = cfg or load_conditions()
    names = {"gl_est": "Glycemic load", "sodium_mg": "Sodium (mg)", "calories": "Calories",
             "sugar_pct_kcal": "Sugar (% kcal)", "sat_fat_pct_kcal": "Sat. fat (% kcal)"}
    return [{"nutrient": names[f], "value": round(float(feats_row[f]), 1),
             "limit": round(lim, 1), "ok": bool(feats_row[f] <= lim)}
            for f, (lim, _) in resolve_limits(profile, cfg).items()]


if __name__ == "__main__":
    P = Path("data/processed")
    rec = pd.read_parquet(P / "recipes_gl.parquet")
    feats = health_features(rec); cfg = load_conditions()
    train = pd.read_parquet(P / "train.parquet")
    liked = train[train.rating >= 4]
    print(f"{'condition':15s} {'all recipes':>12s} {'liked by users':>15s}")
    for c in ("diabetes", "hypertension", "heart_healthy", "weight_loss"):
        ok, _ = score_recipes(feats, Profile([c]), cfg)
        print(f"{c:15s} {ok.mean():12.1%} {ok[liked.item_idx.to_numpy()].mean():15.1%}")
    ok, _ = score_recipes(feats, Profile(["diabetes", "hypertension"]), cfg)
    print(f"{'diabetes+HTN':15s} {ok.mean():12.1%} {ok[liked.item_idx.to_numpy()].mean():15.1%}")
