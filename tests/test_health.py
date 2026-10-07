import pandas as pd
from src.health import Profile, health_features, load_conditions, resolve_limits, score_recipes

CFG = load_conditions()


def _rec(**kw):
    base = dict(calories=400.0, gl_est=8.0, sodium_mg=300.0, sugar_g=5.0, sat_fat_g=3.0)
    base.update(kw)
    return pd.DataFrame([base])


def test_no_condition_passes_everything():
    ok, s = score_recipes(health_features(_rec(sodium_mg=9999)), Profile(["none"]), CFG)
    assert ok[0] and s[0] == 1.0


def test_diabetes_gl_limit():
    lo, _ = score_recipes(health_features(_rec(gl_est=15)), Profile(["diabetes"]), CFG)
    hi, _ = score_recipes(health_features(_rec(gl_est=25)), Profile(["diabetes"]), CFG)
    assert lo[0] and not hi[0]


def test_sodium_scaled_to_meal():
    lim, _ = resolve_limits(Profile(["hypertension"]), CFG)["sodium_mg"]
    assert abs(lim - 2000 * CFG["meal_share"]) < 1e-6


def test_strictest_limit_wins_when_combined():
    both = resolve_limits(Profile(["heart_healthy", "hypertension"]), CFG)
    assert both["sodium_mg"][0] == 2000 * CFG["meal_share"]


def test_score_monotone_in_nutrient():
    f = health_features(pd.concat([_rec(sodium_mg=100), _rec(sodium_mg=500)], ignore_index=True))
    _, s = score_recipes(f, Profile(["hypertension"]), CFG)
    assert s[0] > s[1]


def test_calorie_override():
    lim, _ = resolve_limits(Profile(["weight_loss"], calorie_target=1200), CFG)["calories"]
    assert abs(lim - 1200 * CFG["meal_share"]) < 1e-6
