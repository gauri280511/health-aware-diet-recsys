from src.glycemic import load_gi_table, match_ingredient, recipe_gi

ROWS = load_gi_table()


def test_longest_keyword_wins():
    assert match_ingredient("sweet potato", ROWS)[0] == 63      # not plain potato (78)


def test_ignored_entries_return_none():
    assert match_ingredient("coconut milk", ROWS) is None


def test_recipe_gi_weighted():
    gi, n = recipe_gi(["white rice", "chicken", "tomato"], ROWS)
    assert n == 2 and 60 < gi < 73          # rice (w4) dominates tomato (w1)


def test_no_match():
    assert recipe_gi(["chicken breast", "olive oil"], ROWS)[1] == 0
