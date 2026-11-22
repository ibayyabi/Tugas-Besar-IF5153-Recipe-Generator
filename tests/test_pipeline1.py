"""Pipeline 1 regression guards for behaviour the annotated set only implies."""

from src.pipeline1.query_parser import parse_query


def test_exclusion_clause_removes_earlier_availability():
    result = parse_query("I have potatoes, butter and garlic. Without milk or butter.")
    assert result["available_ingredients"] == ["potato", "garlic"]
    assert result["excluded_ingredients"] == ["milk", "butter"]


def test_upper_bound_is_not_an_exclusion():
    result = parse_query("I have beef and noodles. No more than 700 calories.")
    assert result["excluded_ingredients"] == []
    assert result["calorie_max"] == 700
    assert result["available_ingredients"] == ["beef", "noodle"]


def test_bare_kcal_number_is_not_a_constraint():
    result = parse_query("I have chicken and rice. 450 kcal.")
    assert result["calorie_target"] is None
    assert result["calorie_max"] is None
    assert result["calorie_min"] is None


def test_range_sets_both_bounds():
    result = parse_query("I have chicken, potato, carrot and onion. Between 400 and 600 calories.")
    assert (result["calorie_min"], result["calorie_max"], result["calorie_target"]) == (400, 600, None)


def test_longest_match_wins_over_substring():
    result = parse_query("Available: tomatoes, onions, garlic, olive oil.")
    assert result["available_ingredients"] == ["tomato", "onion", "garlic", "olive oil"]


def test_plural_exclusion_and_generic_nut():
    result = parse_query("I have oats, banana, milk and honey. Around 350 calories, no nuts.")
    assert result["excluded_ingredients"] == ["nut"]
    assert result["calorie_target"] == 350


def test_tilde_approx_target_is_recognized():
    result = parse_query("i have chicken potato carrot onion garlic. ~500 calories")
    assert result["available_ingredients"] == ["chicken", "potato", "carrot", "onion", "garlic"]
    assert result["calorie_target"] == 500


def test_two_exclusion_cues_in_one_clause():
    result = parse_query("I have salmon and rice. No butter and no cream.")
    assert result["excluded_ingredients"] == ["butter", "cream"]


def test_query_without_pantry_still_parses():
    result = parse_query("I want a low calorie dinner.")
    assert result == {
        "available_ingredients": [],
        "excluded_ingredients": [],
        "calorie_target": None,
        "calorie_min": None,
        "calorie_max": None,
    }
 
def test_range_accepts_collapsed_symbol_separator():
    result = parse_query("I have rice. Between 300 & 500 cal.")
    assert (result["calorie_min"], result["calorie_max"]) == (300, 500)


def test_joined_compound_ingredient():
    result = parse_query("I have sweetpotato and spinach.")
    assert result["available_ingredients"] == ["sweet potato", "spinach"]
