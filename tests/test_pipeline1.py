"""Pipeline 1 regression guards for behaviour the annotated set only implies."""

import pytest

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


def test_non_string_query_raises_clear_error():
    with pytest.raises(TypeError, match="query must be a string"):
        parse_query(None)  # type: ignore[arg-type]


def test_exclusion_stops_at_clause_boundaries():
    result = parse_query(
        "I have chicken, rice and garlic but I don't want cheese and I have potato"
    )
    assert result["available_ingredients"] == ["chicken", "rice", "garlic", "potato"]
    assert result["excluded_ingredients"] == ["cheese"]

    result = parse_query("I have chicken and rice. No cheese. I have potato.")
    assert result["available_ingredients"] == ["chicken", "rice", "potato"]
    assert result["excluded_ingredients"] == ["cheese"]


def test_discourse_no_does_not_swallow_the_pantry():
    result = parse_query("No idea, I have chicken and rice")
    assert result["available_ingredients"] == ["chicken", "rice"]
    assert result["excluded_ingredients"] == []

    result = parse_query("no problem, I have chicken")
    assert result["available_ingredients"] == ["chicken"]


def test_hyphenated_no_compound_is_not_an_exclusion_cue():
    result = parse_query("I have no-bake cookie ingredients and oats")
    assert "oat" in result["available_ingredients"]
    assert result["excluded_ingredients"] == []


def test_time_and_volume_units_are_not_calorie_bounds():
    result = parse_query("I have chicken and rice, cook under 20 minutes")
    assert result["calorie_max"] is None

    result = parse_query("I have flour and sugar, over 2 cups of flour")
    assert result["calorie_min"] is None


def test_count_bounds_are_not_calorie_bounds_or_exclusions():
    for query in ("I have eggs, no more than 3 eggs", "I have eggs, between 2 and 3 eggs"):
        result = parse_query(query)
        assert (result["calorie_min"], result["calorie_max"]) == (None, None)
        assert result["excluded_ingredients"] == []


def test_decimal_calorie_value_is_preserved():
    result = parse_query("I have chicken, under 20.5 calories")
    assert result["calorie_max"] == 20.5


def test_non_positive_calorie_values_are_ignored():
    negative = parse_query("I have chicken, under -500 calories")
    zero = parse_query("I have chicken, around 0 calories")
    assert negative["calorie_max"] is None
    assert zero["calorie_target"] is None


def test_calorie_suffix_and_range_phrasings():
    for query in (
        "I have chicken, 500 calories max",
        "I have chicken, 500 calories or less",
        "I have chicken, less than or equal to 500 calories",
        "I have chicken, not exceeding 500 calories",
    ):
        assert parse_query(query)["calorie_max"] == 500

    ranged = parse_query("I have chicken, 400 to 600 calories")
    assert (ranged["calorie_min"], ranged["calorie_max"]) == (400, 600)


def test_reversed_calorie_range_is_normalized():
    result = parse_query("I have chicken, between 700 and 500 calories")
    assert (result["calorie_min"], result["calorie_max"]) == (500, 700)


def test_additional_exclusion_cues():
    for phrase in ("I can't eat cheese", "allergic to cheese", "except cheese", "skip the cheese"):
        result = parse_query(f"I have chicken, {phrase}")
        assert result["available_ingredients"] == ["chicken"]
        assert result["excluded_ingredients"] == ["cheese"]


def test_free_suffix_exclusions():
    assert parse_query("I have chicken, cheese-free")["excluded_ingredients"] == ["cheese"]
    assert parse_query("I have chicken, sugar free")["excluded_ingredients"] == ["sugar"]


def test_dont_have_is_an_exclusion():
    result = parse_query("I don't have cheese, I have chicken")
    assert result["available_ingredients"] == ["chicken"]
    assert result["excluded_ingredients"] == ["cheese"]


def test_unitless_count_before_known_ingredient_is_not_a_calorie_bound():
    for query in (
        "I have potatoes, under 2 potatoes",
        "I have chicken, around 2 eggs",
        "I have chicken, over 3 carrots",
    ):
        result = parse_query(query)
        assert (result["calorie_target"], result["calorie_min"], result["calorie_max"]) == (
            None,
            None,
            None,
        )

    # Bare shorthand without a following ingredient is still a calorie bound.
    assert parse_query("I have chicken, under 500")["calorie_max"] == 500


def test_comma_list_exclusion_keeps_all_items():
    result = parse_query("I have chicken, no cheese, milk")
    assert result["available_ingredients"] == ["chicken"]
    assert result["excluded_ingredients"] == ["cheese", "milk"]

    result = parse_query("I have chicken, no cheese, milk and butter")
    assert result["excluded_ingredients"] == ["cheese", "milk", "butter"]

    result = parse_query("I have chicken, no cheese, I have potato")
    assert result["available_ingredients"] == ["chicken", "potato"]
    assert result["excluded_ingredients"] == ["cheese"]

    result = parse_query("I have chicken, no cheese, milk. I have potato.")
    assert result["available_ingredients"] == ["chicken", "potato"]
    assert result["excluded_ingredients"] == ["cheese", "milk"]

    result = parse_query("I have chicken, no cheese, milk, under 500 calories")
    assert result["excluded_ingredients"] == ["cheese", "milk"]
    assert result["calorie_max"] == 500
