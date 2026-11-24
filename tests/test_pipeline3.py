"""Unit and integration tests for Pipeline 3 (Pantry-Aware Multi-Criteria Reranking).

Compatible with both ``pytest`` and standard-library ``python -m unittest``.
"""
from __future__ import annotations

import ast
import csv
import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline3 import (
    CandidateRecipe,
    ExclusionFilter,
    IngredientNormalizer,
    MultiCriteriaReranker,
    PantryStapleWeights,
    RankingConfig,
    RecipeCalorieEstimator,
    StructuredQuery,
    compute_coverage,
    evaluate_calorie_compatibility,
    extract_servings,
    format_recommendation_card,
    format_recommendations,
    normalize_ingredient,
    parse_ingredient_line,
    recommend,
    to_dict,
)

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
SAMPLE_CSV = REPO_ROOT / "data" / "samples" / "recipes_sample.csv"
ANNOTATED_JSON = REPO_ROOT / "data" / "test" / "annotated_queries.json"


class TestNormalization(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.normalizer = IngredientNormalizer(PROCESSED_DIR)

    def test_plural_folding(self) -> None:
        self.assertEqual(normalize_ingredient("potatoes"), "potato")
        self.assertEqual(normalize_ingredient("tomatoes"), "tomato")
        self.assertEqual(normalize_ingredient("eggs"), "egg")
        self.assertEqual(normalize_ingredient("berries"), "berry")
        self.assertEqual(normalize_ingredient("asparagus"), "asparagus")

    def test_alias_resolution_and_qualifiers(self) -> None:
        self.assertEqual(self.normalizer.canonicalize("scallions"), "green onion")
        self.assertEqual(self.normalizer.canonicalize("capsicum"), "bell pepper")
        self.assertEqual(self.normalizer.canonicalize("white sugar"), "sugar")
        self.assertEqual(self.normalizer.canonicalize("hamburger"), "ground beef")
        self.assertEqual(self.normalizer.canonicalize("fresh minced garlic"), "garlic")
        self.assertEqual(self.normalizer.canonicalize("kosher salt"), "salt")


class TestCoverageAndPantryStaples(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.normalizer = IngredientNormalizer(PROCESSED_DIR)
        cls.staples = PantryStapleWeights.from_csv(
            PROCESSED_DIR / "pantry_staples.csv",
            normalizer=cls.normalizer,
        )

    def test_pantry_staple_weights_loaded_from_csv(self) -> None:
        self.assertAlmostEqual(self.staples.get_weight("water"), 0.05)
        self.assertAlmostEqual(self.staples.get_weight("salt"), 0.15)
        self.assertAlmostEqual(self.staples.get_weight("pepper"), 0.15)
        self.assertAlmostEqual(self.staples.get_weight("cream"), 0.80)
        # Unknown/main ingredients default to 1.0 (data/README.md)
        self.assertAlmostEqual(self.staples.get_weight("chicken"), 1.0)
        self.assertAlmostEqual(self.staples.get_weight("potato"), 1.0)
        self.assertTrue(self.staples.is_staple("salt"))
        self.assertFalse(self.staples.is_staple("chicken"))

    def test_prd_section19_unweighted_coverage_example(self) -> None:
        # PRD §19: U={chicken, potato, carrot, onion, garlic}, R=U+{celery} -> 5/6 = 0.833
        user = ["chicken", "potato", "carrot", "onion", "garlic"]
        recipe = ["chicken", "potato", "carrot", "onion", "garlic", "celery"]
        cov = compute_coverage(
            user,
            recipe,
            staple_weights=self.staples,
            normalizer=self.normalizer,
            use_weights=False,
        )
        self.assertAlmostEqual(cov.unweighted_coverage, 5 / 6, places=4)
        self.assertAlmostEqual(cov.coverage_score, 5 / 6, places=4)
        self.assertAlmostEqual(cov.unweighted_missing_penalty, 1 / 6, places=4)
        self.assertEqual(cov.missing_ingredients, ["celery"])
        self.assertEqual(len(cov.available_matched), 5)

    def test_weighted_coverage_downweights_missing_staples(self) -> None:
        # Missing only salt+water should score much higher than missing celery+cream
        user = ["chicken", "potato"]
        recipe_missing_staples = ["chicken", "potato", "salt", "water"]
        recipe_missing_mains = ["chicken", "potato", "celery", "cream"]

        cov_staples = compute_coverage(
            user, recipe_missing_staples, self.staples, self.normalizer, use_weights=True
        )
        cov_mains = compute_coverage(
            user, recipe_missing_mains, self.staples, self.normalizer, use_weights=True
        )

        # Unweighted coverages are identical (2/4 = 0.5)
        self.assertAlmostEqual(cov_staples.unweighted_coverage, 0.5)
        self.assertAlmostEqual(cov_mains.unweighted_coverage, 0.5)

        # Weighted coverage strongly favors recipe missing only staples
        # staples: (1+1)/(1+1+0.15+0.05) = 2.0 / 2.2 = 0.9091
        # mains  : (1+1)/(1+1+1.0+0.80)  = 2.0 / 3.8 = 0.5263
        self.assertAlmostEqual(cov_staples.weighted_coverage, 2.0 / 2.2, places=4)
        self.assertAlmostEqual(cov_mains.weighted_coverage, 2.0 / 3.8, places=4)
        self.assertGreater(cov_staples.coverage_score, cov_mains.coverage_score)
        self.assertLess(cov_staples.missing_penalty, cov_mains.missing_penalty)
        self.assertEqual(cov_staples.missing_non_staples, [])
        self.assertEqual(set(cov_staples.missing_staples), {"salt", "water"})

    def test_empty_recipe_ingredients_handled_safely(self) -> None:
        cov = compute_coverage(["chicken"], [], self.staples, self.normalizer)
        self.assertEqual(cov.coverage_score, 0.0)
        self.assertEqual(cov.missing_penalty, 0.0)
        self.assertEqual(cov.missing_ingredients, [])


class TestCalorieCompatibility(unittest.TestCase):
    def test_prd_section22_target_examples(self) -> None:
        # PRD §22.2: target=500 -> 490: 0.98, 450: 0.90, 300: 0.60
        c_a = evaluate_calorie_compatibility(490, calorie_target=500)
        c_b = evaluate_calorie_compatibility(450, calorie_target=500)
        c_c = evaluate_calorie_compatibility(300, calorie_target=500)
        c_far = evaluate_calorie_compatibility(1200, calorie_target=500)

        self.assertAlmostEqual(c_a.score, 0.98, places=4)
        self.assertAlmostEqual(c_b.score, 0.90, places=4)
        self.assertAlmostEqual(c_c.score, 0.60, places=4)
        self.assertEqual(c_far.score, 0.0)
        self.assertTrue(c_far.passed_hard_filter)  # target is soft

    def test_max_calorie_hard_filter(self) -> None:
        ok = evaluate_calorie_compatibility(480, calorie_max=500)
        bad = evaluate_calorie_compatibility(650, calorie_max=500)

        self.assertTrue(ok.passed_hard_filter)
        self.assertEqual(ok.score, 1.0)
        self.assertEqual(ok.calorie_distance, 0.0)

        self.assertFalse(bad.passed_hard_filter)
        self.assertAlmostEqual(bad.calorie_distance, 150.0)
        self.assertAlmostEqual(bad.score, 1.0 - 150.0 / 500.0, places=4)

    def test_min_and_range_calorie_constraints(self) -> None:
        inside = evaluate_calorie_compatibility(500, calorie_min=400, calorie_max=600)
        below = evaluate_calorie_compatibility(300, calorie_min=400, calorie_max=600)
        above = evaluate_calorie_compatibility(700, calorie_min=400, calorie_max=600)
        min_only_ok = evaluate_calorie_compatibility(650, calorie_min=600)
        min_only_low = evaluate_calorie_compatibility(450, calorie_min=600)

        self.assertTrue(inside.passed_hard_filter)
        self.assertEqual(inside.score, 1.0)
        self.assertEqual(inside.calorie_distance, 0.0)

        self.assertFalse(below.passed_hard_filter)
        self.assertEqual(below.calorie_distance, 100.0)
        self.assertFalse(above.passed_hard_filter)
        self.assertEqual(above.calorie_distance, 100.0)

        self.assertTrue(min_only_ok.passed_hard_filter)
        self.assertFalse(min_only_low.passed_hard_filter)

    def test_no_constraint_and_unknown_calories(self) -> None:
        none_c = evaluate_calorie_compatibility(500)
        self.assertEqual(none_c.constraint_type, "none")
        self.assertEqual(none_c.score, 1.0)
        self.assertTrue(none_c.passed_hard_filter)

        unk_soft = evaluate_calorie_compatibility(None, calorie_target=500)
        self.assertTrue(unk_soft.passed_hard_filter)
        self.assertAlmostEqual(unk_soft.score, 0.5)

        unk_hard = evaluate_calorie_compatibility(None, calorie_max=500)
        self.assertFalse(unk_hard.passed_hard_filter)

    def test_bounds_sanitize_nan_swap_and_non_positive(self) -> None:
        # min > max is swapped rather than producing an empty range
        swapped = evaluate_calorie_compatibility(500, calorie_min=600, calorie_max=400)
        self.assertEqual(swapped.constraint_type, "range")
        self.assertTrue(swapped.passed_hard_filter)
        self.assertEqual(swapped.score, 1.0)

        # <= 0 bounds are treated as absent
        zero_bounds = evaluate_calorie_compatibility(500, calorie_min=0, calorie_max=-5)
        self.assertEqual(zero_bounds.constraint_type, "none")

        # NaN calories behave like unknown calories
        nan_cal = evaluate_calorie_compatibility(float("nan"), calorie_max=500)
        self.assertIsNone(nan_cal.estimated_calories)
        self.assertFalse(nan_cal.passed_hard_filter)

        nan_target = evaluate_calorie_compatibility(500, calorie_target=float("nan"))
        self.assertEqual(nan_target.constraint_type, "none")


class TestExclusionFilter(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.normalizer = IngredientNormalizer(PROCESSED_DIR)
        cls.ef = ExclusionFilter(normalizer=cls.normalizer)

    def test_exact_and_category_cheese_exclusions(self) -> None:
        # "no cheese" must catch all cheese subtypes even though alias maps cheese->cheddar cheese
        for ing in ("cheese", "cheddar cheese", "mozzarella", "parmesan cheese", "cream cheese", "ricotta cheese"):
            res = self.ef.check(["cheese"], ["chicken", "tomato", ing])
            self.assertTrue(res.violated, msg=f"Expected 'no cheese' to catch {ing!r}")
            self.assertEqual(res.violation_count, 1)

    def test_nut_category_and_false_positive_guards(self) -> None:
        self.assertTrue(self.ef.check(["nut"], ["oats", "banana", "walnuts"]).violated)
        self.assertTrue(self.ef.check(["nut"], ["oats", "banana", "almond"]).violated)
        self.assertTrue(self.ef.check(["nut"], ["oats", "banana", "mixed nuts"]).violated)
        # False-positive guards: nutmeg and butternut squash are NOT nuts
        self.assertFalse(self.ef.check(["nut"], ["oats", "banana", "nutmeg"]).violated)
        self.assertFalse(self.ef.check(["nut"], ["butternut squash", "onion"]).violated)

    def test_butter_and_cream_false_positive_guards(self) -> None:
        self.assertTrue(self.ef.check(["butter"], ["potato", "unsalted butter"]).violated)
        self.assertFalse(self.ef.check(["butter"], ["bread", "peanut butter"]).violated)
        self.assertFalse(self.ef.check(["butter"], ["flour", "buttermilk"]).violated)

        self.assertTrue(self.ef.check(["cream"], ["pasta", "heavy cream"]).violated)
        self.assertFalse(self.ef.check(["cream"], ["egg white", "cream of tartar"]).violated)

    def test_title_and_raw_mentions_are_checked(self) -> None:
        # A cheese mention in the title alone must violate "no cheese"
        self.assertTrue(self.ef.check(["cheese"], ["Mac & Cheese"]).violated)
        # Raw ingredient lines (not yet canonicalized) must also count
        self.assertTrue(self.ef.check(["cheese"], ["1 cup shredded cheddar cheese"]).violated)
        # Non-string entries are skipped, never crash
        self.assertFalse(self.ef.check(["cheese"], [None, 42, ""]).violated)
        self.assertFalse(self.ef.check([None, 3], ["chicken", "rice"]).violated)


class TestMultiCriteriaReranker(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reranker = MultiCriteriaReranker(processed_dir=PROCESSED_DIR)

    def test_prd_section25_exact_formula(self) -> None:
        # PRD §25 numerical example:
        # 0.30(0.78) + 0.35(0.90) + 0.20(0.96) - 0.15(0.10) = 0.726
        # Construct 10 unweighted ingredients (9 matched, 1 missing -> cov=0.90, miss=0.10)
        # and calorie=480 with target=500 -> cal_score=0.96
        user_ings = ["chicken", "potato", "carrot", "broccoli", "spinach", "zucchini", "celery", "beef", "tomato"]
        recipe_ings = user_ings + ["salmon"]  # 9/10 = 0.90 unweighted
        query = StructuredQuery(
            available_ingredients=user_ings,
            excluded_ingredients=["cheese"],
            calorie_target=500,
        )
        cand = CandidateRecipe(
            recipe_id="prd_ex",
            title="Candidate A",
            canonical_ingredients=recipe_ings,
            tfidf_score=0.78,
            total_calories=480.0,
        )
        cfg = RankingConfig(use_pantry_staple_weights=False, use_weighted_missing_penalty=False)
        scored = self.reranker.score_candidate(cand, query, config=cfg)

        self.assertAlmostEqual(scored.coverage.coverage_score, 0.90, places=4)
        self.assertAlmostEqual(scored.coverage.missing_penalty, 0.10, places=4)
        self.assertAlmostEqual(scored.calorie.score, 0.96, places=4)
        self.assertFalse(scored.exclusion.violated)
        self.assertAlmostEqual(scored.final_score, 0.726, places=4)

    def test_reranker_beats_tfidf_baseline_on_pantry_usability(self) -> None:
        query = StructuredQuery(
            available_ingredients=["chicken", "potato", "carrot", "onion", "garlic"],
            excluded_ingredients=["cheese"],
            calorie_target=500,
        )
        candidates = [
            # Highest TF-IDF, but has excluded cheese!
            CandidateRecipe(
                recipe_id=1,
                title="Cheesy Chicken Bake",
                canonical_ingredients=["chicken", "potato", "cheddar cheese", "cream"],
                tfidf_score=0.95,
                total_calories=510,
            ),
            # High TF-IDF, but needs 5 extra main ingredients and way off calories
            CandidateRecipe(
                recipe_id=2,
                title="Loaded Feast Casserole",
                canonical_ingredients=[
                    "chicken", "potato", "carrot", "bacon", "sour cream",
                    "mushroom", "celery", "broccoli", "pasta",
                ],
                tfidf_score=0.88,
                total_calories=1200,
            ),
            # Slightly lower TF-IDF, but high coverage (only missing celery) and ~500 kcal!
            CandidateRecipe(
                recipe_id=3,
                title="Chicken Potato Soup",
                canonical_ingredients=["chicken", "potato", "carrot", "onion", "garlic", "celery"],
                tfidf_score=0.75,
                total_calories=482,
            ),
        ]

        baseline_res = self.reranker.rerank(candidates, query, top_k=2, config=RankingConfig.ablation_model_a())
        proposed_res = self.reranker.rerank(candidates, query, top_k=2, config=RankingConfig.ablation_model_d())

        self.assertEqual(baseline_res.recommendations[0].recipe.recipe_id, 1)
        self.assertEqual(proposed_res.recommendations[0].recipe.recipe_id, 3)
        self.assertEqual(proposed_res.filtered_out_exclusion, 1)
        self.assertFalse(proposed_res.is_fallback)
        # Verify cheese recipe is excluded
        returned_ids = [r.recipe.recipe_id for r in proposed_res.recommendations]
        self.assertNotIn(1, returned_ids)

    def test_risk6_fallback_relaxes_calories_but_never_exclusions(self) -> None:
        query = StructuredQuery(
            available_ingredients=["chicken", "rice"],
            excluded_ingredients=["cheese"],
            calorie_max=300,
        )
        candidates = [
            # Violates calorie_max (450 > 300), but respects exclusion
            CandidateRecipe(
                recipe_id="cal_over",
                title="Chicken Rice Bowl",
                canonical_ingredients=["chicken", "rice", "salt"],
                tfidf_score=0.80,
                total_calories=450,
            ),
            # Violates exclusion (cheese), even though calories <= 300
            CandidateRecipe(
                recipe_id="has_cheese",
                title="Cheesy Chicken Rice",
                canonical_ingredients=["chicken", "rice", "cheddar cheese"],
                tfidf_score=0.90,
                total_calories=280,
            ),
        ]

        res = self.reranker.rerank(candidates, query, top_k=1)
        self.assertTrue(res.is_fallback)
        self.assertIn("No exact recommendation found", res.fallback_message or "")
        self.assertEqual(res.relaxed_constraints, ["calorie_bounds"])
        # Tier 1 must pick calorie-over candidate rather than cheese-violating candidate!
        self.assertEqual(res.recommendations[0].recipe.recipe_id, "cal_over")
        self.assertTrue(res.recommendations[0].is_fallback_candidate)

        # Exclusions are hard: if every candidate violates one, return no result.
        res_tier2 = self.reranker.rerank([candidates[1]], query, top_k=1)
        self.assertFalse(res_tier2.is_fallback)
        self.assertEqual(res_tier2.recommendations, [])
        self.assertIn("excluded", (res_tier2.no_match_reason or "").lower())

    def test_explainable_formatting_and_serialization(self) -> None:
        query = StructuredQuery(
            available_ingredients=["chicken", "potato", "carrot", "onion", "garlic"],
            excluded_ingredients=["cheese"],
            calorie_target=500,
            raw_query="I have chicken, potato, carrot, onion and garlic. Around 500 calories. No cheese.",
        )
        cand = CandidateRecipe(
            recipe_id=123,
            title="Chicken Potato Soup",
            canonical_ingredients=["chicken", "potato", "carrot", "onion", "garlic", "celery", "salt"],
            tfidf_score=0.78,
            total_calories=482.0,
            ingredients=["2 chicken breasts", "2 potatoes", "1 carrot", "1 onion", "2 cloves garlic", "1 stalk celery", "1 tsp salt"],
            directions=["Combine all ingredients in a pot.", "Simmer 30 minutes."],
        )
        result = self.reranker.rerank([cand], query, top_k=3)
        card = format_recommendation_card(
            result.recommendations[0],
            excluded_ingredients=query.excluded_ingredients,
        )
        full_text = format_recommendations(result)
        payload = to_dict(result)

        self.assertIn("#1 Chicken Potato Soup", card)
        self.assertIn("Overall Match", card)
        self.assertIn("Ingredient Coverage", card)
        self.assertIn("Estimated Calories  : 482 kcal", card)
        self.assertIn("YOU HAVE:", card)
        self.assertIn("MISSING:", card)
        self.assertIn("celery", card)
        self.assertIn("salt (pantry staple)", card)
        self.assertIn("no cheese detected", card)
        self.assertIn("no cheese detected", full_text)
        self.assertIn("DIRECTIONS:", full_text)
        self.assertEqual(payload["recommendations"][0]["recipe_id"], 123)

    def test_pipeline1_and_pipeline2_raw_dict_contract(self) -> None:
        # Exact output schema of Pipeline 1 parse_query() (pipeline1-ibay)
        p1_dict = {
            "available_ingredients": ["chicken", "potato", "carrot", "onion", "garlic"],
            "excluded_ingredients": ["cheese"],
            "calorie_target": None,
            "calorie_min": None,
            "calorie_max": 600,
        }
        # Exact output schema of Pipeline 2 CandidateRetriever.retrieve() (pipeline3-hakim)
        p2_candidates = [
            {
                "recipe_id": 101,
                "title": "Unmapped Zero-Calorie Recipe",
                "ingredients": ["1 mystery spice"],
                "canonical_ingredients": ["chicken", "potato"],
                "directions": ["Cook."],
                "total_calories": 0.0,
                "calorie_coverage": 0.0,
                "tfidf_score": 0.92,
            },
            {
                "recipe_id": 102,
                "title": "Cheesy Chicken Gratin",
                "ingredients": ["2 chicken breasts", "1 cup cheddar cheese"],
                "canonical_ingredients": ["chicken", "cheddar cheese", "potato"],
                "directions": ["Bake."],
                "total_calories": 520.0,
                "calorie_coverage": 1.0,
                "tfidf_score": 0.90,
            },
            {
                "recipe_id": 103,
                "title": "Simple Chicken Potato Soup",
                "ingredients": ["2 chicken breasts", "2 potatoes", "1 carrot", "1 onion", "2 cloves garlic"],
                "canonical_ingredients": ["chicken", "potato", "carrot", "onion", "garlic", "salt"],
                "directions": ["Boil and simmer.", "Serves 4."],
                "total_calories": 490.0,
                "calorie_coverage": 0.95,
                "tfidf_score": 0.81,
            },
            {
                "recipe_id": 104,
                "title": "NaN Calorie Edge Case",
                "ingredients": ["1 chicken"],
                "canonical_ingredients": ["chicken"],
                "directions": ["Roast."],
                "total_calories": float("nan"),
                "calorie_coverage": 0.0,
                "tfidf_score": 0.70,
            },
        ]

        res = recommend(p2_candidates, p1_dict, top_k=3, reranker=self.reranker)
        # Strict match is ONLY recipe 103 (101 & 104 have unavailable calories under max=600, 102 has cheese)
        self.assertEqual(res.strict_match_count, 1)
        self.assertEqual(res.filtered_out_exclusion, 1)
        self.assertEqual(res.filtered_out_calorie, 2)
        self.assertEqual(res.recommendations[0].recipe.recipe_id, 103)
        self.assertFalse(res.recommendations[0].is_fallback_candidate)

        payload = to_dict(res)
        self.assertAlmostEqual(payload["recommendations"][0]["calorie_coverage"], 0.95)
        self.assertEqual(payload["recommendations"][0]["estimated_calories"], 490.0)

    def test_optional_per_serving_normalization_from_directions(self) -> None:
        query = {
            "available_ingredients": ["chicken", "potato"],
            "excluded_ingredients": [],
            "calorie_target": 500,
            "calorie_min": None,
            "calorie_max": None,
        }
        cand = {
            "recipe_id": 201,
            "title": "Family Roast Chicken",
            "ingredients": ["1 whole chicken", "4 potatoes"],
            "canonical_ingredients": ["chicken", "potato"],
            "directions": ["Roast for 1 hour.", "Serves 4."],
            "total_calories": 2000.0,
            "calorie_coverage": 1.0,
            "tfidf_score": 0.80,
        }
        cfg_total = RankingConfig(normalize_servings_from_directions=False)
        cfg_serving = RankingConfig(normalize_servings_from_directions=True)

        scored_total = self.reranker.score_candidate(cand, query, config=cfg_total)
        scored_serving = self.reranker.score_candidate(cand, query, config=cfg_serving)

        # Whole recipe (2000 vs target 500 -> score 0.0); per serving (2000/4 = 500 -> score 1.0)
        self.assertEqual(scored_total.calorie.estimated_calories, 2000.0)
        self.assertEqual(scored_total.calorie.score, 0.0)
        self.assertEqual(scored_serving.calorie.estimated_calories, 500.0)
        self.assertEqual(scored_serving.calorie.score, 1.0)

    def test_typed_and_dict_inputs_sanitize_nan_and_none(self) -> None:
        cand = CandidateRecipe(
            recipe_id=7,
            title=None,  # type: ignore[arg-type]
            canonical_ingredients=[None, "chicken", "", 3],  # type: ignore[list-item]
            tfidf_score=float("nan"),
            ingredients=None,  # type: ignore[arg-type]
            directions=None,  # type: ignore[arg-type]
            total_calories=float("nan"),
            calories_per_serving=float("nan"),
        )
        self.assertEqual(cand.title, "")
        self.assertEqual(cand.canonical_ingredients, ["chicken"])
        self.assertEqual(cand.ingredients, [])
        self.assertEqual(cand.directions, [])
        self.assertEqual(cand.tfidf_score, 0.0)
        self.assertIsNone(cand.total_calories)
        self.assertIsNone(cand.calories_per_serving)

        parsed = CandidateRecipe.from_dict(
            {
                "recipe_id": 8,
                "title": "Soup",
                "canonical_ingredients": None,
                "ingredients": None,
                "directions": None,
                "tfidf_score": None,
                "total_calories": float("nan"),
                "calories_per_serving": float("inf"),
            }
        )
        self.assertEqual(parsed.canonical_ingredients, [])
        self.assertEqual(parsed.ingredients, [])
        self.assertEqual(parsed.directions, [])
        self.assertEqual(parsed.tfidf_score, 0.0)
        self.assertIsNone(parsed.total_calories)
        self.assertIsNone(parsed.calories_per_serving)

        sq = StructuredQuery(
            available_ingredients=None,
            excluded_ingredients=None,
            calorie_min=float("nan"),
            calorie_max=-5,
        )
        self.assertEqual(sq.available_ingredients, ())
        self.assertEqual(sq.excluded_ingredients, ())
        self.assertIsNone(sq.calorie_min)
        self.assertIsNone(sq.calorie_max)

        sq_swap = StructuredQuery.from_dict(
            {
                "available_ingredients": [None, "chicken", "", 7],
                "excluded_ingredients": ["cheese"],
                "calorie_min": 600,
                "calorie_max": 400,
            }
        )
        self.assertEqual(sq_swap.available_ingredients, ("chicken",))
        self.assertEqual((sq_swap.calorie_min, sq_swap.calorie_max), (400.0, 600.0))

        with self.assertRaises(TypeError):
            StructuredQuery.from_dict(None)  # type: ignore[arg-type]

    def test_no_match_result_and_reason(self) -> None:
        query = StructuredQuery(available_ingredients=["chicken"], calorie_max=300)

        empty = self.reranker.rerank([], query, top_k=3)
        self.assertEqual(empty.recommendations, [])
        self.assertIsNotNone(empty.no_match_reason)
        self.assertIn("no_match_reason", to_dict(empty))

        hot = CandidateRecipe(
            recipe_id="hot",
            title="Hot Bowl",
            canonical_ingredients=["chicken", "rice", "oil"],
            tfidf_score=0.8,
            total_calories=900.0,
        )
        no_fallback = RankingConfig(enable_fallback=False)
        res = self.reranker.rerank([None, hot], query, top_k=1, config=no_fallback)  # type: ignore[list-item]
        self.assertEqual(res.recommendations, [])
        self.assertEqual(res.filtered_out_calorie, 1)
        self.assertIn("fallback is disabled", res.no_match_reason or "")

    def test_duplicate_titles_are_deduped_and_prefer_best(self) -> None:
        query = StructuredQuery(available_ingredients=["chicken", "rice", "oil"])
        candidates = [
            CandidateRecipe(
                recipe_id=1,
                title="Chicken Rice",
                canonical_ingredients=["chicken", "rice", "oil"],
                tfidf_score=0.50,
                total_calories=500.0,
            ),
            CandidateRecipe(
                recipe_id=2,
                title="  chicken   rice ",
                canonical_ingredients=["chicken", "rice", "oil"],
                tfidf_score=0.90,
                total_calories=500.0,
            ),
        ]
        res = self.reranker.rerank(candidates, query, top_k=3)
        self.assertEqual(len(res.recommendations), 1)
        # Best-scoring duplicate survives, not the first inserted
        self.assertEqual(res.recommendations[0].recipe.recipe_id, 2)

    def test_title_and_raw_cheese_mentions_are_filtered(self) -> None:
        query = StructuredQuery(
            available_ingredients=["pasta", "milk"],
            excluded_ingredients=["cheese"],
        )
        mac = CandidateRecipe(
            recipe_id="mac",
            title="Mac & Cheese",
            canonical_ingredients=["pasta", "milk"],
            ingredients=["1 box macaroni", "2 cups milk"],
            tfidf_score=0.8,
            total_calories=600.0,
        )
        raw_only = CandidateRecipe(
            recipe_id="raw",
            title="Baked Pasta",
            canonical_ingredients=["pasta", "milk"],
            ingredients=["1 box pasta", "1 cup shredded cheddar cheese"],
            tfidf_score=0.7,
            total_calories=600.0,
        )
        res = self.reranker.rerank(
            [mac, raw_only], query, top_k=2, config=RankingConfig(enable_fallback=False)
        )
        self.assertEqual(res.filtered_out_exclusion, 2)
        self.assertEqual(res.recommendations, [])
        self.assertIsNotNone(res.no_match_reason)

    def test_per_serving_calories_preferred_over_total(self) -> None:
        query = StructuredQuery(
            available_ingredients=["chicken", "rice", "oil"],
            calorie_target=500,
        )
        cand = CandidateRecipe(
            recipe_id="per",
            title="Family Roast Chicken",
            canonical_ingredients=["chicken", "rice", "oil"],
            tfidf_score=0.8,
            total_calories=2000.0,
            calories_per_serving=500.0,
            directions=["Serves 4."],
        )
        # Per-serving wins even when direction-based normalization is enabled
        scored = self.reranker.score_candidate(
            cand, query, config=RankingConfig(normalize_servings_from_directions=True)
        )
        self.assertEqual(scored.calorie.estimated_calories, 500.0)
        self.assertEqual(scored.calorie.score, 1.0)

    def test_parse_confidence_scales_coverage_for_short_parses(self) -> None:
        cand = CandidateRecipe(
            recipe_id="short",
            title="Chicken And Salt",
            canonical_ingredients=["chicken", "salt"],
            ingredients=["1 chicken", "1 tsp salt", "1 cup water"],
            tfidf_score=0.5,
            total_calories=400.0,
        )
        scored = self.reranker.score_candidate(
            cand,
            StructuredQuery(["chicken"]),
            config=RankingConfig(use_pantry_staple_weights=False),
        )
        self.assertAlmostEqual(scored.coverage.parse_confidence, 2 / 3, places=4)
        self.assertAlmostEqual(scored.coverage.coverage_score, 0.5 * (2 / 3), places=4)
        # Penalty itself is not reduced by parse confidence
        self.assertAlmostEqual(scored.coverage.missing_penalty, 0.5, places=4)

        full = compute_coverage(
            ["chicken"],
            ["chicken", "salt", "rice"],
            staple_weights=self.reranker.staple_weights,
            normalizer=self.reranker.normalizer,
        )
        self.assertEqual(full.parse_confidence, 1.0)

    def test_fallback_calorie_ordering_prefers_nearest_distance(self) -> None:
        query = StructuredQuery(
            available_ingredients=["chicken", "rice", "oil"],
            calorie_max=300,
        )
        far = CandidateRecipe(
            recipe_id="far",
            title="Loaded Feast",
            canonical_ingredients=["chicken", "rice", "oil"],
            tfidf_score=0.95,
            total_calories=5452.0,
        )
        near = CandidateRecipe(
            recipe_id="near",
            title="Chicken Rice Bowl",
            canonical_ingredients=["chicken", "rice", "oil"],
            tfidf_score=0.70,
            total_calories=481.0,
        )
        res = self.reranker.rerank([far, near], query, top_k=1)
        self.assertTrue(res.is_fallback)
        self.assertEqual(res.recommendations[0].recipe.recipe_id, "near")
        self.assertTrue(res.recommendations[0].is_fallback_candidate)
        self.assertIn("calorie_bounds", res.relaxed_constraints)

    def test_ranking_config_rejects_invalid_weights(self) -> None:
        with self.assertRaises(ValueError):
            RankingConfig(alpha=float("nan"))
        with self.assertRaises(ValueError):
            RankingConfig(beta=-0.1)
        with self.assertRaises(ValueError):
            RankingConfig(delta=float("inf"))
        self.assertEqual(RankingConfig().alpha, 0.30)


class TestCalorieEstimator(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.estimator = RecipeCalorieEstimator(processed_dir=PROCESSED_DIR)

    def test_ingredient_line_parser(self) -> None:
        p1 = parse_ingredient_line("2 Tbsp. butter")
        self.assertAlmostEqual(p1.quantity or 0, 2.0)
        self.assertEqual(p1.unit, "tbsp")
        self.assertEqual(p1.ingredient, "butter")

        p2 = parse_ingredient_line("1 1/2 c. sugar")
        self.assertAlmostEqual(p2.quantity or 0, 1.5)
        self.assertEqual(p2.unit, "cup")
        self.assertEqual(p2.ingredient, "sugar")

        p3 = parse_ingredient_line("1 (16 oz.) can whole cranberry sauce")
        self.assertAlmostEqual(p3.quantity or 0, 16.0)
        self.assertEqual(p3.unit, "oz")
        self.assertIn("cranberry sauce", p3.ingredient.lower())

        p4 = parse_ingredient_line("1/2 tsp. salt")
        self.assertAlmostEqual(p4.quantity or 0, 0.5)
        self.assertEqual(p4.unit, "tsp")
        self.assertEqual(p4.ingredient, "salt")

    def test_extract_servings_and_recipe_estimation(self) -> None:
        self.assertEqual(extract_servings(["Bake 1 hour.", "Serves 16."]), 16)
        self.assertEqual(extract_servings(["Makes 4 to 6 servings."]), 5)

        est = self.estimator.estimate_recipe(
            raw_ingredients=["200 g chicken breast", "200 g potato", "1 tbsp olive oil", "1 tsp salt"],
            directions=["Cook chicken and potatoes in olive oil.", "Serves 2."],
        )
        self.assertIsNotNone(est.total_calories)
        self.assertEqual(est.estimated_servings, 2)
        self.assertEqual(est.servings_source, "parsed")
        # 200g chicken breast (~344) + 200g potato (~154) + 1 tbsp olive oil (~120) + salt (0) ~= 618 kcal
        self.assertGreater(est.total_calories or 0, 450)
        self.assertLess(est.total_calories or 0, 800)
        self.assertAlmostEqual(est.calories_per_serving or 0, (est.total_calories or 0) / 2, delta=0.1)


class TestSampleCorpusIntegration(unittest.TestCase):
    def test_end_to_end_on_sample_recipes_and_annotated_queries(self) -> None:
        self.assertTrue(SAMPLE_CSV.exists(), "recipes_sample.csv must exist")
        self.assertTrue(ANNOTATED_JSON.exists(), "annotated_queries.json must exist")

        normalizer = IngredientNormalizer(PROCESSED_DIR)
        estimator = RecipeCalorieEstimator(processed_dir=PROCESSED_DIR, normalizer=normalizer)
        reranker = MultiCriteriaReranker(processed_dir=PROCESSED_DIR, normalizer=normalizer)

        # Load first 100 sample recipes and estimate calories
        candidates: list[CandidateRecipe] = []
        with SAMPLE_CSV.open(encoding="utf-8", newline="") as fh:
            for idx, row in enumerate(csv.DictReader(fh)):
                if idx >= 100:
                    break
                raw_ings = ast.literal_eval(row["ingredients"])
                ner_ings = ast.literal_eval(row["NER"])
                directions = ast.literal_eval(row["directions"])
                est = estimator.estimate_recipe(raw_ings, ner_ings, directions, title=row["title"])
                candidates.append(
                    CandidateRecipe(
                        recipe_id=idx,
                        title=row["title"],
                        canonical_ingredients=normalizer.canonicalize_list(ner_ings),
                        tfidf_score=0.5,
                        ingredients=raw_ings,
                        directions=directions,
                        total_calories=est.calories_per_serving,
                    )
                )

        # Verify high calorie estimation coverage on real RecipeNLG sample rows
        with_cals = sum(1 for c in candidates if c.total_calories is not None)
        self.assertGreaterEqual(with_cals, 95)

        # Run q041 from annotated_queries.json ("chicken, potato, carrot, onion, garlic; around 500; no cheese")
        data = json.loads(ANNOTATED_JSON.read_text(encoding="utf-8"))
        q041 = next(q for q in data["queries"] if q["id"] == "q041")
        sq = StructuredQuery.from_dict(q041["expected"], raw_query=q041["query"])
        result = reranker.rerank(candidates, sq, top_k=3)

        self.assertEqual(len(result.recommendations), 3)
        for rec in result.recommendations:
            if not rec.is_fallback_candidate:
                self.assertFalse(rec.exclusion.violated)


if __name__ == "__main__":
    unittest.main()
