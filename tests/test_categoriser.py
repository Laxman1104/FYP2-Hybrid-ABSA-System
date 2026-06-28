"""
test_categoriser.py — Unit tests for src/categoriser.py

Tests the two-stage categorisation process:
    Stage 1 — Keyword override lookup
    Stage 2 — Vector similarity against anchor groups

Also tests the on_no_match parameter behaviour and edge cases.

Run with: pytest tests/test_categoriser.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from categoriser import AspectCategoriser


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cat_drop(nlp):
    """Categoriser with on_no_match='drop' — default SemEval behaviour"""
    return AspectCategoriser(nlp_model=nlp, on_no_match="drop")


@pytest.fixture(scope="module")
def cat_other(nlp):
    """Categoriser with on_no_match='other' — dashboard and Malaysian evaluation"""
    return AspectCategoriser(nlp_model=nlp, on_no_match="other")


# ---------------------------------------------------------------------------
# Stage 1: Keyword Override Tests
# These terms were identified in EDA as the 23 unmapped top-50 aspects.
# Override must fire BEFORE vector similarity for these terms.
# ---------------------------------------------------------------------------

class TestKeywordOverrides:

    def test_reservation_maps_to_service(self, cat_drop):
        """'reservation' was incorrectly mapping to price via similarity — override fixes this"""
        result = cat_drop.categorise([("reservation", "positive")])
        assert ("service", "positive") in result

    def test_wait_maps_to_service(self, cat_drop):
        """'wait' was UNMAPPED via similarity — override fixes this"""
        result = cat_drop.categorise([("wait", "negative")])
        assert ("service", "negative") in result

    def test_tables_maps_to_ambiance(self, cat_drop):
        """'tables' was UNMAPPED via similarity — override fixes this"""
        result = cat_drop.categorise([("tables", "negative")])
        assert ("ambiance", "negative") in result

    def test_music_maps_to_ambiance(self, cat_drop):
        """'music' was UNMAPPED via similarity — override fixes this"""
        result = cat_drop.categorise([("music", "positive")])
        assert ("ambiance", "positive") in result

    def test_wine_maps_to_food(self, cat_drop):
        """'wine' maps to food"""
        result = cat_drop.categorise([("wine", "positive")])
        assert ("food", "positive") in result

    def test_ambience_maps_to_ambiance(self, cat_drop):
        """'ambience' spelling variant maps to ambiance category"""
        result = cat_drop.categorise([("ambience", "positive")])
        assert ("ambiance", "positive") in result

    def test_waiters_maps_to_service(self, cat_drop):
        """'waiters' maps to service"""
        result = cat_drop.categorise([("waiters", "negative")])
        assert ("service", "negative") in result

    def test_taste_maps_to_food(self, cat_drop):
        """'taste' maps to food"""
        result = cat_drop.categorise([("taste", "positive")])
        assert ("food", "positive") in result

    def test_quality_maps_to_overall(self, cat_drop):
        """'quality' maps to overall"""
        result = cat_drop.categorise([("quality", "positive")])
        assert ("overall", "positive") in result
    
    def test_choices_maps_to_food(self, cat_other):
        """'choices' maps to food"""
        result = cat_other.categorise([("choices", "many")])
        assert ("food", "many") in result

    def test_choice_maps_to_food(self, cat_other):
        """'choice' maps to food"""
        result = cat_other.categorise([("choice", "many")])
        assert ("food", "many") in result

    def test_ingredients_maps_to_food(self, cat_drop):
        """'ingredients' maps to food"""
        result = cat_drop.categorise([("ingredients", "positive")])
        assert ("food", "positive") in result
    
    def test_price_maps_to_price(self, cat_drop):
        """'price' maps to price"""
        result = cat_drop.categorise([("price", "good")])
        assert ("price", "good") in result
    
    def test_restaurant_maps_to_ambiance(self, cat_other):
        """'restaurant' maps to ambiance"""
        result = cat_other.categorise([("restaurant", "clean")])
        assert ("ambiance", "clean") in result

    def test_override_case_insensitive(self, cat_drop):
        """Keyword override must handle mixed case input"""
        result = cat_drop.categorise([("Reservation", "positive")])
        assert ("service", "positive") in result

    def test_override_strips_whitespace(self, cat_drop):
        """Keyword override must handle leading/trailing whitespace"""
        result = cat_drop.categorise([(" wait ", "negative")])
        assert ("service", "negative") in result

    


# ---------------------------------------------------------------------------
# Stage 2: Vector Similarity Tests
# Terms not in override table should be caught by similarity
# ---------------------------------------------------------------------------

class TestVectorSimilarity:

    def test_food_maps_to_food(self, cat_drop):
        """'food' — direct anchor word match"""
        result = cat_drop.categorise([("food", "positive")])
        assert ("food", "positive") in result

    def test_service_maps_to_service(self, cat_drop):
        """'service' — direct anchor word match"""
        result = cat_drop.categorise([("service", "negative")])
        assert ("service", "negative") in result

    def test_atmosphere_maps_to_ambiance(self, cat_drop):
        """'atmosphere' — anchor word for ambiance category"""
        result = cat_drop.categorise([("atmosphere", "positive")])
        assert ("ambiance", "positive") in result

    def test_price_maps_to_price(self, cat_drop):
        """'price' — direct anchor word match"""
        result = cat_drop.categorise([("price", "negative")])
        assert ("price", "negative") in result

    def test_waiter_maps_to_service(self, cat_drop):
        """'waiter' — anchor word for service"""
        result = cat_drop.categorise([("waiter", "negative")])
        assert ("service", "negative") in result

    def test_meal_maps_to_food(self, cat_drop):
        """'meal' — anchor word for food"""
        result = cat_drop.categorise([("meal", "positive")])
        assert ("food", "positive") in result


# ---------------------------------------------------------------------------
# on_no_match parameter behaviour
# ---------------------------------------------------------------------------

class TestOnNoMatch:

    def test_drop_mode_excludes_unmatched(self, cat_drop):
        """Unrecognised aspect is silently dropped in drop mode"""
        result = cat_drop.categorise([("xyzabc123", "positive")])
        assert result == []

    def test_other_mode_assigns_other(self, cat_other):
        """Unrecognised aspect gets 'other' category in other mode"""
        result = cat_other.categorise([("xyzabc123", "positive")])
        assert ("other", "positive") in result

    def test_drop_mode_keeps_valid_pairs(self, cat_drop):
        """Valid pairs still returned in drop mode alongside dropped ones"""
        result = cat_drop.categorise([
            ("food", "positive"),
            ("xyzabc123", "negative"),
        ])
        assert ("food", "positive") in result
        assert len(result) == 1  # xyzabc123 dropped

    def test_other_mode_keeps_valid_pairs(self, cat_other):
        """Valid pairs still returned in other mode alongside other-assigned ones"""
        result = cat_other.categorise([
            ("food", "positive"),
            ("xyzabc123", "negative"),
        ])
        assert ("food", "positive") in result
        assert ("other", "negative") in result

    def test_invalid_on_no_match_raises(self, nlp):
        """Invalid on_no_match value must raise ValueError at instantiation"""
        with pytest.raises(ValueError):
            AspectCategoriser(nlp_model=nlp, on_no_match="invalid")


# ---------------------------------------------------------------------------
# Multi-pair input
# ---------------------------------------------------------------------------

class TestMultiPairInput:

    def test_multiple_pairs_all_categorised(self, cat_drop):
        """Multiple valid pairs all categorised correctly"""
        result = cat_drop.categorise([
            ("food",    "positive"),
            ("service", "negative"),
            ("price",   "neutral"),
        ])
        assert ("food",    "positive") in result
        assert ("service", "negative") in result
        assert ("price",   "neutral")  in result

    def test_output_deduplicated(self, cat_drop):
        """Duplicate input pairs produce deduplicated output"""
        result = cat_drop.categorise([
            ("food", "positive"),
            ("food", "positive"),
        ])
        assert len(result) == 1

    def test_empty_input_returns_empty_list(self, cat_drop):
        """Empty input list returns empty list"""
        result = cat_drop.categorise([])
        assert result == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_aspect_skipped(self, cat_drop):
        """Empty string aspect must be skipped without crashing"""
        result = cat_drop.categorise([("", "positive")])
        assert result == []

    def test_whitespace_aspect_skipped(self, cat_drop):
        """Whitespace-only aspect must be skipped"""
        result = cat_drop.categorise([("   ", "positive")])
        assert result == []

    def test_none_aspect_skipped(self, cat_drop):
        """None aspect must be skipped without crashing"""
        result = cat_drop.categorise([(None, "positive")])
        assert result == []

    def test_output_is_list(self, cat_drop):
        """Return type must always be a list"""
        result = cat_drop.categorise([("food", "positive")])
        assert isinstance(result, list)

    def test_canonical_category_names(self, cat_drop):
        """All returned categories must be from the canonical set"""
        canonical = {"food", "service", "ambiance", "price", "overall", "other"}
        result = cat_drop.categorise([
            ("food", "positive"),
            ("service", "negative"),
            ("atmosphere", "neutral"),
            ("price", "positive"),
        ])
        for category, _ in result:
            assert category in canonical


# ---------------------------------------------------------------------------
# Test Malay Food Terms
# ---------------------------------------------------------------------------

class TestMalayFoodTerms:

    def test_nasi_kandar_maps_to_food(self, cat_other):
        result = cat_other.categorise([("nasi kandar", "tasty")])
        assert result == [("food", "tasty")]

    def test_mee_goreng_sotong_maps_to_food(self, cat_other):
        result = cat_other.categorise([("mee goreng sotong", "great")])
        assert result == [("food", "great")]

    def test_nasi_lemak_maps_to_food(self, cat_other):
        result = cat_other.categorise([("nasi lemak", "padu")])
        assert result == [("food", "padu")]

    def test_tomyam_maps_to_food(self, cat_other):
        result = cat_other.categorise([("tomyam", "delicious")])
        assert result == [("food", "delicious")]

    def test_roti_canai_maps_to_food(self, cat_other):
        result = cat_other.categorise([("roti canai", "delicious")])
        assert result == [("food", "delicious")]

    def test_satay_maps_to_food(self, cat_other):
        result = cat_other.categorise([("satay", "delicious")])
        assert result == [("food", "delicious")]

    def test_curry_maps_to_food(self, cat_other):
        result = cat_other.categorise([("curry", "delicious")])
        assert result == [("food", "delicious")]
    
    def test_sambal_maps_to_food(self, cat_other):
        result = cat_other.categorise([("sambal", "delicious")])
        assert result == [("food", "delicious")]
    
    def test_laksa_maps_to_food(self, cat_other):
        result = cat_other.categorise([("laksa", "delicious")])
        assert result == [("food", "delicious")]

# ---------------------------------------------------------------------------
# Test Malaysian Context Overrides
# ---------------------------------------------------------------------------   

class TestMalaysianContextOverrides:

    def test_parking_maps_to_ambiance(self, cat_other):
        result = cat_other.categorise([("parking", "limited")])
        assert result == [("ambiance", "limited")]
    
    def test_restaurant_maps_to_ambiance(self, cat_other):
        result = cat_other.categorise([("restaurant", "clean")])
        assert result == [("ambiance", "clean")]
    
    def test_mamak_maps_to_ambiance(self, cat_other):
        result = cat_other.categorise([("mamak", "dirty")])
        assert result == [("ambiance", "dirty")]

    
    


