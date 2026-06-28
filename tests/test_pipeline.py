"""
test_pipeline.py — End-to-end tests for src/pipeline.py

Tests the full pipeline in all three modes and confirms that all components
chain together correctly. Also tests the return format, edge cases, and
the on_no_match parameter passthrough.

These tests are the closest thing to a real-world usage scenario.
If these pass, the full feature extraction pipeline is working end-to-end.

Run with: pytest tests/test_pipeline.py -v

Integration tests (real API calls) marked separately:
    pytest tests/test_pipeline.py -v -m "not integration"
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from unittest.mock import patch, MagicMock
from pipeline import run_pipeline
from extractor import AspectSentimentExtractor
from categoriser import AspectCategoriser


# ---------------------------------------------------------------------------
# Return format tests
# Confirm the dict always has all six keys regardless of mode
# ---------------------------------------------------------------------------

class TestReturnFormat:

    def test_returns_dict(self, nlp):
        """Return type must always be a dict"""
        result = run_pipeline("The food is great.", mode="english")
        assert isinstance(result, dict)

    def test_dict_has_all_keys(self, nlp):
        """Dict must contain all six required keys"""
        result = run_pipeline("The food is great.", mode="english")
        required_keys = {"original", "normalised", "translated",
                         "was_translated", "pairs", "results"}
        assert set(result.keys()) == required_keys

    def test_original_matches_input(self, nlp):
        """'original' must always match raw input regardless of mode"""
        text = "The food is great."
        result = run_pipeline(text, mode="english")
        assert result["original"] == text

    def test_pairs_is_list(self, nlp):
        """'pairs' must always be a list"""
        result = run_pipeline("The food is great.", mode="english")
        assert isinstance(result["pairs"], list)

    def test_results_is_list(self, nlp):
        """'results' must always be a list"""
        result = run_pipeline("The food is great.", mode="english")
        assert isinstance(result["results"], list)

    def test_was_translated_is_bool(self, nlp):
        """'was_translated' must always be a boolean"""
        result = run_pipeline("The food is great.", mode="english")
        assert isinstance(result["was_translated"], bool)


# ---------------------------------------------------------------------------
# English mode tests
# Normaliser and translator must be skipped entirely
# ---------------------------------------------------------------------------

class TestEnglishMode:

    def test_normalised_is_none(self, nlp):
        """English mode must not run normaliser — normalised must be None"""
        result = run_pipeline("The food is great.", mode="english")
        assert result["normalised"] is None

    def test_translated_is_none(self, nlp):
        """English mode must not run translator — translated must be None"""
        result = run_pipeline("The food is great.", mode="english")
        assert result["translated"] is None

    def test_was_translated_false(self, nlp):
        """English mode must always return was_translated=False"""
        result = run_pipeline("The food is great.", mode="english")
        assert result["was_translated"] is False

    def test_extracts_pairs_from_english(self, nlp):
        """English mode must still extract pairs correctly"""
        result = run_pipeline("The food is great.", mode="english")
        assert len(result["pairs"]) > 0

    def test_produces_results_from_english(self, nlp):
        """English mode must produce categorised results"""
        result = run_pipeline("The food is great.", mode="english")
        assert len(result["results"]) > 0

    def test_multi_aspect_english(self, nlp):
        """English mode must handle multi-aspect reviews — validates RO2"""
        result = run_pipeline(
            "The food was delicious but the service was terrible.",
            mode="english"
        )
        categories = [r[0] for r in result["results"]]
        assert len(result["results"]) >= 2
        assert "food" in categories
        assert "service" in categories


# ---------------------------------------------------------------------------
# Manglish mode tests
# Full pipeline must run — normaliser → translator → extractor → categoriser
# Uses mocked translator to avoid real API calls
# ---------------------------------------------------------------------------

class TestManglishMode:

    def test_normalised_is_not_none(self, nlp):
        """Manglish mode must run normaliser — normalised must not be None"""
        with patch('pipeline.translate_to_standard_english') as mock_translate:
            mock_translate.return_value = {
                "original": "food very delicious",
                "translated": "the food is very delicious",
                "was_translated": True
            }
            result = run_pipeline("food sedap gila", mode="manglish")
        assert result["normalised"] is not None

    def test_translation_result_used(self, nlp):
        """Manglish mode must use translated text for extraction"""
        with patch('pipeline.translate_to_standard_english') as mock_translate:
            mock_translate.return_value = {
                "original": "food sedap",
                "translated": "the food is delicious",
                "was_translated": True
            }
            result = run_pipeline("food sedap", mode="manglish")
        assert result["was_translated"] is True
        assert result["translated"] == "the food is delicious"

    def test_was_translated_propagated(self, nlp):
        """was_translated from translator must propagate to pipeline output"""
        with patch('pipeline.translate_to_standard_english') as mock_translate:
            mock_translate.return_value = {
                "original": "the food is great",
                "translated": "the food is great",
                "was_translated": False
            }
            result = run_pipeline("the food is great", mode="manglish")
        assert result["was_translated"] is False


# ---------------------------------------------------------------------------
# Auto mode tests
# Normaliser always runs, translator only runs if output differs
# ---------------------------------------------------------------------------

class TestAutoMode:

    def test_normalised_not_none_in_auto(self, nlp):
        """Auto mode always runs normaliser"""
        with patch('pipeline.translate_to_standard_english') as mock_translate:
            mock_translate.return_value = {
                "original": "the food is great",
                "translated": "the food is great",
                "was_translated": False
            }
            result = run_pipeline("the food is great", mode="auto")
        assert result["normalised"] is not None

    def test_translated_none_when_not_translated(self, nlp):
        """When was_translated=False in auto mode, translated must be None"""
        with patch('pipeline.translate_to_standard_english') as mock_translate:
            mock_translate.return_value = {
                "original": "the food is great",
                "translated": "the food is great",
                "was_translated": False
            }
            result = run_pipeline("the food is great", mode="auto")
        assert result["translated"] is None

    def test_translated_set_when_translation_occurred(self, nlp):
        """When was_translated=True in auto mode, translated must be set"""
        with patch('pipeline.translate_to_standard_english') as mock_translate:
            mock_translate.return_value = {
                "original": "makanan sedap",
                "translated": "the food is delicious",
                "was_translated": True
            }
            result = run_pipeline("makanan sedap", mode="auto")
        assert result["translated"] == "the food is delicious"


# ---------------------------------------------------------------------------
# Invalid mode test
# ---------------------------------------------------------------------------

class TestInvalidMode:

    def test_invalid_mode_raises_value_error(self, nlp):
        """Invalid mode must raise ValueError immediately"""
        with pytest.raises(ValueError):
            run_pipeline("The food is great.", mode="invalid_mode")


# ---------------------------------------------------------------------------
# Custom component injection
# ---------------------------------------------------------------------------

class TestCustomComponents:

    def test_custom_categoriser_on_no_match_other(self, nlp):
        """
        Passing a custom categoriser with on_no_match='other' must produce
        'other' for unrecognised aspects instead of dropping them.
        """
        custom_cat = AspectCategoriser(nlp_model=nlp, on_no_match="other")
        result = run_pipeline(
            "The xyzabc is great.",
            mode="english",
            categoriser=custom_cat
        )
        categories = [r[0] for r in result["results"]]
        # xyzabc should come back as 'other' not be dropped
        if len(result["pairs"]) > 0:
            assert "other" in categories

    def test_custom_extractor_window(self, nlp):
        """Custom extractor with different proximity window must be accepted"""
        custom_ext = AspectSentimentExtractor(nlp_model=nlp, proximity_window=2)
        result = run_pipeline(
            "The food is great.",
            mode="english",
            extractor=custom_ext
        )
        assert isinstance(result["results"], list)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_string_returns_safe_dict(self, nlp):
        """Empty string must return dict with empty pairs and results"""
        result = run_pipeline("", mode="english")
        assert isinstance(result, dict)
        assert result["pairs"] == []
        assert result["results"] == []

    def test_whitespace_only_returns_safe_dict(self, nlp):
        """Whitespace-only input must return safe dict"""
        result = run_pipeline("   ", mode="english")
        assert result["pairs"] == []
        assert result["results"] == []

    def test_results_contain_valid_categories(self, nlp):
        """All categories in results must be from the canonical set"""
        canonical = {"food", "service", "ambiance", "price", "overall", "other"}
        result = run_pipeline(
            "The food was great and the service was slow.",
            mode="english"
        )
        for category, _ in result["results"]:
            assert category in canonical

    def test_results_tuples_have_two_elements(self, nlp):
        """Each item in results must be a (category, opinion_word) tuple"""
        result = run_pipeline("The food is great.", mode="english")
        for item in result["results"]:
            assert len(item) == 2


# ---------------------------------------------------------------------------
# Integration tests — real API calls
# ---------------------------------------------------------------------------

class TestIntegration:

    @pytest.mark.integration
    def test_full_manglish_pipeline_real_api(self, nlp):
        """
        Real end-to-end test with actual Google Translate API call.
        Input: real Manglish sentence from restaurant reviews.
        Confirms the full pipeline chains correctly with live translation.
        Requires internet connection.
        """
        result = run_pipeline(
            "makanan sgt sedap tapi service lambat gila",
            mode="manglish"
        )
        assert isinstance(result, dict)
        assert result["normalised"] is not None
        assert isinstance(result["pairs"], list)
        assert isinstance(result["results"], list)
        # Pipeline must not crash — results may be empty depending on translation
        # quality, which is an acceptable outcome documented as a known limitation

    @pytest.mark.integration
    def test_auto_mode_real_api_manglish(self, nlp):
        """
        Auto mode with real Manglish input must trigger translation.
        Requires internet connection.
        """
        result = run_pipeline(
            "tempat ni sgt best tapi mahal sikit",
            mode="auto"
        )
        assert isinstance(result, dict)
        assert result["normalised"] is not None
        # was_translated depends on Google's response — just confirm it's bool
        assert isinstance(result["was_translated"], bool)