"""
test_translator.py — Unit tests for src/translator.py

Tests are split into two categories:

    1. Mocked tests — patch the GoogleTranslator to avoid real API calls.
       These run fast, work offline, and test the function logic in isolation.

    2. Integration test — makes a real API call to confirm end-to-end behaviour.
       Marked with @pytest.mark.integration so they can be skipped in CI:
           pytest tests/test_translator.py -m "not integration"

Run all tests:
    pytest tests/test_translator.py -v

Run without integration tests:
    pytest tests/test_translator.py -v -m "not integration"
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from unittest.mock import patch, MagicMock

from translator import translate_to_standard_english


# ---------------------------------------------------------------------------
# Helper — builds a mock translator that returns a fixed string
# ---------------------------------------------------------------------------

def make_mock_translator(return_value: str):
    mock = MagicMock()
    mock.translate.return_value = return_value
    return mock


# ---------------------------------------------------------------------------
# Return format tests
# All tests confirm the dict always has exactly three keys with correct types
# ---------------------------------------------------------------------------

class TestReturnFormat:

    def test_returns_dict(self):
        """Return type must always be a dict"""
        with patch('translator._TRANSLATOR', make_mock_translator("hello")):
            result = translate_to_standard_english("hello")
        assert isinstance(result, dict)

    def test_dict_has_three_keys(self):
        """Dict must contain exactly: original, translated, was_translated"""
        with patch('translator._TRANSLATOR', make_mock_translator("hello")):
            result = translate_to_standard_english("hello")
        assert set(result.keys()) == {"original", "translated", "was_translated"}

    def test_original_matches_input(self):
        """'original' field must always match the raw input"""
        with patch('translator._TRANSLATOR', make_mock_translator("the food is great")):
            result = translate_to_standard_english("the food is great")
        assert result["original"] == "the food is great"

    def test_translated_is_string(self):
        """'translated' field must be a string"""
        with patch('translator._TRANSLATOR', make_mock_translator("delicious")):
            result = translate_to_standard_english("sedap")
        assert isinstance(result["translated"], str)

    def test_was_translated_is_bool(self):
        """'was_translated' field must be a boolean"""
        with patch('translator._TRANSLATOR', make_mock_translator("hello")):
            result = translate_to_standard_english("hello")
        assert isinstance(result["was_translated"], bool)


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_empty_string_returns_safe_dict(self):
        """Empty string must return dict with empty fields and was_translated=False"""
        result = translate_to_standard_english("")
        assert result["original"] == ""
        assert result["translated"] == ""
        assert result["was_translated"] is False

    def test_whitespace_only_returns_safe_dict(self):
        """Whitespace-only input must return safe dict without calling API"""
        result = translate_to_standard_english("   ")
        assert result["was_translated"] is False

    def test_none_input_returns_safe_dict(self):
        """None input must return safe dict without crashing"""
        result = translate_to_standard_english(None)
        assert result["original"] == ""
        assert result["translated"] == ""
        assert result["was_translated"] is False

    def test_integer_input_returns_safe_dict(self):
        """Non-string input must return safe dict without crashing"""
        result = translate_to_standard_english(123)
        assert result["was_translated"] is False


# ---------------------------------------------------------------------------
# was_translated flag behaviour
# ---------------------------------------------------------------------------

class TestWasTranslatedFlag:

    def test_flag_true_when_output_differs(self):
        """was_translated=True when output meaningfully differs from input"""
        with patch('translator._TRANSLATOR', make_mock_translator("the food is delicious")):
            result = translate_to_standard_english("makanan sedap")
        assert result["was_translated"] is True

    def test_flag_false_when_output_identical(self):
        """was_translated=False when output matches input (already English)"""
        with patch('translator._TRANSLATOR', make_mock_translator("the food is great")):
            result = translate_to_standard_english("the food is great")
        assert result["was_translated"] is False

    def test_flag_false_on_failure(self):
        """was_translated=False when all retries fail"""
        with patch('translator._TRANSLATOR') as mock_translator:
            mock_translator.translate.side_effect = Exception("API error")
            result = translate_to_standard_english("makanan sedap")
        assert result["was_translated"] is False

    def test_original_returned_on_failure(self):
        """On API failure, translated field must equal original text"""
        with patch('translator._TRANSLATOR') as mock_translator:
            mock_translator.translate.side_effect = Exception("API error")
            result = translate_to_standard_english("makanan sedap")
        assert result["translated"] == "makanan sedap"


# ---------------------------------------------------------------------------
# Retry logic tests
# ---------------------------------------------------------------------------

class TestRetryLogic:

    def test_retries_on_exception(self):
        """Translator must retry up to 3 times on exception"""
        with patch('translator._TRANSLATOR') as mock_translator:
            with patch('translator.time.sleep'):  # skip actual sleep in tests
                mock_translator.translate.side_effect = [
                    Exception("attempt 1"),
                    Exception("attempt 2"),
                    "the food is delicious"  # succeeds on 3rd attempt
                ]
                result = translate_to_standard_english("makanan sedap")

        assert mock_translator.translate.call_count == 3
        assert result["translated"] == "the food is delicious"

    def test_fails_after_max_retries(self):
        """After 3 failed attempts, must return original text gracefully"""
        with patch('translator._TRANSLATOR') as mock_translator:
            with patch('translator.time.sleep'):
                mock_translator.translate.side_effect = Exception("persistent error")
                result = translate_to_standard_english("makanan sedap")

        assert mock_translator.translate.call_count == 3
        assert result["translated"] == "makanan sedap"
        assert result["was_translated"] is False

    def test_sleep_called_between_retries(self):
        """Sleep must be called between retry attempts"""
        with patch('translator._TRANSLATOR') as mock_translator:
            with patch('translator.time.sleep') as mock_sleep:
                mock_translator.translate.side_effect = [
                    Exception("attempt 1"),
                    Exception("attempt 2"),
                    "success"
                ]
                translate_to_standard_english("makanan sedap")

        # Sleep called between attempt 1→2 and 2→3 (not after final attempt)
        assert mock_sleep.call_count == 2


# ---------------------------------------------------------------------------
# Integration test — real API call
# Skip with: pytest -m "not integration"
# ---------------------------------------------------------------------------

class TestIntegration:

    @pytest.mark.integration
    def test_real_malay_translation(self):
        """
        Real API call — confirms Google Translate integration works end to end.
        Input: normalised Manglish. Expected: meaningful English output.
        Requires internet connection.
        """
        result = translate_to_standard_english("makanan very delicious but service very slow")
        assert isinstance(result, dict)
        assert result["original"] == "makanan very delicious but service very slow"
        assert isinstance(result["translated"], str)
        assert len(result["translated"]) > 0
        # was_translated depends on whether Google changed the text
        # We only assert it is a boolean — not its specific value
        assert isinstance(result["was_translated"], bool)

    @pytest.mark.integration
    def test_english_input_not_meaningfully_changed(self):
        """
        Real API call — pure English input should return was_translated=False
        since Google Translate returns the same text for English input.
        Requires internet connection.
        """
        result = translate_to_standard_english("the food was absolutely delicious")
        assert result["was_translated"] is False
        assert result["translated"].lower().strip() == "the food was absolutely delicious"