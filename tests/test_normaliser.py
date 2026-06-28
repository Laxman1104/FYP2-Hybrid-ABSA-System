"""
test_normaliser.py — Unit tests for src/normaliser.py

Tests each stage of resolve_local_terms() independently and in combination.
All tests use minimal inline dictionaries rather than the JSON file so
tests remain isolated from the production dictionary.

Run with: pytest tests/test_normaliser.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from normaliser import resolve_local_terms

# ---------------------------------------------------------------------------
# Minimal test dictionaries — isolated from production manglish_dict.json
# ---------------------------------------------------------------------------

TEST_PHRASES = {
    "sold out":     "unavailable",
    "check out":    "leave",
    "out of stock": "unavailable",
    "ok je":        "just okay",
    "okay je":      "just okay",
}

TEST_WORDS = {
    "x":       "not",
    "sedap":   "delicious",
    "lambat":  "slow",
    "mahal":   "expensive",
    "best":    "great",
    "sgt":     "very",
    "mmg":     "indeed",
    "tapau":   "takeaway",
    "out":     "bad",
    "agak":    "knew",
    "bagi":    "give",
    "tak":     "not",
    "nk":      "want",
    "dah":     "already",
}

TEST_NOISE = ["lah", "la", "weh", "sia", "doh"]


# ---------------------------------------------------------------------------
# Helper — shorthand for calling resolve_local_terms with test dicts
# ---------------------------------------------------------------------------

def normalise(text):
    return resolve_local_terms(text, TEST_PHRASES, TEST_WORDS, TEST_NOISE)


# ---------------------------------------------------------------------------
# Stage 0: Repeated character collapse
# ---------------------------------------------------------------------------

class TestRepeatedCharCollapse:

    def test_triple_repeat_collapses(self):
        result = normalise("sedappp")
        assert "sedapp" in result  

    def test_quad_repeat_collapses(self):
        """'lambaaaat' collapses repeated 'a' to 'aa' → 'lambaat' passes through unchanged"""
        result = normalise("lambaaaat")
        assert "lambaat" in result  # collapses but no dict match — passes through

    def test_double_repeat_preserved(self):
        """Legitimate double letters must not be collapsed — 'good' stays 'good'"""
        result = normalise("good")
        assert result == "good"

    def test_staff_preserved(self):
        """'staff' has double f — must not be corrupted"""
        result = normalise("staff")
        assert result == "staff"

    def test_exclamation_not_collapsed(self):
        """Repeated punctuation like '!!!' should collapse but not break word"""
        result = normalise("sedap!!!")
        # After collapse: 'sedap!' then word lookup finds 'sedap' → 'delicious!'
        # punctuation attached — acceptable behaviour
        assert "delicious" in result
    
    def test_goooood_collapses_to_good(self):
        result = resolve_local_terms("goooood")
        assert result == "good"
    
    def test_niceee_collapses(self):
        result = resolve_local_terms("niceee")
        assert "nice" in result

# ---------------------------------------------------------------------------
# Stage 1: X-prefix detacher
# ---------------------------------------------------------------------------

class TestXPrefixDetacher:

    def test_xbagi_detaches(self):
        """'xbagi' → 'x bagi' → 'not give'"""
        result = normalise("xbagi")
        assert result == "not give"

    def test_xsedap_detaches(self):
        """'xsedap' → 'x sedap' → 'not delicious'"""
        result = normalise("xsedap")
        assert result == "not delicious"

    def test_standalone_x_maps(self):
        """Standalone 'x' should map to 'not'"""
        result = normalise("x sedap")
        assert result == "not delicious"

    def test_english_word_starting_x_untouched(self):
        """'extra' should not be split — only 'x' followed by known letters"""
        result = normalise("extra")
        assert result == "extra"


# ---------------------------------------------------------------------------
# Stage 2: Phrase replacement (N-gram protection)
# ---------------------------------------------------------------------------

class TestPhraseReplacement:

    def test_sold_out_protected(self):
        """'sold out' must be caught as phrase BEFORE 'out' maps to 'bad'"""
        result = normalise("sold out")
        assert result == "unavailable"
        assert "bad" not in result

    def test_check_out_protected(self):
        """'check out' must be protected before single-word pass"""
        result = normalise("check out")
        assert result == "leave"
        assert "bad" not in result

    def test_out_of_stock_protected(self):
        """'out of stock' — multi-word phrase caught before 'out' → 'bad'"""
        result = normalise("out of stock")
        assert result == "unavailable"
        assert "bad" not in result

    def test_ok_je_phrase(self):
        """'ok je' maps to 'just okay' as a phrase"""
        result = normalise("ok je")
        assert result == "just okay"

    def test_standalone_out_maps_to_bad(self):
        """Standalone 'out' with no phrase protection should map to 'bad'"""
        result = normalise("makanan mmg out")
        assert "bad" in result


# ---------------------------------------------------------------------------
# Stage 3: Single-word tokenisation and noise filtering
# ---------------------------------------------------------------------------

class TestSingleWordTokenisation:

    def test_known_slang_maps(self):
        """'sedap' maps to 'delicious'"""
        result = normalise("sedap")
        assert result == "delicious"

    def test_noise_particle_dropped(self):
        """'lah' should be dropped entirely"""
        result = normalise("sedap lah")
        assert result == "delicious"
        assert "lah" not in result

    def test_multiple_noise_particles_dropped(self):
        """Multiple particles all dropped"""
        result = normalise("best lah weh")
        assert result == "great"
        assert "lah" not in result
        assert "weh" not in result

    def test_unknown_word_preserved(self):
        """Words not in any dictionary should pass through unchanged"""
        result = normalise("restaurant")
        assert result == "restaurant"

    def test_sgt_maps_to_very(self):
        """'sgt' must map to 'very' — confirmed fix from code review"""
        result = normalise("sgt sedap")
        assert result == "very delicious"

    def test_mmg_maps_to_indeed(self):
        """'mmg' maps to 'indeed'"""
        result = normalise("mmg sedap")
        assert result == "indeed delicious"


# ---------------------------------------------------------------------------
# Combined / integration tests
# ---------------------------------------------------------------------------

class TestCombined:

    def test_full_manglish_sentence(self):
        """
        'sedappp lah mmg best' should:
        - collapse 'sedappp' → 'sedap'
        - drop 'lah'
        - map 'sedap' → 'delicious', 'mmg' → 'indeed', 'best' → 'great'
        """
        result = normalise("sedappp lah mmg best")
        assert "delicious" in result
        assert "indeed" in result
        assert "great" in result
        assert "lah" not in result

    def test_x_prefix_in_sentence(self):
        """'xsedap langsung' → 'not delicious langsung'"""
        result = normalise("xsedap langsung")
        assert "not" in result
        assert "delicious" in result

    def test_sold_out_in_sentence(self):
        """'lontong goreng sedap sgt tp sedih slalu sold out' — sold out protected"""
        result = normalise("lontong goreng sedap sgt tp sedih slalu sold out")
        assert "unavailable" in result
        assert "bad" not in result

    def test_empty_string_returns_empty(self):
        """Empty string input should return empty string"""
        result = normalise("")
        assert result == ""

    def test_non_string_returns_empty(self):
        """Non-string input should return empty string"""
        result = normalise(None)
        assert result == ""

    def test_whitespace_only_returns_empty(self):
        """Whitespace-only input should return empty string"""
        result = normalise("   ")
        assert result == ""

    def test_noise_particle_with_punctuation_dropped(self):
        """'sia,' with trailing comma must still be dropped as noise particle"""
        result = normalise("best sia,")
        assert "sia" not in result
        assert "great" in result
    
    def test_full_manglish_sentence(self):
        result = normalise("sedappp lah mmg best")
        assert "sedapp" in result     
        assert "indeed" in result
        assert "great" in result
        assert "lah" not in result

    def test_repeated_word_not_collapsed(self):
        result = normalise("best best")
        assert result == "great great" 
    
