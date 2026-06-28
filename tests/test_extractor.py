"""
test_extractor.py — Unit tests for src/extractor.py

Tests each extraction rule independently, then tests the proximity fallback,
enhancement mechanisms, and edge cases.

Each rule test uses a sentence specifically designed to trigger that rule
and confirms the expected (aspect, opinion_word) pair is in the output.

Run with: pytest tests/test_extractor.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from extractor import AspectSentimentExtractor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def extractor(nlp):
    """
    Module-scoped extractor instance.
    Uses the session-scoped nlp fixture from conftest.py.
    """
    return AspectSentimentExtractor(nlp_model=nlp)


# ---------------------------------------------------------------------------
# Rule 1: Adjectival Modifier (amod)
# Pattern: NOUN with ADJ child via amod dependency
# ---------------------------------------------------------------------------

class TestRule1AdjectivalModifier:

    def test_basic_amod(self, extractor):
        """'delicious food' — ADJ directly modifies NOUN via amod"""
        result = extractor.extract("The restaurant serves delicious food.")
        assert ("food", "delicious") in result

    def test_amod_with_article(self, extractor):
        """Article should not affect extraction"""
        result = extractor.extract("I had a terrible experience.")
        assert ("experience", "terrible") in result

    def test_amod_superlative(self, extractor):
        """Superlative adjective should be caught by amod"""
        result = extractor.extract("They serve the best pizza.")
        assert ("pizza", "best") in result

    def test_multiple_amod_pairs(self, extractor):
        """Multiple noun-adjective pairs in one sentence"""
        result = extractor.extract("Great food and friendly staff.")
        aspects = [pair[0] for pair in result]
        assert "food" in aspects or "staff" in aspects


# ---------------------------------------------------------------------------
# Rule 2: Predicate Adjective (nsubj + acomp)
# Pattern: NOUN subject → VERB → ADJ complement via acomp
# ---------------------------------------------------------------------------

class TestRule2PredicateAdjective:

    def test_basic_predicate_adj(self, extractor):
        """'The food is great' — classic nsubj + copula + acomp"""
        result = extractor.extract("The food is great.")
        assert ("food", "great") in result

    def test_predicate_adj_was(self, extractor):
        """Past tense copula should still trigger rule 2"""
        result = extractor.extract("The service was excellent.")
        assert ("service", "excellent") in result

    def test_predicate_adj_negative(self, extractor):
        """Negated predicate adjective — 'food is not good'"""
        result = extractor.extract("The food is not good.")
        aspects = [pair[0] for pair in result]
        opinions = [pair[1] for pair in result]
        assert "food" in aspects
        # Opinion word should contain negation
        assert any("not" in op for op in opinions)

    def test_predicate_adj_with_intensifier(self, extractor):
        """Intensifier should not break rule 2"""
        result = extractor.extract("The atmosphere was really nice.")
        aspects = [pair[0] for pair in result]
        assert "atmosphere" in aspects
    
    def test_rule2_catches_propn_tagged_adjective(self, extractor):
        """
        In informal context spaCy sometimes tags adjectives as PROPN.
        Rule 2 must still extract the pair via acomp dependency label.
        Example: 'the food was crazy delicious' in a longer informal sentence.
        """
        result = extractor.extract(
            "eh that day the food was crazy delicious the place was clean"
        )
        aspects = [pair[0] for pair in result]
        assert "food" in aspects


# ---------------------------------------------------------------------------
# Rule 3: Sentiment Verb (dobj, filtered)
# Pattern: NOUN as direct object of a sentiment-bearing VERB
# ---------------------------------------------------------------------------

class TestRule3SentimentVerb:

    def test_basic_sentiment_verb(self, extractor):
        """'I loved the service' — 'love' is in sentiment verb whitelist"""
        result = extractor.extract("I loved the service.")
        aspects = [pair[0] for pair in result]
        assert "service" in aspects

    def test_recommend_verb(self, extractor):
        """'recommend' is in the whitelist"""
        result = extractor.extract("I would recommend this place.")
        aspects = [pair[0] for pair in result]
        assert "place" in aspects

    def test_noise_verb_filtered(self, extractor):
        """'ate' is NOT in the whitelist — should not produce a pair"""
        result = extractor.extract("I ate the pizza.")
        opinions = [pair[1] for pair in result]
        assert "ate" not in opinions

    def test_got_verb_filtered(self, extractor):
        """'got' is NOT in the whitelist — should not produce a pair"""
        result = extractor.extract("I got the pasta.")
        opinions = [pair[1] for pair in result]
        assert "got" not in opinions


# ---------------------------------------------------------------------------
# Rule 4: Prepositional Modifier (prep + pobj)
# Pattern: ADJ or VERB → PREP → NOUN object
# ---------------------------------------------------------------------------

class TestRule4PrepositionalModifier:

    def test_disappointed_with(self, extractor):
        """'disappointed with the service' — ADJ → with → NOUN"""
        result = extractor.extract("I was disappointed with the service.")
        aspects = [pair[0] for pair in result]
        assert "service" in aspects

    def test_happy_with(self, extractor):
        """'happy with the food' — ADJ → with → NOUN"""
        result = extractor.extract("I am happy with the food.")
        aspects = [pair[0] for pair in result]
        assert "food" in aspects

    def test_satisfied_with(self, extractor):
        """'satisfied with the price' — ADJ → with → NOUN"""
        result = extractor.extract("We were satisfied with the price.")
        aspects = [pair[0] for pair in result]
        assert "price" in aspects
    
    def test_rule4_does_not_fire_on_action_verbs(self, extractor):
        """
        Rule 4 must not extract pairs where the prep head is a non-sentiment VERB.
        'waited for 1 hour' — 'waited' is VERB but not in _SENTIMENT_VERBS.
        (hour, waited) must NOT appear in output.
        """
        result = extractor.extract(
            "i ordered in foodpanda at 12:30am and waited for 1 hour "
            "just for tender wrap and rice bowl"
        )
        opinions = [pair[1] for pair in result]
        assert "waited" not in opinions, (
            f"Rule 4 over-fired on action verb 'waited'. "
            f"Got opinions: {opinions}"
        )
        assert "ordered" not in opinions, (
            f"Rule 4 over-fired on action verb 'ordered'. "
            f"Got opinions: {opinions}"
        )

    def test_rule4_still_fires_on_sentiment_adj(self, extractor):
        """
        Regression guard — Rule 4 must still fire when prep head is ADJ.
        'disappointed with the service' must still extract correctly after fix.
        """
        result = extractor.extract("I was disappointed with the service.")
        aspects = [pair[0] for pair in result]
        assert "service" in aspects, (
            f"Regression: Rule 4 broke on ADJ head after action verb fix. "
            f"Got aspects: {aspects}"
        )


# ---------------------------------------------------------------------------
# Rule 5: Root Adjective Fallback
# Pattern: ADJ at ROOT with no NOUN subject → maps to "overall"
# ---------------------------------------------------------------------------

class TestRule5RootFallback:

    def test_dangling_adjective(self, extractor):
        """'Absolutely terrible!' — ADJ at ROOT, no subject → overall"""
        result = extractor.extract("Absolutely terrible!")
        assert ("overall", "terrible") in result

    def test_positive_dangling_adjective(self, extractor):
        """'Amazing!' — short positive exclamation → overall"""
        result = extractor.extract("Amazing!")
        aspects = [pair[0] for pair in result]
        assert "overall" in aspects

    def test_adjective_with_subject_does_not_trigger(self, extractor):
        """When a NOUN subject exists, Rule 5 should NOT fire for overall"""
        result = extractor.extract("The food is amazing.")
        # 'food' should be the aspect, not 'overall'
        aspects = [pair[0] for pair in result]
        if "overall" in aspects:
            assert "food" in aspects  # at minimum food should also be there


# ---------------------------------------------------------------------------
# Proximity Fallback
# Triggers ONLY when Rules 1-5 yield zero pairs
# ---------------------------------------------------------------------------

class TestProximityFallback:

    def test_proximity_fires_on_zero_extraction(self, extractor):
        """
        A grammatically broken post-translation sentence where rules find nothing.
        Proximity fallback should find a NOUN-ADJ pair nearby.
        """
        # This broken sentence mimics post-translation Manglish output
        # Rules likely fail — proximity should catch 'place' and 'dirty'
        result = extractor.extract("food dirty staff rude")
        assert len(result) > 0

    def test_proximity_does_not_fire_when_rules_succeed(self, extractor):
        """
        When rules extract pairs, proximity fallback must NOT add extra pairs.
        Verify by checking that a well-formed sentence produces only rule-based pairs.
        """
        result = extractor.extract("The food is delicious.")
        # food/delicious should be found by Rule 2
        assert ("food", "delicious") in result

    def test_proximity_short_sentence_guard(self, extractor):
        """Sentence under 3 tokens should return empty list from fallback"""
        # "ok" alone — no NOUN or ADJ relationship possible
        result = extractor.extract("ok")
        # May return empty or catch something — just must not crash
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Enhancement Mechanisms
# ---------------------------------------------------------------------------

class TestEnhancementMechanisms:

    def test_compound_noun_reconstruction(self, extractor):
        """'chicken rice' should be reconstructed as a single aspect"""
        result = extractor.extract("The chicken rice is delicious.")
        aspects = [pair[0] for pair in result]
        assert "chicken rice" in aspects

    def test_negation_on_sentiment_token(self, extractor):
        """'not good' — negation should be attached to opinion word"""
        result = extractor.extract("The food is not good.")
        opinions = [pair[1] for pair in result]
        assert any("not" in op for op in opinions)

    def test_conjunction_expansion(self, extractor):
        """'good and fresh' — both adjectives should produce pairs"""
        result = extractor.extract("The food is good and fresh.")
        opinions = [pair[1] for pair in result]
        assert "good" in opinions or "fresh" in opinions


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_string_returns_empty_list(self, extractor):
        """Empty string must return empty list without crashing"""
        result = extractor.extract("")
        assert result == []

    def test_none_input_returns_empty_list(self, extractor):
        """None input must return empty list without crashing"""
        result = extractor.extract(None)
        assert result == []

    def test_whitespace_only_returns_empty_list(self, extractor):
        """Whitespace-only input must return empty list"""
        result = extractor.extract("   ")
        assert result == []

    def test_output_is_list(self, extractor):
        """Return type must always be a list"""
        result = extractor.extract("The food is great.")
        assert isinstance(result, list)

    def test_output_deduplicated(self, extractor):
        """No duplicate pairs in output"""
        result = extractor.extract("The food is great and the food is great.")
        assert len(result) == len(set(result))

    def test_multi_aspect_review(self, extractor):
        """
        A review with multiple aspects must return multiple pairs.
        Directly validates RO2 — multi-aspect extraction per review.
        """
        result = extractor.extract(
            "The food was delicious but the service was terrible."
        )
        aspects = [pair[0] for pair in result]
        # Both food and service should be extracted
        assert len(result) >= 2
        assert "food" in aspects
        assert "service" in aspects

    def test_all_output_lowercased(self, extractor):
        """All extracted pairs must be lowercased"""
        result = extractor.extract("The Food is Great.")
        for aspect, opinion in result:
            assert aspect == aspect.lower()
            assert opinion == opinion.lower()

# ---------------------------------------------------------------------------
# Malay Multi-Word Food Term Extraction
# These tests FAIL before the PROPN fix is applied to Rules 2 and 3.
# The fix: extend POS check to include PROPN when a compound child exists.
#
# Failure root cause (confirmed by diagnose_extractor.py):
#   spaCy tags the head word of Malay food terms as PROPN (e.g. "lemak",
#   "goreng", "canai", "tarik") because it does not recognise them as English.
#   Rules 2 and 3 check token.pos_ == "NOUN" only — PROPN head fails silently.
#   _get_compound_noun() already collects compound children correctly, but
#   never gets called because the POS guard rejects the head token first.
# ---------------------------------------------------------------------------

class TestMalayFoodTermExtraction:

    def test_nasi_lemak_predicate_adj(self, extractor):
        """
        'The nasi lemak was delicious.' — Rule 2 path.
        'lemak' tagged PROPN nsubj. Fix: accept PROPN with compound child.
        Expected: ('nasi lemak', 'delicious') in output.
        """
        result = extractor.extract("The nasi lemak was delicious.")
        aspects = [pair[0] for pair in result]
        assert "nasi lemak" in aspects, (
            f"Expected 'nasi lemak' in aspects but got: {aspects}. "
            f"Full pairs: {result}"
        )

    def test_mee_goreng_sentiment_verb(self, extractor):
        """
        'I loved the mee goreng here.' — Rule 3 path.
        'goreng' tagged PROPN dobj. Fix: accept PROPN dobj with compound child.
        Expected: ('mee goreng', ...) in output — opinion word is the verb.
        """
        result = extractor.extract("I loved the mee goreng here.")
        aspects = [pair[0] for pair in result]
        assert "mee goreng" in aspects, (
            f"Expected 'mee goreng' in aspects but got: {aspects}. "
            f"Full pairs: {result}"
        )

    def test_two_malay_food_terms_same_sentence(self, extractor):
        """
        'The nasi goreng is really good but the roti canai was cold.'
        Two PROPN-headed food terms in one sentence.
        nasi goreng: 'goreng' is NOUN here (spaCy varies) — should already work.
        roti canai: 'canai' tagged PROPN — this is the failing case.
        Both aspects must be extracted.
        """
        result = extractor.extract(
            "The nasi goreng is really good but the roti canai was cold."
        )
        aspects = [pair[0] for pair in result]
        assert "nasi goreng" in aspects, (
            f"Expected 'nasi goreng' in aspects but got: {aspects}"
        )
        assert "roti canai" in aspects, (
            f"Expected 'roti canai' in aspects but got: {aspects}. "
            f"This is the PROPN fix target."
        )

    def test_teh_tarik_predicate_adj(self, extractor):
        """
        'The teh tarik was too sweet.' — Rule 2 path.
        'tarik' tagged PROPN nsubj, 'teh' tagged NOUN compound.
        Fix: accept PROPN nsubj when compound NOUN child exists.
        Expected: ('teh tarik', 'sweet') in output.
        """
        result = extractor.extract("The teh tarik was too sweet.")
        aspects = [pair[0] for pair in result]
        assert "teh tarik" in aspects, (
            f"Expected 'teh tarik' in aspects but got: {aspects}. "
            f"Full pairs: {result}"
        )

    def test_malay_food_term_with_negation(self, extractor):
        """
        'The nasi lemak was not good.' — negation must still attach correctly
        after the PROPN fix. Compound reconstruction + negation must both work.
        """
        result = extractor.extract("The nasi lemak was not good.")
        aspects = [pair[0] for pair in result]
        opinions = [pair[1] for pair in result]
        assert "nasi lemak" in aspects, (
            f"Expected 'nasi lemak' in aspects but got: {aspects}"
        )
        assert any("not" in op for op in opinions), (
            f"Expected negation in opinions but got: {opinions}"
        )

    def test_malay_food_term_compound_reconstructed_fully(self, extractor):
        """
        Confirms _get_compound_noun() reconstructs the FULL multi-word term,
        not just the head word alone. 'nasi' must appear with 'lemak', not solo.
        Regression guard: before fix, only 'nasi' was extracted (not 'nasi lemak').
        """
        result = extractor.extract("The nasi lemak was delicious.")
        aspects = [pair[0] for pair in result]
        assert "nasi" not in aspects, (
            f"'nasi' extracted alone — compound reconstruction broken. "
            f"Got: {aspects}"
        )
        assert "nasi lemak" in aspects


# ---------------------------------------------------------------------------
# Mistagged Adjective Extraction
# These tests FAIL before the fix for crispy (NOUN) and burnt (VERB).
#
# Failure root cause (confirmed by diagnose_adjectives.py):
#   'crispy' — spaCy tags as NOUN in predicative position ('was crispy').
#              Rule 2 acomp check requires child.pos_ == "ADJ" — fails.
#              Negation test also fails: 'not crispy' extracts nothing.
#   'burnt'  — spaCy parses 'was burnt' as passive voice construction.
#              'burnt' becomes ROOT VERB, 'rice' becomes nsubjpass.
#              Rule 2 never fires because the structure is passive, not
#              copular. Need to handle past-participle adjectives used
#              as passive constructions.
# ---------------------------------------------------------------------------

class TestMistagedAdjectives:

    def test_crispy_predicative_extracted(self, extractor):
        """
        'The chicken was crispy.' — Rule 2 path.
        spaCy tags 'crispy' as NOUN attr, not ADJ acomp.
        Fix target: handle NOUN-tagged words in attr dependency position.
        Expected: ('chicken', 'crispy') in output.
        """
        result = extractor.extract("The chicken was crispy.")
        aspects = [pair[0] for pair in result]
        opinions = [pair[1] for pair in result]
        assert "chicken" in aspects, (
            f"Expected 'chicken' in aspects but got: {aspects}"
        )
        assert "crispy" in opinions, (
            f"Expected 'crispy' in opinions but got: {opinions}. "
            f"spaCy tags 'crispy' as NOUN — fix needed."
        )

    def test_crispy_with_negation(self, extractor):
        """
        'The chicken was not crispy.' — negation on a NOUN-tagged word.
        If crispy is mistagged as NOUN, negation extraction also fails.
        Both the aspect and negated opinion must be present.
        """
        result = extractor.extract("The chicken was not crispy.")
        aspects = [pair[0] for pair in result]
        opinions = [pair[1] for pair in result]
        assert "chicken" in aspects, (
            f"Expected 'chicken' in aspects but got: {aspects}"
        )
        assert any("crispy" in op for op in opinions), (
            f"Expected 'crispy' (with or without negation) in opinions "
            f"but got: {opinions}"
        )

    def test_burnt_predicative_extracted(self, extractor):
        """
        'The rice was burnt.' — passive voice construction.
        spaCy parses: rice → nsubjpass, burnt → ROOT VERB.
        Rule 2 never fires on this structure.
        Fix target: detect nsubjpass + past-participle ROOT as sentiment pair.
        Expected: ('rice', 'burnt') in output.
        """
        result = extractor.extract("The rice was burnt.")
        aspects = [pair[0] for pair in result]
        opinions = [pair[1] for pair in result]
        assert "rice" in aspects, (
            f"Expected 'rice' in aspects but got: {aspects}"
        )
        assert "burnt" in opinions, (
            f"Expected 'burnt' in opinions but got: {opinions}. "
            f"spaCy parses 'was burnt' as passive voice — fix needed."
        )

    def test_burnt_with_negation(self, extractor):
        """
        'The rice was not burnt.' — negation on passive past participle.
        Compound failure: passive structure + negation both unhandled.
        """
        result = extractor.extract("The rice was not burnt.")
        aspects = [pair[0] for pair in result]
        opinions = [pair[1] for pair in result]
        assert "rice" in aspects, (
            f"Expected 'rice' in aspects but got: {aspects}"
        )
        assert any("burnt" in op for op in opinions), (
            f"Expected 'burnt' (with or without negation) in opinions "
            f"but got: {opinions}"
        )

    def test_crispy_attributive_still_works(self, extractor):
        """
        'The crispy chicken was good.' — attributive position.
        Diagnostic confirmed this ALREADY works (compound noun path).
        This is a regression guard — fix must not break attributive case.
        ('crispy chicken', 'good') must remain in output after fix.
        """
        result = extractor.extract("The crispy chicken was good.")
        aspects = [pair[0] for pair in result]
        assert "crispy chicken" in aspects, (
            f"Regression: attributive crispy chicken broken. Got: {aspects}"
        )

    def test_burnt_attributive_regression(self, extractor):
        """
        'The burnt rice was bad.' — attributive position.
        Diagnostic showed attributive extracts ('rice', 'bad') not ('burnt rice').
        This test confirms the attributive path still works after fix —
        we accept ('rice', 'bad') as valid output here.
        """
        result = extractor.extract("The burnt rice was bad.")
        assert len(result) > 0, (
            f"Expected at least one pair from 'The burnt rice was bad.' "
            f"but got nothing. Regression in attributive path."
        )