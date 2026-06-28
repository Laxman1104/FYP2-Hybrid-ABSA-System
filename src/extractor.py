"""
extractor.py — Aspect-Sentiment Pair Extraction Engine

Extracts (aspect, opinion_word) pairs from standardised English text
using dependency grammar rules. Each rule is implemented as an independent
private method for testability. extract() orchestrates all rules.

IMPORTANT: Expects standardised English input — text must have passed through
           standardize_text() and, for Manglish input, resolve_local_terms()
           and translate_to_standard_english() before reaching this module.

Extraction rules (in priority order):
    Rule 1 — Adjectival modifier (amod)         : "delicious food"
    Rule 2 — Predicate adjective (nsubj + acomp): "The food is great"
              Extended: accepts PROPN nsubj with compound child (Malay food terms)
              Extended: accepts attr-dep NOUN/PROPN complement (e.g. "crispy")
    Rule 3 — Sentiment verb (dobj, filtered)    : "I loved the service"
              Extended: accepts PROPN dobj with compound child (Malay food terms)
    Rule 4 — Prepositional modifier (prep+pobj) : "disappointed with the service"
    Rule 5 — Root fallback (dangling adjective) : "Absolutely terrible!"
    Rule 6 — Passive past participle (nsubjpass) : "The rice was burnt"

Proximity fallback      
    Triggers ONLY when Rules 1–6 yield zero pairs.
    Finds nearest NOUN–ADJ pair within `proximity_window` tokens.
    Handles post-translation Manglish where parser structure is unreliable.
    Window size is configurable at instantiation (default: 4).

Known limitation:
    en_core_web_md is trained on standard English. Parser accuracy degrades
    on heavy Manglish even after normalisation and translation — acknowledged
    as a system limitation in the report.

Change log:
    Rule 2 extended — two additions:
        (a) PROPN nsubj accepted when token has at least one compound child.
            Fixes Malay multi-word food terms ("nasi lemak", "teh tarik") where
            spaCy tags the head word as PROPN due to OOV status.
        (b) attr dependency accepted alongside acomp for the sentiment complement.
            Fixes words spaCy misstags as NOUN in attr position ("crispy").
    Rule 3 extended — one addition:
        PROPN dobj accepted when token has at least one compound child.
        Fixes "I loved the mee goreng" where "goreng" is PROPN dobj.
    Rule 6 added — passive past participle:
        Handles nsubjpass + ROOT VERB pattern ("The rice was burnt").
        spaCy parses passive voice as VERB ROOT, not ADJ acomp — Rule 2
        never fired on this structure. Rule 6 detects it directly.
"""

import spacy


# ---------------------------------------------------------------------------
# Sentiment verb whitelist for Rule 3
# Only pairs where the governing verb is in this set are extracted.
# Prevents noise pairs like (pizza, ate) or (table, got).
# ---------------------------------------------------------------------------

_SENTIMENT_VERBS = {
    "love", "hate", "enjoy", "recommend", "avoid",
    "prefer", "dislike", "appreciate", "disappoint",
    "like", "adore", "detest", "regret", "suggest"
}


# ---------------------------------------------------------------------------
# Extractor Class
# ---------------------------------------------------------------------------

class AspectSentimentExtractor:
    """
    Extracts (aspect, opinion_word) pairs using dependency grammar rules.

    Args:
        nlp_model        : Loaded spaCy model (en_core_web_md recommended).
        proximity_window : Token window for proximity fallback (default: 4).
                           Tune this during evaluation without changing code.
    """

    def __init__(self, nlp_model: spacy.language.Language, proximity_window: int = 4):
        self.nlp = nlp_model
        self.proximity_window = proximity_window

    # -----------------------------------------------------------------------
    # Helper Methods
    # -----------------------------------------------------------------------

    def _get_compound_noun(self, token) -> str:
        """
        Reconstructs multi-word aspects by collecting compound children.
        e.g. "chicken" + "rice" → "chicken rice"
        Works for both NOUN and PROPN head tokens.
        """
        compounds = [child.text for child in token.lefts if child.dep_ == "compound"]
        compounds.append(token.text)
        return " ".join(compounds)

    def _has_compound_child(self, token) -> bool:
        """
        Returns True if token has at least one compound-dependency child.
        Used as a guard when accepting PROPN tokens — a standalone PROPN
        (restaurant name, person name) is rejected; a PROPN with a compound
        child is accepted as the head of a multi-word food term.
        """
        return any(child.dep_ == "compound" for child in token.lefts)

    def _get_negation(self, token) -> str:
        """
        Attaches negation prefix if a neg child exists.
        e.g. "not friendly" instead of "friendly"
        """
        for child in token.children:
            if child.dep_ == "neg":
                return f"{child.text} {token.text}"
        return token.text

    def _get_conjunctions(self, token) -> list:
        """
        Expands conjunctions so "good and fresh" produces two separate pairs.
        Returns list of conjoined tokens (excluding the base token itself).
        """
        conjs = [child for child in token.rights if child.dep_ == "conj"]
        if token.dep_ == "conj":
            conjs.append(token.head)
        return list(set(conjs))

    def _process_sentiment_targets(self, aspect: str, base_sentiment_token,
                                   parent_verb=None) -> list:
        """
        Resolves negation and conjunctions for a given sentiment token.
        Handles cases like "not friendly" and "good and fresh".

        Returns list of (aspect, opinion_word) tuples.
        """
        extracted = []
        all_sentiment_tokens = [base_sentiment_token] + self._get_conjunctions(base_sentiment_token)

        # Check for negation on the parent verb (e.g. "did not enjoy")
        verb_negation = ""
        if parent_verb:
            for child in parent_verb.children:
                if child.dep_ == "neg":
                    verb_negation = f"{child.text} "
                    break

        for sent_token in all_sentiment_tokens:
            # Token-level negation takes priority over verb-level negation
            token_negation = ""
            for child in sent_token.children:
                if child.dep_ == "neg":
                    token_negation = f"{child.text} "
                    break

            final_negation = token_negation if token_negation else verb_negation
            final_sentiment_str = f"{final_negation}{sent_token.text}"

            extracted.append((aspect.lower(), final_sentiment_str.lower()))

        return extracted

    # -----------------------------------------------------------------------
    # Extraction Rules
    # -----------------------------------------------------------------------

    def _apply_rule1_amod(self, doc) -> list:
        """
        Rule 1: Adjectival modifier (amod)
        Pattern : NOUN with ADJ child via amod dependency
        Example : "delicious food" → (food, delicious)
        """
        pairs = []
        for token in doc:
            if token.pos_ == "NOUN":
                for child in token.children:
                    if child.dep_ == "amod" and child.pos_ == "ADJ":
                        aspect = self._get_compound_noun(token)
                        pairs.extend(self._process_sentiment_targets(aspect, child))
        return pairs

    def _apply_rule2_predicate_adj(self, doc) -> list:
        """
        Rule 2: Predicate adjective (nsubj + acomp) — extended.

        Original pattern:
            NOUN nsubj → head verb → ADJ/PROPN acomp
            Example: "The food is great" → (food, great)

        Extension (a) — PROPN nsubj with compound child:
            Accepts PROPN nsubj when it has at least one compound child.
            Fixes Malay food terms: "The nasi lemak was delicious"
            spaCy tags "lemak" as PROPN nsubj; "nasi" is NOUN compound child.
            Guard: standalone PROPN (no compound child) is still rejected.

        Extension (b) — attr dependency alongside acomp:
            Accepts sentiment complement in attr position as well as acomp.
            Fixes words spaCy misstags as NOUN in attr position.
            Example: "The chicken was crispy" — spaCy: crispy → NOUN attr.
            The attr token is used as the opinion word directly.
        """
        pairs = []
        for token in doc:
            # Accept NOUN nsubj (original) OR PROPN nsubj with compound child
            is_noun_subject = token.dep_ == "nsubj" and token.pos_ == "NOUN"
            is_propn_subject = (
                token.dep_ == "nsubj"
                and token.pos_ == "PROPN"
                and self._has_compound_child(token)
            )

            if not (is_noun_subject or is_propn_subject):
                continue

            head_verb = token.head
            aspect = self._get_compound_noun(token)

            for child in head_verb.children:
                # Original: acomp with ADJ or PROPN
                if child.dep_ == "acomp" and child.pos_ in ["ADJ", "PROPN"]:
                    pairs.extend(
                        self._process_sentiment_targets(aspect, child, parent_verb=head_verb)
                    )
                # Extension (b): attr position — covers mistagged words like "crispy"
                elif child.dep_ == "attr" and child.pos_ in ["ADJ", "NOUN", "PROPN"]:
                    pairs.extend(
                        self._process_sentiment_targets(aspect, child, parent_verb=head_verb)
                    )

        return pairs

    def _apply_rule3_sentiment_verb(self, doc) -> list:
        """
        Rule 3: Sentiment verb (dobj, filtered) — extended.

        Original pattern:
            NOUN dobj of sentiment VERB
            Example: "I loved the service" → (service, loved)

        Extension — PROPN dobj with compound child:
            Accepts PROPN dobj when it has at least one compound child.
            Fixes Malay food terms: "I loved the mee goreng"
            spaCy tags "goreng" as PROPN dobj; "mee" is PROPN compound child.
            Guard: standalone PROPN dobj (no compound child) still rejected.

        IMPORTANT: Verb must be in _SENTIMENT_VERBS whitelist.
        Prevents noise pairs like (pizza, ate) or (table, got).
        """
        pairs = []
        for token in doc:
            is_noun_dobj = token.dep_ == "dobj" and token.pos_ == "NOUN"
            is_propn_dobj = (
                token.dep_ == "dobj"
                and token.pos_ == "PROPN"
                and self._has_compound_child(token)
            )

            if not (is_noun_dobj or is_propn_dobj):
                continue

            head_verb = token.head
            if head_verb.pos_ == "VERB" and head_verb.lemma_.lower() in _SENTIMENT_VERBS:
                aspect = self._get_compound_noun(token)
                pairs.extend(
                    self._process_sentiment_targets(aspect, head_verb, parent_verb=head_verb)
                )

        return pairs

    def _apply_rule4_prep_modifier(self, doc) -> list:
        """
        Rule 4: Prepositional modifier (prep + pobj)
        Pattern : ADJ or VERB → PREP → NOUN object
        Example : "disappointed with the service" → (service, disappointed)
        """
        pairs = []
        for token in doc:
            if token.dep_ == "prep":
                head_sentiment = token.head
                if head_sentiment.pos_ == "ADJ" or (
                    head_sentiment.pos_ == "VERB"
                    and head_sentiment.lemma_.lower() in _SENTIMENT_VERBS
                ):
                    for child in token.children:
                        if child.dep_ == "pobj" and child.pos_ == "NOUN":
                            aspect = self._get_compound_noun(child)
                            pairs.extend(
                                self._process_sentiment_targets(aspect, head_sentiment)
                            )
        return pairs

    def _apply_rule5_root_fallback(self, doc) -> list:
        """
        Rule 5: Root fallback for dangling adjectives
        Pattern : ADJ at ROOT with no NOUN subject → maps to "overall"
        Example : "Absolutely terrible!" → (overall, terrible)
        """
        pairs = []
        for token in doc:
            if token.pos_ == "ADJ" and token.dep_ == "ROOT":
                has_noun_subject = any(child.pos_ == "NOUN" for child in token.children)
                if not has_noun_subject:
                    pairs.extend(self._process_sentiment_targets("overall", token))
        return pairs

    def _apply_rule6_passive_participle(self, doc) -> list:
        """
        Rule 6: Passive past participle (nsubjpass + ROOT VERB)
        Pattern : NOUN nsubjpass → past-participle ROOT VERB
        Example : "The rice was burnt" → (rice, burnt)

        Motivation:
            spaCy parses "The rice was burnt" as passive voice:
                rice → nsubjpass (passive subject)
                burnt → ROOT VERB (past participle acting as predicate)
            Rule 2 never fires on this structure because:
                (a) "rice" is nsubjpass, not nsubj
                (b) "burnt" is ROOT VERB, not acomp
            Rule 6 detects this pattern directly.

        Guard — only fires when ROOT verb has passive subject (nsubjpass).
        Prevents false matches on active-voice ROOT verbs.

        Negation is handled via _get_negation() on the ROOT verb token,
        which checks for neg children (e.g. "not burnt" → "not burnt").
        """
        pairs = []
        for token in doc:
            if token.dep_ == "nsubjpass" and token.pos_ == "NOUN":
                head_verb = token.head
                # Only fire when head is ROOT — avoids subordinate clauses
                if head_verb.dep_ == "ROOT" and head_verb.pos_ == "VERB":
                    aspect = self._get_compound_noun(token)
                    opinion = self._get_negation(head_verb)
                    pairs.append((aspect.lower(), opinion.lower()))
        return pairs

    def _proximity_fallback(self, doc) -> list:
        """
        Proximity fallback — triggers ONLY when Rules 1–6 yield zero pairs.

        Scans the token sequence for NOUN–ADJ pairs within `proximity_window`
        tokens of each other. Handles post-translation Manglish where the
        dependency parse structure is too unreliable for rule-based extraction.

        Window size is set at class instantiation (default: 4).
        Tune during evaluation without changing code.
        """

        if len(doc) < 3:          # ← guard lives HERE only
            return []
        pairs = []
        tokens = list(doc)

        for i, token in enumerate(tokens):
            if token.pos_ == "NOUN":
                window_start = max(0, i - self.proximity_window)
                window_end   = min(len(tokens), i + self.proximity_window + 1)

                for j in range(window_start, window_end):
                    if i == j:
                        continue
                    candidate = tokens[j]
                    if candidate.pos_ == "ADJ":
                        aspect  = self._get_compound_noun(token)
                        opinion = self._get_negation(candidate)
                        pairs.append((aspect.lower(), opinion.lower()))

        return list(set(pairs))

    # -----------------------------------------------------------------------
    # Public Interface
    # -----------------------------------------------------------------------

    def extract(self, text: str) -> list:
        if not isinstance(text, str) or not text.strip():
            return []

        doc = self.nlp(text)

        pairs = []
        pairs += self._apply_rule1_amod(doc)
        pairs += self._apply_rule2_predicate_adj(doc)
        pairs += self._apply_rule3_sentiment_verb(doc)
        pairs += self._apply_rule4_prep_modifier(doc)
        pairs += self._apply_rule5_root_fallback(doc)
        pairs += self._apply_rule6_passive_participle(doc)

        # Proximity fallback — only when all rules return nothing
        if not pairs:
            pairs = self._proximity_fallback(doc)   # guard fires inside here

        return list(set(pairs))
        # ← NO length check here at all