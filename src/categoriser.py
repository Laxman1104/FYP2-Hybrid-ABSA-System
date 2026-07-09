"""
categoriser.py — Aspect Categorisation Engine

Maps raw extracted aspect terms to one of five macro-categories using
a two-stage process:
    Stage 1 — Keyword override lookup (exact match, O(1))
    Stage 2 — spaCy vector similarity against anchor word groups

Macro-categories (canonical names — must match annotation schema exactly):
    food | service | ambiance | price | overall | other

IMPORTANT: "other" is only produced when on_no_match="other" is set at
           instantiation. Default behaviour is "drop" (preserves SemEval
           training data quality). Use "other" for dashboard and Malaysian
           gold standard evaluation.

Keyword overrides (Stage 1) were derived from EDA findings — 23 terms
in the top 50 SemEval aspects that fell below the vector similarity
threshold and were being dropped incorrectly. Overrides ensure these
high-frequency terms are always correctly categorised.
"""

import spacy


# ---------------------------------------------------------------------------
# Keyword Override Table
# Stage 1 — fires before vector similarity. Derived from EDA top-50 analysis.
# Covers 23 confirmed unmapped terms. Extend here if new gaps are found.
# ---------------------------------------------------------------------------

_KEYWORD_OVERRIDES = {
    "wine":        "food",
    "ambience":    "ambiance",
    "fish":        "food",
    "priced":      "price",
    "price":      "price",
    "wine list":   "food",
    "waiters":     "service",
    "owner":       "service",
    "reservation": "service",
    "wait":        "service",
    "bagels":      "food",
    "appetizers":  "food",
    "rice":        "food",
    "wait staff":  "service",
    "thai food":   "food",
    "served":      "service",
    "tables":      "ambiance",
    "music":       "ambiance",
    "salads":      "food",
    "sauce":       "food",
    "ingredients": "food",
    "taste":       "food",
    "quality":     "overall",
    "appetizer":   "food",
    "shop" :       "ambiance",
    "kedai" :      "ambiance",
    "order" :      "service",
    "employee" :   "service",
    "parking":    "ambiance",
    "restaurant": "ambiance",
    "mamak":      "ambiance",
    "food stall": "ambiance",
    "choices":    "food",
    "choice": "food",
    
}

# Load Malaysian food terms dynamically and add to keyword overrides
import os

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_FOOD_TERMS_PATH = os.path.join(_CURRENT_DIR, "malaysian_food_terms.txt")

if os.path.exists(_FOOD_TERMS_PATH):
    try:
        with open(_FOOD_TERMS_PATH, "r", encoding="utf-8") as _f:
            for _line in _f:
                _term = _line.strip().lower()
                if _term:
                    # Map local terms to "food" category if not already overridden
                    if _term not in _KEYWORD_OVERRIDES:
                        _KEYWORD_OVERRIDES[_term] = "food"
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Anchor Groups for Vector Similarity
# Stage 2 — fires only when keyword override yields no match.
# Multiple anchor words per category broaden the similarity target area.
# Canonical category names must match annotation schema exactly.
# ---------------------------------------------------------------------------

_ANCHOR_GROUPS = {
    "food":     ["food", "meal", "dish", "taste", "drink"],
    "service":  ["service", "staff", "waiter", "waitress", "management"],
    "ambiance": ["ambiance", "atmosphere", "place", "cleanliness", "vibe","shop"],
    "price":    ["price", "cost", "value", "money"],
    "overall":  ["overall", "experience", "visit"],
}


# ---------------------------------------------------------------------------
# Categoriser Class
# ---------------------------------------------------------------------------

class AspectCategoriser:
    """
    Maps (aspect, opinion_word) pairs to (macro_category, opinion_word) pairs.

    Two-stage process:
        Stage 1 — Keyword override lookup (exact string match)
        Stage 2 — spaCy vector similarity against anchor word groups

    Args:
        nlp_model            : Loaded spaCy model (en_core_web_md required
                               for word vectors — sm model has no vectors).
        similarity_threshold : Minimum cosine similarity to accept a category
                               match in Stage 2 (default: 0.45).
        on_no_match          : Behaviour when both stages fail to match.
                               "drop"  — pair is silently excluded (default).
                                         Use for SemEval training data.
                               "other" — pair is assigned category "other".
                                         Use for dashboard and Malaysian
                                         gold standard evaluation.
    """

    def __init__(self,
                 nlp_model: spacy.language.Language,
                 similarity_threshold: float = 0.45,
                 on_no_match: str = "drop"):

        if on_no_match not in ("drop", "other"):
            raise ValueError(f"on_no_match must be 'drop' or 'other', got '{on_no_match}'")

        self.nlp       = nlp_model
        self.threshold = similarity_threshold
        self.on_no_match = on_no_match

        # Pre-vectorise all anchor words once on instantiation
        # Avoids re-computing vectors on every categorise() call
        self.anchor_tokens = {
            category: [self.nlp(word) for word in words]
            for category, words in _ANCHOR_GROUPS.items()
        }

    def _stage1_keyword_override(self, aspect: str):
        """
        Stage 1: Exact keyword lookup against override table.
        Returns category string if matched, None otherwise.
        O(1) lookup — always runs before vector similarity.
        """
        return _KEYWORD_OVERRIDES.get(aspect.lower().strip())

    def _stage2_vector_similarity(self, aspect: str):
        """
        Stage 2: spaCy vector similarity against anchor groups.
        Returns (category, score) for best match above threshold,
        or (None, 0.0) if no anchor group meets the threshold.

        Requires en_core_web_md — sm model has no word vectors.
        """
        aspect_token = self.nlp(aspect)

        if not aspect_token.has_vector:
            return None, 0.0

        best_match   = None
        highest_score = 0.0

        for category, anchor_tokens in self.anchor_tokens.items():
            for anchor_token in anchor_tokens:
                score = aspect_token.similarity(anchor_token)
                if score > highest_score:
                    highest_score = score
                    best_match    = category

        if highest_score >= self.threshold:
            return best_match, highest_score

        return None, highest_score

    def categorise(self, raw_pairs: list) -> list:
        """
        Maps raw (aspect, opinion_word) pairs to (macro_category, opinion_word).

        Two-stage resolution per pair:
            1. Keyword override — if matched, category assigned immediately
            2. Vector similarity — if above threshold, category assigned
            3. No match — dropped (on_no_match="drop") or assigned "other"

        Args:
            raw_pairs : List of (aspect, opinion_word) tuples from extractor.

        Returns:
            Deduplicated list of (macro_category, opinion_word) tuples.
            Canonical category names: food | service | ambiance | price |
                                      overall | other
        """
        structured_data = []

        for aspect, sentiment in raw_pairs:
            if not aspect or not isinstance(aspect, str) or aspect.isspace():
                continue

            # Stage 1 — keyword override
            category = self._stage1_keyword_override(aspect)

            if category is not None:
                structured_data.append((category, sentiment))
                continue

            # Stage 2 — vector similarity
            category, _ = self._stage2_vector_similarity(aspect)

            if category is not None:
                structured_data.append((category, sentiment))
                continue

            # No match — drop or assign "other" based on instantiation setting
            if self.on_no_match == "other":
                structured_data.append(("other", sentiment))
            # if "drop": append nothing — SemEval training behaviour preserved

        return list(set(structured_data))