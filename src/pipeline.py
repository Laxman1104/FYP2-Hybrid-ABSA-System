"""
pipeline.py — Feature Extraction Pipeline

Chains all src/ components into a single callable for feature extraction.
Produces (category, opinion_word) pairs ready for model inference.

This module is the feature extraction layer only. It does NOT run sentiment
classification. Sentiment polarity (positive/negative/neutral) is assigned
by the trained model in the Streamlit app after run_pipeline() returns.

Correct usage in Sandbox dashboard:
    # Step 1 — extract features
    result = run_pipeline(text, mode="manglish")

    # Step 2 — model inference per pair
    for category, opinion_word in result["results"]:
        context   = build_context_window(result["translated"], opinion_word)
        sentiment = model.predict([context])[0]
        # assemble (category, opinion_word, sentiment) for display

Pipeline modes:
    "english"  — skip normaliser and translator entirely.
                 Use for SemEval batch processing and model training.
    "manglish" — run full pipeline: normalise → translate → extract → categorise.
                 Use for Malaysian batch processing.
    "auto"     — run normaliser always, translate only if output differs from input.
                 Use for Sandbox dashboard where input language is unknown.

Return format (always a dict regardless of mode or caller):
    {
        "original":       str        — raw input text
        "normalised":     str|None   — after normaliser (None if mode="english")
        "translated":     str|None   — after translator (None if mode="english"
                                       or was_translated=False in auto mode)
        "was_translated": bool       — True if translation meaningfully changed text
        "pairs":          list       — raw (aspect_term, opinion_word) from extractor
        "results":        list       — (category, opinion_word) after categorisation
    }

Callers read only the keys they need:
    Model training notebook  — reads "results" only
    Sandbox dashboard        — reads all keys to render each pipeline step
"""

import spacy

from normaliser import resolve_local_terms, PHRASE_LOOKUP, WORD_LOOKUP, NOISE_PARTICLES
from translator import translate_to_standard_english
from extractor import AspectSentimentExtractor
from categoriser import AspectCategoriser
from utils import standardize_text


# ---------------------------------------------------------------------------
# Module-level component initialisation
# Loaded once on import — not re-instantiated on every run_pipeline() call
# ---------------------------------------------------------------------------

_NLP = spacy.load("en_core_web_md")

_EXTRACTOR  = AspectSentimentExtractor(nlp_model=_NLP)
_CATEGORISER = AspectCategoriser(nlp_model=_NLP, on_no_match="drop")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

_VALID_MODES = ("english", "manglish", "auto")


def run_pipeline(text: str,
                 mode: str = "english",
                 extractor: AspectSentimentExtractor  = None,
                 categoriser: AspectCategoriser        = None) -> dict:
    """
    Runs the full feature extraction pipeline on a single review string.

    Args:
        text        : Raw input text (any language/mode).
        mode        : Pipeline mode — "english" | "manglish" | "auto".
                      See module docstring for full description.
        extractor   : Optional custom AspectSentimentExtractor instance.
                      Defaults to module-level _EXTRACTOR.
        categoriser : Optional custom AspectCategoriser instance.
                      Defaults to module-level _CATEGORISER.
                      Pass AspectCategoriser(on_no_match="other") for
                      Malaysian gold standard and dashboard evaluation.

    Returns:
        Dict with keys: original, normalised, translated,
                        was_translated, pairs, results.
        See module docstring for full return format specification.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got '{mode}'")

    # Use module-level defaults if no custom instances passed
    _ext = extractor  or _EXTRACTOR
    _cat = categoriser or _CATEGORISER

    # Initialise intermediate values — None signals step was skipped
    normalised    = None
    translated    = None
    was_translated = False

    # ------------------------------------------------------------------
    # Step 1 — Text standardisation (always runs regardless of mode)
    # ------------------------------------------------------------------
    standardised = standardize_text(text)

    # ------------------------------------------------------------------
    # Step 2 — Normalisation and translation (mode-dependent)
    # ------------------------------------------------------------------

    if mode == "english":
        # Skip normaliser and translator entirely
        # text_for_extraction is the standardised text
        text_for_extraction = standardised

    elif mode == "manglish":
        # Full pipeline — normalise then translate unconditionally
        normalised = resolve_local_terms(
            standardised, PHRASE_LOOKUP, WORD_LOOKUP, NOISE_PARTICLES
        )
        translation_result = translate_to_standard_english(normalised)
        translated         = translation_result["translated"]
        was_translated     = translation_result["was_translated"]
        text_for_extraction = translated

    elif mode == "auto":
        # Always normalise — translate only if output differs from input
        normalised = resolve_local_terms(
            standardised, PHRASE_LOOKUP, WORD_LOOKUP, NOISE_PARTICLES
        )
        translation_result = translate_to_standard_english(normalised)
        was_translated     = translation_result["was_translated"]

        if was_translated:
            translated          = translation_result["translated"]
            text_for_extraction = translated
        else:
            # Translation changed nothing — use normalised text directly
            text_for_extraction = normalised

    # ------------------------------------------------------------------
    # Step 3 — Aspect-opinion pair extraction
    # ------------------------------------------------------------------
    pairs = _ext.extract(text_for_extraction)

    # ------------------------------------------------------------------
    # Step 4 — Aspect categorisation
    # ------------------------------------------------------------------
    results = _cat.categorise(pairs)

    return {
        "original":       text,
        "normalised":     normalised,
        "translated":     translated,
        "was_translated": was_translated,
        "pairs":          pairs,
        "results":        results,
    }