"""
normaliser.py — Manglish Slang Normalisation Engine

Converts Manglish (Malaysian English code-switching) into a form
suitable for downstream translation and NLP processing.

IMPORTANT: All functions in this module expect lowercase input.
           standardize_text() from the data pipeline must be called
           before any function here is invoked.

Dictionary is loaded from data/manglish_dict.json on module startup.
Known limitations (deliberate design decisions, not bugs):
    - "agak" → "knew"  : context-dependent; correct for "dah agak dah"
    - "out"  → "bad"   : context-dependent; correct for "makanan tu mmg out"
      Phrase protection entries ("sold out", "out of stock") prevent the
      most common false positives — remaining cases noted as error analysis.
"""

import re
import json
import os

# ---------------------------------------------------------------------------
# Dictionary Loading
# ---------------------------------------------------------------------------

_DICT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "..", "data", "manglish_dict.json"
)


def _load_dict(path: str) -> tuple[dict, dict, list]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return (
        data["phrase_lookup"],
        data["word_lookup"],
        data["noise_particles"],
    )

PHRASE_LOOKUP, WORD_LOOKUP, NOISE_PARTICLES = _load_dict(_DICT_PATH)


# ---------------------------------------------------------------------------
# Core Normalisation Function
# ---------------------------------------------------------------------------

def resolve_local_terms(text: str,
                        phrase_map: dict = PHRASE_LOOKUP,
                        word_map: dict   = WORD_LOOKUP,
                        noise: list      = NOISE_PARTICLES) -> str:
    """
    Four-stage Manglish normalisation pipeline.

    Expects lowercase input — must be called after standardize_text().

    Stages:
        Step 0 — Repeated character collapse  : "gooood" → "good"
        Step 1 — X-prefix detacher            : "xbagi"  → "x bagi"
        Step 2 — Phrase replacement (N-grams) : fires before single-word
                 pass so protected phrases ("sold out") are caught first
        Step 3 — Single-word tokenisation     : slang lookup + noise drop

    Args:
        text       : Lowercase, standardised input string.
        phrase_map : N-gram slang dictionary (default: loaded from JSON).
        word_map   : Unigram slang dictionary (default: loaded from JSON).
        noise      : Discourse particles to drop entirely (default: from JSON).

    Returns:
        Normalised string ready for translation.
    """
    if not isinstance(text, str):
        return ""

    # Step 0: Repeated character collapse
    # Fires only on 3+ repetitions — preserves legitimate doubles (staff, good)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # Step 1: X-prefix detacher
    # "xbagi" → "x bagi" so Step 3 can resolve "x" → "not" and "bagi" → "give"
    text = re.sub(r'\bx([a-z]+)\b', r'x \1', text)

    # Step 2: Phrase replacement (N-grams)
    # Must run before Step 3 to protect multi-word expressions from
    # single-token substitution (e.g. "sold out" caught here before
    # "out" → "bad" fires in Step 3)
    for phrase, replacement in phrase_map.items():
        text = re.sub(rf'\b{re.escape(phrase)}\b', replacement, text)

    # Step 3: Single-word tokenisation + noise filtering
    tokens = text.split()
    normalised_tokens = []

    for token in tokens:
        # Strip trailing punctuation first — applies to both noise check and word lookup
        stripped = token.rstrip('!?.,;:')
        punctuation = token[len(stripped):]

        # Drop discourse particles entirely — carry no semantic content
        if stripped in noise:
            continue

        # Dictionary lookup on stripped token
        mapped = word_map.get(stripped, stripped)

        # Reattach punctuation
        normalised_tokens.append(mapped + punctuation)

    return " ".join(normalised_tokens)