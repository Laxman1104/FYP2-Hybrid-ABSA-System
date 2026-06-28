"""
translator.py — Translation Engine

Translates normalised Manglish/Malay text to standard English using
Google Translate via the deep_translator library.

IMPORTANT: This module expects input that has already passed through
           normaliser.py. Do not call this on raw, un-normalised text.

Returns a dictionary with three fields so the Sandbox dashboard can
render pipeline steps selectively:
    {
        "original"       : str   — the input text passed to this function
        "translated"     : str   — the translation result (or original on failure)
        "was_translated" : bool  — True if output meaningfully differs from input
    }

`was_translated` is a UI display flag only. It has no effect on the
extraction or classification pipeline. The dashboard uses it to decide
whether to render the translation step — if False, the step is skipped
entirely to avoid showing identical text twice.
"""

import html
import time

from deep_translator import GoogleTranslator

# ---------------------------------------------------------------------------
# Module-level translator instance
# Instantiated once on import — not on every function call
# ---------------------------------------------------------------------------

_TRANSLATOR = GoogleTranslator(source='malay', target='en')

_MAX_RETRIES = 3
_RETRY_DELAY = 1 


# ---------------------------------------------------------------------------
# Core Translation Function
# ---------------------------------------------------------------------------

def translate_to_standard_english(text: str) -> dict:
    """
    Translates normalised Manglish/Malay text to standard English.

    Expects lowercase, normalised input — must be called after
    resolve_local_terms() from normaliser.py.

    Retry logic: 3 attempts with 1-second delay between each.
    On final failure, returns original text with was_translated=False
    so the pipeline continues uninterrupted.

    Args:
        text : Normalised input string.

    Returns:
        dict with keys:
            "original"       — input text as received
            "translated"     — translated output (or original on failure)
            "was_translated" — True if output meaningfully differs from input
    """
    # Input validation — return consistent dict format immediately
    if not isinstance(text, str) or not text.strip():
        return {"original": "", "translated": "", "was_translated": False}

    last_exception = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            translated_text = _TRANSLATOR.translate(text)
            translated_text = html.unescape(translated_text)

            # was_translated: True only if output meaningfully differs from input
            # Purely a dashboard UI flag — does not affect model or categoriser
            # Known edge case: Google occasionally makes trivial normalisation
            # changes to English text, causing spurious True — acceptable limitation
            was_translated = translated_text.strip().lower() != text.strip().lower()

            return {
                "original":       text,
                "translated":     translated_text,
                "was_translated": was_translated,
            }

        except Exception as e:
            last_exception = e
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY)

    # All retries exhausted — fail-safe: return original text, pipeline continues
    print(f"Translation Engine Error after {_MAX_RETRIES} attempts: {last_exception}")
    return {
        "original":       text,
        "translated":     text,
        "was_translated": False,
    }