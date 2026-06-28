"""
conftest.py — Shared pytest fixtures

Loads the spaCy model once for the entire test session.
All test files that need the nlp model receive it via the `nlp` fixture
rather than loading it independently — prevents repeated slow model loads.
"""

import pytest
import spacy


@pytest.fixture(scope="session")
def nlp():
    """
    Loads en_core_web_md once per test session.
    Shared across all test files via pytest fixture injection.
    """
    return spacy.load("en_core_web_md")