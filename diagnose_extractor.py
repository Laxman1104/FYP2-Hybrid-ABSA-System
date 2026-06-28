"""
diagnose_extractor.py — Extractor Failure Mapper

Run this from your project root (same level as src/):
    python diagnose_extractor.py

What this does:
    1. Feeds controlled sentences into spaCy en_core_web_md
    2. Prints the full dependency parse for each token
    3. Runs the actual extractor and shows what it captures vs misses
    4. Groups results into EXTRACTED / MISSED so you get a clean failure map

No manual input needed — just run and read.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import spacy
from extractor import AspectSentimentExtractor

# ── Colour helpers ───────────────────────────────────────────────────────────

def cyan(t):    return f"\033[96m{t}\033[0m"
def green(t):   return f"\033[92m{t}\033[0m"
def red(t):     return f"\033[91m{t}\033[0m"
def yellow(t):  return f"\033[93m{t}\033[0m"
def bold(t):    return f"\033[1m{t}\033[0m"
def dim(t):     return f"\033[2m{t}\033[0m"

# ── Test sentences ───────────────────────────────────────────────────────────
# Each entry: (sentence, expected_aspects, notes)
# expected_aspects: list of aspect strings we SHOULD extract
# If you are unsure, set expected_aspects=[] and just read the parse output

TEST_CASES = [

    # --- Malay multi-word food terms (compound noun problem) ---
    ("The nasi lemak was delicious.",
     ["nasi lemak"],
     "Malay compound food noun — does spaCy parse nasi as compound?"),

    ("I loved the mee goreng here.",
     ["mee goreng"],
     "Malay compound food noun — Rule 3 sentiment verb path"),

    ("The nasi goreng is really good but the roti canai was cold.",
     ["nasi goreng", "roti canai"],
     "Two Malay food terms in one sentence"),

    ("Their char kway teow is the best in town.",
     ["char kway teow"],
     "Three-word Malay food term"),

    ("The teh tarik was too sweet.",
     ["teh tarik"],
     "Two-word drink term"),

    # --- Standard English — should already work (sanity checks) ---
    ("The food was delicious.",
     ["food"],
     "Sanity check — Rule 2 predicate adj"),

    ("The chicken was overcooked.",
     ["chicken"],
     "Sanity check — Rule 2"),

    ("I loved the service.",
     ["service"],
     "Sanity check — Rule 3 sentiment verb"),

    ("The waiter was rude.",
     ["waiter"],
     "Sanity check — Rule 2"),

    # --- English compound nouns (should work via _get_compound_noun) ---
    ("The chicken rice was amazing.",
     ["chicken rice"],
     "English compound — does compound dep fire for chicken rice?"),

    ("The fried chicken was crispy.",
     ["fried chicken"],
     "Adj+noun — check if fried parsed as amod or compound"),

    # --- Mixed sentence structures ---
    ("Nasi lemak sedap tapi service lambat.",
     [],
     "Raw Manglish — no translation, parser likely to fail badly"),

    ("The food was good but the service was terrible.",
     ["food", "service"],
     "Two aspects one sentence — conjunction handling"),

    ("Service was slow and the food was cold.",
     ["service", "food"],
     "Two subjects, two predicates"),

    ("The staff were friendly and helpful.",
     ["staff"],
     "Conjunction on opinion side — should produce two pairs"),

    # --- Negation ---
    ("The food was not good.",
     ["food"],
     "Simple negation — should capture not good"),

    ("The staff were not friendly.",
     ["staff"],
     "Negation on predicate adj"),

    # --- Known trouble cases from CLAUDE.md ---
    ("The place was nice.",
     ["place"],
     "place tagged as VERB in short contexts — known issue"),

    ("The ambiance was great.",
     ["ambiance"],
     "Ambiance — low training data category"),

    ("The price was reasonable.",
     ["price"],
     "Price aspect"),

    ("Absolutely terrible!",
     ["overall"],
     "Rule 5 root fallback — dangling adjective"),
]

# ── Parse printer ────────────────────────────────────────────────────────────

def print_parse(doc):
    print(f"\n  {'TOKEN':<18} {'LEMMA':<18} {'POS':<8} {'DEP':<12} {'HEAD':<18} {'HEAD POS'}")
    print(f"  {'-'*18} {'-'*18} {'-'*8} {'-'*12} {'-'*18} {'-'*8}")
    for token in doc:
        print(f"  {token.text:<18} {token.lemma_:<18} {token.pos_:<8} "
              f"{token.dep_:<12} {token.head.text:<18} {token.head.pos_}")

# ── Main diagnostic ──────────────────────────────────────────────────────────

def main():
    print(bold("\n" + "=" * 70))
    print(bold("  EXTRACTOR DIAGNOSTIC — spaCy Parse + Extraction Failure Map"))
    print(bold("=" * 70))

    print(f"\n  Loading en_core_web_md...", end="", flush=True)
    nlp = spacy.load("en_core_web_md")
    extractor = AspectSentimentExtractor(nlp_model=nlp)
    print(f" {green('ready')}\n")

    extracted_ok  = []
    missed        = []
    partial       = []

    for sentence, expected, note in TEST_CASES:
        doc   = nlp(sentence)
        pairs = extractor.extract(sentence)
        got_aspects = [p[0] for p in pairs]

        print(bold("=" * 70))
        print(f"  {bold('SENTENCE:')} {cyan(sentence)}")
        print(f"  {dim('Note:')} {note}")

        print_parse(doc)

        print(f"\n  {bold('Extractor output:')} ", end="")
        if pairs:
            print()
            for aspect, opinion in pairs:
                print(f"    {green('→')} ({cyan(aspect)}, {yellow(opinion)})")
        else:
            print(red("(nothing extracted)"))

        # Compare against expected
        if expected:
            missing = [e for e in expected if not any(e in g for g in got_aspects)]
            found   = [e for e in expected if any(e in g for g in got_aspects)]

            if missing and found:
                status = yellow("PARTIAL")
                partial.append((sentence, missing, note))
            elif missing:
                status = red("MISSED")
                missed.append((sentence, missing, note))
            else:
                status = green("OK")
                extracted_ok.append(sentence)

            print(f"\n  Status: {status}")
            if missing:
                print(f"  {red('Missing aspects:')} {missing}")
        else:
            print(f"\n  Status: {dim('(no expected — parse inspection only)')}")

        print()

    # ── Summary ──────────────────────────────────────────────────────────────
    print(bold("=" * 70))
    print(bold("  FAILURE MAP SUMMARY"))
    print(bold("=" * 70))

    print(f"\n  {green('EXTRACTED OK')} ({len(extracted_ok)}):")
    for s in extracted_ok:
        print(f"    ✓ {s}")

    print(f"\n  {yellow('PARTIAL')} ({len(partial)}):")
    for s, missing, note in partial:
        print(f"    ~ {s}")
        print(f"      missing: {missing}")
        print(f"      note: {note}")

    print(f"\n  {red('FULLY MISSED')} ({len(missed)}):")
    for s, missing, note in missed:
        print(f"    ✗ {s}")
        print(f"      missing: {missing}")
        print(f"      note: {note}")

    total_with_expected = len(extracted_ok) + len(partial) + len(missed)
    print(f"\n  Overall: {len(extracted_ok)}/{total_with_expected} fully correct, "
          f"{len(partial)} partial, {len(missed)} fully missed\n")
    print(bold("=" * 70 + "\n"))


if __name__ == "__main__":
    main()