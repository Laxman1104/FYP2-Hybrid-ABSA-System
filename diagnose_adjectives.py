"""
diagnose_adjectives.py — FnB Adjective POS Tagging Diagnostic

Tests how spaCy en_core_web_md tags common restaurant-domain adjectives
in two grammatical positions:

  1. PREDICATIVE: "The chicken was crispy."
     → Relevant to Rule 2 (nsubj + acomp) — needs ADJ tag on acomp
     → Also tests negation: "The chicken was not crispy."

  2. ATTRIBUTIVE: "The crispy chicken was good."
     → Relevant to Rule 1 (amod) — needs ADJ tag on amod child

If spaCy mistaggs the adjective as NOUN, VERB, or anything other than ADJ,
both rules will silently fail to extract the pair.

Run from project root:
    python diagnose_adjectives.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import spacy
from extractor import AspectSentimentExtractor

# ── Colour helpers ───────────────────────────────────────────────────────────

def green(t):  return f"\033[92m{t}\033[0m"
def red(t):    return f"\033[91m{t}\033[0m"
def yellow(t): return f"\033[93m{t}\033[0m"
def cyan(t):   return f"\033[96m{t}\033[0m"
def bold(t):   return f"\033[1m{t}\033[0m"
def dim(t):    return f"\033[2m{t}\033[0m"

# ── Adjective list ───────────────────────────────────────────────────────────
# (adjective, food_noun_for_context)

ADJECTIVES = [
    # Added — common FnB descriptors likely to appear in Malaysian reviews
    ("crispy",      "chicken"),
    ("overcooked",  "chicken"),
    ("undercooked", "chicken"),
    ("bland",       "food"),
    ("juicy",       "chicken"),
    ("fresh",       "food"),
    ("stale",       "bread"),
    ("tender",      "meat"),
    ("flavourful",  "food"),
    ("raw",         "chicken"),
    ("soggy",       "fries"),
    ("greasy",      "food"),
    ("spicy",       "curry"),
    ("salty",       "food"),
    ("sweet",       "dessert"),
    ("sour",        "sauce"),
    ("oily",        "food"),
    ("dry",         "chicken"),
    ("burnt",       "rice"),
    ("tasteless",   "food"),
    ("unpleasant",  "smell"),
    ("mediocre",    "food"),
    ("decent",      "food"),
    ("satisfying",  "meal"),
    ("disappointing", "food"),
]

# ── Sentence templates ───────────────────────────────────────────────────────

def make_predicative(adj, noun):
    return f"The {noun} was {adj}."

def make_predicative_negated(adj, noun):
    return f"The {noun} was not {adj}."

def make_attributive(adj, noun):
    return f"The {adj} {noun} was good."

# ── Get POS of target adjective in sentence ──────────────────────────────────

def get_adj_token(doc, adj):
    """Find the target adjective token in parsed doc."""
    for token in doc:
        if token.text.lower() == adj.lower():
            return token
    return None

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(bold("\n" + "=" * 72))
    print(bold("  FnB ADJECTIVE POS DIAGNOSTIC"))
    print(bold("  Checks how spaCy tags each adjective in predicative + attributive positions"))
    print(bold("=" * 72))

    print(f"\n  Loading en_core_web_md...", end="", flush=True)
    nlp = spacy.load("en_core_web_md")
    extractor = AspectSentimentExtractor(nlp_model=nlp)
    print(f" {green('ready')}\n")

    mislabelled   = []   # ADJ tagged as something else
    correct       = []   # ADJ tagged correctly in all positions
    partial_fail  = []   # ADJ correct in one position but not the other

    print(f"  {'ADJECTIVE':<16} {'PREDICATIVE POS':<18} {'ATTRIBUTIVE POS':<18} "
          f"{'PRED EXTRACTED?':<18} {'ATTR EXTRACTED?':<18} STATUS")
    print(f"  {'-'*16} {'-'*18} {'-'*18} {'-'*18} {'-'*18} {'-'*8}")

    for adj, noun in ADJECTIVES:
        pred_sent  = make_predicative(adj, noun)
        attr_sent  = make_attributive(adj, noun)

        pred_doc   = nlp(pred_sent)
        attr_doc   = nlp(attr_sent)

        pred_token = get_adj_token(pred_doc, adj)
        attr_token = get_adj_token(attr_doc, adj)

        pred_pos   = pred_token.pos_ if pred_token else "NOT FOUND"
        attr_pos   = attr_token.pos_ if attr_token else "NOT FOUND"

        pred_pairs = extractor.extract(pred_sent)
        attr_pairs = extractor.extract(attr_sent)

        pred_extracted = bool(pred_pairs)
        attr_extracted = bool(attr_pairs)

        pred_ok = pred_pos == "ADJ"
        attr_ok = attr_pos == "ADJ"

        if pred_ok and attr_ok:
            status = green("OK")
            correct.append(adj)
        elif not pred_ok and not attr_ok:
            status = red("BOTH FAIL")
            mislabelled.append((adj, noun, pred_pos, attr_pos, pred_pairs, attr_pairs))
        else:
            status = yellow("PARTIAL")
            partial_fail.append((adj, noun, pred_pos, attr_pos, pred_pairs, attr_pairs))

        pred_pos_str = green(pred_pos) if pred_ok else red(pred_pos)
        attr_pos_str = green(attr_pos) if attr_ok else red(attr_pos)
        pred_ext_str = green("YES") if pred_extracted else red("NO")
        attr_ext_str = green("YES") if attr_extracted else red("NO")

        print(f"  {adj:<16} {pred_pos:<18} {attr_pos:<18} "
              f"{'YES' if pred_extracted else 'NO':<18} "
              f"{'YES' if attr_extracted else 'NO':<18} "
              f"{adj} -> {pred_pos}/{attr_pos}")

    # ── Detailed breakdown for failures ─────────────────────────────────────

    if mislabelled or partial_fail:
        print(f"\n{bold('=' * 72)}")
        print(bold("  DETAILED FAILURE BREAKDOWN"))
        print(bold("=" * 72))

        for adj, noun, pred_pos, attr_pos, pred_pairs, attr_pairs in mislabelled + partial_fail:
            print(f"\n  {bold(adj.upper())} — predicative POS: {red(pred_pos)}, attributive POS: {red(attr_pos)}")

            pred_sent = make_predicative(adj, noun)
            attr_sent = make_attributive(adj, noun)
            pred_doc  = nlp(pred_sent)
            attr_doc  = nlp(attr_sent)

            print(f"\n  Predicative: \"{pred_sent}\"")
            print(f"  {'TOKEN':<16} {'POS':<10} {'DEP':<12} {'HEAD'}")
            for token in pred_doc:
                marker = " ← TARGET" if token.text.lower() == adj.lower() else ""
                print(f"  {token.text:<16} {token.pos_:<10} {token.dep_:<12} {token.head.text}{marker}")

            print(f"\n  Attributive: \"{attr_sent}\"")
            print(f"  {'TOKEN':<16} {'POS':<10} {'DEP':<12} {'HEAD'}")
            for token in attr_doc:
                marker = " ← TARGET" if token.text.lower() == adj.lower() else ""
                print(f"  {token.text:<16} {token.pos_:<10} {token.dep_:<12} {token.head.text}{marker}")

            print(f"\n  Predicative extraction: {pred_pairs if pred_pairs else red('(nothing)')}")
            print(f"  Attributive extraction: {attr_pairs if attr_pairs else red('(nothing)')}")

    # ── Negation spot check for mislabelled words ────────────────────────────

    if mislabelled:
        print(f"\n{bold('=' * 72)}")
        print(bold("  NEGATION CHECK — for mislabelled adjectives"))
        print(bold("  (negation compounds the problem if base form already fails)"))
        print(bold("=" * 72))
        for adj, noun, *_ in mislabelled:
            neg_sent  = make_predicative_negated(adj, noun)
            neg_pairs = extractor.extract(neg_sent)
            print(f"\n  \"{neg_sent}\"")
            print(f"  Extracted: {neg_pairs if neg_pairs else red('(nothing)')}")

    # ── Summary ──────────────────────────────────────────────────────────────

    print(f"\n{bold('=' * 72)}")
    print(bold("  SUMMARY"))
    print(bold("=" * 72))
    print(f"\n  {green('CORRECTLY TAGGED')} ({len(correct)}): {', '.join(correct)}")

    if partial_fail:
        print(f"\n  {yellow('PARTIAL FAIL')} ({len(partial_fail)}):")
        for adj, noun, pred_pos, attr_pos, *_ in partial_fail:
            print(f"    {adj}: predicative={pred_pos}, attributive={attr_pos}")

    if mislabelled:
        print(f"\n  {red('BOTH POSITIONS FAIL')} ({len(mislabelled)}):")
        for adj, noun, pred_pos, attr_pos, *_ in mislabelled:
            print(f"    {adj}: predicative={red(pred_pos)}, attributive={red(attr_pos)}")

    total = len(correct) + len(partial_fail) + len(mislabelled)
    print(f"\n  Overall: {len(correct)}/{total} fully correct, "
          f"{len(partial_fail)} partial, {len(mislabelled)} both positions failing\n")
    print(bold("=" * 72 + "\n"))


if __name__ == "__main__":
    main()