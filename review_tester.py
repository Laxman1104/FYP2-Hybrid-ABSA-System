"""
review_tester.py — Manual Review Tester

Paste a review, see every pipeline step clearly.
Always runs in AUTO mode with on_no_match=OTHER.
No prompts, no choices — just paste and go.

Usage:
    python review_tester.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pipeline import run_pipeline
from categoriser import AspectCategoriser
import spacy


# ── Colour helpers ───────────────────────────────────────────────────────────

def cyan(text):    return f"\033[96m{text}\033[0m"
def green(text):   return f"\033[92m{text}\033[0m"
def yellow(text):  return f"\033[93m{text}\033[0m"
def magenta(text): return f"\033[95m{text}\033[0m"
def bold(text):    return f"\033[1m{text}\033[0m"
def dim(text):     return f"\033[2m{text}\033[0m"
def red(text):     return f"\033[91m{text}\033[0m"

def divider():
    print(dim("─" * 60))


# ── Display pipeline result ──────────────────────────────────────────────────

def display_result(result):
    print()
    divider()
    print(bold("  PIPELINE OUTPUT"))
    divider()

    # Step 1 — Original input
    print(f"\n  {bold('Step 1 — Original Input')}")
    print(f"  {cyan(result['original'])}")

    # Step 2 — Normalisation
    print(f"\n  {bold('Step 2 — After Normalisation')}")
    if result["normalised"] is None or result["normalised"] == result["original"]:
        print(f"  {dim('(no changes — text already clean)')}")
    else:
        print(f"  {yellow(result['normalised'])}")

    # Step 3 — Translation
    print(f"\n  {bold('Step 3 — After Translation')}")
    if result["translated"] is not None:
        print(f"  {green(result['translated'])}")
    else:
        print(f"  {dim('(no translation needed — text already in English)')}")

    # Step 4 — Extracted raw pairs
    print(f"\n  {bold('Step 4 — Extracted (Raw Aspect, Opinion) Pairs')}")
    if result["pairs"]:
        for aspect, opinion in result["pairs"]:
            print(f"  {magenta('→')} raw aspect: {cyan(aspect)}  |  opinion: {yellow(opinion)}")
    else:
        print(f"  {red('(no pairs extracted — extraction engine returned nothing)')}")

    # Step 5 — Categorised results ordered to match Step 4
    print(f"\n  {bold('Step 5 — Categorised Results')}")
    if result["pairs"]:
        opinion_to_cat = {op: cat for cat, op in result["results"]}
        shown = set()
        for aspect, opinion in result["pairs"]:
            key = (aspect, opinion)
            if key in shown:
                continue
            shown.add(key)
            category = opinion_to_cat.get(opinion)
            if category is None:
                print(f"  {dim('✗')} [{dim('DROPPED')}]  raw: {cyan(aspect)}  |  opinion: {yellow(opinion)}  {dim('(below similarity threshold)')}")
            else:
                print(f"  {green('✓')} [{bold(category.upper())}]  raw: {cyan(aspect)}  |  opinion: {yellow(opinion)}")
    else:
        print(f"  {dim('(no results)')}")

    print()
    divider()
    print()


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    print()
    print(bold("=" * 60))
    print(bold("  ABSA Review Tester"))
    print(bold("  Mode: AUTO  |  Unmatched: OTHER"))
    print(bold("=" * 60))
    print(f"\n  {dim('Loading spaCy model...')}", end="", flush=True)

    nlp = spacy.load("en_core_web_md")
    categoriser = AspectCategoriser(nlp_model=nlp, on_no_match="other")
    print(f" {green('ready')}")

    while True:
        print(f"\n  {dim('─' * 58)}")
        text = input(f"\n  {bold('Paste review')} {dim('(or q to quit)')}: ").strip()

        if text.lower() in ("q", "quit", "exit"):
            print(f"\n  {green('Done.')}\n")
            break

        if not text:
            print(f"  {red('Please enter some text.')}")
            continue

        print(f"\n  {dim('Running pipeline...')}")

        try:
            result = run_pipeline(text, mode="auto", categoriser=categoriser)
            display_result(result)

        except Exception as e:
            print(f"\n  {red(f'Pipeline error: {e}')}\n")


if __name__ == "__main__":
    main()