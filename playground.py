"""
playground.py — Interactive Pipeline Tester

Run this from the project root to manually test the full pipeline
on any review text you want.

Usage:
    python playground.py

Then type any review text and choose a mode to see every pipeline
step laid out clearly — from raw input all the way to extracted
aspect-opinion pairs.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pipeline import run_pipeline
from categoriser import AspectCategoriser
import spacy

# ── Colour helpers for terminal output ──────────────────────────────────────

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

def display_result(result, mode):
    print()
    divider()
    print(bold("  PIPELINE OUTPUT"))
    divider()

    # Step 1 — Original input
    print(f"\n  {bold('Step 1 — Original Input')}")
    print(f"  {cyan(result['original'])}")

    # Step 2 — Normalisation (only in manglish / auto mode)
    if result["normalised"] is not None:
        print(f"\n  {bold('Step 2 — After Normalisation')}")
        if result["normalised"] == result["original"]:
            print(f"  {dim('(no changes — text already clean)')}")
        else:
            print(f"  {yellow(result['normalised'])}")
    else:
        print(f"\n  {bold('Step 2 — Normalisation')}")
        print(f"  {dim('(skipped — english mode)')}")

    # Step 3 — Translation (only when was_translated=True)
    if result["translated"] is not None:
        print(f"\n  {bold('Step 3 — After Translation')}")
        print(f"  {green(result['translated'])}")
    else:
        if mode == "manglish":
            print(f"\n  {bold('Step 3 — Translation')}")
            print(f"  {dim('(no change detected — text already in English)')}")
        elif mode == "english":
            print(f"\n  {bold('Step 3 — Translation')}")
            print(f"  {dim('(skipped — english mode)')}")

    # Step 4 — Extracted pairs
    print(f"\n  {bold('Step 4 — Extracted (Aspect, Opinion) Pairs')}")
    if result["pairs"]:
        for aspect, opinion in result["pairs"]:
            print(f"  {magenta('→')} aspect: {cyan(aspect)}  |  opinion: {yellow(opinion)}")
    else:
        print(f"  {dim('(no pairs extracted)')}")

    # Step 5 — Categorised results
    print(f"\n  {bold('Step 5 — Categorised Results')}")
    if result["results"]:
        for category, opinion in result["results"]:
            print(f"  {green('✓')} [{bold(category.upper())}]  opinion: {yellow(opinion)}")
    else:
        print(f"  {dim('(no results — all aspects fell below similarity threshold)')}")

    print()
    divider()
    print()


# ── Mode selection ───────────────────────────────────────────────────────────

def choose_mode():
    print(f"\n  Choose pipeline mode:")
    print(f"  {cyan('1')} — english   (skip normaliser + translator)")
    print(f"  {cyan('2')} — manglish  (full pipeline)")
    print(f"  {cyan('3')} — auto      (normalise always, translate if needed)")

    while True:
        choice = input(f"\n  Enter 1, 2, or 3: ").strip()
        if choice == "1":
            return "english"
        elif choice == "2":
            return "manglish"
        elif choice == "3":
            return "auto"
        else:
            print(f"  {red('Invalid choice. Enter 1, 2, or 3.')}")


# ── on_no_match selection ────────────────────────────────────────────────────

def choose_on_no_match():
    print(f"\n  Unmatched aspects:")
    print(f"  {cyan('1')} — drop   (exclude from results)")
    print(f"  {cyan('2')} — other  (assign category 'other')")

    while True:
        choice = input(f"\n  Enter 1 or 2: ").strip()
        if choice == "1":
            return "drop"
        elif choice == "2":
            return "other"
        else:
            print(f"  {red('Invalid choice. Enter 1 or 2.')}")


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    print()
    print(bold("=" * 60))
    print(bold("  ABSA Pipeline Playground"))
    print(bold("  Type a review, see every step of the pipeline"))
    print(bold("=" * 60))
    print(f"\n  {dim('Loading spaCy model (en_core_web_md)...')}", end="", flush=True)

    # Warm up the pipeline by importing — this loads the spaCy model
    nlp = spacy.load("en_core_web_md")
    print(f" {green('ready')}")

    while True:
        print(f"\n  {dim('─' * 58)}")
        text = input(f"\n  {bold('Enter a review')} {dim('(or q to quit)')}: ").strip()

        if text.lower() in ("q", "quit", "exit"):
            print(f"\n  {green('Goodbye.')}\n")
            break

        if not text:
            print(f"  {red('Please enter some text.')}")
            continue

        mode         = choose_mode()
        on_no_match  = choose_on_no_match()

        # Build custom categoriser if on_no_match is not default
        custom_cat = None
        if on_no_match == "other":
            custom_cat = AspectCategoriser(nlp_model=nlp, on_no_match="other")

        print(f"\n  {dim('Running pipeline...')}")

        try:
            result = run_pipeline(
                text,
                mode=mode,
                categoriser=custom_cat
            )
            display_result(result, mode)

        except Exception as e:
            print(f"\n  {red(f'Pipeline error: {e}')}\n")


if __name__ == "__main__":
    main()