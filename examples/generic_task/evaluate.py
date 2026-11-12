#!/usr/bin/env python3
"""Task-specific evaluator for a tiny non-code task.

The artifact is a one-sentence tagline. It is valid when it contains the three
required keywords and stays within the word limit. The evaluator also supplies
its own natural-language feedback, used by the ``evaluator`` feedback strategy.
"""

import argparse
import json
import re
from pathlib import Path

REQUIRED = ("fast", "reliable", "secure")


def evaluate(text):
    words = re.findall(r"[a-z0-9']+", text.lower())
    missing = [word for word in REQUIRED if word not in words]
    valid = not missing and len(words) <= 20
    quality = (len(REQUIRED) - len(missing)) - 0.01 * max(0, len(words) - 20)

    if missing:
        feedback = "The tagline is missing these required words: " + ", ".join(missing)
    elif len(words) > 20:
        feedback = f"The tagline is valid but too long ({len(words)} words); shorten it."
    else:
        feedback = "The tagline is valid. Improve its quality while keeping it under 20 words."

    return {
        "quality": quality,
        "valid": valid,
        "baseline_score": 2.0,
        "missing_keywords": missing,
        "word_count": len(words),
        "feedback": feedback,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", required=True)
    parser.add_argument("--reference", default=None)
    args = parser.parse_args()

    result = evaluate(Path(args.solution).read_text())
    print("=== Evaluation Result ===")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
