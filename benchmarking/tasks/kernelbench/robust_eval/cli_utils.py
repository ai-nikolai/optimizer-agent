"""Small argparse helpers shared by the CLI entry points."""

from __future__ import annotations

import argparse


def positive_int(value: str) -> int:
    """argparse type for counts that must be at least 1."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as e:
        raise argparse.ArgumentTypeError(f"expected an integer >= 1, got {value!r}") from e
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"expected an integer >= 1, got {parsed}")
    return parsed


def non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as e:
        raise argparse.ArgumentTypeError(f"expected an integer >= 0, got {value!r}") from e
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected an integer >= 0, got {parsed}")
    return parsed


def non_negative_float(value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as e:
        raise argparse.ArgumentTypeError(f"expected a number >= 0, got {value!r}") from e
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a number >= 0, got {parsed}")
    return parsed
