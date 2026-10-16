"""Tiny parsing helpers.

Everything here is dependency-free (no torch / numpy). Two jobs:

* pull the fenced code block out of an LLM response,
* pull the result JSON out of an evaluator's stdout.

Both helpers fail *soft*: on anything they can't parse they return a sentinel
(the ``MISSING`` dict carrying ``__unparsed__: True``) instead of raising, so
the search loop never crashes on a malformed response or evaluator log line.
The caller decides what "unparsed" means.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from optimizer_agent import EVAL_MARKER

# The single file ending we write the model's new code to.
CORE_RESPONSE = "core_response"

# Sentinel returned when we can't parse a result at all. Carrying an
# `__unparsed__` flag lets the caller tell "evaluator ran, bad result"
# apart from "we couldn't get a result at all".
MISSING: Dict[str, Any] = {"__unparsed__": True}


def extract_final_code_response(response: str) -> str:
    """Return the ```code``` block from an LLM response, or the whole thing.

    If there is no code block we simply return the response unchanged, so the
    loop keeps working with whatever the model produced.
    """
    if not isinstance(response, str):
        return ""
    start = response.find("```")
    if start == -1:
        return response
    end = response.find("```", start + 3)
    if end == -1:
        return response[start + 3:].strip()
    code = response[start + 3:end]
    first_newline = code.find("\n")
    if first_newline != -1:
        code = code[first_newline + 1:]
    return code.strip()


def extract_eval_results(stdout: str) -> Dict[str, Any]:
    """Extract the JSON printed by ``evaluate.py`` after the marker.

    The convention: a line ``=== Evaluation Result ===`` followed immediately by
    one JSON object. We search for the first ``{`` and the last ``}``` after the
    marker. Anything missing (no marker, malformed JSON) yields the empty
    ``MISSING`` sentinel -- a failing result, never a crash.
    """
    if not stdout:
        return MISSING

    parts = stdout.split(EVAL_MARKER)
    if not parts:
        return MISSING

    payload = parts[-1].strip()
    start = payload.find("{")
    end = payload.rfind("}")
    if start == -1 or end == -1 or end < start:
        return MISSING
    try:
        result = json.loads(payload[start:end + 1])
    except json.JSONDecodeError:
        return MISSING
    return result if isinstance(result, dict) else MISSING


def score_of(result: Dict[str, Any], *, default: float = -float("inf")) -> float:
    """The scalar the search compares. ``combined_score`` if present, else inf-neg."""
    value = result.get("combined_score", default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def write(path, content: str) -> str:
    """Write ``content`` to ``path`` (as text) and return ``path``."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return str(path)
