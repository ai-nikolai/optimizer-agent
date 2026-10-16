"""Extraction helpers shared across the agent.

This module is deliberately dependency-light (no torch) so it can be imported
anywhere. Scoring-classification rules live in :mod:`optimizer_agent.prompts_v2`
(now that the package is generic rather than CUDA-specific).
"""

from __future__ import annotations

import json
from typing import Any, Dict


def extract_final_code_response(response: str) -> str:
    """Pull the final fenced ```code``` block out of an LLM response string.

    Returns the extracted code block, or the original response if no code block
    is found (so we never fail to return *something* useful).
    """
    if not isinstance(response, str):
        return response

    start_marker = "```"
    code_start = response.find(start_marker)
    if code_start == -1:
        return response

    code_end = response.find(start_marker, code_start + len(start_marker))
    if code_end == -1:
        return response[code_start + len(start_marker) :]

    code_content_start = code_start + len(start_marker)
    first_newline = response.find("\n", code_content_start)
    if first_newline != -1 and first_newline < code_end:
        code_content_start = first_newline + 1
    return response[code_content_start:code_end]


def extract_eval_results(stdout: str) -> Dict[str, Any]:
    """Parse the JSON object printed by an evaluator subprocess.

    The convention is that the evaluator prints a marker line
    ``=== Evaluation Result ===`` followed immediately by a JSON object.
    Everything between the first ``{`` and the last ``}``` after the marker is
    treated as the result. Missing the marker (or any JSON) yields a failing
    ``{"combined_score": -inf}`` record instead of raising.
    """
    marker = "=== Evaluation Result ==="
    default_eval_results = {"combined_score": -float("inf")}

    if marker not in stdout:
        print(f"Warning: EVAL_MARKER '{marker}' not found in stdout")
        print(f"Stdout: {stdout[:500]}...")
        return default_eval_results

    evaluation_output = stdout.split(marker)[-1].strip()
    try:
        result = json.loads(evaluation_output)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        start_idx = evaluation_output.find("{")
        end_idx = evaluation_output.rfind("}") + 1
        if start_idx != -1 and end_idx != 0:
            json_str = evaluation_output[start_idx:end_idx]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

    print(f"Warning: No JSON found after EVAL_MARKER in stdout")
    print(f"Evaluation output: {evaluation_output[:500]}...")
    return default_eval_results


# The default evaluator exit status codes (kept in sync with prompts_v2).
STATUS_OK = "ok"
STATUS_COMPILE_ERROR = "compile_error"
STATUS_RUNTIME_ERROR = "runtime_error"
STATUS_INCORRECT = "incorrect"
STATUS_EMPTY = "empty"
STATUS_EVAL_ERROR = "eval_error"
