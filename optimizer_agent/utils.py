"""Response/eval parsing, scoring helpers and the evaluator subprocess runner.

Two independent jobs:

1. Extract an artifact from a model response (fenced code, raw text or JSON).
2. Run the user's evaluator and parse the JSON it prints after
   ``=== Evaluation Result ===``.

Both fail softly: a bad model response or a crashing evaluator becomes a
``-inf`` candidate, never a crashed search.
"""

import json
import math
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from optimizer_agent import EVAL_MARKER

DEFAULT_EVAL_TIMEOUT = 600
DEFAULT_OUTPUT_TAIL = 20000

_CODE_LANGS = {
    "python",
    "py",
    "python3",
    "cuda",
    "cpp",
    "c++",
    "c",
    "cxx",
    "cu",
}


# ---------------------------------------------------------------------------
# Response extraction
# ---------------------------------------------------------------------------
def _fenced_blocks(response):
    """Return ``[(language, content), ...]`` for every fenced block."""
    if not isinstance(response, str):
        return []
    blocks = []
    cursor = 0
    while True:
        start = response.find("```", cursor)
        if start == -1:
            break
        end = response.find("```", start + 3)
        if end == -1:
            break
        raw = response[start + 3 : end]
        language = ""
        content = raw
        first_newline = raw.find("\n")
        if first_newline != -1:
            first_line = raw[:first_newline].strip().lower()
            if first_line and " " not in first_line:
                language = first_line
                content = raw[first_newline + 1 :]
        blocks.append((language, content.strip()))
        cursor = end + 3
    return blocks


def _is_code_language(language):
    return (language or "").lower().strip() in _CODE_LANGS


def match_last(response):
    """Content of the last fenced block, or the raw response."""
    blocks = _fenced_blocks(response)
    return blocks[-1][1] if blocks else response


def match_largest(response):
    """Content of the largest fenced block, or the raw response."""
    blocks = _fenced_blocks(response)
    if not blocks:
        return response
    return max(blocks, key=lambda item: len(item[1]))[1]


def match_python(response):
    """Content of the last Python/code-tagged block, falling back to last."""
    blocks = _fenced_blocks(response)
    if not blocks:
        return response
    code_blocks = [block for block in blocks if _is_code_language(block[0])]
    return (code_blocks or blocks)[-1][1]


def match_largest_code(response):
    """Content of the largest Python/code-tagged block, falling back to largest."""
    blocks = _fenced_blocks(response)
    if not blocks:
        return response
    code_blocks = [block for block in blocks if _is_code_language(block[0])]
    return max(code_blocks or blocks, key=lambda item: len(item[1]))[1]


def match_last_code(response):
    """Prefer the last Python/code-tagged block, then the last block."""
    blocks = _fenced_blocks(response)
    if not blocks:
        return response
    code_blocks = [block for block in blocks if _is_code_language(block[0])]
    return (code_blocks or blocks)[-1][1]


EXTRACTORS = {
    "last_code": match_last_code,
    "last": match_last,
    "largest": match_largest,
    "code": match_largest_code,
    "python": match_python,
}


def _extract_json_block(text):
    """Best-effort extraction of the outermost JSON object/array."""
    if not isinstance(text, str):
        return ""
    candidates = [(text.find("{"), text.rfind("}")), (text.find("["), text.rfind("]"))]
    candidates = [(start, end) for start, end in candidates if start != -1 and end > start]
    if not candidates:
        return text.strip()
    start, end = min(candidates, key=lambda pair: pair[0])
    return text[start:end + 1].strip()


def extract_response(response, response_format="last_code"):
    """Extract the artifact from a model response.

    ``response_format``:
      - ``last_code`` (default): last code-tagged block, else last block
      - ``last``: last fenced block
      - ``largest``: largest fenced block
      - ``code``: largest code-tagged block, else largest block
      - ``raw`` / ``text`` / ``none``: the whole response
      - ``json``: fenced block or the outermost JSON object/array
    """
    if not isinstance(response, str):
        return ""
    fmt = (response_format or "last_code").strip().lower()
    if fmt in ("raw", "text", "none"):
        return response.strip()
    if fmt == "json":
        for _, content in reversed(_fenced_blocks(response)):
            try:
                json.loads(content)
                return content.strip()
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        return _extract_json_block(response)
    extractor = EXTRACTORS.get(fmt, match_last_code)
    return extractor(response).strip()


def extract_final_code_response(response):
    """Backwards-compatible alias for the default extractor."""
    return extract_response(response, "last_code")


# ---------------------------------------------------------------------------
# Eval parsing and scoring
# ---------------------------------------------------------------------------
def extract_eval_results(stdout):
    """Return the result dict after ``EVAL_MARKER``, or ``None`` if unusable."""
    if not stdout:
        return None
    if EVAL_MARKER not in stdout:
        return None
    payload = stdout.split(EVAL_MARKER)[-1].strip()
    start = payload.find("{")
    end = payload.rfind("}")
    if start == -1 or end < start:
        return None
    try:
        result = json.loads(payload[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    return result if isinstance(result, dict) else None


def score_of(scores, default=float("-inf"), key="combined_score"):
    """The scalar the search compares: ``scores[key]`` or ``default``.

    Non-finite values (NaN, +/-Infinity) are treated as ``default`` so a bad
    evaluator result cannot win selection by accident.
    """
    try:
        value = float((scores or {}).get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def rank_score_of(scores, key="combined_score", minimize=False, default=float("-inf")):
    """Selection score, always higher-is-better.

    ``minimize=True`` negates a finite raw score. The failure sentinel ``-inf``
    stays at the bottom in both modes.
    """
    raw = score_of(scores, default=default, key=key)
    if raw == float("-inf"):
        return float("-inf")
    return -raw if minimize else raw


# ---------------------------------------------------------------------------
# Evaluator subprocess
# ---------------------------------------------------------------------------
def _failed(
    status,
    stdout="",
    stderr="",
    returncode=None,
    error=None,
    stdout_path=None,
    stderr_path=None,
):
    scores = {"combined_score": float("-inf")}
    if error:
        scores["error"] = error
    return {
        "status": status,
        "scores": scores,
        "stdout": stdout or "",
        "stderr": stderr or "",
        "returncode": returncode,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
    }


def _tail(path, limit=DEFAULT_OUTPUT_TAIL):
    """Read at most the last ``limit`` characters of a text file."""
    path = Path(path)
    if not path.exists():
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            return handle.read()
    except OSError:
        return ""


def run_evaluation(
    evaluate_path,
    solution_path,
    reference_path=None,
    timeout=DEFAULT_EVAL_TIMEOUT,
    stdout_path=None,
    stderr_path=None,
    tail_chars=DEFAULT_OUTPUT_TAIL,
):
    """Run ``evaluate.py --solution ... [--reference ...]`` and parse the result.

    Full stdout/stderr are written to ``stdout_path`` / ``stderr_path`` when
    provided (the agent passes per-sample log paths). Only the last ``tail_chars``
    are returned in ``stdout``/``stderr`` so prompts and memory stay bounded.
    """
    command = [sys.executable, str(evaluate_path), "--solution", str(solution_path)]
    if reference_path:
        command += ["--reference", str(reference_path)]

    temporary = None
    if stdout_path is None or stderr_path is None:
        temporary = tempfile.TemporaryDirectory(prefix="optagent_eval_")
        stdout_path = stdout_path or str(Path(temporary.name) / "stdout.txt")
        stderr_path = stderr_path or str(Path(temporary.name) / "stderr.txt")

    try:
        Path(stdout_path).parent.mkdir(parents=True, exist_ok=True)
        Path(stderr_path).parent.mkdir(parents=True, exist_ok=True)
        with open(stdout_path, "w", encoding="utf-8", errors="replace") as out_handle, open(
            stderr_path, "w", encoding="utf-8", errors="replace"
        ) as err_handle:
            try:
                proc = subprocess.Popen(
                    command,
                    stdout=out_handle,
                    stderr=err_handle,
                    text=True,
                    start_new_session=True,
                )
            except Exception as exc:  # noqa: BLE001 - evaluator errors must not stop search
                return _failed(
                    "eval_error",
                    error=str(exc),
                    stdout_path=str(stdout_path),
                    stderr_path=str(stderr_path),
                )

            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                return _failed(
                    "timeout",
                    stdout=_tail(stdout_path, tail_chars),
                    stderr=_tail(stderr_path, tail_chars),
                    error=f"evaluator timed out after {timeout}s",
                    stdout_path=str(stdout_path),
                    stderr_path=str(stderr_path),
                )

        stdout_tail = _tail(stdout_path, tail_chars)
        stderr_tail = _tail(stderr_path, tail_chars)
        # Parse a larger window than the feedback tail: the marker can be
        # followed by a lot of evaluator logging.
        parse_text = _tail(stdout_path, max(tail_chars, 1_000_000))
        parsed = extract_eval_results(parse_text)

        if proc.returncode != 0:
            result = _failed(
                "eval_error",
                stdout_tail,
                stderr_tail,
                proc.returncode,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
            )
            if parsed is not None:
                parsed = dict(parsed)
                parsed["combined_score"] = float("-inf")
                result["scores"] = parsed
            return result

        if parsed is None:
            return _failed(
                "eval_error",
                stdout_tail,
                stderr_tail,
                proc.returncode,
                error="no parseable result after EVAL_MARKER",
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
            )

        return {
            "status": "ok",
            "scores": parsed,
            "stdout": stdout_tail,
            "stderr": stderr_tail,
            "returncode": proc.returncode,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }
    finally:
        if temporary is not None:
            temporary.cleanup()


# ---------------------------------------------------------------------------
# Prompt/result helpers
# ---------------------------------------------------------------------------
def _truncate(text, limit=1200):
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def eval_detail(eval_result, max_chars=1200, include_output=None):
    """Human-readable detail for a failed evaluation (for prompts/debug)."""
    if not eval_result:
        return ""
    parts = []
    scores = eval_result.get("scores") or {}
    structured = False
    for key in ("error", "message", "error_message", "traceback", "main_error"):
        if scores.get(key):
            parts.append(str(scores[key]))
            structured = True
            break
    status = eval_result.get("status", "ok")
    if include_output is None:
        include_output = status != "ok" or structured
    if include_output:
        if eval_result.get("stderr"):
            parts.append("stderr:\n" + str(eval_result["stderr"]))
        elif eval_result.get("stdout"):
            parts.append("stdout:\n" + str(eval_result["stdout"]))
    return _truncate("\n".join(parts), max_chars)


def summarize_result(eval_result, max_chars=1200):
    """One compact result line, injected into ``prompt_data["result"]``."""
    if not eval_result:
        return ""
    status = eval_result.get("status", "eval_error")
    scores = eval_result.get("scores") or {}

    if status != "ok":
        message = f"Evaluation failed ({status})."
        detail = eval_detail(eval_result, max_chars=max_chars)
        return _truncate(f"{message}\n{detail}" if detail else message, max_chars)

    preferred = ("combined_score", "correctness", "speedup", "runtime", "error")
    parts = []
    for key in preferred:
        if key in scores:
            parts.append(f"{key}={scores[key]}")
    for key, value in scores.items():
        if key not in preferred and isinstance(value, (int, float, str, bool)):
            parts.append(f"{key}={value}")
    return _truncate("Result: " + ", ".join(parts[:10]), max_chars)


def read_file(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def write_file(directory, name, content):
    """Atomically write text to ``directory/name``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(directory)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content if isinstance(content, str) else str(content))
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
