"""Apply model-produced diffs to a base artifact.

Two formats are supported:

- ``unified-diff``: standard ``---`` / ``+++`` / ``@@`` patch text.
- ``json``: ``{"edits": [{"old": "...", "new": "..."}, ...]}``.

Application failures are surfaced to the caller, which decides whether to fall
back to treating the response as a complete artifact.
"""

import json
import subprocess
import tempfile
from pathlib import Path

from optimizer_agent.utils import extract_response


class DiffApplyError(ValueError):
    """Raised when a diff/edit cannot be applied to the base artifact."""


def extract_unified_diff(response):
    """Return the likely unified diff text from a model response."""
    fenced = extract_response(response, "last")
    candidates = [fenced, response]
    for candidate in candidates:
        if isinstance(candidate, str) and ("@@" in candidate or "--- " in candidate):
            return candidate.strip()
    return str(response or "").strip()


def apply_unified_diff(base_text, response, extension="txt"):
    """Apply a unified diff; return the patched text or raise ``DiffApplyError``."""
    diff_text = extract_unified_diff(response)
    if not diff_text:
        raise DiffApplyError("empty diff")

    with tempfile.TemporaryDirectory(prefix="optagent_diff_") as tmp:
        tmp_path = Path(tmp)
        artifact = tmp_path / f"artifact.{extension.lstrip('.') or 'txt'}"
        diff_path = tmp_path / "change.diff"
        diff_path.write_text(diff_text + ("\n" if not diff_text.endswith("\n") else ""))
        for strip in ("-p0", "-p1"):
            artifact.write_text(base_text)
            try:
                proc = subprocess.run(
                    [
                        "patch",
                        "--batch",
                        "--forward",
                        "--no-backup-if-mismatch",
                        strip,
                        "-i",
                        str(diff_path),
                        str(artifact),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise DiffApplyError(f"patch failed to run: {exc}") from exc
            if proc.returncode == 0:
                patched = artifact.read_text()
                if patched != base_text:
                    return patched
        raise DiffApplyError(
            "unified diff did not apply"
            + (f": {proc.stderr.strip()[:400]}" if proc.stderr else "")
        )


def apply_json_edits(base_text, response):
    """Apply ``{"edits":[{"old","new"}]}``; return the patched text or raise."""
    payload = extract_response(response, "json")
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise DiffApplyError(f"edit response is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("edits"), list):
        raise DiffApplyError("expected a JSON object with an 'edits' list")

    current = base_text
    for index, edit in enumerate(data["edits"]):
        if not isinstance(edit, dict):
            raise DiffApplyError(f"edit {index} is not an object")
        old = edit.get("old")
        new = edit.get("new")
        if not isinstance(old, str) or not isinstance(new, str) or not old:
            raise DiffApplyError(f"edit {index} needs non-empty string 'old' and 'new'")
        if old not in current:
            raise DiffApplyError(f"edit {index}: 'old' text not found")
        # Decision: ambiguous matches replace the first occurrence.
        current = current.replace(old, new, 1)
    return current


def apply_diff(base_text, response, mode="unified-diff", extension="txt"):
    """Apply a diff in the selected mode; raises ``DiffApplyError`` on failure."""
    mode = (mode or "unified-diff").strip().lower()
    if mode in ("json", "json-edits", "edits"):
        return apply_json_edits(base_text, response)
    return apply_unified_diff(base_text, response, extension=extension)
