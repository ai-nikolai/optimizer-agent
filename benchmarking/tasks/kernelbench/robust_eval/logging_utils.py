"""Small logging configuration helper.

The pipeline historically used bare ``print`` calls.  That is fine for a script you
watch in a terminal, but it makes failures impossible to triage after the fact (Slurm
logs interleave several processes) and impossible to silence in tests.

This module provides a single ``setup_logging`` entry point plus ``get_logger``.
Progress bars keep using ``tqdm``: they write to stdout, log records go to stderr, so
the two do not fight over the same line.
"""

from __future__ import annotations

import logging
import os
import sys

DEFAULT_FORMAT = "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def parse_log_level(level: str | int) -> int:
    """Translate ``"debug"``/``"INFO"``/``10`` (and friends) into a logging level int."""
    if isinstance(level, int):
        return level
    text = str(level).strip().upper()
    if text.isdigit():
        return int(text)
    resolved = logging.getLevelName(text)
    if not isinstance(resolved, int):
        raise ValueError(f"Unknown log level: {level!r}")
    return resolved


def setup_logging(level: str | int = "INFO", log_file: str | None = None, force: bool = False) -> logging.Logger:
    """Configure the root logger once and return it.

    Safe to call repeatedly: subsequent calls are no-ops unless ``force=True``, which
    is what the CLI does so that ``--log-level`` always wins.
    """
    global _configured

    root = logging.getLogger()
    if _configured and not force:
        return root

    resolved_level = parse_log_level(level)

    # Remove handlers this module installed previously so ``force`` does not stack
    # duplicate handlers on top of each other.
    for handler in list(root.handlers):
        if getattr(handler, "_robust_kb_handler", False):
            root.removeHandler(handler)
            handler.close()

    if force or not root.handlers:
        root.setLevel(resolved_level)

    formatter = logging.Formatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATEFMT)

    stream_handler = logging.StreamHandler(stream=sys.stderr)
    stream_handler.setFormatter(formatter)
    stream_handler._robust_kb_handler = True  # type: ignore[attr-defined]
    root.addHandler(stream_handler)

    if log_file:
        directory = os.path.dirname(os.path.abspath(log_file))
        if directory:
            os.makedirs(directory, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler._robust_kb_handler = True  # type: ignore[attr-defined]
        root.addHandler(file_handler)

    _configured = True
    return root


def get_logger(name: str = "robust_kernelbench") -> logging.Logger:
    """Return a logger.  Does not configure handlers (call ``setup_logging`` first)."""
    return logging.getLogger(name)


def is_configured() -> bool:
    """Whether ``setup_logging`` already installed handlers."""
    return _configured
