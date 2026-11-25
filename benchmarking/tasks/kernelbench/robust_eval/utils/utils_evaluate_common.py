"""Compile/execute generated code while capturing both Python and native output.

A generated kernel can print from Python **and** spawn compilers (``nvcc``,
``ninja``) that write directly to file descriptors.  :class:`OutputCapture`
redirects both levels so the diagnostics land in the result metadata.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from collections.abc import Callable
from typing import Any


class OutputCapture:
    """Capture Python ``sys.stdout/stderr`` and native fd 1/2.

    Python output is written to ``StringIO`` buffers.  Native output (including
    subprocesses such as nvcc) is redirected to temporary files.  The two are
    concatenated when the context exits.
    """

    def __init__(self) -> None:
        self._python_stdout = io.StringIO()
        self._python_stderr = io.StringIO()
        self._saved_stdout = None
        self._saved_stderr = None
        self._saved_stdout_fd = None
        self._saved_stderr_fd = None
        self._stdout_file = None
        self._stderr_file = None
        self._stdout_path = None
        self._stderr_path = None
        self.combined_stdout = ""
        self.combined_stderr = ""

    def __enter__(self) -> "OutputCapture":
        self._saved_stdout = sys.stdout
        self._saved_stderr = sys.stderr

        self._stdout_file = tempfile.NamedTemporaryFile(mode="w+b", delete=False)
        self._stderr_file = tempfile.NamedTemporaryFile(mode="w+b", delete=False)
        self._stdout_path = self._stdout_file.name
        self._stderr_path = self._stderr_file.name

        self._saved_stdout_fd = os.dup(1)
        self._saved_stderr_fd = os.dup(2)
        os.dup2(self._stdout_file.fileno(), 1)
        os.dup2(self._stderr_file.fileno(), 2)

        sys.stdout = self._python_stdout
        sys.stderr = self._python_stderr
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        # Flush Python and native output before restoring the original fds.
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os.dup2(self._saved_stdout_fd, 1)
            os.dup2(self._saved_stderr_fd, 2)
            os.close(self._saved_stdout_fd)
            os.close(self._saved_stderr_fd)
            sys.stdout = self._saved_stdout
            sys.stderr = self._saved_stderr

        self._stdout_file.close()
        self._stderr_file.close()
        self.combined_stdout = self._python_stdout.getvalue() + self._read(self._stdout_path)
        self.combined_stderr = self._python_stderr.getvalue() + self._read(self._stderr_path)
        self._unlink(self._stdout_path)
        self._unlink(self._stderr_path)

    @staticmethod
    def _read(path: str) -> str:
        try:
            with open(path, "rb") as handle:
                return handle.read().decode("utf-8", errors="replace")
        except OSError:
            return ""

    @staticmethod
    def _unlink(path: str) -> None:
        try:
            os.unlink(path)
        except OSError:
            pass

    def get_output(self) -> str:
        return self.combined_stdout

    def get_error(self) -> str:
        return self.combined_stderr


def _capture(call: Callable[[], Any]) -> tuple[Any, str, BaseException | None]:
    """Run ``call`` inside :class:`OutputCapture` and return its value/output/error."""
    value = None
    error = None
    output = ""
    try:
        with OutputCapture() as capture:
            try:
                value = call()
            except (Exception, SystemExit) as exc:  # noqa: BLE001 - reported to the caller
                error = exc
        output = capture.get_output()
    except Exception as exc:  # noqa: BLE001 - output capture itself failed
        error = exc
        output = f"output capture failed: {exc}"
    return value, output, error


def compile_and_get_output(source_string: str, *args):
    """Compile Python source and return ``(code, captured_output, error)``."""
    return _capture(lambda: compile(source_string, *args))


def exec_and_get_output(source_string: str, context: dict):
    """Execute Python source and return ``(result, captured_output, error)``."""
    return _capture(lambda: exec(source_string, context))


def format_exception_string(exception_or_string, truncate: bool = True, max_length: int = 200) -> str:
    if isinstance(exception_or_string, str):
        return exception_or_string
    text = str(exception_or_string)
    if truncate and len(text) > max_length:
        text = text[: max_length - 3] + "..."
    return f"{type(exception_or_string)}({text})"


def register_and_format_exception(
    exception_type: str,
    exception_msg: Exception | str,
    metadata: dict,
    verbose: bool = False,
    truncate: bool = False,
    max_length: int = 200,
):
    """Store an exception message in ``metadata`` and return it for the result row."""
    text = str(exception_msg)
    if truncate and len(text) > max_length:
        text = text[: max_length - 3] + "..."
    if verbose:
        print(f"[Exception {exception_type}] {text}")
    metadata[exception_type] = text
    return metadata, text
