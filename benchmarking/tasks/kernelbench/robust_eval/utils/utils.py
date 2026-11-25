"""Filesystem layout helpers shared by the whole pipeline.

Everything here is deliberately dependency-free (stdlib only) so that the very
light-weight modules (and the tests) can import it without pulling in torch.
"""

from __future__ import annotations

import os
from pathlib import Path

from .. import __version__

VERSION = __version__


def get_folder_path(experiment_name, trial=None, postfix=None, prefix="experiments"):
    """Creates the folder and returns path"""
    if trial:
        trial_name = f"trial_{trial}"
        folder_path = os.path.join(experiment_name, trial_name)
    else:
        folder_path = os.path.join(experiment_name)

    if postfix:
        folder_path = os.path.join(folder_path, postfix)

    if prefix:
        folder_path = os.path.join(prefix, folder_path)

    Path(folder_path).mkdir(parents=True, exist_ok=True)

    return folder_path


def get_file_name(problem_id, sample_id=None, suffix=None):
    """Creates the filename"""
    if sample_id:
        file_name = f"{problem_id}__sample_{sample_id}"
    else:
        file_name = f"{problem_id}__sample_"

    if suffix and type(suffix) == str:
        file_name = file_name + "_" + suffix

    return file_name


def get_problem_id_from_file_name(file_name):
    """Returns problem_id"""
    problem_id = int(file_name.split("__")[0])
    return problem_id


def read_file(file_path) -> str:
    """Read a text file, returning "" (and logging a warning) when it is missing.

    Historically this lived in ``evaluate_single`` and was imported *from there* by the
    compile drivers, which dragged torch into modules that only needed a file read.
    """
    if not os.path.exists(file_path):
        import logging

        logging.getLogger(__name__).warning("File %s does not exist", file_path)
        return ""

    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:  # pragma: no cover - io failure path
        import logging

        logging.getLogger(__name__).warning("Error reading file %s: %s", file_path, e)
        return ""


def write_code(fullpath, code_src):
    """Write source code to ``fullpath``."""
    with open(fullpath, "w") as file:
        file.write(code_src)


def find_ending_digit_and_string(name):
    """Split a trailing decimal suffix off ``name``.

    ``"9__sample_12"`` -> ``("9__sample_", "12")``.  A name without a trailing
    digit returns ``(name, "")``; the previous implementation sliced the name
    incorrectly in that case.
    """
    idx = len(name)
    while idx > 0 and name[idx - 1].isdigit():
        idx -= 1
    return name[:idx], name[idx:]


def construct_generation_dict(folder):
    """
    Group generated kernel files by problem.

    Assumes files named according to the KernelBench problem names, e.g.
    ``1_Square_matrix_multiplication_0.py``, and returns::

        {'1_Square_matrix_multiplication_': ['..._0.py', '..._1.py', ...]}

    The per-problem list is sorted by the **numeric** sample suffix.  A plain
    lexicographic sort puts ``_10.py`` before ``_2.py``, which silently shifts
    every sample_id after the ninth sample.
    """
    out_dict: dict[str, list[str]] = {}

    for file in sorted(os.listdir(folder)):
        if not file.endswith(".py"):
            continue
        file_basename = file[:-3]
        file_basename_root, _file_digit = find_ending_digit_and_string(file_basename)
        out_dict.setdefault(file_basename_root, []).append(file)

    for solutions in out_dict.values():
        solutions.sort(key=_sample_sort_key)
    return out_dict


def _sample_sort_key(filename: str) -> tuple[int, str]:
    """Numeric sample suffix first, with the filename as a stable tie-breaker."""
    stem = filename[:-3] if filename.endswith(".py") else filename
    _, digits = find_ending_digit_and_string(stem)
    return (int(digits) if digits else -1, filename)
