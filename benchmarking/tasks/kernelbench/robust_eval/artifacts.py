"""Filesystem and CSV artefacts for one KernelBench experiment trial.

Every stage reads and writes the same four files under
``experiments/<experiment>/trial_<N>/``:

``generations.csv``
    One row per prompted problem/sample.
``compilations.csv``
    One row per compiled problem/sample.
``evaluations.csv``
    One row per evaluated problem/sample.
``metadata.json``
    Generation settings for the trial.

Keeping the paths, row counts and append logic here means the stages cannot drift
apart (in particular, they all agree on how a CSV is appended to after a crash).
"""

from __future__ import annotations

import csv
import os
import sys
from typing import Iterable

import pandas as pd

from .utils.utils import (
    construct_generation_dict,
    get_folder_path,
    get_problem_id_from_file_name,
)

GENERATIONS_CSV = "generations.csv"
COMPILATIONS_CSV = "compilations.csv"
EVALUATIONS_CSV = "evaluations.csv"
METADATA_JSON = "metadata.json"

# Kernel sources live inside CSV cells; the 128 KiB csv default truncates them.
csv.field_size_limit(sys.maxsize)

KernelKey = tuple[int, int]


def trial_dir(experiment_name: str, trial: int) -> str:
    """Directory holding one trial's CSV files and build trees."""
    return get_folder_path(experiment_name, trial)


def csv_path(experiment_name: str, trial: int, filename: str) -> str:
    return os.path.join(trial_dir(experiment_name, trial), filename)


def ensure_trailing_newline(path: str) -> None:
    """Make sure ``path`` ends with a newline before another row is appended.

    A child killed mid-append can leave a truncated last line; appending directly
    after it would merge two CSV rows and corrupt every later read.  Best effort:
    an I/O error here is not allowed to stop the run.
    """
    try:
        with open(path, "rb+") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                return
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                handle.write(b"\n")
    except OSError:
        # The caller will notice the real failure when it writes the row.
        pass


def append_rows(df: pd.DataFrame, path: str) -> None:
    """Append ``df`` to a CSV, creating it with a header when necessary."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        ensure_trailing_newline(path)
        df.to_csv(path, index=False, header=False, mode="a")
    else:
        df.to_csv(path, index=False)


def count_rows(path: str) -> int:
    """Number of data rows in ``path`` (0 when absent)."""
    if not os.path.exists(path):
        return 0
    with open(path, newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def read_csv(path: str) -> pd.DataFrame | None:
    """Read a trial CSV, returning ``None`` when it does not exist."""
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def generated_kernel_keys(experiment_name: str, trial: int) -> set[KernelKey]:
    """``(problem_id, sample_id)`` pairs that have a generated ``.py`` file."""
    code_dir = get_folder_path(experiment_name, trial, postfix="code")
    grouped = construct_generation_dict(code_dir)
    keys: set[KernelKey] = set()
    for file_stem, solutions in grouped.items():
        problem_id = get_problem_id_from_file_name(file_stem)
        keys.update((problem_id, sample_id) for sample_id in range(len(solutions)))
    return keys


def _keys_from_csv(path: str, successful_only: bool = False) -> set[KernelKey]:
    frame = read_csv(path)
    if frame is None or frame.empty:
        return set()
    if successful_only:
        if "model_new_available" not in frame.columns:
            return set()
        frame = frame[frame["model_new_available"].fillna(False).astype(bool)]
    if not {"problem_id", "sample_id"}.issubset(frame.columns):
        return set()
    return {
        (int(problem_id), int(sample_id))
        for problem_id, sample_id in frame[["problem_id", "sample_id"]].itertuples(
            index=False, name=None
        )
    }


def compilation_keys(experiment_name: str, trial: int, successful_only: bool = False) -> set[KernelKey]:
    return _keys_from_csv(csv_path(experiment_name, trial, COMPILATIONS_CSV), successful_only)


def evaluation_keys(experiment_name: str, trial: int) -> set[KernelKey]:
    return _keys_from_csv(csv_path(experiment_name, trial, EVALUATIONS_CSV))


def format_keys(keys: Iterable[KernelKey], limit: int = 8) -> str:
    """Short, stable rendering of missing keys for error messages."""
    ordered = sorted(keys)
    shown = ", ".join(f"{pid}/{sid}" for pid, sid in ordered[:limit])
    if len(ordered) > limit:
        shown += f", ... ({len(ordered)} total)"
    return shown
