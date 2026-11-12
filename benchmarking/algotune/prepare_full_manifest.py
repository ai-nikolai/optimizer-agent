#!/usr/bin/env python3
"""Build the full AlgoTune task manifest from the official HF dataset.

The official AlgoTune dataset stores one train/test pair per task, with the
official size parameter encoded in the filename:

    data/<task>/<task>_T100ms_n<n>_size100_train.jsonl

This script reads the Hugging Face file list (metadata only -- the 328 GB of
instance arrays are never downloaded), joins it with the local task
descriptions, and writes ``tasks_full.json`` plus the official named task lists.

Usage:
    python benchmarking/algotune/prepare_full_manifest.py \
        --algotune-root /hx2-weka/home/nr1713/AlgoTune

    python benchmarking/algotune/prepare_dataset.py \
        --algotune-root /hx2-weka/home/nr1713/AlgoTune \
        --manifest benchmarking/algotune/tasks_full.json \
        --output benchmarking/algotune/dataset_full
"""

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ALGOTUNE_ROOT = "/hx2-weka/home/nr1713/AlgoTune"
DEFAULT_HF_REPO = "oripress/AlgoTune"
FILENAME_RE = re.compile(
    r"^data/(?P<task>[^/]+)/(?P=task)_T(?P<target>\d+)ms_n(?P<n>\d+)_size(?P<size>\d+)_train\.jsonl$"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algotune-root", default=DEFAULT_ALGOTUNE_ROOT)
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument(
        "--target-ms",
        type=int,
        default=100,
        help="Official target reference runtime encoded in the dataset filenames.",
    )
    parser.add_argument("--seeds", default="1,2")
    parser.add_argument("--output", default=str(HERE / "tasks_full.json"))
    parser.add_argument(
        "--task-lists-output", default=str(HERE / "task_lists.json")
    )
    parser.add_argument(
        "--file-list",
        default=None,
        help="Optional local JSON list of HF repo files, to avoid a network call.",
    )
    return parser.parse_args()


def load_file_list(repo, cache_path):
    if cache_path:
        return json.loads(Path(cache_path).read_text())
    from huggingface_hub import list_repo_files

    return list_repo_files(repo, repo_type="dataset")


def category_for(task_dir):
    description = task_dir / "description.txt"
    if not description.exists():
        return ""
    match = re.search(
        r"^Category:\s*(\S+)", description.read_text(errors="replace"), re.MULTILINE
    )
    return match.group(1) if match else ""


def main():
    args = parse_args()
    root = Path(args.algotune_root).resolve()
    tasks_root = root / "AlgoTuneTasks"
    if not tasks_root.is_dir():
        raise SystemExit(f"missing task directory: {tasks_root}")

    files = load_file_list(args.hf_repo, args.file_list)
    official = {}
    for path in files:
        match = FILENAME_RE.match(path)
        if not match:
            continue
        if int(match.group("target")) != args.target_ms:
            continue
        task = match.group("task")
        official[task] = {
            "n": int(match.group("n")),
            "target_time_ms": int(match.group("target")),
            "official_dataset_size": int(match.group("size")),
        }

    local_tasks = sorted(
        path.name
        for path in tasks_root.iterdir()
        if path.is_dir() and not path.name.startswith("__")
    )
    missing = sorted(set(local_tasks) - set(official))
    extra = sorted(set(official) - set(local_tasks))
    if missing:
        print(f"WARNING: local tasks missing from HF metadata: {missing}", file=sys.stderr)
    if extra:
        print(f"WARNING: HF tasks missing locally: {extra}", file=sys.stderr)

    seeds = [int(part) for part in args.seeds.split(",") if part.strip()]
    manifest = []
    for task in local_tasks:
        info = official.get(task)
        if info is None:
            continue
        manifest.append(
            {
                "name": task,
                "n": info["n"],
                "seeds": seeds,
                "category": category_for(tasks_root / task),
                "target_time_ms": info["target_time_ms"],
            }
        )

    output = Path(args.output)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(manifest)} tasks to {output}")

    # Save the official named task lists for reference / subset runs.
    sys.path.insert(0, str(root))
    try:
        from AlgoTuner.task_lists import TASK_LISTS
    except Exception as exc:  # noqa: BLE001 - optional metadata
        print(f"WARNING: could not import official task lists: {exc}", file=sys.stderr)
        TASK_LISTS = {}
    lists_path = Path(args.task_lists_output)
    lists_path.write_text(json.dumps(TASK_LISTS, indent=2) + "\n")
    print(f"Wrote official task lists to {lists_path}")


if __name__ == "__main__":
    main()
