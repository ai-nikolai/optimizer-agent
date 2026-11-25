#!/usr/bin/env python3
"""Prepare a lightweight AlgoTune task dataset for optimizer-agent / HF.

The official AlgoTune repository contains the 154 task definitions. This script
does NOT duplicate the official 328 GB instance dataset; it creates a portable
task manifest containing:

    dataset/
      manifest.json
      tasks.jsonl
      tasks/<task>/description.txt
      tasks/<task>/task_source.py
      tasks/<task>/baseline.py
      tasks/<task>/reference.txt

The baseline is a correct-but-unoptimized ``Solver`` that calls the official
task's solve method. The agent's job is to replace it with a faster solver.

Usage:
    python prepare_dataset.py --algotune-root /path/to/AlgoTune
    python prepare_dataset.py --algotune-root /path/to/AlgoTune \
        --push-to-hf your-username/AlgoTuneTasks
"""

import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "dataset"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algotune-root",
        required=True,
        help="Path to the official AlgoTune checkout (contains AlgoTuneTasks/).",
    )
    parser.add_argument("--manifest", default=str(HERE / "tasks.json"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--push-to-hf",
        default=None,
        help="Optional HF dataset repo id, e.g. your-name/AlgoTuneTasks.",
    )
    return parser.parse_args()


def baseline_source(task_name):
    return f'''"""Baseline AlgoTune Solver for {task_name}.

This is intentionally correct but unoptimized: it delegates to the official
AlgoTune reference implementation. Replace the body of solve with a faster
implementation.
"""

from AlgoTuneTasks.factory import TaskFactory


class Solver:
    def __init__(self):
        self.task = TaskFactory("{task_name}")

    def solve(self, problem, **kwargs):
        return self.task.solve(problem)
'''


def main():
    args = parse_args()
    root = Path(args.algotune_root).resolve()
    manifest = json.loads(Path(args.manifest).read_text())
    output = Path(args.output)
    tasks_root = output / "tasks"
    tasks_root.mkdir(parents=True, exist_ok=True)

    records = []
    for entry in manifest:
        name = entry["name"]
        source_dir = root / "AlgoTuneTasks" / name
        source_file = source_dir / f"{name}.py"
        description_file = source_dir / "description.txt"
        if not source_file.exists():
            raise FileNotFoundError(f"missing official task source: {source_file}")
        description = description_file.read_text() if description_file.exists() else name

        task_dir = tasks_root / name
        task_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, task_dir / "task_source.py")
        (task_dir / "description.txt").write_text(description)
        (task_dir / "baseline.py").write_text(baseline_source(name))
        (task_dir / "reference.txt").write_text(
            description.rstrip()
            + "\n\n## Official reference implementation source\n\n"
            + source_file.read_text()
        )

        record = {
            "task": name,
            "category": entry.get("category", ""),
            "n": entry.get("n"),
            "seeds": entry.get("seeds", [1, 2]),
            "description": description,
            "baseline": str((task_dir / "baseline.py").relative_to(output)),
            "reference": str((task_dir / "reference.txt").relative_to(output)),
            "task_source": str((task_dir / "task_source.py").relative_to(output)),
        }
        records.append(record)

    (output / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    with open(output / "tasks.jsonl", "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    print(f"Wrote {len(records)} tasks to {output}")

    if args.push_to_hf:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(
            repo_id=args.push_to_hf, repo_type="dataset", exist_ok=True
        )
        api.upload_folder(
            repo_id=args.push_to_hf,
            repo_type="dataset",
            folder_path=str(output),
        )
        print(f"Uploaded to https://huggingface.co/datasets/{args.push_to_hf}")


if __name__ == "__main__":
    main()
