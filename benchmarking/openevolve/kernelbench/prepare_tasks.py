#!/usr/bin/env python3
"""Prepare OpenEvolve KernelBench tasks from the KernelBench dataset.

For each problem this writes a self-contained task directory:

    reference.py        original KernelBench problem (Model, get_inputs, ...)
    initial_program.py  reference code + a correct ModelNew wrapper
    evaluator.py        robust_kernelbench-backed OpenEvolve evaluator
    config.yaml         strong OpenEvolve search config + local vLLM
    meta.json           problem metadata

Examples:
    python benchmarking/openevolve/kernelbench/prepare_tasks.py --level 1 --limit 10
    python benchmarking/openevolve/kernelbench/prepare_tasks.py \
        --level 1 --problem-ids 100,99,98
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
ROBUST_KB_ROOT = Path(
    "/hx2-weka/home/nr1713/robust_kernelbench"
)
if str(ROBUST_KB_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBUST_KB_ROOT))

from robust_kernelbench.inference_common import load_dataset_for_level  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--problem-ids", default=None, help="Comma-separated IDs")
    parser.add_argument("--output", default=None)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--model-url", default="http://localhost:30000/v1")
    parser.add_argument(
        "--model-name", default="Qwen/Qwen3-Coder-30B-A3B-Instruct"
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=8192)
    return parser.parse_args()


def slug(text):
    import re

    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(text)).strip("_")[:60] or "problem"


def initial_program(reference_code):
    if "class ModelNew" in reference_code:
        raise ValueError("reference problem already defines ModelNew")
    wrapper = '''

class ModelNew(nn.Module):
    """Correct but unoptimized starting point; evolve this class."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = Model(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
'''
    return (
        "# EVOLVE-BLOCK-START\n"
        "import torch\n"
        "import torch.nn as nn\n"
        f"{reference_code.rstrip()}\n"
        f"{wrapper}"
        "# EVOLVE-BLOCK-END\n"
    )


def config_text(args):
    return f'''# OpenEvolve KernelBench task
max_iterations: 100
checkpoint_interval: 10
log_level: INFO
random_seed: 42
diff_based_evolution: true
max_code_length: 20000

llm:
  api_base: "{args.model_url}"
  api_key: "dummy"
  models:
    - name: "{args.model_name}"
      weight: 1.0
  temperature: {args.temperature}
  max_tokens: {args.max_tokens}
  timeout: 300
  retries: 3

prompt:
  system_message: |
    You are an expert CUDA kernel engineer. The program defines a reference
    Model, get_inputs and get_init_inputs, plus a ModelNew class. Optimize
    ModelNew so it produces the same outputs as Model but runs faster. You may
    use torch, triton, torch.utils.cpp_extension.load_inline, or any other
    available package. Preserve the ModelNew interface and return complete,
    runnable Python code. Correctness comes first; invalid or slow kernels score
    zero.
  num_top_programs: 3
  num_diverse_programs: 2
  include_artifacts: true

database:
  population_size: 1000
  archive_size: 100
  num_islands: 4
  elite_selection_ratio: 0.1
  exploration_ratio: 0.3
  exploitation_ratio: 0.6
  feature_bins: 10
  migration_interval: 20
  migration_rate: 0.1

evaluator:
  timeout: 600
  max_retries: 3
  cascade_evaluation: true
  cascade_thresholds: [0.5, 0.8]
  parallel_evaluations: 1
'''


def main():
    args = parse_args()
    output = Path(args.output) if args.output else HERE / "tasks" / f"level_{args.level}"
    if args.clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = sorted(load_dataset_for_level(args.level), key=lambda row: int(row["problem_id"]))
    if args.problem_ids:
        wanted = {int(part) for part in args.problem_ids.split(",") if part.strip()}
        rows = [row for row in rows if int(row["problem_id"]) in wanted]
    else:
        rows = rows[args.start :]
        if args.limit is not None:
            rows = rows[: args.limit]

    evaluator_source = (HERE / "evaluator_template.py").read_text()
    for row in rows:
        pid = int(row["problem_id"])
        name = row.get("name") or f"problem_{pid}"
        task_dir = output / f"pid_{pid}"
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "reference.py").write_text(row["code"])
        (task_dir / "initial_program.py").write_text(initial_program(row["code"]))
        (task_dir / "evaluator.py").write_text(evaluator_source)
        (task_dir / "config.yaml").write_text(config_text(args))
        (task_dir / "meta.json").write_text(
            json.dumps(
                {
                    "problem_id": pid,
                    "name": name,
                    "level": args.level,
                    "slug": slug(name),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"prepared pid={pid} {name} -> {task_dir}")

    print(f"Prepared {len(rows)} tasks in {output}")


if __name__ == "__main__":
    main()
