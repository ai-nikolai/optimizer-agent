#!/usr/bin/env python3
"""Prepare OpenEvolve AlgoTune tasks for a local vLLM model.

Copies the official OpenEvolve AlgoTune examples and adapts them:

- evaluator finds the local AlgoTune checkout via ALGOTUNE_PATH
- LLM points at the local OpenAI-compatible vLLM server
- ``algotune.data_size`` uses the same official n as the optimizer-agent run
- OpenEvolve's native strong search settings are preserved (100 iterations,
  population 1000, 4 islands, 5 correctness trials, 3 timing runs)

Example:
    python benchmarking/openevolve/algotune/prepare_tasks.py
    python benchmarking/openevolve/algotune/prepare_tasks.py --tasks affine_transform_2d,svm
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_SOURCE = Path("/hx2-weka/home/nr1713/openevolve/examples/algotune")
DEFAULT_MANIFEST = HERE.parents[1] / "algotune" / "tasks.json"
DEFAULT_ALGOTUNE_ROOT = "/hx2-weka/home/nr1713/AlgoTune"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(HERE / "tasks"))
    parser.add_argument("--tasks", default=None, help="Comma-separated subset")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--model-url", default="http://localhost:30000/v1")
    parser.add_argument(
        "--model-name", default="Qwen/Qwen3-Coder-30B-A3B-Instruct"
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--algotune-root", default=DEFAULT_ALGOTUNE_ROOT)
    return parser.parse_args()


def patch_evaluator(path, algotune_root):
    text = path.read_text()
    block = f'''    possible_algotune_paths = [
        Path(os.environ.get("ALGOTUNE_PATH", "{algotune_root}")),
        Path("{algotune_root}"),
        Path(__file__).parent.parent.parent.parent / "AlgoTune",
    ]'''
    new_text, count = re.subn(
        r"    possible_algotune_paths = \[.*?\]",
        block,
        text,
        count=1,
        flags=re.DOTALL,
    )
    if count == 0:
        raise SystemExit(f"could not patch AlgoTune path list in {path}")
    path.write_text(new_text)


def patch_config(path, args, n):
    config = yaml.safe_load(path.read_text())
    config["llm"] = {
        "api_base": args.model_url,
        "api_key": "dummy",
        "models": [{"name": args.model_name, "weight": 1.0}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout": 300,
        "retries": 3,
    }
    evaluator = config.setdefault("evaluator", {})
    evaluator["parallel_evaluations"] = 1
    evaluator["timeout"] = max(int(evaluator.get("timeout", 200)), 600)
    algotune = config.setdefault("algotune", {})
    algotune["data_size"] = n
    algotune["num_trials"] = 5
    algotune["num_runs"] = 3
    algotune["warmup_runs"] = 1
    config["max_iterations"] = 100
    path.write_text(yaml.safe_dump(config, sort_keys=False))


def main():
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    manifest = json.loads(Path(args.manifest).read_text())
    if args.tasks:
        wanted = {name.strip() for name in args.tasks.split(",") if name.strip()}
        manifest = [row for row in manifest if row["name"] in wanted]

    if args.clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    for row in manifest:
        name = row["name"]
        src = source / name
        dst = output / name
        if not (src / "initial_program.py").exists():
            print(f"skipping {name}: no OpenEvolve example at {src}")
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(
            src,
            dst,
            ignore=shutil.ignore_patterns("__pycache__", "openevolve_output"),
        )
        patch_evaluator(dst / "evaluator.py", args.algotune_root)
        patch_config(dst / "config.yaml", args, int(row.get("n", 100)))
        (dst / "meta.json").write_text(
            json.dumps(
                {
                    "task": name,
                    "n": row.get("n"),
                    "seeds": row.get("seeds", [1, 2]),
                    "category": row.get("category", ""),
                    "source": str(src),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"prepared {name} n={row.get('n')} -> {dst}")


if __name__ == "__main__":
    main()
