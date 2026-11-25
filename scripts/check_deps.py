#!/usr/bin/env python3
"""Check the external checkouts this repository expects.

Run this first on a new machine:

    python scripts/check_deps.py

`robust_kernelbench` is required for KernelBench evaluation and must be pulled
manually — it is imported, not vendored. AlgoTune and OpenEvolve are only needed
for their own benchmarks.
"""

import importlib
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

CHECKS = [
    {
        "module": "robust_kernelbench",
        "env": "ROBUST_KERNELBENCH_ROOT",
        "default": "$HOME/robust_kernelbench",
        "repo": "https://github.com/ai-nikolai/robust_kernelbench",
        "required": True,
        "why": "KernelBench evaluation (imported by tasks/kernelbench/evaluate.py)",
        "install": "pip install -e $HOME/robust_kernelbench",
        "min_feature": ("evaluate_single", "error_kind"),
    },
    {
        "module": "AlgoTuneTasks",
        "env": "ALGOTUNE_ROOT",
        "default": "$HOME/AlgoTune",
        "repo": "https://github.com/oripress/AlgoTune",
        "required": False,
        "why": "AlgoTune task implementations",
        "install": "# no install needed; the checkout is used via ALGOTUNE_ROOT",
    },
    {
        "module": "openevolve",
        "env": "OPENEVOLVE_ROOT",
        "default": "$HOME/openevolve",
        "repo": "https://github.com/ai-nikolai/openevolve",
        "required": False,
        "why": "OpenEvolve competitor baselines",
        "install": "pip install -e $HOME/openevolve",
    },
]


def check_agent() -> bool:
    try:
        import optimizer_agent
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] optimizer_agent        {type(exc).__name__}: {exc}")
        print("       pip install -e .")
        return False
    print(f"[ok]   optimizer_agent        {optimizer_agent.__file__}")
    return True


def check(spec: dict) -> bool:
    env = os.environ.get(spec["env"])
    root = Path(env).expanduser() if env else None
    if root and root.exists() and str(root) not in sys.path:
        sys.path.insert(0, str(root))

    label = spec["module"]
    try:
        module = importlib.import_module(label)
    except Exception as exc:  # noqa: BLE001
        level = "FAIL" if spec["required"] else "warn"
        print(f"[{level}] {label:22s} not importable ({type(exc).__name__})")
        print(f"       needed for: {spec['why']}")
        print(f"       git clone {spec['repo']} {spec['default']}")
        print(f"       export {spec['env']}={spec['default']}")
        print(f"       {spec['install']}")
        return not spec["required"]

    path = Path(getattr(module, "__file__", "?"))
    version = getattr(module, "__version__", "")
    print(f"[ok]   {label:22s} {version:8s} {path}")

    feature = spec.get("min_feature")
    if feature:
        mod_name, attr = feature
        try:
            target = importlib.import_module(f"{label}.{mod_name}")
            source = Path(target.__file__).read_text()
        except Exception:  # noqa: BLE001
            source = ""
        if attr not in source:
            print(
                f"[warn] {label}: this checkout predates the structured correctness fix "
                f"(no `{attr}`); update it before running KernelBench."
            )
    return True


def main() -> int:
    print(f"repository: {REPO}")
    ok = check_agent()
    for spec in CHECKS:
        ok = check(spec) and ok
    print()
    if ok:
        print("required dependencies present")
    else:
        print("missing required dependencies — see AGENTS.md §2")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
