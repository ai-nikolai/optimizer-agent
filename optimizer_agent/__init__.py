"""Optimizer Agent: an LLM-driven search loop over artifacts.

The package intentionally keeps a small surface:

- ``main.Agent``      -- the search loop (iterations and generations)
- ``candidates``      -- predecessor selection policies
- ``feedback``        -- classification + pluggable feedback strategies
- ``diffs``           -- unified-diff / JSON-edit application
- ``llm``             -- OpenAI-compatible chat client (and a fake one)
- ``prompts``         -- config-driven prompt builder + default prompts
- ``utils``           -- response/eval parsing and the evaluator subprocess
- ``cli``             -- the ``optimizer-agent`` command
"""

from importlib.metadata import PackageNotFoundError, version

#: Evaluator contract: the result JSON is printed after this marker.
EVAL_MARKER = "=== Evaluation Result ==="

try:
    __version__ = version("optimizer-agent")
except PackageNotFoundError:  # importing from a source checkout
    __version__ = "0.1.0.0"
