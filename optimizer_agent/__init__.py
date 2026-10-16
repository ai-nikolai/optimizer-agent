from importlib.metadata import version

__version__ = version("optimizer-agent")

# The evaluator prints its JSON result immediately after this marker line.
EVAL_MARKER = "=== Evaluation Result ==="
