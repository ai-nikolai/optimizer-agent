"""Vendored copy of the robust_kernelbench evaluator (AGPL-3.0).

Source: https://github.com/ai-nikolai/robust_kernelbench (v0.2.0)
Vendored so KernelBench evaluation in this repository needs no external
checkout. Only the evaluation path is included: ``evaluate_single`` plus the
modules it imports. Local imports were rewritten to relative imports; the
evaluation logic is unchanged.
"""

__version__ = "0.2.0"
