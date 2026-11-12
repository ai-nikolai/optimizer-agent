"""Baseline AlgoTune Solver for dst_type_II_scipy_fftpack.

This is intentionally correct but unoptimized: it delegates to the official
AlgoTune reference implementation. Replace the body of solve with a faster
implementation.
"""

from AlgoTuneTasks.factory import TaskFactory


class Solver:
    def __init__(self):
        self.task = TaskFactory("dst_type_II_scipy_fftpack")

    def solve(self, problem, **kwargs):
        return self.task.solve(problem)
