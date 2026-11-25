"""Baseline AlgoTune Solver for unit_simplex_projection.

This is intentionally correct but unoptimized: it delegates to the official
AlgoTune reference implementation. Replace the body of solve with a faster
implementation.
"""

from AlgoTuneTasks.factory import TaskFactory


class Solver:
    def __init__(self):
        self.task = TaskFactory("unit_simplex_projection")

    def solve(self, problem, **kwargs):
        return self.task.solve(problem)
