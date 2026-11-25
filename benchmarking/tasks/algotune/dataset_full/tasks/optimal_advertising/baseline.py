"""Baseline AlgoTune Solver for optimal_advertising.

This is intentionally correct but unoptimized: it delegates to the official
AlgoTune reference implementation. Replace the body of solve with a faster
implementation.
"""

from AlgoTuneTasks.factory import TaskFactory


class Solver:
    def __init__(self):
        self.task = TaskFactory("optimal_advertising")

    def solve(self, problem, **kwargs):
        return self.task.solve(problem)
