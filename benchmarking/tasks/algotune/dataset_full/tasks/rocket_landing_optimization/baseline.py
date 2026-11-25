"""Baseline AlgoTune Solver for rocket_landing_optimization.

This is intentionally correct but unoptimized: it delegates to the official
AlgoTune reference implementation. Replace the body of solve with a faster
implementation.
"""

from AlgoTuneTasks.factory import TaskFactory


class Solver:
    def __init__(self):
        self.task = TaskFactory("rocket_landing_optimization")

    def solve(self, problem, **kwargs):
        return self.task.solve(problem)
