"""Baseline AlgoTune Solver for ode_lotkavolterra.

This is intentionally correct but unoptimized: it delegates to the official
AlgoTune reference implementation. Replace the body of solve with a faster
implementation.
"""

from AlgoTuneTasks.factory import TaskFactory


class Solver:
    def __init__(self):
        self.task = TaskFactory("ode_lotkavolterra")

    def solve(self, problem, **kwargs):
        return self.task.solve(problem)
