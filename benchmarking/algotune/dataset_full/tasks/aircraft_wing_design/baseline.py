"""Baseline AlgoTune Solver for aircraft_wing_design.

This is intentionally correct but unoptimized: it delegates to the official
AlgoTune reference implementation. Replace the body of solve with a faster
implementation.
"""

from AlgoTuneTasks.factory import TaskFactory


class Solver:
    def __init__(self):
        self.task = TaskFactory("aircraft_wing_design")

    def solve(self, problem, **kwargs):
        return self.task.solve(problem)
