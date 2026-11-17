# EVOLVE-BLOCK-START
"""
Kernel Density Estimation (KDE) baseline.

This is the unoptimized reference implementation that OpenEvolve will improve.
"""
import logging
from numbers import Real

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KernelDensity


class KernelDensityEstimation:
    def __init__(self):
        self.available_kernels = [
            "gaussian",
            "tophat",
            "epanechnikov",
            "exponential",
            "linear",
            "cosine",
        ]

    def solve(self, problem):
        try:
            X = np.array(problem["data_points"])
            X_q = np.array(problem["query_points"])
            kernel = problem["kernel"]
            bandwidth = problem["bandwidth"]

            if X.ndim != 2 or X_q.ndim != 2:
                raise ValueError("Data points or query points are not 2D arrays.")
            if X.shape[0] == 0:
                raise ValueError("No data points provided.")
            if X_q.shape[0] == 0:
                return {"log_density": []}
            if X.shape[1] != X_q.shape[1]:
                raise ValueError("Data points and query points have different dimensions.")
            if not isinstance(bandwidth, Real) or bandwidth <= 0:
                raise ValueError("Bandwidth must be positive.")
            bandwidth = float(bandwidth)
            if kernel not in self.available_kernels:
                raise ValueError(f"Unknown kernel: {kernel}")

            kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
            kde.fit(X)
            log_density = kde.score_samples(X_q)
            return {"log_density": log_density.tolist()}

        except KeyError as e:
            logging.error("Missing key in problem dictionary: %s", e)
            return {"error": f"Missing key: {e}"}
        except (ValueError, TypeError, NotFittedError, np.linalg.LinAlgError) as e:
            logging.error("Error during KDE computation: %s", e)
            return {"error": f"Computation error: {e}"}
        except Exception as e:
            logging.error("An unexpected error occurred during solve: %s", e, exc_info=True)
            return {"error": f"Unexpected error: {e}"}

    def is_solution(self, problem, solution):
        try:
            if not isinstance(solution, dict) or "log_density" not in solution:
                return False
            expected = self.solve(problem)
            if "error" in expected:
                return False
            actual = np.asarray(solution["log_density"], dtype=float).squeeze()
            reference = np.asarray(expected["log_density"], dtype=float).squeeze()
            if actual.shape != reference.shape:
                return False
            if not np.array_equal(np.isfinite(actual), np.isfinite(reference)):
                return False
            finite = np.isfinite(actual) & np.isfinite(reference)
            return bool(np.allclose(actual[finite], reference[finite], rtol=1e-5, atol=1e-8))
        except Exception:
            return False


def run_solver(problem):
    return KernelDensityEstimation().solve(problem)
# EVOLVE-BLOCK-END
