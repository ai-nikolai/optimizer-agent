"""Candidate populations and predecessor selection.

The agent keeps, per island, one of these populations. When it needs predecessors
for the next generation it asks the population for its ``get_best()``. Three
strategies are provided (all domain-agnostic -- nothing about CUDA/scores here,
just ordering by ``score``):

* ``bestN_last``    -- keep only the most recent generation as the candidate pool.
* ``bestN_all``     -- keep every candidate ever seen.
* ``map_elites``    -- cluster similar code (string Jaccard) into "cells" and
  keep a diverse set of top-scoring cells.

``textdistance`` is imported lazily so the package stays importable without it;
the other two strategies don't need it at all.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class SingleCandidate:
    """One model output for one prompt: code + its score + context."""

    def __init__(self, path, scores, other=None, prompt_data=None):
        self.path = path
        self.score = float(scores.get("combined_score", float("-inf")))
        self.scores = scores
        self.other = other  # the map-elites cell key this candidate belongs to
        self.prompt_data = prompt_data or {}

    def __repr__(self):
        return f"<Candidate score={self.score}>"


class Population:
    """Base class: a bucket of candidates with a notion of "the best"."""

    name = "population"

    def get_best(self) -> Dict[int, SingleCandidate]:
        raise NotImplementedError

    def update_candidates(self, candidate, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        parts = []
        for key, candidate in self.get_best().items():
            parts.append(f'"{key}" : "{candidate.score}"')
        return f"-- {self.name} --\n{{\n" + "\n".join(parts) + "\n--end--"


class CandidatesBestNLast(Population):
    """Only the latest generation is a predecessor candidate."""

    name = "bestN_last"

    def __init__(self, **kwargs):
        self.candidates = []
        self.best = {}
        self.current_iteration = None

    def get_best(self):
        return self.best

    def update_candidates(self, candidate, current_iteration, **kwargs):
        if self.current_iteration is None or current_iteration > self.current_iteration:
            self.candidates = [candidate]
        else:
            self.candidates.append(candidate)
        self.current_iteration = current_iteration
        if not self.best or candidate.score > self.best[self.name].score:
            self.best[self.name] = candidate


class CandidatesBestNAll(Population):
    """Every candidate ever seen is a predecessor candidate."""

    name = "bestN_all"

    def __init__(self, **kwargs):
        self.candidates = []
        self.best = {}

    def get_best(self):
        return self.best

    def update_candidates(self, candidate, **kwargs):
        self.candidates.append(candidate)
        if not self.best or candidate.score > self.best[self.name].score:
            self.best[self.name] = candidate


class MapElitesCandidates(Population):
    """Diverse top candidates: cluster by code similarity, keep the best per cell."""

    name = "map_elites"
    cluster_threshold = 0.2

    def __init__(self, num_map_elites=1, **kwargs):
        self.num_map_elites = num_map_elites
        self.cell_best = {}      # cell key -> best candidate so far
        self.cell_scores = []    # (cell key, best score) for ranking
        self.best = {}

    def get_best(self):
        self.cell_scores.sort(key=lambda kv: kv[1], reverse=True)
        self.best = {}
        for key, _ in self.cell_scores[: self.num_map_elites]:
            self.best[key] = self.cell_best[key]
        return self.best

    def update_candidates(self, candidate, **kwargs):
        key = candidate.other
        prev = self.cell_best.get(key)
        if prev is None or candidate.score > prev.score:
            self.cell_best[key] = candidate
            current = candidate.score
            for idx, (existing, score) in enumerate(self.cell_scores):
                if existing == key:
                    self.cell_scores[idx] = (key, current)
                    break
            else:
                self.cell_scores.append((key, current))

    def get_key(self, new_generation, current_other):
        """Return the cell key to use for ``new_generation`` (string-Jaccard clustering)."""
        if new_generation == current_other:
            return current_other, 0.0
        if not self.cell_best:
            return current_other, -1.0
        try:
            import textdistance
        except ImportError:  # pragma: no cover - only if textdistance is missing
            return new_generation, 0.0
        best_key, best_distance = current_other, None
        for key in self.cell_best:
            distance = textdistance.jaccard.normalized_distance(key, new_generation)
            if best_distance is None or distance < best_distance:
                best_key, best_distance = key, distance
        if best_distance > self.cluster_threshold:
            return new_generation, best_distance
        return best_key, best_distance


_SELECTION_CLASSES = {
    "bestN_all": CandidatesBestNAll,
    "bestN_last": CandidatesBestNLast,
    "map_elites": MapElitesCandidates,
}
