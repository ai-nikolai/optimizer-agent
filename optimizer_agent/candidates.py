"""Candidate populations and predecessor selection.

The agent owns one population. Selection policies:

- ``bestN_last`` -- best candidate from the previous generation
- ``bestN_all``  -- best candidate across all generations
- ``map_elites`` -- top cells, using one of:
    * ``cluster``  -- numerical similarity (Jaccard or embeddings)
    * ``fixed``    -- LLM assigns to a configured taxonomy
    * ``dynamic``  -- LLM assigns to an existing class or creates a new one

All comparisons use a higher-is-better rank score; minimization is applied when
the candidate is created.
"""

import math


class SingleCandidate:
    """One model output: artifact path, raw scores, and its selection rank."""

    def __init__(
        self,
        path,
        scores,
        other=None,
        prompt_data=None,
        score_key="combined_score",
        minimize=False,
    ):
        self.path = path
        self.scores = scores or {}
        self.raw_score = _raw_score(self.scores, score_key)
        self.score = _rank_score(self.raw_score, minimize)
        self.other = other
        self.other_vector = None
        self.prompt_data = prompt_data or {}

    def __repr__(self):
        return (
            f"<Candidate raw_score={self.raw_score} "
            f"rank_score={self.score}>"
        )


def _raw_score(scores, key="combined_score"):
    try:
        value = float((scores or {}).get(key, float("-inf")))
    except (TypeError, ValueError):
        return float("-inf")
    return value if math.isfinite(value) else float("-inf")


def _rank_score(raw, minimize=False):
    if raw == float("-inf"):
        return float("-inf")
    return -raw if minimize else raw


def _insert_top(candidates, candidate, limit):
    """Insert into a descending, bounded list; earlier candidates win ties."""
    if limit <= 0:
        return
    index = len(candidates)
    for position, existing in enumerate(candidates):
        if candidate.score > existing.score:
            index = position
            break
    if index >= limit:
        return
    candidates.insert(index, candidate)
    del candidates[limit:]


class Population:
    name = "population"

    def get_best(self):
        raise NotImplementedError

    def update_candidates(self, candidate, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        lines = [f"-- {self.name} --", "{"]
        for key, candidate in self.get_best().items():
            lines.append(f'"{key}" : "{candidate.raw_score}",')
        lines.append("}\n--end--")
        return "\n".join(lines)


class CandidatesBestNAll(Population):
    """Best candidate across all generations; keeps a bounded top-K pool."""

    name = "bestN_all"

    def __init__(self, max_candidates=20, **kwargs):
        self.max_candidates = max(1, int(max_candidates))
        self.candidates = []
        self.best = None

    def get_best(self):
        return {self.name: self.best} if self.best is not None else {}

    def update_candidates(self, candidate, **kwargs):
        _insert_top(self.candidates, candidate, self.max_candidates)
        if self.best is None or candidate.score > self.best.score:
            self.best = candidate


class CandidatesBestNLast(Population):
    """Best candidate from the previous generation; keeps a bounded top-K pool."""

    name = "bestN_last"

    def __init__(self, max_candidates=20, **kwargs):
        self.max_candidates = max(1, int(max_candidates))
        self.candidates = []
        self.current_iteration = None

    def get_best(self):
        if not self.candidates:
            return {}
        return {self.name: self.candidates[0]}

    def update_candidates(self, candidate, current_iteration=0, **kwargs):
        if self.current_iteration is None or current_iteration > self.current_iteration:
            self.candidates = []
            self.current_iteration = current_iteration
        _insert_top(self.candidates, candidate, self.max_candidates)


class MapElitesCandidates(Population):
    """Keep the best candidate per cell; return the top ``num_map_elites`` cells."""

    name = "map_elites"

    def __init__(
        self,
        num_map_elites=1,
        method="cluster",
        cluster_metric="jaccard",
        taxonomy=None,
        classify=None,
        embed=None,
        max_cells=100,
        cluster_threshold=0.2,
        **kwargs,
    ):
        self.num_map_elites = max(1, int(num_map_elites))
        self.method = (method or "cluster").strip().lower()
        self.cluster_metric = (cluster_metric or "jaccard").strip().lower()
        self.taxonomy = dict(taxonomy or {})
        self.classify = classify
        self.embed = embed
        self.max_cells = max(1, int(max_cells))
        self.cluster_threshold = float(cluster_threshold)

        if self.method not in ("cluster", "fixed", "dynamic"):
            raise ValueError(
                f"Unknown map-elites method {self.method!r}; "
                "expected cluster, fixed or dynamic"
            )
        if self.method in ("fixed", "dynamic") and self.classify is None:
            raise ValueError(
                f"map-elites method {self.method!r} requires a classifier"
            )
        if self.method == "fixed" and not self.taxonomy:
            raise ValueError("fixed map-elites requires a non-empty taxonomy")
        if self.method == "cluster" and self.cluster_metric == "embedding" and self.embed is None:
            raise ValueError("cluster-metric=embedding requires an embedding function")

        self.cell_best = {}
        self.cell_scores = []
        self.cell_vectors = {}
        self.best = {}
        self.last_key_vector = None

    def get_best(self):
        self.cell_scores.sort(key=lambda item: item[1], reverse=True)
        self.best = {
            key: self.cell_best[key]
            for key, _ in self.cell_scores[: self.num_map_elites]
        }
        return self.best

    def update_candidates(self, candidate, **kwargs):
        key = candidate.other
        previous = self.cell_best.get(key)
        if previous is not None and candidate.score <= previous.score:
            return

        self.cell_best[key] = candidate
        if self.cluster_metric == "embedding" and candidate.other_vector is not None:
            self.cell_vectors[key] = candidate.other_vector

        for index, (existing, _) in enumerate(self.cell_scores):
            if existing == key:
                self.cell_scores[index] = (key, candidate.score)
                break
        else:
            self.cell_scores.append((key, candidate.score))

    # ------------------------------------------------------------------
    # Cell assignment
    # ------------------------------------------------------------------
    def get_key(self, new_generation, current_other):
        """Return the cell key for ``new_generation`` and a distance."""
        self.last_key_vector = None
        if self.method in ("fixed", "dynamic"):
            return self._taxonomy_key(new_generation), 0.0
        if self.cluster_metric == "embedding":
            return self._embedding_key(new_generation)
        return self._jaccard_key(new_generation, current_other)

    def _taxonomy_key(self, artifact):
        result = self.classify(artifact, dict(self.taxonomy), self.method)
        if not isinstance(result, dict):
            result = {}
        name = str(result.get("class") or result.get("name") or "").strip()
        description = str(result.get("description") or "").strip()
        is_new = result.get("new")
        if isinstance(is_new, str):
            is_new = is_new.strip().lower() in ("true", "1", "yes", "new")
        else:
            is_new = bool(is_new)

        if self.method == "fixed":
            return name if name in self.taxonomy else "unclassified"

        if name and name in self.taxonomy:
            return name
        if is_new and name:
            if name not in self.taxonomy and len(self.taxonomy) < self.max_cells:
                self.taxonomy[name] = description or name
                return name
        return "unclassified"

    def _jaccard_key(self, new_generation, current_other):
        if new_generation == current_other:
            return current_other, 0.0
        if not self.cell_best:
            return current_other, -1.0
        try:
            import textdistance
        except ImportError as exc:  # textdistance is a declared dependency
            raise RuntimeError("cluster-metric=jaccard requires textdistance") from exc

        best_key, best_distance = current_other, None
        for key in self.cell_best:
            distance = textdistance.jaccard.normalized_distance(key, new_generation)
            if best_distance is None or distance < best_distance:
                best_key, best_distance = key, distance

        if best_distance > self.cluster_threshold:
            return new_generation, best_distance
        return best_key, best_distance

    def _embedding_key(self, new_generation):
        vector = self.embed(new_generation)
        self.last_key_vector = vector
        if not self.cell_vectors:
            key = f"cell_{len(self.cell_vectors)}"
            self.cell_vectors[key] = vector
            return key, 0.0

        best_key, best_distance = None, None
        for key, representative in self.cell_vectors.items():
            distance = _cosine_distance(vector, representative)
            if best_distance is None or distance < best_distance:
                best_key, best_distance = key, distance

        if best_key is None or best_distance > self.cluster_threshold:
            key = f"cell_{len(self.cell_vectors)}"
            self.cell_vectors[key] = vector
            return key, best_distance if best_distance is not None else 0.0
        return best_key, best_distance


def _cosine_distance(left, right):
    if not left or not right or len(left) != len(right):
        return 1.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 1.0
    similarity = dot / (left_norm * right_norm)
    return 1.0 - max(-1.0, min(1.0, similarity))
