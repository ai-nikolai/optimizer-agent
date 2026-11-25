# Benchmark comparison matrix

Fair comparison between optimizer-agent settings and OpenEvolve-only
competitors. All runs use the same local model
(`Qwen/Qwen3-Coder-30B-A3B-Instruct`) served by vLLM.

## AlgoTune

| Setting | Search | Rounds / iterations | Samples per round | Feedback / strategy | Notes |
|---|---|---|---|---|---|
| **optimizer-agent** | `bestN_last` | 10 | 1 | `rules_llm`, generic, no strategy/debug | sequential repair chain |
| **optimizer-agent: advanced** | `map_elites`, dynamic, 3 cells | 4 | 1 | `rules_llm`, generic, strategy + debug | quality-diversity search |
| **OpenEvolve** | OpenEvolve MAP-Elites, 4 islands, population 1000 | 100 iterations | OpenEvolve native | OpenEvolve native prompts | OpenEvolve only, 5 correctness trials |

Task set for the first comparison: the 8 curated AlgoTune tasks in
`benchmarking/tasks/algotune/tasks.json`, with the same `n` per task. The full
154-task manifests are available; scale the same commands with
`MANIFEST=... DATASET_ROOT=...`.

Commands:

```bash
# optimizer-agent, sequential
ROUNDS=5 sbatch benchmarking/agents/optimizer-agent/algotune/run_slurm.sh

# optimizer-agent, advanced
ROUNDS=5 SELECTION=map_elites MAP_ELITES_METHOD=dynamic NUM_MAP_ELITES=3 \
  USE_STRATEGY=1 USE_DEBUG=1 sbatch benchmarking/agents/optimizer-agent/algotune/run_slurm.sh

# OpenEvolve
sbatch benchmarking/agents/openevolve/algotune/run_slurm.sh
```

## KernelBench

| Setting | Search | Rounds / iterations | Samples per round | Feedback / strategy | Notes |
|---|---|---|---|---|---|
| **recursive** | `bestN_last` | 5 | 1 | kernel `rules_llm`, debug | `benchmarking/agents/recursive` |
| **recursive: advanced** | `map_elites`, dynamic, 3 cells | 5 | 1 | kernel `rules_llm`, strategy + debug | `agents/recursive`, AKD newsettings profile |
| **OpenEvolve** | OpenEvolve MAP-Elites, 4 islands, population 1000 | 50–100 iterations | OpenEvolve native | OpenEvolve native prompts | OpenEvolve only, robust evaluator |

The KernelBench recursive/advanced runs are the AKD `kernel_eng2` loop over the
100-problem dataset in `benchmarking/tasks/kernelbench/dataset`, reproduced by
`benchmarking/agents/recursive`. The OpenEvolve KernelBench baseline uses the same
KernelBench level-1 problem family with the robust evaluator. Start with a
subset and scale:

```bash
# smoke
PROBLEM_LIMIT=1 ITERATIONS=1 sbatch benchmarking/agents/openevolve/kernelbench/run_slurm.sh

# baseline
PROBLEM_LIMIT=10 ITERATIONS=50 sbatch benchmarking/agents/openevolve/kernelbench/run_slurm.sh
```

## Fairness notes

- Same model and vLLM serving settings across methods.
- Same task sizes / problem IDs where possible.
- Same hardened KernelBench evaluator policy (5 correctness trials, 3 timing
  trials, L2 flush, dtype-aware tolerances).
- OpenEvolve keeps its native search/prompt configuration; it is not reduced to
  a one-shot or repair-only loop.
- OpenEvolve AlgoTune uses its native 5-trial correctness check, which is
  stronger than the optimizer-agent 2-seed AlgoTune check. This favours
  OpenEvolve, not the other way around.

## Token accounting

- **optimizer-agent runs**: each `history` entry now carries
  `token_usage` (`prompt_tokens`, `completion_tokens`, `total_tokens`,
  `calls`); `summary.json` carries the run total; the AlgoTune/KernelBench
  driver CSVs expose the same four columns.
- **OpenEvolve runs**: each task writes `token_usage.jsonl` from the
  OpenAI-compatible response usage; the driver CSV exposes the summed
  `prompt_tokens`, `completion_tokens`, `total_tokens`, `llm_calls`.
- **robust_kernelbench**: already writes `in_tokens` and `out_tokens` in each
  `generations.csv`.
- **Reference-solver baseline**: no LLM calls, so token counts are zero.

Runs that completed before the tracking patch do not have exact token counts;
the vLLM logs only report aggregate throughput, not per-request usage.

### De-duplicating repeated attempts

`token_usage.jsonl` is appended by every attempt on a task, so an OpenEvolve
driver row holds the **cumulative** total for that task, not that attempt's
spend. Summing rows across shards therefore over-counts (run `1689375`
reported 87.0M this way; the true spend was 20.3M).

`benchmarking/aggregate_results.py` produces the canonical table. Its rules:

- **optimizer-agent** runs are one row per task, so totals are summed directly.
- **OpenEvolve** runs count one attempt per task: the first attempt with a
  measurable speedup, and tokens = cumulative total / number of attempts.
  The total actually spent across every attempt is printed separately.
- A task counts as valid when its measured speedup is finite and positive.

## Results

Generated with `python benchmarking/aggregate_results.py`. Rows are grouped by
task set and ordered from the simplest search (repair-only, one sample per
round) to the hardest (OpenEvolve's native MAP-Elites). `Tokens/task` is the
total spend divided by the number of tasks actually run.

### AlgoTune (154 tasks)

| Run | Tasks | Valid | Median | fast@1 | Tokens in | Tokens out | Tokens total | Tokens/task | Calls |
|---|---|---|---|---|---|---|---|---|---|
| optimizer-agent sequential (10 rounds, `bestN_last`) | 154/154 | 136 | 1.042 | 125 | 10,095,168 | 1,754,788 | 11,849,956 | 76,948 | 3,080 |
| optimizer-agent advanced (4 rounds, `map_elites` x3) | 154/154 | 141 | 1.037 | 127 | 12,719,352 | 2,280,951 | 15,000,303 | 97,405 | 4,498 |
| OpenEvolve (20 iterations, checkpoint/iter) | 154/154 | 129 | 1.035 | 115 | 19,525,944 | 5,099,172 | 24,625,117 | 159,903 | 2,823 |

### KernelBench level 1 (100 tasks)

| Run | Tasks | Valid | Median | fast@1 | Tokens in | Tokens out | Tokens total | Tokens/task | Calls |
|---|---|---|---|---|---|---|---|---|---|
| optimizer-agent sequential (5 rounds, `bestN_last`) | 100/100 | 57 | 1.000 | 15 | 4,209,864 | 1,974,518 | 6,184,382 | 61,844 | 1,670 |
| optimizer-agent advanced (5 rounds, `map_elites` x3) | 100/100 | 62 | 1.000 | 15 | 9,417,163 | 4,310,096 | 13,727,259 | 137,273 | 3,770 |
| OpenEvolve (50 iterations) | 100/100 | 100 | 1.000 | 7 | 17,868,685 | 6,349,661 | 24,218,346 | 242,183 | 4,999 |
| robust_kernelbench v11 (3-GPU TP=2) | 90/100 | 11 | 0.437 | 2 | 1,044,914 | 563,336 | 1,608,250 | 17,869 | — |
| robust_kernelbench v11 fair (2-GPU TP=1) | 91/100 | 0 | — | 0 | 1,031,377 | 572,217 | 1,603,594 | 17,622 | — |

Coverage of the OpenEvolve AlgoTune run (154 attempted):

```text
129  scored           correctness passed and timed
 15  best program incorrect (run completed, correctness_score = 0)
 10  evaluator stage-2 timeout (600 s per candidate, task killed at 3 h)
```

The 10 timeouts are `channel_capacity`, `chebyshev_center`, `lasso`, `lp_box`,
`lp_centering`, `lyapunov_stability`, `minimum_volume_ellipsoid`,
`ode_lorenz96_nonchaotic`, `ode_stiff_robertson`, `qp`. Their token counts are
tiny (5k-155k) because almost all of the 3 h went into evaluator timeouts, so
they are a harness limit rather than a search-budget result.

Notes:

- Run `1689375` was mis-sharded (all 8 shards started at task 1), producing 8
  independent attempts on the first 22 tasks; `1709405` ran the remaining 132
  once each with fixed sharding. The 22 duplicated tasks were run
  independently, not resumed (19 of 22 produced a different best program), so
  only one attempt each is counted above. All attempts together spent 42.2M
  tokens, against 24.6M for one attempt per task.
- Recursive KernelBench validity is low (57 and 62 of 100): the failures are 6 compile
  failures, 12 load failures and 25 correctness failures on the best candidate
  per task (sequential profile).
- Outliers above 10x (e.g. `communicability` 116x, `count_riemann_zeta_zeros`
  510x) should be treated with suspicion: they are consistent with solutions
  that exploit the in-process timing harness (caching across the 5 trials).
  The median is the headline number.
