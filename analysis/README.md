# Analysis

Tools for inspecting optimizer-agent experiment output.

## Install

```bash
python -m pip install -r analysis/requirements.txt
```

## 1. Collect summaries

`collect_results.py` walks an experiments directory and flattens every
`summary.json` history entry into analysis CSVs.

```bash
python analysis/collect_results.py \
  --experiments-dir experiments \
  --version v5_0
```

It accepts both run layouts produced by the benchmarks and the older AKD
experiments:

```text
experiments/<run>/P12/summary.json
experiments/<run>/problem_pid_12/summary.json
experiments/<run>/level_1_pid_12/summary.json
```

Outputs, written next to the experiments directory by default:

```text
experiment_results_<version>.csv
experiment_results_summary_deb_<version>.csv
experiment_results_summary_<version>.csv
experiment_results_summary_agg_<version>.csv
```

## 2. Build candidate lineage

`collect_lineage.py` reconstructs parent → child candidate links from the saved
prompts and candidate artifacts:

```bash
python analysis/collect_lineage.py \
  --experiments-dir experiments/algotune_adv8_1794491181 \
  --version adv8
```

This writes `lineage_<version>.csv` next to the run. The viewer exposes it in
the **Lineage** tab, including parent/child scores and score deltas.

## 3. Open the interactive viewer

```bash
python analysis/interactive.py \
  --data-dir experiments \
  --version v5_0
```

Then open the URL printed by Dash (default `http://127.0.0.1:8050`).

Useful options:

```bash
--host 0.0.0.0 --port 8050 --debug
```

The viewer plots the summary CSVs by method, with filters for selection method
(`bestN_last`, `bestN_all`, `map_elites`), strategy, feedback, aggregation
(individual vs cumulative maximum), and color grouping.

## Legacy AKD CSVs

If you still have the old `experiment_results_summary_<version>.csv` files from
`automatic_knowledge_discovery`, point the viewer at them directly:

```bash
python analysis/interactive.py \
  --data-dir /path/to/kernel_eng2 \
  --version v4_2_0_0
```

## Notes

- The viewer was moved from
  `automatic_knowledge_discovery/kernel_eng2/4_interactive.py`.
- `collect_results.py` replaces the old `1_data_collection.py` +
  `2_analysis.py` CSV pipeline and works directly on optimizer-agent
  `summary.json` files.
- `combined_score` is evaluator-defined: the legacy kernel evaluator emits a
  negative runtime, while the hardened KernelBench evaluator emits a speedup.

## Token accounting

`collect_results.py` writes `prompt_tokens`, `completion_tokens`,
`total_tokens` and `llm_calls` into the raw and per-iteration summary CSVs, and
a per-method total to `experiment_results_tokens_<version>.csv`. These columns
come from `token_usage` entries in optimizer-agent `summary.json` history.
`interactive.py` exposes them as additional metrics.
