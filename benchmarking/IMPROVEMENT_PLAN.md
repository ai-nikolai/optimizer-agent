# Improving optimizer-agent: evidence-based plan

Built from the four full runs (`ANALYSIS.md`) plus per-stage token accounting.
Every number below is reproducible from the `summary.json` files.

## What the data says in one table

| Metric | KernelBench seq | KernelBench adv | AlgoTune seq | AlgoTune adv |
|---|---|---|---|---|
| Tasks | 100 | 100 | 154 | 154 |
| Valid at the end | 57 | 62 | 136 | 141 |
| Valid on round 0 | 36% | 35% | 51% | 68%* |
| Rounds where the main candidate succeeded | 165/500 (33%) | 215/700 (31%) | — | 523/924 (57%) |
| Debug rescue rate (failed round → valid) | **37/335 (11%)** | **73/485 (15%)** | — | **173/401 (43%)** |
| Share of tokens spent in debug | **86%** | **81%** | 0% (off) | **61%** |
| Tokens per debug call vs main call | **3.0×** | 3.0× | — | 2.0× |
| Debug calls repeating an error already seen for that task | **48%** | **44%** | — | **50%** |
| Tokens spent on tasks that never validate | 61% | 49% | 17% | 13% |
| Format/parse failures | 7.2% | 18.0% | 0.7% | 0.2% |
| Best candidate found at round 0 | 56% | 68% | 22% | 49% |

`*` advanced round 0 has 3 samples per task, so 68% is not comparable to the
one-sample rows; the fair comparison is the paired test below.

## Finding 1 — generation quality, not search, is the bottleneck

Evidence:

- The first candidate is valid only **36% (KB)** / **51% (AlgoTune)** of the time.
- **56–68%** of KernelBench tasks reach their best score at round 0; extra rounds
  improve breadth (more tasks become valid) but not the median.
- Paired comparison, same task, sequential vs advanced (map_elites ×3 +
  strategy + more samples):
  - AlgoTune: advanced wins 62, **loses 69**, median ratio **0.999** — a coin
    flip at 1.27× the tokens.
  - KernelBench: advanced wins 21, loses 17, ties 9, median ratio 1.000 — flat
    at **2.2×** the tokens.
- Every main-candidate success avoids a debug cycle worth **86% of the KB
  budget**; the payoff for generation quality is therefore disproportionate.

**Proposal 1a — pre-evaluation format gate + cheap re-ask.**
Run parse + compile + contract checks (`class ModelNew` / `class Solver`) before
the expensive evaluator. On failure, re-ask *once* with only the parse error,
not the full debug prompt.

- Evidence: 7.2%/18% of KB candidates fail on format; a re-ask costs ~5k tokens
  versus ~15.9k for a debug call (3× more) that currently handles them.
- Evidence: only 1–2% of candidates reach the 8192-token cap, so this is a
  parsing/contract problem, not a length problem. Raising `max_tokens` would not
  help; the prompt contract and the gate would.

**Proposal 1b — safety-net first candidate for KernelBench.**
Round 0 should emit a *correct, unoptimized* `ModelNew` (e.g. a thin torch
implementation) before any CUDA rewrite; optimisation starts at round 1.

- Evidence: round-0 valid kernels have median speedup **0.63×** — the aggressive
  first rewrite is usually both broken and slow. Repair (rounds 1–4) recovers to
  1.0×; starting from a valid kernel would remove that recovery cost.
- Evidence: OpenEvolve, which stays close to the reference, is **100/100 valid**
  but only 7/100 faster; ours is 57–62/100 valid but 15/100 faster. A
  conservative first candidate captures both: 100% validity, then the aggressive
  loop supplies the speed.

**Proposal 1c — task-aware first prompt.** For combinatorial categories
(cover/packing/routing/tsp), prepend a feasibility checklist; for
`matrix_operations`, cap the effort (see Finding 5).

## Finding 2 — debug is the cost centre and mostly fails

Evidence:

- 81–86% of KernelBench tokens and 61% of AlgoTune-advanced tokens go to debug.
- Rescue rate: 11% (KB seq), 15% (KB adv), 43% (AlgoTune adv).
- Debug candidates cost 2–3× a main candidate (15.9k vs 5.2k tokens on KB),
  because the debug prompt repeats the full reference, artifact and result.
- **44–50% of debug calls repeat an error signature already seen for that task.**
- Debug-rescued bests are worse than main bests: AlgoTune advanced median
  **1.022** (debug) vs **1.064** (main). Debug buys validity, not speed.
- Still, debug produced 19/57 (KB seq) and 32/62 (KB adv) of the *valid* bests —
  it cannot simply be removed.

**Proposal 2a — de-duplicate debug by error signature.** Normalise the error
(`ModelNew not defined`, `bias exists`, `illegal memory access`, `mismatch`,
`timeout`, …), keep a per-task set, and skip a debug attempt when that signature
has already been attempted twice. Expected: ~25–30% fewer debug calls on KB
(half of 44–50% repeats are exact repeats).

**Proposal 2b — route mechanical failures to a format fix, not a debug call.**
`ModelNew` missing, unterminated fence, `SyntaxError` on line 1 and
`attribute 'bias' already exists` are mechanical. A short, targeted repair with
the file header and the error (no reference, no result) should fix them at a
fraction of the cost.

**Proposal 2c — shrink the debug context.** Send the failing candidate, the
error, and only the relevant reference signature (not the whole reference and
the full evaluation summary). A 2× reduction in debug tokens would cut KB spend
by roughly 40% at the current rescue rate.

**Proposal 2d — stop after a valid candidate.** A task that becomes valid should
not keep paying debug costs; spend remaining budget on optimisation rounds.

## Finding 3 — round budget is poorly allocated

Evidence (cumulative best if the run stopped at round *k*):

```text
AlgoTune seq   k=0 1.003 (79 valid)   k=2 1.013 (121)   k=4 1.017 (128)
               k=6 1.020 (132)        k=8 1.040 (134)   k=9 1.042 (136)
KernelBench    k=0 0.632 (36)         k=2 0.997 (48)    k=4 1.000 (57)
```

- AlgoTune rounds 5–9 cost **45%** of the budget and add **+0.024 median**
  (2.3%) and +4 valid tasks.
- KernelBench gains nothing after round 3 in median; only breadth (53→57 valid)
  and fast@1 (14→15).
- 22% of AlgoTune tasks are already optimal at round 0 and never improve.

**Proposal 3a — adaptive stopping.** Stop a task when it has a valid candidate
and no score improvement for 3 consecutive rounds; keep running only tasks that
are still improving. Expected: ~25–35% token saving on AlgoTune with a median
loss under 1%, and a larger KB saving.

**Proposal 3b — reallocate, don't just cut.** Send the saved rounds to tasks
that are still improving (AlgoTune rounds 8–9 peak for 25% of tasks). A simple
scheduler: after round 4, continue only tasks whose best improved in the last
two rounds, up to a global round budget.

**Proposal 3c — abandon never-valid tasks.** Cap spend per task (e.g. 2× the
median cost) and stop. Expected: reclaim ~60% of KernelBench tokens.

## Finding 4 — feedback carries almost no information

Evidence:

- AlgoTune's evaluator returns `"candidate output failed correctness check"` —
  no field, no magnitude, no tolerance. Replaying the failed candidates against
  the reference shows two classes: near misses (lp_centering 2.3e-5 against a
  1e-5 tolerance; pagerank rel 3.3e-3) and gross errors (edge_expansion 2×;
  infeasible covers returning 7 of 8 elements).
- KernelBench's mismatch path returned a harness `TypeError` for 125/126
  rejected candidates (reported upstream), and even after the fix the message is
  the constant `"Output mismatch"` until the report's issue 2 lands.
- Repair does work when it has signal: 57 AlgoTune tasks and 19–32 KB tasks were
  rescued by later rounds.

**Proposal 4a — structured failure payload.** Feed the agent a JSON block:
`{failure_stage, error_kind, metric, best_so_far, max_abs_diff, max_rel_diff,
atol, rtol, shape_expected, shape_actual, constraint_violated}`. For
combinatorial tasks, name the violated constraint instead of a generic mismatch.

**Proposal 4b — always show the delta.** Include `previous_best` and
`this_score` so the model knows whether it regressed; today the prompt shows the
result but the agent has to infer whether it beat the best.

**Proposal 4c — pass the evaluator's raw message through unchanged.** The
adapter should never replace a specific message with a generic one (the upstream
`metadata["correctness_issue"] = "Output mismatch"` line is exactly this bug).

## Finding 5 — no headroom / no feasibility detection

Evidence:

- `matrix_operations` is 29/154 tasks (19%) with median **1.017×** and 28/29
  faster than baseline; at ~77k tokens/task it is roughly a fifth of the budget
  for ~0 gain.
- `differential_equation` (2.23×) and `graph` (1.61×) have large headroom.
- 8 tasks fail in *both* AlgoTune runs; the sampled ones split into near-miss
  numerics and infeasible solutions, not random model error.

**Proposal 5a — per-category round budgets.** 1–2 rounds for
`matrix_operations`-like categories, more for graph/differential equations.
Expected: ~15% AlgoTune token saving with no median loss (those tasks already sit
at 1.0).

**Proposal 5b — feasibility check in the loop.** After generating a candidate
for a combinatorial task, run a cheap feasibility check (shape, coverage,
constraints) before the timing evaluation; on failure, feed "infeasible: 7 of 8
elements" back instead of a generic mismatch.

## Finding 6 — instrumentation gaps made this analysis slow

We inferred the stage from the filename (`core_response_debug.py`) and the
failure class from string matching. Add to every history entry:

```json
{"stage": "improve|debug|format_repair",
 "parse_ok": true, "error_kind": "mismatch|compile|load|oom|timeout|feasibility",
 "tokens_by_stage": {...}, "improved_over_best": false}
```

This makes the next analysis a query instead of a script.

## Update: per-failure-class analysis (see `RECURSIVE_ADVANCED_ANALYSIS.md`)

Reading the per-attempt scores (not the merged per-round history) changes the
priority order:

| Main failure | Debug rescue rate | Tokens per rescue | Verdict |
|---|---|---|---|
| compile | 31% (recursive) / 22% (advanced) | 55k / 108k | keep |
| load | **0%** (0/93 attempts) | ∞ | replace with a mechanical fixer |
| mismatch | 3.5% / 7% | 483k / 359k | stop debugging it |

Two changes therefore outrank everything in the table below:

1. **Mechanical load-error fixer** — 40/46 and 45/48 load failures are
   `attribute already exists`; fixing them is worth +12–15 valid tasks per run.
2. **Route debug by failure class** — compile-only debug cuts KernelBench debug
   spend by roughly two thirds with ≤3–5 lost conversions.

## Recommended order of work

| # | Change | Expected effect (estimate) | Evidence strength |
|---|---|---|---|
| 1 | Format gate + one-shot re-ask | +7–18% main success, −10–20% debug spend | strong (format failure rates, token ratios) |
| 2 | Structured feedback payload | converts near-miss failures; fewer wasted rounds | strong (near-miss sample, rescue counts) |
| 3 | Debug dedupe by error signature + context trim | −30–40% debug tokens | strong (44–50% repeats, 2–3× cost) |
| 4 | Adaptive stopping + abandon policy | −25–60% tokens, median loss <1% | strong (round curves, 61% never-valid spend) |
| 5 | Safety-net first candidate (KB) | validity toward 100% without losing speed headroom | medium (OE contrast, round-0 0.63×) |
| 6 | Per-category budgets + feasibility checks | −15% AlgoTune tokens, fewer infeasible rounds | medium (category table, 8 shared failures) |
| 7 | Selective map_elites (only after valid + headroom) | keep diversity where it pays | strong (paired 62W/69L) |

Already applied in the `clean_` batch and therefore tested by it: the shared
prompt contracts (`ModelNew`, complete file, feasibility, tolerance), the shared
feedback config, and debug enabled everywhere. The remaining items below need
code changes and are not yet in the batch.

Do not start with search-space changes: the paired test shows map_elites at 1.27×
(AlgoTune) / 2.2× (KernelBench) tokens is a wash on median. The leverage is in
the first candidate and the information the repair loop receives.

## How to validate (A/B protocol)

The `clean_*` batch is the baseline. For each change:

1. Run the same tasks with the same seeds and budgets on the same shard layout.
2. Compare: **valid@1**, **valid@end**, **median**, **fast@1**, **tokens per
   valid task**, **debug rescue rate**, **repeat-error rate**.
3. Accept a change only if it improves tokens per valid task without lowering
   the median; a median gain with a validity loss is a regression (see the
   KernelBench/OpenEvolve trade-off).

Minimum sample: the 8-task AlgoTune and 10-problem KernelBench sets for
iteration, then one full 154/100 run before publishing.

## Caveats

- History stores one result per round (main if it succeeded, otherwise the debug
  result), so "main validity" is only observable as round validity, not per
  candidate.
- The AlgoTune evaluator used 2 seeds and a 120 s solve timeout; the near-miss
  classification was verified on one seed per sampled task.
- Speedups above 10× remain suspect (timing-harness exploitation) and are
  excluded from the recommendations; they do not change any conclusion here.
