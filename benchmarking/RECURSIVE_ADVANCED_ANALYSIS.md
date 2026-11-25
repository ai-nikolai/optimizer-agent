# Recursive and advanced modes: deeper analysis

Follow-up to `IMPROVEMENT_PLAN.md`, using the per-attempt artefacts
(`solutions/iteration_it_N_samp_M/scores.json` and `scores_debug.json`) rather
than the per-round `summary.json`, which merges main and debug outcomes.

## 0. What the two modes actually are

**Both are optimizer-agent.** They share the same `Agent` loop, evaluator
contract, feedback mechanism and token accounting; they differ in selection,
sample count and prompts.

| | recursive | advanced |
|---|---|---|
| Selection | `bestN_last` | `map_elites`, dynamic, 3 cells |
| Samples per round | 1 | 3 |
| Prompts | shared `prompts/kernelbench.json` | same + strategy stage |
| Debug | 1 iteration per failed round | same |
| Runner | `agents/recursive/run_slurm.sh` → `run_level.py --dataset-dir` | same + selection flags |
| Task set | 100-problem KernelBench-derived `tasks/kernelbench/dataset` | same |

**Relationship to `robust_kernelbench`.** It is the *evaluator* only
(`evaluate_single.eval_kernel_against_ref`, 5 correctness trials, 3 timing
trials, L2 flush). The agent loop is optimizer-agent's. The "recursive"
behaviour (round N repairs round N-1) reproduces what the AKD `kernel_eng2`
experiments did, which is the same policy family as robust_kernelbench's
test-time-scaling loop — but we do **not** run robust_kernelbench's own agent.
So: same policy shape, same evaluator, different implementation of the loop.

## 1. Where the 57 / 62 valid kernels come from

First time a task becomes valid, by source:

| First valid from | recursive | advanced |
|---|---|---|
| main candidate | **42** | 27 |
| debug after a **compile** failure | 12 | **30** |
| debug after a **mismatch** | 3 | 5 |
| debug after a **load** failure | **0** | **0** |
| never valid | 43 | 38 |

Two facts stand out: compile-debug is the productive repair path, and
load-failure debug has never converted a single task (0/93 attempts).

## 2. Debug economics by failure class

Classified from `scores.json` (main) and `scores_debug.json` (debug):

| Main failure | Debug attempts | Rescued | Rate | Tokens (est.) | Tokens per rescue |
|---|---|---|---|---|---|
| compile | 96 | 30 | **31%** | 1.65M | **55k** |
| load | 45 | **0** | **0%** | 0.77M | ∞ |
| mismatch | 169 | 6 | **3.5%** | 2.91M | **483k** |

Advanced:

| Main failure | Debug attempts | Rescued | Rate | Tokens (est.) | Tokens per rescue |
|---|---|---|---|---|---|
| compile | 250 | 56 | 22% | 6.06M | 108k |
| load | 48 | **0** | **0%** | 1.16M | ∞ |
| mismatch | 163 | 11 | 7% | 3.95M | 359k |

So the 81–86% of tokens that go to debug are spent mostly on the class with the
*worst* return. Compile-debug is worth its cost; mismatch-debug is ~9× more
expensive per rescue; load-debug is pure waste.

Caveat: these runs predate the `robust_kernelbench` fix, so mismatch messages
were the old `TypeError`. With the new `error_kind` + numeric diffs, the
mismatch rescue rate may improve. Re-measure before deleting mismatch debug.

## 3. Load failures are one mechanical bug

| Subtype | recursive | advanced |
|---|---|---|
| `attribute 'bias'/'weight' already exists` | **40** | **45** |
| `ModelNew` missing / init signature | 3 | 2 |
| other | 3 | 1 |

The dominant pattern is code that re-assigns a parameter/buffer that the module
(or a copy of the reference) already registers, e.g.

```python
self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
self.bias   = nn.Parameter(torch.randn(out_channels))
...
self.conv1d = nn.Conv1d(..., bias=bias)
```

It is deterministic, detectable from the error string, and has **never** been
fixed by the generic debug prompt in 93 attempts. It also blocks ~12–15 tasks
per run outright — those are tasks whose *final* error is this one.

## 4. Rounds do improve already-valid kernels

| | recursive | advanced |
|---|---|---|
| Tasks valid at iteration 0 | 29 | 17 |
| Of those, improved later | **19 (66%)** | **12 (71%)** |

So "stop as soon as it is valid" is wrong. The repair loop is doing two jobs —
validity conversion *and* optimisation — and the second job is productive.

## 5. Advanced's three samples: diverse, but rarely decisive

- Pairwise code similarity between the 3 samples at iteration 0: **median 0.43**
  (p25 0.31) — they are genuinely different attempts, not near-duplicates.
- Of the 27 tasks with at least one valid sample at iteration 0, the best sample
  was index 0 only **10 times** (1 → 8, 2 → 9).
- But **73 of 100 tasks had zero valid samples at iteration 0** (9 had one, 14
  had two, 4 had three).

Interpretation: the extra samples do help when the problem is tractable, but for
most problems all three candidates fail for the same reason (the prompt/format
problem), so tripling the sample count multiplies cost without multiplying
success.

## 6. Stuck tasks are easy to detect

For tasks whose main candidate never validated (48 recursive, 49 advanced):

| Distinct main-failure kinds across rounds | recursive | advanced |
|---|---|---|
| 1 (same failure every round) | 11 | 26 |
| 2 | 27 | 20 |
| 3 | 10 | 3 |

Rounds executed after the first repeated failure kind: **119 (recursive)**,
**122 (advanced)**. Those rounds changed nothing; they are the clearest
candidate for early stopping.

## 7. Proposals

Ranked by expected effect per unit of work.

### R1. Route debug by failure class (largest token saving)

- compile → keep the debug prompt (31% / 22% rescue).
- load → do **not** call the model; run a mechanical fixer (see R2) or stop.
- mismatch → replace the generic debug call with either (a) a second sample at
  a higher temperature, or (b) an optimisation prompt seeded with the numeric
  diff. Both are cheaper than 483k tokens per rescue.

Expected: cut KernelBench debug spend from ~86% to ~25% of the budget and lose
≤3–5 converted tasks, all of which are better targeted by R2/R3.

### R2. Mechanical fixer for load failures (largest validity gain)

Detect `attribute already exists` / `ModelNew not defined` and apply a
deterministic repair step: remove the duplicate parameter assignment, or wrap
the implementation as `ModelNew`. Then re-evaluate.

Expected: up to **+12–15 valid tasks per run** (12–15% of the suite) at ~5k
tokens each instead of ~17k for a failed debug call. This is the single biggest
available validity improvement in either mode.

### R3. Split validity from optimisation

- Phase 1 (validity): debug enabled, compile-only, stop when a valid kernel
  exists.
- Phase 2 (optimisation): no debug, one sample per round, prompt variant
  "improve speed without changing semantics", keep the best valid kernel.

Evidence: 66–71% of tasks that start valid improve further; debug never runs on
a valid main (1/1 and 0/0 observed), so phase 2 costs no debug tokens.

### R4. Advanced: taper the sample count

Use 3 samples for the first iteration (validity search) and 1 sample per
optimisation round.

Evidence: samples are diverse (0.43) and win 63% of the time when a valid sample
exists, but 73% of first iterations have no valid sample at all, and sample 0 is
the winner 37% of the time among valid tasks.

Expected: ~30–40% fewer advanced tokens for the same median, with validity
preserved by the first-iteration search.

### R5. Stop on repeated failure kinds

Stop a task after two consecutive rounds with the same main-failure kind and no
valid candidate.

Evidence: 119–122 rounds were executed after the first repeated kind; 11–26
tasks never change failure class at all.

### R6. Ablate the advanced profile instead of assuming it helps

The advanced mode changes three things at once (selection, sample count,
strategy prompt). Paired results are a wash (AlgoTune 62W/69L, KernelBench
21W/17L/9T) at 1.27–2.2× tokens. Run a 2×2 ablation on 20 problems:

```text
selection ∈ {bestN_last, map_elites×3}
strategy  ∈ {off, on}
fixed: 5 rounds, 1 sample (except the map_elites arm needs its cells), debug=1
```

Only adopt map_elites/strategy per component if it wins on that grid.

## 8. Expected effect summary

| Change | Validity | Median | Tokens |
|---|---|---|---|
| R2 mechanical load fix | **+12–15 tasks** | neutral | +small |
| R1 debug routing | −0–5 tasks | neutral | **−50–65%** |
| R3 phase split | neutral/+ | +? (valid tasks optimise longer) | − |
| R4 taper samples | neutral | neutral | −30–40% (advanced) |
| R5 stop on repeats | neutral | neutral | −10–20% |
| R6 ablation | — | decides whether advanced is worth its cost | — |

## 9. How to validate

Run on 20 KernelBench problems (P1–P20) with the same seeds and 5 rounds:

1. baseline: current recursive and advanced.
2. R2 only → measure how many of the load-blocked tasks become valid.
3. R1+R2 → debug tokens and rescued-task count.
4. R3 on the tasks that became valid → does phase 2 beat phase 1's score?

Accept when tokens per valid kernel fall and no validity is lost; then run the
full 100-problem suite before publishing.
