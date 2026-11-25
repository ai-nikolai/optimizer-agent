# Where optimizer-agent loses

A qualitative read of the two full 154/100-task runs:
`experiments/algotune_full_1689335_*` (sequential, 10 rounds),
`experiments/algotune_full_1689334_*` (advanced, 4 rounds),
`kernel_eng2/experiment_v5_1_robust_tokens_optagent` (recursive, 5 rounds),
`kernel_eng2/experiment_v5_1_newsettings_tokens_optagent` (recursive advanced).

Definitions: a task is *valid* when its best candidate passes correctness and is
timed (`speedup > 0`). A round is *valid* when its candidate is accepted. All
numbers below are reproducible from the `summary.json` files.

## Headline: the budget goes where the agent cannot succeed

| Run | Tasks | Never valid | Tokens | Spent on never-valid |
|---|---|---|---|---|
| AlgoTune sequential | 154 | 18 (12%) | 11.85M | 2.00M (17%) |
| AlgoTune advanced | 154 | 13 (8%) | 15.00M | 1.99M (13%) |
| KernelBench sequential | 100 | 43 (43%) | 6.18M | 3.76M (61%) |
| KernelBench advanced | 100 | 38 (38%) | 13.73M | 6.76M (49%) |

On KernelBench the agent burns most of its budget on problems it never solves:
43% of tasks, 61% of tokens. AlgoTune is far healthier — 88–92% of tasks produce
a result, and the wasted share is 13–17%.

Effective cost per valid result:

```text
AlgoTune   sequential  11.85M / 136 =  87k tokens per valid task
AlgoTune   advanced    15.00M / 141 = 106k
KernelBench sequential  6.18M /  57 = 108k
KernelBench advanced   13.73M /  62 = 221k
```

## AlgoTune

### The first attempt usually fails; repair is doing the work

| | sequential | advanced |
|---|---|---|
| Valid on round 0 | 79 / 154 (51%) | 104 / 154 (68%) |
| Valid at the end | 136 / 154 (88%) | 141 / 154 (92%) |
| Rescued (round 0 invalid → valid) | 57 | 37 |
| Improved over round 0 | 63 | 90 |

Per-round validity rises then flattens:

```text
round      0     1     2     3     4     5     6     7     8     9
seq      79/154 95  104  102  106  110  110  105  111  110
```

From round 4 on, only ~71% of tasks have a valid candidate in any given round.
Rounds 5–9 add roughly nothing to validity; they only occasionally improve a
score. **The repair loop saturates after ~5 rounds** — consistent with the
10-round run buying +0.023 median (1.019 → 1.042) and +7 valid tasks over the
5-round run, nothing more.

Where the best candidate appears:

```text
round   0   1   2   3   4   5   6   7   8   9
seq    34  13   8   8  16   9  13  14  20  19
adv    75  35  23  21   -   -   -   -   -   -
```

22% of sequential tasks are already optimal at round 0 and never improve; 25%
peak at rounds 8–9, i.e. the long tail only pays off for a minority.

### Why candidates are rejected

| Reason | sequential | advanced |
|---|---|---|
| correctness mismatch | 323 | 154 |
| evaluation timeout (120 s) | 13 | 10 |
| truncated / no closing code fence → `SyntaxError` | 11 | — |
| broadcast/shape bugs | 10 | 8 |
| `AttributeError: 'list' object has no attribute 'shape'` | 9 | 5 |
| Numba typing errors | — | 5 |

Two distinct failure shapes, checked by re-running the saved candidates against
the reference on seed 1:

**Near misses** — algorithm is right, numerics are off by a hair:

```text
lp_centering     maxabs 2.3e-5   (task rtol 1e-5)   → fails by ~2x tolerance
pagerank         maxabs 2.0e-6   (rel 3.3e-3 on the score field) → fails tolerance
```

**Structurally wrong or infeasible output** — the agent optimises the objective
and stops enforcing the constraints:

```text
min_dominating_set   output length 7 vs reference 8   (returned a smaller, infeasible cover)
vertex_cover         output length 66 vs 69
tsp                  objective off by 25 (rel 0.96)
edge_expansion       5.89 vs 11.87  (factor 2, counting convention)
qp                   maxabs 123, objective finite-pattern differs
```

This is the clearest qualitative pattern in AlgoTune: on **combinatorial tasks
(cover / packing / routing / tsp)** the agent trades feasibility for speed. It
knows the objective is "smaller is better" and returns a smaller solution than
allowed.

### The evaluator gives no signal to repair with

`tasks/algotune/evaluate.py` reports only:

```text
"candidate output failed correctness check"
```

No max-abs error, no relative error, no tolerance, no failing field, no shape.
The agent's debug/repair prompt therefore sees exactly the same string whether
it missed by `2e-5` or returned a solution of the wrong size. Since repair is
what rescues 41 tasks, this is the single highest-leverage message to improve —
a two-line change that adds `max_abs`, `rel`, `field` and `rtol/atol` would let
the agent fix the near-miss class.

### Category effects

| Category | n | median | fast@1 |
|---|---|---|---|
| differential_equation | 11 | 2.23x | 11 |
| graph | 14 | 1.61x | 14 |
| computational_geometry | 4 | 1.27x | 4 |
| discrete_optimization | 13 | 1.27x | 10 |
| statistics | 9 | 1.15x | 8 |
| convex_optimization | 22 | 1.04x | 20 |
| signal_processing | 13 | 1.03x | 12 |
| **matrix_operations** | **29** | **1.017x** | **28** |
| numerical_methods | 5 | 1.013x | 5 |

`matrix_operations` is 19% of the suite and is essentially a numpy-call
wrapper; the agent cannot beat it, so 28/29 tasks land at ~1.0. That single
category holds the headline median down. The agent is not failing there — the
task simply has little headroom.

## KernelBench

### Validity funnel: half the problems never compile into something correct

| | sequential | advanced |
|---|---|---|
| Valid on round 0 | 36 / 100 | 35 / 100 |
| Valid at the end | 57 / 100 | 62 / 100 |
| Never valid | 43 | 38 |
| Best candidate at round 0 | 56 | 68 |

Per-round validity: `36 → 36 → 39 → 44 → 47` (sequential). It plateaus at ~50%.
The advanced profile spreads 3 samples per round but ends at 51/100 — MAP-Elites
adds validity, not speed: **68% of advanced tasks have their best kernel at
iteration 0**, so the exploration phase mostly fails to beat the first answer.

### Why candidates are rejected

| Reason | sequential | advanced |
|---|---|---|
| `TypeError: 'str' object does not support item assignment` | 125 | 126 |
| `RuntimeError: ModelNew was not defined` | 12 | 114 |
| `KeyError("attribute 'bias' already exists")` | 26 | 31 |
| CUDA illegal memory access | 34 | 23 |
| `no parseable result after EVAL_MARKER` | 24 | 9 |

Three of these are harness/prompt problems, not model-capability problems.

**1. A real bug in the evaluation path.** Every output mismatch raised
this internal error instead of returning a mismatch result:

```python
# robust_kernelbench/evaluate_single.py
exception_str, metadata = register_and_format_exception(...)   # returns (metadata, text)
metadata["correctness_issue"] = "Output mismatch"              # metadata is now a str → TypeError
```

The unpack order was swapped in the `not matched` branch (the runtime-error
branch above it is correct). The exception then escapes into the outer handler,
which reports the formatted `TypeError` as the candidate's error. Consequence:
125/126 candidates failed with a message that says nothing about *what* was
wrong, so the debug stage repaired blind. The one-line fix (swap the tuple) is reported in
`ROBUST_KERNELBENCH_FEEDBACK.md`; correctness verdicts are unaffected, only the
diagnostic is lost until the maintainer lands it.

**2. `ModelNew was not defined` — 114 times in the advanced profile.**
The candidates contain a full implementation but name it `Model` (the
reference's class) instead of `ModelNew`. They are not copies of the reference
(median similarity 0.278, only 12/114 near-identical), so this is a
**format-adherence** failure: the model rewrites the problem and forgets the
required class name. The advanced profile's strategy stage makes it 10x more
likely than sequential (114 vs 12), which is the main reason advanced pays 2.2x
tokens for the same median.

**3. `attribute 'bias' already exists`** (26/31): code that re-registers a
parameter/buffer on a module that already has it — a mechanical PyTorch mistake
that a stricter prompt or a load-time check would catch.

The remaining failures are genuine: illegal memory accesses (34/23) and kernels
that crash under the correctness runner.

### "1.0x" is not "did nothing"

Valid KernelBench candidates are genuinely rewritten code (median similarity to
the baseline 0.31 sequential / 0.24 advanced, zero identical files), but they do
not beat the PyTorch baseline. The agent writes plausible custom kernels that
are no faster — the modal outcome on this suite.

### Contrast with OpenEvolve

| | optimizer-agent | OpenEvolve |
|---|---|---|
| KernelBench valid | 57–62 / 100 | 100 / 100 |
| KernelBench fast@1 | 15 | 7 |
| KernelBench median | 1.000x | 1.000x |
| Tokens/task | 62k–137k | 242k |

OpenEvolve never breaks correctness but almost never speeds anything up (7/100);
optimizer-agent takes real risks — 40% of kernels are broken, but it produces
twice as many faster-than-baseline kernels and the large algorithmic wins. The
same split appears on AlgoTune, where optimizer-agent is both cheaper and more
valid than OpenEvolve (77k/97k vs 160k, 136/141 vs 129 valid).

## Cross-cutting failure modes

1. **Evaluators return verdicts, not evidence.** AlgoTune: one opaque string.
   KernelBench: a harness TypeError (now fixed to a mismatch string, but still
   only "Output mismatch"). The agent spends 5–10 rounds repairing against
   `False`.
2. **Format adherence is a first-class failure.** `ModelNew` missing (114),
   unterminated code fences (11 AlgoTune `SyntaxError`s), and truncation at
   `max_tokens=8192`. These are recoverable with a strict re-ask, yet they
   currently consume a whole round.
3. **Premature optimisation on combinatorial tasks.** The agent returns
   infeasible-but-smaller solutions for cover/packing/routing, losing 6+ tasks.
4. **No abandonment policy.** 61% of the KernelBench budget goes to tasks that
   never validate. A per-task token cap, or stopping after N identical failures,
   would free more than half of the KernelBench budget.
5. **Search depth saturates.** AlgoTune becomes valid-flat after ~5 rounds;
   KernelBench flattens after ~3. Extra rounds buy tail improvements only
   (rounds 8–9 peak for 25% of AlgoTune tasks), so round count should be
   allocated per task rather than uniformly.

## Fixes, in order of expected value

1. Return diagnostics from both evaluators: failing field, max-abs and relative
   error, tolerance, and the first mismatching index/shape.
2. Fix the `ModelNew` class-name failure with a format-repair retry (cheap, and
   worth ~114 candidates in the advanced profile alone).
3. Detect truncated generations (unterminated fence, no `ModelNew`) before
   evaluation and re-ask instead of scoring them.
4. Add a feasibility check to the AlgoTune prompt and feedback for the
   combinatorial categories (cover, packing, routing, tsp).
5. Cap per-task spend and abandon tasks after repeated identical failures to
   reclaim the KernelBench budget.
6. For KernelBench, prefer more rounds with fewer samples: the advanced
   profile's MAP-Elites cells mostly do not beat the first candidate.

## Caveats

- Validity is measured with the harnesses in this repo: AlgoTune uses the
  official `is_solution` with the configured seeds; KernelBench uses the
  hardened evaluator (5 correctness / 3 timing trials).
- The near-miss analysis re-ran the saved best candidate on one seed; a
  candidate could pass on another seed. The tolerance argument holds regardless.
- Speedups above 10x (`communicability` 116x, `count_riemann_zeta_zeros` 510x)
  are consistent with timing-harness exploitation and are excluded from the
  qualitative conclusions; the medians are the reliable signal.
