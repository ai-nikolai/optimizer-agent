# Feedback for `robust_kernelbench` (evaluator)

Context: `benchmarking/tasks/kernelbench/evaluate.py` now imports
`robust_kernelbench.evaluate_single.eval_kernel_against_ref` directly (the
previous vendored copy has been removed), so every KernelBench score in this
repository comes from that function. The issues below were found while analysing
100-problem runs and are reported here for the maintainer.

Severity: **1** = wrong results / lost diagnostics, **2** = degraded agent
feedback, **3** = ergonomics.

---

## 1. Critical — a correctness mismatch raises `TypeError` instead of returning

**Location:** `robust_kernelbench/evaluate_single.py:435-439`

```python
matched, comparison_message = _compare_output_sequences(...)
if not matched:
    exception_str, metadata = register_and_format_exception(   # BUG: returns (metadata, text)
        "correctness_issue", comparison_message, metadata
    )
    metadata["correctness_issue"] = "Output mismatch"          # metadata is now a str
```

`register_and_format_exception` returns `(metadata, text)`, but this branch
unpacks it the other way round (the runtime-error branch at line 455 does it
correctly). `metadata` therefore holds the message string, and the very next
line raises `TypeError: 'str' object does not support item assignment`.

The exception is then caught by the surrounding `except Exception` at line 452,
which calls `register_and_format_exception("runtime_error", e, metadata)` with
the same string and raises again, so it escapes `run_and_check_correctness` and
is reported as a harness-level error.

**Observed impact:** 125 of 126 rejected candidates in a 100-task run carried
`main_error = "<class 'TypeError'>('str' object does not support item
assignment)"` instead of a mismatch message. The correctness verdict is still
`False`, so scores are unaffected, but the agent's repair loop is blind.

**Fix:** swap the unpack order in that branch.

```python
metadata, exception_str = register_and_format_exception(
    "correctness_issue", comparison_message, metadata
)
```

**Regression test:** a candidate that returns a wrong tensor must produce
`correctness=False, correctness_success=True` and a `main_error` containing the
comparison message — never a `TypeError`.

---

## 2. High — the mismatch diagnostic is overwritten immediately

**Location:** `evaluate_single.py:438`

The comparison message (which usually contains the max-abs difference and the
failing element) is stored under `metadata["correctness_issue"]` and then
replaced by the constant `"Output mismatch"` on the next line. Even after fixing
issue 1, callers only receive the useless constant.

**Suggestion:** keep the detailed message, e.g.
`metadata["correctness_issue"] = comparison_message` (and, if a short label is
needed, add `metadata["error_kind"] = "mismatch"`).

---

## 3. High — mismatch results contain no tolerances or per-trial context

For a near-miss the agent currently cannot tell whether it is off by `1e-7` or
by a factor of two. Please include, on the mismatch path:

```python
metadata["atol"], metadata["rtol"]      # resolved for the dtype (already computed at line 343)
metadata["max_abs_diff"]                # max |expected - actual|
metadata["max_rel_diff"]                # max relative difference
metadata["mismatch_index"]              # first offending index
metadata["trial"]                       # which correctness trial failed
metadata["seed"]                        # trial seed
```

`metadata["atol"]`/`["rtol"]` are already written in `_compare_output_sequences`
(line 343); the rest are cheap and make the result usable as feedback.

---

## 4. High — no timeout around candidate execution

`eval_kernel_against_ref` has no per-call timeout. A kernel that hangs (spin
loop, deadlock, `cudaDeviceSynchronize` never returning) hangs the whole
evaluation, and the caller has no way to distinguish "slow" from "stuck".

**Suggestion:** accept `timeout_s: float | None` and wrap the three risky calls
(load, correctness forward, timed runs) in a timeout that returns
`metadata["error_kind"] = "timeout"` instead of raising. Our adapter currently
has to add its own subprocess timeout on top.

---

## 5. Medium — error taxonomy is a single bucket

Every exception inside the correctness loop is registered as `runtime_error`
(line 455), so OOM, illegal memory access, wrong shapes and a Python `TypeError`
in the candidate all look the same to a caller. Please add a machine-readable
`error_kind` with at least:

```text
compile | load | oom | illegal_access | timeout | mismatch | runtime | nan
```

This also lets consumers decide whether to retry (environment OOM) or repair
(kernel bug). During our runs, OOM surfaced only as the generic message
`CUDA driver error: out of memory` inside `main_error`, which required string
matching to detect.

---

## 6. Medium — unstable metadata schema

`metadata` is a free-form dict, and different outcomes populate different keys
(`compilation_error`, `runtime_error`, `correctness_issue`, `atol`, ...). Our
adapter has to probe several keys to find an error, and there is no `error` key
on the mismatch path at all.

**Suggestion:** always set `metadata["error"]` (short, human-readable) and
`metadata["error_kind"]`, in addition to the specific fields.

---

## 7. Low — only the first failing trial is reported

The function returns at the first mismatch or error. A candidate that passes 4
of 5 trials returns exactly the same shape as one that fails all 5. Returning a
compact `trial_results` list (pass/fail per trial, plus the first failure detail)
would make flaky kernels visible.

---

## 8. Low — document the two-boolean contract

`KernelExecResult` documents that `correctness_success` means "the check ran",
not "the output is correct", but the names invite the opposite reading
(`correctness_success=True, correctness=False` on a mismatch). At minimum keep
the docstring prominent; ideally rename to `correctness_checked` or add a single
`correctness_status: pass | mismatch | error`.

---

## What we changed on our side

- Removed the vendored copy; `tasks/kernelbench/evaluate.py` imports this
  package via `ROBUST_KERNELBENCH_ROOT`.
- Every KernelBench agent (recursive, optimizer-agent, OpenEvolve) now calls the
  same CLI, so all three see identical verdicts.
- The launchers export `ROBUST_KERNELBENCH_ROOT` and add it to `PYTHONPATH`.

Once issues 1–3 land upstream, no change is needed here: the adapter already
passes `metadata`/`main_error` through to the agent's feedback.
