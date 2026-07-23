# P0-C Implementation Report

Status: complete.

## Implemented

- Compiler v3 counterbalances action-choice order, reverses action/position assignments across pairs, uses lexically disjoint near-neighbor contexts, and writes a leakage audit with frozen hashes.
- Formal tiers enforce calibrated provenance, exact counts, permitted splits, complete pairs, lesson-type balance, and action-token balance. Explicit lesson overrides are development/smoke only.
- Calibration stage 1 contains one fact and one procedure lesson and uses the preregistered write-time tie-break. Its dedicated runner evaluates only N/P/rollback and shares validated No-write results across candidates.
- Model revisions, compiler artifacts, code hashes, calibration-report hashes, dependencies, CUDA, and GPU identity are pinned or checked before resuming.
- Failed units are immutable. Interrupted training recovers only from atomically written metadata whose config, data, model revision, and adapter bundle hashes validate.
- Aggregation rejects partial or mixed runs and preserves seed/category measurements. It reports lesson-level arm CIs, median/IQR efficiency, equivalence checks, seed stability, pair diagnostics, and action-token diagnostics.
- Action parsing now accepts exactly one normalized action token. Latency measurement performs an unmeasured per-arm warm-up.
- NF4 uses BF16 only on capable GPUs and otherwise FP16. Adapter size includes only weights and adapter configuration; P0-C adapters do not duplicate tokenizer files.
- The Colab notebook pins the first resolved code commit in Drive and uses new compiler-v3 output directories.

## Verification

- 48 unit/regression tests passed.
- Ruff lint and scoped Ruff formatting passed.
- Python compileall, Notebook JSON validation, lockfile validation, CLI loading, and `git diff --check` passed.
- A real cached tiny Llama smoke completed N/E/P/B/rollback for two lessons:
  - 260 strict result rows;
  - all adapter units `verified`;
  - rollback exact-match rate 1.0;
  - strict aggregation completed;
  - no tokenizer files in adapter directories.
- A second invocation preserved adapter mtimes, confirming zero-work resume.
- A real one-lesson calibration-only loop completed N/P/rollback with 78 strict rows.

## Remaining Environment-Specific Check

The local machine has no CUDA runtime, so the NF4 BF16/FP16 branches are unit-tested but the final Colab GPU smoke must still be run before starting the 60-adapter calibration.
