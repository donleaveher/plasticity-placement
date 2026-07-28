# Task Plan: P0-D2H-CAL oracle calibration

## Goal
Implement an independent base-only calibration experiment that compares no-write,
verified-memory external, and answer-copy oracle prompts across one or more frozen
model revisions without training or loading adapters.

## Phases
- [x] Phase 1: Establish scope and preserve current worktree changes
- [x] Phase 2: Inspect reusable P0-D2H components and freeze schemas/gates
- [x] Phase 3: Implement package, CLI, manifest, runtime, aggregation, and tests
- [x] Phase 4: Build the independent Colab notebook and protocol
- [x] Phase 5: Run focused/full validation and prepare the handoff

## Key Questions
1. How should answer-copy oracle prompts remain deterministic and leak only in the
   explicitly labelled oracle arm?
2. How should multiple model revisions share the same immutable probe bank while
   retaining separate model-level results?
3. Which gates distinguish output-copy failure, verified-memory-use failure, and
   a model-scale bottleneck?
4. How can the notebook run base-only canaries without importing any LoRA path?

## Decisions Made
- Use a new `plasticity_placement.p0d2hc` package and `plasticity-p0d2hc` CLI.
- Read the verified P0-D2H-R manifest/probe bank as immutable source provenance.
- Write only to a new `hard-probe-calibration/v1` Drive namespace.
- Keep deterministic greedy generation and lesson-clustered paired bootstrap.
- Do not relax the existing external threshold retrospectively.
- Treat the verified P0-D2H-R manifest, summary, hard-probe hashes, and compiled
  source lessons as immutable inputs; calibration never reads adapter bundles.
- Use fixed model IDs `source_model` and `scale_canary`, with the source model
  always included and the canary optional.
- Evaluate each model once with greedy decoding across 24 lessons × 16 probes ×
  three arms, producing 1,152 rows per model.
- Emit model-level diagnoses (`oracle_failed`, `oracle_pass_external_failed`, or
  `calibrated`) and a separate cross-scale diagnosis.

## Errors Encountered
- Local shell `python` was unavailable; validation uses the repository-managed
  `uv run python` environment.

## Status
**Complete** - P0-D2H-CAL implementation, generated Colab, protocol, focused tests,
and full repository validation are complete. Formal GPU results remain to be run
in Colab.
