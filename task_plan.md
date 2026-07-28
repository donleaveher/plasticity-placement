# Task Plan: P0-D2H-CAL-FC format-stable forced-choice calibration

## Goal
Implement and verify an independent, base-only forced-choice calibration that
scores the four frozen actions by candidate-only conditional log-likelihood,
without changing prior experiments or starting training.

## Phases
- [x] Phase 0: Commit the pre-task working tree as an explicit baseline
- [x] Phase 1: Read the canonical protocols, layout, notebooks, package, and tests
- [x] Phase 2: Freeze the package architecture, schemas, manifest identity, gates,
  provenance, and output contracts in the new protocol
- [x] Phase 3: Implement candidate scoring, source audit, atomic persistence,
  aggregation, gates, CLI, and report generation in `src/plasticity_placement/p0d2hfc/`
- [x] Phase 4: Add the independent notebook generator, generated Colab, README,
  project documentation, and CLI entry point
- [x] Phase 5: Add focused tests for scoring, provenance, matrix integrity, gates,
  persistence semantics, notebook structure, and absence of training paths
- [x] Phase 6: Run focused tests, full tests, Ruff, notebook regeneration,
  real-tokenizer CPU audit where available, and `git diff --check`
- [x] Phase 7: Review scientific boundaries, repository status, and handoff

## Key Questions
1. How does the existing calibration render prompts and identify immutable source
   rows, models, lessons, probes, and actions?
2. How can prompt/candidate concatenation be audited without accidentally scoring
   prompt tokens or silently accepting tokenizer boundary changes?
3. Which persisted identities are required for safe resume and immutable attempts?
4. How should overall and category gates distinguish eligibility from
   `category_calibration_failed` without triggering downstream work?
5. Can the generated Colab keep all preflight work CPU-only and make the first
   CUDA-loading cell explicit?

## Decisions Made
- Baseline commit: `db185e5` (`docs: prepare format-stable calibration handoff`).
- Preserve all P0-D2H-R, P0-D2H-CAL, and invalid-audit artifacts as read-only.
- Use `sum_logprob` as the frozen primary prediction score and `mean_logprob`
  only as a tokenizer/length-sensitivity diagnostic.
- Do not use subagents, adapter discovery/loading, training, narrow scans, RLVR,
  or automatic next-stage actions.
- Do not commit or push the implementation unless the user separately requests it.
- Use the complete verified P0-D2H-CAL run as the direct source and call its
  canonical validator before deriving the forced-choice matrix.
- Freeze the canonical continuation prefix to the empty string: the real Qwen
  chat template already ends with the assistant-turn newline, and both prompt
  prefix and standalone candidate token IDs remain exact without added whitespace.
- Refuse formal scoring unless an immutable full candidate-token audit already
  exists; formal scoring re-tokenizes and checks every frozen ID before inference.
- Store explicit non-finite/tie/error rows so aggregation can report and gate them,
  while infrastructure/model failures retain immutable failed-unit semantics.

## Errors Encountered
- Initial focused Ruff check reported three unused imports in the new runtime;
  removed them immediately. Package syntax compilation already passed.
- Raw non-finite floats would have serialized as non-standard JSON `NaN`/`Inf`;
  normalized them to JSON `null` with an explicit `score_finite=false` flag.
- First end-to-end fake-backend audit used a character tokenizer and correctly
  failed 1,920/2,304 decisions at the frozen 512-token full-sequence limit.
  Replaced the unrealistic fixture with deterministic word-level tokenization;
  no production threshold or fallback was changed.

## Status
**Complete** - The independent format-stable calibration implementation,
prospectively frozen protocol, generated Colab, tests, and validation evidence are
ready. No formal GPU result, downstream training, commit, or push was produced.
