# Notes: P0-D2H-R evaluation repair

## Observed run

- `late-matched` beat `full-base` by 10.07 percentage points on hard accuracy.
- `binding_decoys` was at a common 25% floor for every parametric/no-write arm.
- `external` had 33.33% invalid outputs and 0% `long_context` accuracy.
- External prompts averaged about 30 more input tokens than other arms.

## Initial code evidence

- External memory is prepended to the probe.
- Tokenization currently uses truncation with the source run's `max_length`.
- The frozen locus gate consumes only the late-vs-full contrast and does not express
  suite-quality readiness.

## Implementation constraints

- Existing source manifests, adapters, aggregates, and hard-probe outputs are
  read-only.
- Configuration/probe changes require a new attempt namespace.
- Any new quality thresholds must be explicit in provenance and aggregate output.

## Design selected

- Configuration schema v2 records both `source_training_max_length` and
  `evaluation_max_length`.
- Default repaired evaluation length is 512 tokens.
- A deterministic prompt-token audit is written before inference and hashed into
  the manifest.
- Raw rows record untruncated length, truncation count, instruction preservation,
  prompt variant, and evaluation length.
- External memory is inserted after the challenge body and immediately before the
  strict output instruction.
- The frozen locus label remains intact. A separate suite-quality result checks:
  no truncation, a working external anchor, and absence of common parametric
  category floors.
- The repaired notebook uses `pipeline-r1/hard_probe-r1`.

## Verification evidence

- 105 repository tests pass.
- Ruff lint and `git diff --check` pass.
- Real `Qwen/Qwen2.5-0.5B-Instruct` tokenizer audit over 768 prompt variants:
  - maximum plain prompt: 259 tokens;
  - maximum repaired external prompt: 303 tokens;
  - prompts truncated at 512: 0;
  - output instructions missing at 512: 0.
- This confirms that the old 256-token evaluation limit could truncate some plain
  hard prompts as well as the longer external prompts.

---

# Notes: P0-D2H-CAL oracle calibration

## Fixed scope

- Base models only; no adapter discovery, activation, training, or rollback.
- Three arms: `no_write`, `external`, and `answer_copy_oracle`.
- Reuse the verified P0-D2H-R 24-lesson, 16-probe hard suite read-only.
- Support a primary 0.5B model and an optional same-family larger-model canary.
- Independent manifest schema, raw rows, aggregate schema, Colab directory, and
  Drive namespace.

## Frozen default gates

- Answer-copy oracle: overall accuracy at least 0.90, every category at least
  0.80, invalid rate at most 0.01.
- External: overall accuracy at least 0.75 and invalid rate at most 0.05.
- Every prompt must be untruncated and preserve the strict output instruction.

## Required decisions

- Oracle fails: output-copy/instruction interface bottleneck.
- Oracle passes but external fails: verified-memory use under distractors fails.
- 0.5B fails while larger model passes: model-scale canary supports a capacity
  bottleneck.
- Oracle and external pass: anchor calibration is eligible for a separately
  frozen training-complexity experiment.

## Implementation evidence

- New package and CLI: `plasticity_placement.p0d2hc` /
  `plasticity-p0d2hc`.
- Independent Colab and Drive root:
  `notebooks/p0d2h_calibration/` and
  `plasticity-p0d/hard-probe-calibration/v1/`.
- Source-only matrix: 1,152 rows; source plus canary: 2,304 rows.
- Atomic lesson-model outputs recover normally after Colab interruption.
- No adapter path, training command, narrow scan, or automatic next-stage action
  is exposed by the CLI or notebook.
- Focused tests: 10 passed. Full repository: 115 passed. Ruff and diff checks
  passed.

---

# Notes: P0-D2H-CAL invalid-output audit

## Observed calibration result

- Source 0.5B: oracle `0.7891`, invalid `0.0`; frozen status `oracle_failed`.
- Scale canary 1.5B: oracle `0.9948`, invalid `0.0`; oracle gate passes.
- Scale canary external: strict accuracy `0.6901`, invalid `0.2318`.
- Scale canary no-write: strict accuracy `0.1276`, invalid `0.2448`.
- The strict parser accepts only a generated string exactly equal to one frozen
  action token; invalid does not by itself imply that no correct action was
  mentioned.

## Audit constraints

- No new inference.
- No raw-row or manifest mutation.
- No retrospective gate or threshold change.
- Supplementary semantic recovery must remain visibly separate from strict
  exact-action accuracy.

## Frozen taxonomy

- `empty_output`
- `expected_action_with_extra_text`
- `wrong_action_with_extra_text`
- `multiple_actions_including_expected`
- `multiple_actions_excluding_expected`
- `no_allowed_action_at_generation_limit`
- `no_allowed_action_other`

Semantic recovery is conservative: exactly one distinct allowed action must occur,
and it must equal the expected action. Mentioning the expected action alongside a
distractor is not counted as semantically correct.

## Implemented artifacts

- CLI: `plasticity-p0d2hc audit-invalid`.
- Source module: `p0d2hc/invalid_audit.py`.
- Independent outputs: summary JSON, Markdown report, and classified invalid
  JSONL.
- Summaries: model × arm and model × arm × category, including lesson-clustered
  strict/semantic intervals and paired recovery-gain intervals.
- Independent CPU-only Colab:
  `notebooks/p0d2h_invalid_audit/p0d2h_invalid_output_audit_colab.ipynb`.
- Source manifest and raw tree are hashed before and after the Colab audit.
- Focused P0-D2H-CAL tests: 24 passed.
- Full repository validation: 129 passed; Ruff, notebook regeneration, and diff
  checks passed.
