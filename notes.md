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
