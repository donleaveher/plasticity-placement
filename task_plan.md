# Task Plan: P0-D2H-R evaluation repair

## Goal
Implement an isolated evaluation-only repair that detects truncation, separates
evaluation context length from source training length, validates the external
anchor and category floors, and preserves all existing P0-D2/P0-D2H artifacts.

## Phases
- [x] Phase 1: Establish scope and inspect repository state
- [x] Phase 2: Map the current configuration, runtime, aggregation, notebook, and tests
- [x] Phase 3: Implement the repair with focused tests
- [x] Phase 4: Regenerate the Colab notebook and run full verification
- [x] Phase 5: Review changes and prepare the handoff

## Key Questions
1. How can evaluation length change without weakening source-training provenance?
2. Which token-audit fields are required to distinguish truncation from task failure?
3. How should suite-quality checks remain distinct from the frozen locus decision?
4. How should a repaired attempt be isolated from `pipeline-a2`?

## Decisions Made
- Reuse the frozen read-only adapters; do not retrain.
- Preserve the original paired locus gate and add explicit suite-quality diagnostics
  instead of silently redefining the preregistered estimand.
- Require a new pipeline/attempt namespace for the repaired evaluation.
- Use a default evaluation context of 512 tokens while preserving the source
  training context as separate provenance.
- Render verified external memory immediately before the action-choice instruction
  so long-context distractors cannot push the anchor out of the retained suffix.
- Treat provenance validity, frozen locus status, and suite readiness as three
  separate outputs.

## Errors Encountered
- The first real-tokenizer smoke check used an incomplete offline Hugging Face
  cache without `tokenizer_config.json`, so the chat template was unavailable.
  Retrying after fetching the tokenizer completed successfully; no repository
  change was needed.

## Status
**Complete** - Implementation, notebook regeneration, real-tokenizer audit, full
tests, lint, and handoff documentation are finished.
