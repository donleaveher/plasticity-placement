# Task Plan: P0-D2H-CAL result backfill and next-session handoff

## Goal
Record the verified calibration and invalid-output audit results in the canonical
protocol documents, then prepare a self-contained prompt for implementing the next
format-stable calibration in a new Codex session.

## Phases
- [x] Phase 1: Verify repository state and locate canonical result documents
- [x] Phase 2: Reconcile supplied audit output with frozen calibration results
- [x] Phase 3: Backfill verified tables, interpretations, and claim boundaries
- [x] Phase 4: Write the next-session implementation prompt
- [x] Phase 5: Review cross-document consistency and validate the diff

## Key Questions
1. Which observed failures are output-format failures versus semantic selection
   failures?
2. Which claims remain blocked by the frozen strict gate?
3. What new endpoint can isolate action selection without introducing LoRA
   training or changing the old gate?
4. What independent package, notebook, manifest, and Drive boundaries must the
   next implementation preserve?

## Decisions Made
- Treat the supplied 2,304-row audit as verified post-hoc supplementary evidence.
- Preserve `mixed_scale_result`, `calibration_followup_required`, and an empty
  eligible-model list from the frozen P0-D2H-CAL gate.
- Interpret 1.5B `long_context` external failure as predominantly formatting,
  while retaining `conditional_route` as a semantic/compositional failure.
- Make the next stage a new base-only forced-choice candidate-scoring calibration.
- Do not include LoRA adapters, parameter-budget conditions, training, or an
  automatic narrow scan in the next stage.

## Errors Encountered
- None.

## Status
**Complete** - Verified results, claim boundaries, roadmap status, and the
next-session implementation prompt are documented; local links and diff checks
pass.
