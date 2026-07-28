# Task Plan: P0-D2H-CAL invalid-output audit

## Goal
Implement a read-only supplementary audit that distinguishes strict output-format
failure from semantic action-selection failure in existing P0-D2H-CAL raw rows,
without changing frozen gates or rerunning inference.

## Phases
- [x] Phase 1: Preserve current notebook-fix changes and define audit scope
- [x] Phase 2: Inspect raw-row/parser contracts and freeze taxonomy
- [x] Phase 3: Implement aggregate audit artifacts and tests
- [x] Phase 4: Add Colab review output and protocol documentation
- [x] Phase 5: Run focused/full validation and prepare handoff

## Key Questions
1. Which invalid classes are mutually exclusive and reconstructable from existing
   `generated_text` and frozen action tokens?
2. How should semantic-recovery accuracy be reported without replacing strict
   exact-action accuracy?
3. Which category/model/arm breakdowns are needed to distinguish formatting from
   mapping-use failure?
4. How can the Colab aggregate display the audit without modifying raw rows?

## Decisions Made
- Keep all frozen primary gates and `next_stage_status` unchanged.
- Treat the new analysis as explicitly post-hoc and supplementary.
- Read only verified raw rows already accepted by aggregate provenance validation.
- Write deterministic aggregate artifacts only; do not alter raw rows or manifest.
- Count semantic recovery only when the generated text contains exactly one
  distinct allowed action and it is the expected action.
- Use mutually exclusive invalid classes: empty; expected-only with extra text;
  wrong-only with extra text; multiple including expected; multiple excluding
  expected; no allowed action at generation limit; and no allowed action otherwise.
- Write the audit to an independent output directory and expose it through a
  CPU-only `audit-invalid` CLI action.

## Errors Encountered
- None.

## Status
**Complete** - Read-only invalid-output classification, independent artifacts,
CPU-only Colab, documentation, and repository-wide validation are complete.
