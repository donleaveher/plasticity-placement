# Task Plan: RAB paired error-topology audit

## Goal
Implement, review, verify, and publish a Colab-ready CPU-only audit that freezes and reports the
error topology of the completed receipt/action binding experiment.

## Phases
- [x] Phase 1: Freeze source qualification, taxonomy, margins, transitions, and factor slices
- [x] Phase 2: Implement immutable plan/run/verify lifecycle and analysis package
- [x] Phase 3: Add protocol, Colab, project status, and regression tests
- [x] Phase 4: Run focused and full verification
- [ ] Phase 5: Independent review, revision lock, commits, and GitHub push

## Key Questions
1. Are ON errors dominated by action-prior locking, receipt invariance, binding invariance,
   systematic inversion, or mixed partial compliance?
2. Which OFF→ON transitions account for the net 12-row gain?
3. Do errors concentrate by pair, lesson type, route variant, receipt, orientation, action token,
   or candidate position?
4. Does the adapter increase selected-minus-counterfactual score margin even where top-1 remains
   wrong?

## Decisions Made
- This is a descriptive post-hoc analysis with frozen categories; it has no qualification gate.
- It accepts only the exact completed `binding_not_supported` RAB source and revalidates all
  published and preflight artifacts.
- It uses existing raw/paired rows only: no model load, inference, training, or 1/4/8 scan.
- Planning and diagnostic execution are separate; no external approval is required.

## Errors Encountered
- Initial Ruff pass found two long lines and one import-order issue; reformatted and fixed.
- Two new regression assertions initially mutated an already-top rank and rewrote an artifact with
  identical bytes; corrected the fixtures so both tests exercise real corruption.
- Independent review found three provenance/validation gaps: complete verification omitted
  preflight revalidation, result rows were not joined to the frozen binding bank, and token counts
  accepted malformed values. All three were fixed with regression tests before publication.

## Status
**Currently in Phase 5** - conducting independent review and locking the Colab revision.
