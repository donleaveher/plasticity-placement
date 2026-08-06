# Task Plan: RAB error localization reader

## Goal
Implement, verify, independently review, and publish a Colab-ready CPU-only reader that localizes
the completed RAB error-topology result without inference, training, or gate changes.

## Phases
- [x] Phase 1: Freeze scope, source identity, estimands, and interpretation boundary
- [x] Phase 2: Implement source validation, localization analysis, and immutable lifecycle
- [x] Phase 3: Add protocol, Colab, status updates, and regression tests
- [x] Phase 4: Run focused and repository-wide verification
- [ ] Phase 5: Independent review, revision lock, commits, and GitHub push

## Key Questions
1. Which factor levels account for the 21 `C→W` and 48 `W→W` cells?
2. Where did the 30 base `single_action_locked` units move after adapter activation?
3. Do adverse transitions coincide with negative or weakened selected-minus-counterfactual margins?
4. Are adverse cells concentrated by pair, lesson type, route variant, receipt, orientation,
   expected action, candidate position, or token count?

## Decisions Made
- Read only the exact completed `p0d2hrabd-a82d35757b` output and revalidate every artifact.
- Keep all slices descriptive; introduce no new statistical qualification gate.
- Use frozen, exhaustive factor levels and publish zero-count levels where defined by source data.
- Planning and localization execution are separate; no external approval is required.

## Errors Encountered
- None yet.

## Status
**Currently in Phase 5** - conducting independent review and locking the Colab revision.
