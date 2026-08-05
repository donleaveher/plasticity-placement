# Task Plan: Route-state handoff audit

## Goal
Implement, verify, review, and publish a Colab-ready inference-only experiment that tests whether
explicitly externalizing a predicted route state rescues action selection.

## Phases
- [x] Phase 1: Audit reusable RTB and same-runtime components
- [x] Phase 2: Freeze replay qualification, endpoints, estimands, gates, and lifecycle
- [x] Phase 3: Implement package, CLI, protocol, and dedicated-control-cell Colab
- [x] Phase 4: Add tests and run focused verification
- [x] Phase 5: Independent code review, revision lock, commit, and push

## Key Questions
1. Can the RTB route/action gap be localized to a same-prompt handoff failure rather than route
   computation or action retrieval?
2. Can the first-stage predicted slot be chained into a second-stage action without using ground
   truth?
3. How should expected-compatible slot ties be handled prospectively without revising RTB's
   historical zero-tie decision?

## Decisions Made
- RTB remains `scoring_integrity_failed`; this experiment cannot reclassify it.
- A CPU-only replay qualification must bind RTB and q2 artifacts before authorization.
- The GPU audit is inference-only on the immutable cpr2 adapter and uses one loaded base runtime.
- Training and 1/4/8 remain unauthorized under every result.

## Errors Encountered
- Initial Ruff pass found two unused imports and ten overlong lines in the new runtime module;
  corrected before the first focused test pass.
- Independent review found four release blockers: production ties use a null prediction, shared
  chain/oracle receipts require coupled bounds, selected prompts need row-level RTB anchors, and
  completion display must verify every published artifact. All four were fixed with regressions.

## Status
**Complete** - the implementation was independently reviewed, the Colab was locked to the exact
implementation revision, the full test suite passed, and the branch was prepared for GitHub push.
