# Task Plan: RTB zero-artifact recovery

## Goal
Implement and publish a one-time, externally authorized recovery for a stale `running` RTB
attempt with no persisted inference or analysis artifacts.

## Phases
- [x] Phase 1: Inspect existing recovery and RTB state contracts
- [x] Phase 2: Freeze eligibility, authorization, and immutability rules
- [x] Phase 3: Implement CLI/runtime/notebook recovery and dedicated control cell
- [x] Phase 4: Add tests, review, and run full verification
- [x] Phase 5: Commit and push

## Key Questions
1. How can a zero-artifact infrastructure interruption be retried without enabling result-based
   selection or changing the frozen experiment?
2. How can recovery code run while the experiment itself remains locked to its original code?
3. How should the Colab expose human authorization without mixing controls and execution logic?

## Decisions Made
- No manifest editing or automatic retry.
- Recovery must require a separate immutable external authorization and recovery claim.
- Eligibility requires `state == running` and absence of every raw, paired, summary, report, and
  audit-manifest artifact.
- Recovery governance runs from separately locked recovery code; the retried audit still executes
  the original experiment commit and must pass its original `code_sha256` check.
- The recovered manifest keeps the same run ID, output directory, preregistration, bank, adapter,
  source snapshot, thresholds, and original authorization.

## Errors Encountered
- One large context-sensitive notebook patch did not apply after the source had changed; replaced
  it with smaller exact block patches. No generated file was affected by the failed patch.

## Status
**Complete** - the reviewed recovery implementation, immutable revision pin, Colab notebook, and
protocol are committed and pushed to `agent/add-lora-evaluation`.
