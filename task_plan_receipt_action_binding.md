# Task Plan: Receipt/action binding audit

## Goal
Implement, review, verify, and publish a Colab-ready audit that first diagnoses the completed RSH
paired outcomes and then tests causal receipt-to-action binding with two valid slot mappings.

## Phases
- [x] Phase 1: Freeze the claim, source qualification, factorial bank, estimands, and gates
- [x] Phase 2: Implement CPU RSH quadrant diagnostic and GPU receipt/action audit
- [x] Phase 3: Add protocol, Colab workflow, project status, and regression tests
- [x] Phase 4: Run focused and full verification
- [ ] Phase 5: Independent review, revision lock, commit, and GitHub push

## Key Questions
1. Does the RSH wrong-receipt failure reflect receipt neglect or an invalid decoy slot with no
   uniquely correct alternative action?
2. When both slots contain valid mappings, does changing only the receipt causally change the
   selected action?
3. Does the effect survive swapping which valid mapping is assigned to slot A versus slot B?

## Decisions Made
- The completed RSH result remains `handoff_not_supported`; this audit cannot reclassify it.
- The existing four-quadrant diagnostic is descriptive and CPU-only.
- The GPU audit is inference-only on the same frozen adapter, with one loaded runtime and OFF→ON
  scoring per prompt.
- Training and 1/4/8 remain unauthorized under every outcome.

## Errors Encountered
- After making CUDA conditional on `RUN_AUDIT`, the notebook lifecycle-order test matched the
  earlier setup guard instead of the final audit branch. Updated it to assert against the last
  `RUN_AUDIT` block; production code was unaffected.
- Independent review identified seven issues: RSH preflight provenance, per-unit factorial
  integrity, scorer-contract validation, revision pinning, conservative compliance subtraction,
  incomplete persistent gate reporting, and unconditional CUDA setup. Six implementation issues
  are fixed and regression-tested; revision pinning follows the first implementation commit.

## Status
**Currently in Phase 5** - independent review is running before revision lock and GitHub push.
