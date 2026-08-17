# Task Plan: G1-C R-OPCD Colab runner

## Goal
Package the completed G0-v2/R-OPCD planning implementation as a reproducible, safe-by-default Colab workflow, validate it, and publish the intended first-part changes to GitHub.

## Phases
- [x] Phase 1: Inspect source-of-truth plans, existing implementation, tests, and notebook conventions
- [x] Phase 2: Freeze the Colab execution contract and record design decisions
- [x] Phase 3: Implement the notebook builder, generated notebook, documentation, and tests
- [x] Phase 4: Run focused and full verification and inspect the complete diff
- [ ] Phase 5: Stage the intended scope, commit, push, and open a draft PR if needed

## Key Questions
1. Which existing CLI commands constitute the safe first-part Colab lifecycle?
2. Which files must be copied into the notebook as a frozen protocol snapshot?
3. How should Drive persistence and immutable bundle verification be exposed without enabling GPU/training actions?

## Decisions Made
- The notebook must retain the existing default prohibition on training, GPU, P0, kill/reserve, and path comparison.
- The generated notebook must follow the repository authorization-control convention even if the current phase exposes no state-changing control.
- Existing G0-v1 and G1/P0 artifacts remain immutable and outside the new output namespace.
- Expose only `inspect`, `prepare`, and `verify`; the default pass is non-writing inspection.
- Use a CPU Colab runtime and base `uv sync --frozen`; do not install training dependencies.
- Reuse the checked-in frozen G0-v1 snapshot instead of copying or modifying it.

## Errors Encountered
- Initial `gh auth status` reported an expired keyring token; the user re-authenticated successfully before implementation resumed.
- The first temporary plan write targeted generic tracked files; both were restored byte-for-byte from HEAD before any implementation change.
- No implementation errors in the first focused pass; all 10 G1-C and notebook tests and focused Ruff passed.

## Status
**Currently in Phase 5** - Staging only the G1-C first-part and Colab files, then committing and publishing the current branch.
