# Task Plan: G1 training audit and Recipe B

## Goal
Identify code-level causes of the failed G1 parametric qualification, implement a predeclared second qualification recipe and diagnostics without weakening frozen gates, and leave P0 blocked until the new recipe genuinely passes.

## Phases
- [x] Phase 1: Establish scope and preserve the failed Recipe A evidence
- [x] Phase 2: Audit training text, tokenization, optimizer, adapter lineage, and scoring parity
- [x] Phase 3: Reproduce suspected interface mismatches with CPU tokenizer-level tests
- [x] Phase 4: Implement the minimal correction, Recipe B configuration, diagnostics, and Colab controls
- [x] Phase 5: Run focused and full regression verification
- [x] Phase 6: Write the audit report and delivery instructions

## Key Questions
1. Does the completion text optimized during LoRA training exactly match the continuation scored at qualification time?
2. Does the current-only anchor use the intended parent/root adapter and fresh optimizer state?
3. Are candidate lengths, candidate priors, and per-action margins exposed well enough to distinguish underfitting from an interface bug?
4. Can Recipe B be run under a new immutable identity while preserving Recipe A and keeping P0 authorization strict?

## Decisions Made
- Treat the existing `pathmem-v1` G1 summary as immutable failed Recipe A evidence.
- Do not lower thresholds or create a manual P0 bypass.
- Prefer correcting verified train/eval contract mismatches before changing rank, learning rate, or optimizer steps.

## Errors Encountered
- The first Recipe B draft increased the event budget to 32 optimizer steps. Diff-level review found that this conflicts with the frozen G0 event blocks, which specify 16 steps. The draft was not run; Recipe B was corrected to a capacity-only change with 16 steps.
- One focused pytest command referenced a nonexistent `tests/test_pathmem_manifest.py`. The existing three PathMem freeze suites were located with `rg --files` and passed 18/18.

## Status
**Complete** - Recipe B is prospectively locked for the next run, Colab is regenerated, strict integrity checks are executable, and the full CPU regression suite passes.
