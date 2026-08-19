# Task Plan: RAB receipt/slot-label disentanglement

## Goal

Implement and verify a separately frozen, inference-only audit that crosses receipt token and
selected slot label independently, while retaining display-order and candidate-order controls.

## Phases

- [x] Phase 1: Inspect the completed counterbalancing result, existing RAB/RABC packages,
  notebook lifecycle, and repository constraints.
- [x] Phase 2: Freeze the new factorial, source qualification, analysis, authorization, and
  interpretation boundary in a protocol.
- [x] Phase 3: Implement the probe bank, paired analysis, lifecycle runtime, and CLI.
- [x] Phase 4: Add a generated Colab, README, entry point, and project documentation.
- [x] Phase 5: Add focused tests for factorial identifiability, attribution, lifecycle safety,
  and notebook parity.
- [x] Phase 6: Run focused/full tests, Ruff, notebook regeneration, compile checks, and diff checks.
- [x] Phase 7: Review scope and hand off without running GPU inference or training.

## Key Questions

1. How can receipt token A/B be crossed independently from selected Slot A/B without changing the
   underlying two-record binding task?
2. How can the new audit retain the same adapter/runtime provenance and bind the completed RABC
   result that motivated it?
3. Which effects can be attributed separately, and which higher-order interactions must remain
   composition-level findings?

## Decisions Made

- Use an explicit receipt-to-slot codebook with canonical and crossed mappings. This makes receipt
  token and selected slot label independently balanced while preserving a valid routing task.
- Keep orientation, display order, and candidate rotation fully crossed within all 48 frozen RAB
  units: `2 × 2 × 2 × 2 × 4 = 64` prompts per unit, or 3,072 prompts / 6,144 OFF-ON decisions.
- Treat the completed RABC output as the direct qualifying source and its original RAB output as
  the transitive model/adapter source.
- Keep training, historical RAB reclassification, and 1/4/8 authorization disabled.

## Errors Encountered

- Initial focused Ruff check found one unused import and five overlong lines in the new package;
  corrected them without changing the frozen design.
- The first generated notebook parse test exposed three newline escapes interpreted by the builder
  string; escaped them at the builder layer and regenerated the notebook.
- The first optional real-tokenizer audit summary used obsolete token-count field names and raised
  `KeyError`; inspected the scoring schema and reran with `untruncated_prompt_token_count` and
  `full_input_token_count`. Production code was unaffected.

## Status

**Complete** - implementation and local verification are finished. Formal Colab planning,
authorization, and GPU inference have not been run; training and 1/4/8 remain unauthorized.
