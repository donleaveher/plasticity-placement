# Notes: RAB receipt/slot-label disentanglement

## Motivation from the completed counterbalancing audit

- Source run: `p0d2hrabc-c4f96876bb`.
- Attribution: `multiple_mechanisms`.
- Receipt B minus A adapter accuracy: `-0.1745 [-0.2370, -0.1146]`.
- Selected slot second minus first: `-0.2839 [-0.3750, -0.2005]`.
- Candidate position 0 minus positions 1/2/3: `-0.5260 [-0.6372, -0.4132]`.
- Receipt and Slot A/B labels remained linked, so the receipt-related contrast could not distinguish
  receipt-token bias from selected-slot-label bias.

## Frozen follow-up concept

- Add a visible routing codebook:
  - canonical: `Receipt A -> Slot A; Receipt B -> Slot B`;
  - crossed: `Receipt A -> Slot B; Receipt B -> Slot A`.
- Cross codebook with receipt A/B, slot-content orientation, display AB/BA, and four candidate
  rotations.
- Primary separable effects: receipt token, selected slot label, selected display position, and
  candidate position 0.
- Preserve pair-cluster uncertainty and tie-identified accuracy bounds.
- This remains inference-only and cannot authorize remediation training.

## Implemented surface

- Package and CLI: `p0d2hrabx` / `plasticity-p0d2hrabx`.
- Lifecycle: `environment`, `plan`, `authorize`, `run`, `status`, and `verify`.
- Direct source is restricted to complete RABC run `p0d2hrabc-c4f96876bb`; validation follows its
  exact RAB source and frozen adapter provenance.
- The generated Colab uses one dedicated controls cell, safe planning defaults, an external
  authorization artifact, a locked code revision, and one same-runtime OFF→ON GPU pass.
- Published output includes immutable OFF/ON rows, paired records, summary/report, raw checkpoint,
  and audit manifest.

## Verification evidence

- Bank audit: 3,072 conditions, 768 distinct prompt texts at multiplicity four, all checks passed.
- Factor counts: receipt A/B, selected Slot A/B, and selected display position are 1,536/1,536;
  exact expected candidate positions are 768 each.
- Synthetic attribution tests separately recover a receipt-token-only failure and a
  selected-slot-label-only failure.
- Cached frozen Qwen tokenizer audit: 3,072/3,072 prompts passed; maximum formatted prompt/full
  lengths were 251/254 tokens under the frozen 512-token ceiling.
- Focused package/notebook tests: 9 passed.
- Full repository suite: 366 passed.
- Full Ruff, compileall, notebook regeneration/parity, CLI help, and `git diff --check`: passed.

## Execution boundary

- No formal GPU inference was run locally.
- No training or 1/4/8 experiment was authorized or started.
- `REQUESTED_CODE_REVISION` remains optional until this implementation is committed; the first
  Colab planning pass resolves and locks the exact branch revision in the new pipeline namespace.
