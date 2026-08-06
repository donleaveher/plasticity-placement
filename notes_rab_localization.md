# Notes: RAB error localization reader

## Bound source result

- Source run: `p0d2hrabd-a82d35757b`.
- Cell transitions: `C→C=90`, `C→W=21`, `W→C=33`, `W→W=48`.
- Unit taxonomy OFF→ON:
  - `single_action_locked`: `30→10`;
  - `receipt_invariant`: `3→13`;
  - `partial_mixed`: `13→23`;
  - `fully_compliant`: `1→2`.
- Mean selected-minus-counterfactual margin change:
  `+0.814450 [0.690749, 0.925130]` by pair-cluster bootstrap.
- Historical RAB decision, training authorization, and 1/4/8 authorization remain unchanged.

## Analysis constraints

- Do not infer localization from aggregate counts alone.
- Validate source summary, cell/unit records, factor slices, preregistration, and audit hashes before
  reading rows.
- Treat factor and hotspot outputs as descriptive diagnostics, not confirmatory subgroup claims.
- Preserve all C/W transitions and all 48 units; do not filter after observing results.

## Initial verification

- Reader analysis/lifecycle tests: 12 passed.
- Reader plus generated-notebook tests: 15 passed.
- Focused Ruff and CLI smoke tests: passed.
- Repository-wide Ruff and pytest: passed.

## Independent review

- One P1 provenance finding was fixed by cross-checking manifest boundary flags, audit source hash,
  summary source metadata, and before/after snapshots against the preregistration identity.
- Factor×transition profiles were expanded to base/adapter/change mean, median, and complete
  positive/zero/negative counts; slices remain descriptive-only.
- Post-fix focused suite: 20 passed.
- Reviewer recheck: approved with no remaining findings.
- Colab exact revision: `36f9d4e957f6b8e4ca5da0f9d88656fde40b5aef`.
