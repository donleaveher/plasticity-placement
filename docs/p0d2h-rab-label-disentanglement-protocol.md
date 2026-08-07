# P0-D2H frozen RAB receipt/slot-label disentanglement protocol

## Purpose and boundary

The completed RAB counterbalancing audit separated joint receipt/slot-label, serial-position,
candidate-position, and interaction effects, but it intentionally kept receipt token A/B equal to
the selected Slot A/B label. This follow-up inference-only audit introduces an explicit routing
codebook so that receipt token and selected slot label can vary independently.

It accepts the exact verified completed RABC result with run id `p0d2hrabc-c4f96876bb`, together
with its transitively bound RAB source. It keeps the same 48 RAB units, adapter, model revision,
precision, tokenizer, and same-runtime OFF/ON scoring path. It cannot revise either historical
result, train a remediation adapter, or authorize the 1/4/8 mapping experiment.

## Frozen factorial

Every prompt contains one of two valid receipt-to-slot codebooks:

- canonical: `Receipt A selects Slot A; Receipt B selects Slot B`;
- crossed: `Receipt A selects Slot B; Receipt B selects Slot A`.

Within every frozen RAB unit, the audit crosses:

- slot-content orientation `canonical/swapped`;
- receipt token `A/B`;
- codebook `canonical/crossed`;
- slot display order `AB/BA`;
- four cyclic action-candidate rotations.

This produces `48 × 2 × 2 × 2 × 2 × 4 = 3072` prompts and 6144 paired OFF/ON decisions. The
selected slot label is derived only from receipt and codebook. Each of the 32
  `receipt × selected-slot-label × selected-display-position × exact-candidate-position` cells is
  balanced within every `pair_id` cluster.

The four source route variants already rotate the candidate panel. Crossing them with four new
rotations creates expected textual aliasing: 3072 condition IDs correspond to 768 distinct prompt
strings, each with multiplicity four. Uncertainty is clustered by the 12 semantic `pair_id` values,
not by aliased condition rows.

## Preflight invariants

Planning must pass before model loading. It verifies:

1. the completed RABC result, its causal-attribution flags, and its complete transitive RAB source;
2. 48 units, 3072 expanded conditions, and unique condition IDs;
3. the complete 64-cell factorial within every unit;
4. exact balance of receipt token, selected slot label, selected display position, and candidate
   position;
5. independent crossing of receipt token and selected slot label across all 32 joint primary cells;
6. semantic equivalence of canonical and crossed codebooks when they select the same slot;
7. display-only slot-line reordering and cyclic candidate-set preservation;
8. full-string candidate tokenization for every formatted prompt;
9. the expected `768 × multiplicity 4` prompt-text aliasing pattern.

## Scoring and analysis

Each prompt is scored OFF and then ON in one loaded-model process. The adapter is activated once;
`disable_adapter()` supplies the OFF score. The runtime preserves base-model object identity and
repeats an OFF sentinel after all pairs. Ties are never resolved by candidate order. Accuracy is
reported as lower/upper identified bounds with a midpoint used only for clustered contrasts.

Negative primary contrasts mean that the named level is disadvantaged:

- receipt-token effect: `Receipt B − Receipt A`;
- slot-label effect: `selected Slot B − selected Slot A`;
- serial-position effect: `selected slot second − selected slot first`;
- candidate-position effect: `expected at position 0 − expected at positions 1/2/3`.

For OFF and ON states, the report includes correctness and selected-minus-counterfactual margin.
It also reports adapter-excess contrasts, crossed strata, all pairwise interactions among the four
binary primary factors, their four three-way interactions, and the four-way interaction.
Confidence intervals use 10,000 pair-cluster bootstrap samples.

A persistent main-factor penalty requires both the midpoint confidence interval and the
conservative identified interval to be below zero. An interaction is supported when its clustered
midpoint interval excludes zero. Multiple supported mechanisms are reported as
`multiple_mechanisms`.

This design separates the literal receipt-token effect from the selected-slot-label effect under
the explicit codebook. It does not claim that the rewritten codebook prompt is identical to the
original RAB interface, and it does not turn a diagnostic attribution into training authorization.

## Lifecycle

```bash
plasticity-p0d2hrabx plan \
  --rabc-output /path/to/completed-rab-counterbalancing \
  --output /path/to/new-rab-label-disentanglement
```

Review the bank, token, source, and analysis-plan artifacts. Adopt an external human authorization
in a separate pass, then run and verify:

```bash
plasticity-p0d2hrabx authorize \
  --output /path/to/new-rab-label-disentanglement \
  --authorization /path/to/external-approval.json

plasticity-p0d2hrabx run --output /path/to/new-rab-label-disentanglement
plasticity-p0d2hrabx verify --output /path/to/new-rab-label-disentanglement
```

Primary outputs are `summary.json`, `report.md`, `paired_records.jsonl`, and immutable OFF/ON raw
rows under `results/`.

If a terminated Colab session leaves the manifest at `running` before any result artifact is
published, do not edit or delete the manifest. Follow the separately governed
[zero-artifact recovery protocol](p0d2h-rab-label-zero-artifact-recovery-protocol.md). Recovery
requires six hours of staleness, a separate inspection and approval pass, explicit confirmation
that the original runtime terminated, and reuse of the original experiment commit and run ID.
