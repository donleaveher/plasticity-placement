# P0-D2H frozen RAB counterbalancing audit protocol

## Purpose and boundary

This inference-only audit accepts the exact verified completed RAB result whose historical
decision is `binding_not_supported`. It keeps the same 48 units, adapter, model revision,
precision, tokenizer, and same-runtime OFF/ON scoring path, while independently crossing:

- `canonical/swapped × receipt A/B`;
- slot display order `AB/BA`;
- four cyclic rotations of the frozen four-action candidate panel.

The result is 1536 condition rows and 3072 raw decisions. The audit cannot revise the historical
RAB decision, train a remediation adapter, or authorize the 1/4/8 mapping experiment.

## Important textual-aliasing invariant

The four RAB `route_variant` values already cycle the action panel. Crossing each of those frozen
units with another four cyclic rotations produces 1536 condition IDs but only 384 distinct prompt
strings; every distinct string occurs exactly four times. This is an expected property of the
requested design, not hidden replication. Preflight verifies and reports it. Uncertainty is
clustered by the 12 frozen `pair_id` values, never by the 1536 condition rows.

## Preflight invariants

Planning must pass before model loading. It verifies:

1. the complete RAB result and every bound upstream artifact;
2. 48 units, 192 source probes, 1536 expanded conditions, and unique condition IDs;
3. the complete 32-cell factorial within every unit;
4. exact balance of receipt, selected display position, rotation, and expected candidate position;
5. all 16 `receipt × selected-display-position × expected-candidate-position` cells at 96 rows;
6. byte-identical recovery of every source RAB prompt at `display=AB, rotation=0`;
7. cyclic candidate-set preservation and display-only slot-line reordering;
8. full-string candidate tokenization for every formatted prompt;
9. the expected `384 × multiplicity 4` prompt-text aliasing pattern.

## Scoring

Each condition is scored OFF and then ON in the same loaded model process. The frozen adapter is
activated once; `disable_adapter()` supplies the OFF score. The runtime verifies base-model object
identity and repeats the first OFF sentinel after all 1536 pairs.

Tied candidate scores are not broken using candidate order because candidate order is itself under
audit. Accuracy is propagated as an identified interval: a tied expected action contributes lower
bound zero unless it is the unique winner and upper bound one when it belongs to the tied top set.

## Analysis

Negative contrasts mean the named condition is disadvantaged:

- receipt/slot label: `receipt B − receipt A`;
- serial position: `selected slot second − selected slot first`;
- candidate position: `expected at position 0 − expected at positions 1/2/3`.

The report includes OFF and ON contrasts, the adapter-excess contrast, selected-minus-
counterfactual log-probability margins, crossed-stratum estimates, all pairwise interactions, and
the receipt × display × candidate-position-0 third-order interaction. All confidence intervals are
12-`pair_id` cluster bootstraps with 10,000 samples.

A persistent main-factor attribution requires both its midpoint CI and conservative identified
interval to be below zero in the balanced factorial. Crossed-stratum estimates remain visible to
expose heterogeneity. A composition interaction is flagged when a pair-cluster interval for a
two- or three-way interaction excludes zero. Multiple supported mechanisms are reported as
`multiple_mechanisms`.

Receipt token `B` and selected slot label `B` remain linked in this design. Therefore a persistent
B effect identifies a joint receipt/slot-label binding obstacle, not receipt-token bias and
slot-label bias separately.

## Lifecycle

```bash
plasticity-p0d2hrabc plan \
  --rab-output /path/to/completed-rab \
  --output /path/to/new-rab-counterbalancing
```

Review `preflight/bank_audit.json`, `preflight/token_audit.json`,
`preflight/analysis_plan.json`, `preregistration.json`, and
`authorization.template.json`. Create the approval outside the output tree, changing only:

- `decision` to `approved`;
- `approved_by` to the responsible human identifier;
- `approved_at` to a timezone-aware timestamp.

Then run:

```bash
plasticity-p0d2hrabc authorize \
  --output /path/to/new-rab-counterbalancing \
  --authorization /path/to/external-approval.json

plasticity-p0d2hrabc run --output /path/to/new-rab-counterbalancing
plasticity-p0d2hrabc verify --output /path/to/new-rab-counterbalancing
```

Primary outputs are `summary.json`, `report.md`, `paired_records.jsonl`, and immutable OFF/ON raw
rows under `results/`.
