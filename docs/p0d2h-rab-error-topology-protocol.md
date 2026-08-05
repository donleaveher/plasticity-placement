# P0-D2H RAB paired error-topology protocol

## Scope

This CPU-only post-hoc audit accepts the exact completed RAB result whose valid decision is
`binding_not_supported`. It reads the 192 paired records and 384 immutable raw score rows, but
does not load a model, run inference, train, revise RAB, or authorize the 1/4/8 experiment.

The audit is descriptive. Its taxonomy and summaries are frozen before row-level execution, but
it introduces no new qualification threshold and cannot reclassify the historical RAB result.

## Source qualification

Planning verifies the RAB complete manifest, audit-manifest digest, every published artifact,
raw checkpoint counts and hashes, preregistration self-hash, and all RAB preflight hashes. It
requires:

- `binding_not_supported` with scoring integrity true;
- binding accuracy and orientation robustness false;
- causal specificity and adapter noninferiority true;
- historical RSH decision unchanged;
- training and 1/4/8 flags false.

The upstream completed RSH source is revalidated, and a full RAB tree hash is frozen before
diagnostic execution.

## Cell-level reconstruction

For each frozen prompt, the audit joins the paired record to exactly one OFF and one ON raw row.
It verifies all static fields, predictions, candidate order, and selected/counterfactual outcomes.
It then records for both states:

- selected correctness;
- selected and counterfactual sum log-probabilities;
- selected-minus-counterfactual margin;
- chosen action class: `selected`, `counterfactual`, or `other`;
- candidate position and token count;
- correctness transition: `C→C`, `C→W`, `W→C`, or `W→W`;
- margin-sign transition: negative, zero, or positive.

Overall OFF, ON, and ON−OFF mean margins use a 10,000-sample pair-cluster bootstrap. All factor
slices below are explicitly descriptive and do not receive multiplicity-adjusted claims.

## Frozen mutually exclusive unit taxonomy

Each of 48 units must contain exactly `canonical_A`, `canonical_B`, `swapped_A`, and `swapped_B`.
For each state, categories are assigned in this priority order:

1. `invalid_or_tied`: at least one joined row is tied; any malformed/invalid scorer row instead
   fails reconstruction before taxonomy assignment;
2. `other_action_intrusion`: at least one winner is neither valid pair action;
3. `fully_compliant`: all four selected bindings are correct;
4. `fully_inverted`: all four winners are the counterfactual action;
5. `single_action_locked`: the same valid action wins all four cells;
6. `receipt_invariant`: within both orientations, receipt A and B produce the same action;
7. `binding_invariant`: for both receipt labels, canonical and swapped bindings produce the same
   action;
8. `partial_mixed`: every remaining valid pattern.

Every unit also reports its four-character prediction signature in fixed `CA,CB,SA,SB` order,
correct-count distribution, OFF→ON taxonomy transition, receipt-switch compliance, binding-swap
compliance, and full-factorial compliance.

## Frozen descriptive slices

The audit reports row counts, OFF/ON accuracy, four correctness transitions, OFF/ON mean margin,
and mean margin change for every observed level of:

- `pair_id`;
- `lesson_type` recovered from the frozen RSH probe bank;
- `route_variant`;
- `orientation`;
- `receipt`;
- `expected_action`;
- `candidate_position`;
- `expected_token_count`.

It additionally reports expected-versus-predicted action counts for OFF and ON. No level is
selected or removed after seeing its result.

## Lifecycle

```bash
uv run plasticity-p0d2hrabd plan \
  --output /new/rab-error-topology-rabd1 \
  --rab-output /completed/receipt-action-binding-rab1

uv run plasticity-p0d2hrabd run --output /new/rab-error-topology-rabd1
uv run plasticity-p0d2hrabd verify --output /new/rab-error-topology-rabd1
```

Planning and execution are separate stages. No external approval is needed because execution is a
deterministic read-only transformation of locked artifacts. The manifest permits one diagnostic
run and records `training_authorized=false` and `mappings_per_adapter_authorized=false` under all
outcomes.

[Open the RAB error-topology diagnostic in Google Colab](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/p0d2h_rab_error_topology/p0d2h_rab_error_topology_colab.ipynb).

## Interpretation boundary

The audit can identify concentrated descriptive failure modes and nominate the next remediation.
It cannot establish a new confirmatory effect, retune historical gates, label a remediation as
successful, or authorize the storage-capacity experiment.
