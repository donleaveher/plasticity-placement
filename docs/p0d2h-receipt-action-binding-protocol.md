# P0-D2H receipt/action binding audit protocol

## Scope and boundary

The completed RSH result remains `handoff_not_supported`: its primary handoff rescue passed, but
its wrong-receipt specificity safeguard did not reach `0.20`. That panel's opposite slot was an
archived advisory decoy without a uniquely valid alternative mapping. This protocol therefore
defines a new inference-only audit in which both slots contain valid mappings. It cannot
reclassify RSH, train an adapter, or authorize the 1/4/8 mappings-per-adapter experiment.

## Source qualification and CPU diagnostic

Planning is CPU-only. It verifies every RSH artifact and its audit-manifest hash, requires the
frozen `handoff_not_supported` decision with supported primary rescue and failed receipt
specificity, binds the full RSH and upstream RTB snapshots, and checks that all historical
training and 1/4/8 flags remain false.

The planner then derives a descriptive four-quadrant table from all 96 RSH paired records,
separately for adapter OFF and ON:

1. oracle and wrong receipt both select the historical target;
2. only the oracle receipt selects the target;
3. only the wrong receipt selects the target;
4. neither receipt selects the target.

Rows affected by compatible action ties are reported separately. This diagnostic does not change
the historical RSH decision and is not a new inferential gate.

## Frozen counterfactual bank

The 24 RSH lessons form 12 frozen `a/b` pairs. Each pair shares a four-action panel and supplies
two different verified query-to-action mappings. For each pair and each of four route variants,
the bank creates one unit and crosses:

- binding orientation: `canonical` (`A→lesson a`, `B→lesson b`) versus `swapped`
  (`A→lesson b`, `B→lesson a`);
- signed receipt: `A` versus `B`.

Both verified mappings appear in every prompt in fixed `a,b` order. Candidate order is inherited
from the frozen lesson-a RSH probe for that route variant and is held fixed across the unit's four
prompts. Only the signed receipt changes within an orientation; only the A/B assignment of the two
valid queries changes across orientations.

This yields 48 units, 192 frozen prompts, and 384 raw decisions when every prompt is scored adapter
OFF then ON in one loaded base-model runtime.

## Endpoints and tie handling

For every prompt, the selected binding action is the action mapped to the receipt-selected slot;
the counterfactual action is the other valid slot's action. The audit reports:

- selected-binding forced-choice accuracy;
- counterfactual-action selection rate;
- selected-minus-counterfactual causal specificity gap;
- accuracy by binding orientation and receipt label;
- receipt-pair compliance: both A and B receipts correct within an orientation;
- binding-swap compliance: canonical and swapped prompts correct for one receipt;
- full 2×2 compliance: all four prompts correct for a unit;
- adapter ON−OFF differences with pair-cluster bootstrap intervals.

A top tie has no fabricated winner. Selected-action and counterfactual-action lower/upper bounds
are derived from the complete tied top set. All primary and safeguard decisions use conservative
bounds; frozen candidate order is used only for separately labelled descriptive accuracy.

## Preregistered decision

The audit returns `binding_supported` only when scoring integrity passes and all of the following
hold:

1. adapter-ON conservative selected-binding accuracy is at least `0.75`, with its 95% pair-cluster
   bootstrap lower bound strictly above `0.625`;
2. adapter-ON conservative selected-minus-counterfactual specificity is at least `0.20`, with its
   95% pair-cluster bootstrap lower bound strictly above zero;
3. conservative adapter-ON accuracy is at least `0.75` in both canonical and swapped orientations;
4. adapter ON is noninferior to OFF within `0.05`, using the conservative ON−OFF interval lower
   bound.

Any non-finite score, incomplete factorial, candidate mismatch, source mutation, runtime identity
failure, or unpropagated tie yields `scoring_integrity_failed`. Otherwise the result is
`binding_supported` or `binding_not_supported`.

The `0.75`, `0.625`, `0.20`, and `0.05` constants are frozen before inference. They reuse the
project's historical conditional-route qualification level, meaningful lower bar, specificity
materiality threshold, and noninferiority margin rather than being selected from this result.

## Lifecycle

```bash
uv run plasticity-p0d2hrab plan \
  --output /new/receipt-action-binding-rab1 \
  --rsh-output /completed/route-state-handoff-rsh1

uv run plasticity-p0d2hrab authorize \
  --output /new/receipt-action-binding-rab1 \
  --authorization /external/authorization-rab1.json

uv run plasticity-p0d2hrab run --output /new/receipt-action-binding-rab1
uv run plasticity-p0d2hrab verify --output /new/receipt-action-binding-rab1
```

The authorization permits exactly one frozen inference audit. It explicitly forbids training,
prompt revision after results, historical RSH reclassification, and the 1/4/8 scan.

## Result template

| Endpoint | OFF | ON | ON−OFF 95% pair-cluster CI | Gate |
|---|---:|---:|---:|---|
| Selected-binding accuracy | TBD | TBD | TBD | primary |
| Counterfactual-action rate | TBD | TBD | TBD | diagnostic |
| Selected−counterfactual specificity | TBD | TBD | TBD | primary |
| Canonical accuracy | TBD | TBD | TBD | safeguard |
| Swapped accuracy | TBD | TBD | TBD | safeguard |
| Receipt-pair compliance | TBD | TBD | TBD | diagnostic |
| Binding-swap compliance | TBD | TBD | TBD | diagnostic |
| Full 2×2 compliance | TBD | TBD | TBD | diagnostic |

No counterfactual result has been generated here. All `TBD` values must come from the locked run.
