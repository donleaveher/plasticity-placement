# P0-D2H route-state handoff audit protocol

## Status

The experiment is preregistration-ready but has not been run. Every result field remains TBD.

## Historical boundary

The source RTB result remains `scoring_integrity_failed`. This protocol defines a new,
inference-only audit and cannot reclassify RTB, retrain an adapter, change the historical
conditional-route gate, or authorize the 1/4/8 mappings-per-adapter experiment.

## Source qualification

Planning is CPU-only and must bind the complete RTB manifest, summary, raw checkpoint, OFF/ON
rows, paired records, report, and audit manifest. It verifies the reviewed RTB signature: 26
primary state-level ties, split 18 OFF and 8 ON, all expected-compatible with zero signed margin,
and positive conservative ON-OFF bounds in every slot cell. It also joins the 96 exact external
action rows to q2 by source probe ID and records score/decision replay drift without changing
either historical result.

Authorization is unavailable unless prompt hashes, expected candidates, candidate order, adapter
identity, RTB artifact hashes, q2 provenance, and all replay checks pass.

## Frozen bank

The audit uses the same 24 lessons and four route variants, for 96 matched units. Each unit has
four token-audited prompts:

1. `slot_readout`: exact RTB `slot_default_natural_external` prompt;
2. `direct_action`: exact RTB `external_action` prompt;
3. `receipt_A`: exact external payload with an authoritative receipt selecting slot A;
4. `receipt_B`: the same payload with the receipt selecting slot B.

The existing RTB `forced_slot_action` must equal the receipt corresponding to the target slot.
The other receipt changes only the receipt slot label. Both receipt prompts are scored before
analysis, so the predicted chain selects an already frozen and scored prompt rather than creating
a post-result prompt.

All four prompts are scored adapter OFF then ON in one loaded base-model object. This yields 768
raw decisions. The following endpoints are derived per state:

- `predicted_slot_chain`: choose receipt A/B using that state's slot-readout prediction;
- `oracle_slot_action`: choose the target-slot receipt;
- `wrong_slot_control`: choose the opposite receipt and measure target-action selection;
- `direct_action`: use the exact monolithic external prompt.

The predicted chain never substitutes the expected slot. For a top slot tie, every tied top slot
is propagated through its already scored receipt to construct conservative lower and upper
bounds. The separately reported observed path resolves a tie by the frozen candidate order; this
expected-independent convention is descriptive only and cannot replace the tie-robust bounds.

## Estimands and decision

The single primary estimand is the lesson-clustered difference-in-differences

```text
handoff rescue = (predicted-chain − direct) ON − (predicted-chain − direct) OFF.
```

The primary claim requires the conservative compatible-tie lower estimand to be at least `0.10`
and its two-sided 95% lesson-cluster bootstrap interval lower bound to exceed zero. The observed
deterministic chain is reported separately and cannot replace the conservative result.

Two safeguards are required for a `handoff_supported` decision:

- adapter-ON predicted chain is noninferior to oracle receipt within margin `0.05` using the
  conservative lower bound and its 95% clustered interval;
- adapter-ON oracle receipt reduces target-action selection relative to the wrong receipt by at
  least `0.20`, with clustered interval lower bound above zero.

Any non-finite score, missing prompt, incompatible provenance, source mutation, runtime identity
failure, or invalid tie bound yields `scoring_integrity_failed`. Otherwise the decision is either
`handoff_supported` or `handoff_not_supported`.

## Lifecycle

```bash
uv run plasticity-p0d2hrsh plan \
  --output /new/route-state-handoff-rsh1 \
  --rtb-output /frozen/route-transfer-rtb1

uv run plasticity-p0d2hrsh authorize \
  --output /new/route-state-handoff-rsh1 \
  --authorization /outside/authorization-rsh1.json

uv run plasticity-p0d2hrsh run --output /new/route-state-handoff-rsh1
```

Planning, authorization, and the single GPU audit are separate passes. There is no automatic
retry, training path, qualification-gate mutation, or 1/4/8 entry point.

## Result template

| Endpoint/contrast | OFF | ON | Estimate | 95% lesson-cluster CI | Status |
|---|---:|---:|---:|---:|---|
| Direct action | TBD | TBD | TBD | TBD | diagnostic |
| Predicted-slot chain | TBD | TBD | TBD | TBD | primary input |
| Oracle-slot action | TBD | TBD | TBD | TBD | safeguard |
| Wrong-slot target-action rate | TBD | TBD | TBD | TBD | negative control |
| Conservative handoff rescue | — | — | TBD | TBD | primary |
| ON chain minus oracle | — | — | TBD | TBD | noninferiority |
| ON oracle minus wrong receipt | — | — | TBD | TBD | specificity |

No experimental result has been generated here. All TBD values must come from the single locked
audit.
