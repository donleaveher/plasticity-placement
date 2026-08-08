# P0-D2H-CBR-v1 counterbalanced binding-remediation protocol

## Purpose and boundary

The completed RABX audit independently identified receipt-token, selected-slot-label,
serial-position, candidate-position-0, and composition-interaction penalties. CBR-v1 asks whether
a counterbalanced post-training bank can produce stable receipt→slot→action binding without
damaging the already measured retrieval and route→retrieve→action chain, and whether the same
trainable-parameter budget behaves differently when placed across all transformer layers or only
the late third.

CBR-v1 is a new experiment. It does not resume RR1 or CPR-v1 and does not modify RAB, RABC, or
RABX. The RABX result is a required read-only causal source. Planning does not authorize training;
all 12 training units require one external authorization bound to the exact preregistration.
Neither a successful unit nor the final matrix automatically authorizes 1/4/8 mappings.

`full_depth` means LoRA modules are inserted across all transformer layers. It is not
full-parameter fine-tuning. CBR-v1 expressly forbids updating the frozen base weights.

## Frozen 2 × 2 × 3 matrix

| Curriculum | Placement | LoRA rank/alpha | Seeds | Purpose |
|---|---|---:|---:|---|
| coupled | full_depth | 8/16 | 3 | coupled-label, full-depth control |
| disentangled | full_depth | 8/16 | 3 | data-balance effect at full depth |
| coupled | late_matched | 24/48 | 3 | coupled-label late placement control |
| disentangled | late_matched | 24/48 | 3 | primary counterbalanced late candidate |

The model is the exact frozen `Qwen/Qwen2.5-1.5B-Instruct` revision. All arms target only
`q_proj` and `v_proj`, use NF4, learning rate `2e-4`, batch size 4, gradient accumulation 4,
one natural epoch / 72 optimizer steps, zero dropout, and the fixed final checkpoint. Seeds are
`20260810`, `20260811`, and `20260812`. There is no hyperparameter search, early stopping,
checkpoint selection, retry seed, or failed-unit overwrite.

The late third uses three times the rank and alpha so its nominal LoRA budget matches full depth.
Aggregation additionally requires the actual trainable-parameter count to differ by no more than
1% for every curriculum/seed pair.

## Training banks

Each curriculum contains 48 train groups × 24 rows = 1,152 rows and 12 structurally held-out
groups × 24 rows = 288 rows. Every group has eight `route_only`, eight `retrieval_only`, and eight
`combined` rows. Selected Slot A/B, slot display AB/BA, completion exposure, and expected answer
position are exactly balanced in both curricula.

- `coupled`: Receipt A selects Slot A and Receipt B selects Slot B; only the diagonal
  receipt×selected-slot cells occur.
- `disentangled`: canonical/crossed codebooks independently cross receipt A/B and selected Slot
  A/B; all four joint cells occur equally.

The structural dev bank is never used for checkpoint selection. Generated group identifiers,
opaque records, and complete prompt texts are disjoint from RAB/RABC/RABX.

## Independent RAB-GEN-v1 panel

The primary panel is frozen before training:

```text
24 held-out semantic groups
× orientation canonical/swapped
× receipt A/B
× codebook canonical/crossed
× display AB/BA
× four cyclic candidate rotations
= 1,536 prompts per adapter
```

Every group contains the complete 64-cell factorial. Receipt token, selected Slot label, selected
display position, and exact expected-candidate position form 32 joint primary cells with 48 rows
each. Candidate continuations are full-string token-audited before authorization.

Every adapter is evaluated in its own single loaded-model process. For each prompt the scorer
runs adapter OFF immediately before adapter ON. The same process also scores:

1. all 192 historical RAB prompts;
2. all 96 frozen external conditional-route prompts;
3. all 1,536 CRD route-only/retrieval-only/combined prompts;
4. one OFF sentinel before and after the complete matrix.

Ties are never resolved by candidate order. Accuracy gates use pessimistic tie bounds.

## Unit gates

A training unit is a remediation candidate only when every check passes:

- RAB-GEN tie-worst adapter accuracy ≥ 0.75;
- historical RAB conservative selected-binding accuracy ≥ 0.75;
- the 95% clustered interval for each of Receipt B−A, selected Slot B−A, selected-slot
  second−first, and candidate-position-0−others lies wholly inside `[−0.05, +0.05]`;
- external conditional-route tie-worst adapter accuracy ≥ 0.75;
- retrieval-only tie-worst adapter accuracy = 1.0;
- combined ON−OFF lesson-cluster CI lower bound ≥ −0.02;
- combined pessimistic tie-bound delta ≥ −0.02;
- guardrail correct→wrong rate 95% CI upper bound ≤ 0.02;
- all scoring errors/ties satisfy the frozen compatible-tie contract and the OFF sentinel passes.

Confidence intervals use 10,000 semantic-group or lesson-cluster bootstrap samples. Prompts are
not treated as independent replicates.

## Matrix interpretation

All three fixed seeds must qualify before an arm is called qualified. The final aggregate may
report:

- `late_placement_reduces_drift`;
- `binding_remediation_supported_scope_tradeoff`;
- `full_depth_required_for_binding`;
- `late_binding_remediation_supported`;
- `balanced_data_advantage_not_supported`;
- `no_condition_qualified`.

`late_placement_reduces_drift` additionally requires seedwise late-minus-full held-out binding
non-inferiority within −0.02 and no larger correct→wrong rate for every seed. Other curriculum and
placement contrasts remain reported seedwise and are not converted into significance claims from
only three training seeds.

## Lifecycle

```bash
plasticity-p0d2hcbr plan \
  --output /new/cbr-training-output \
  --rabx-output /verified/rabx-output \
  --source-code-revision-lock /rabx-pipeline/code_revision.txt \
  --spec configs/p0d2hcbr-counterbalanced-binding-remediation-v1.json

plasticity-p0d2hcbr authorize \
  --output /new/cbr-training-output \
  --authorization /outside/approved-cbr.json

plasticity-p0d2hcbr train --output /new/cbr-training-output

plasticity-p0d2hcbr evaluate \
  --output /new/cbr-training-output \
  --analysis-root /independent/cbr-evaluations

plasticity-p0d2hcbr aggregate \
  --output /new/cbr-training-output \
  --evaluations-root /independent/cbr-evaluations

plasticity-p0d2hcbr verify --output /new/cbr-training-output
```

Training and evaluation are resumable only at immutable unit boundaries. A completed unit is
validated and reused. A failed or partially published unit is never deleted or overwritten; a
new preregistered experiment attempt is required.
