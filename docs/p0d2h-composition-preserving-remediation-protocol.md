# P0-D2H-CPR-v1 composition-preserving remediation protocol

## Status and rationale

RR1 learned a partial marker→slot behavior but did not qualify the full reading chain. In the
same loaded base-model object, adapter ON versus OFF changed external conditional-route from
`0.6354` to `0.6562`, route-only from `0.5052` to `0.6458`, retrieval-only remained `1.0000`,
and combined changed from `0.7904` to `0.7305` (`95%` lesson-cluster CI for ON−OFF
`[-0.1042, -0.0182]`). The OFF replay sentinel passed. Compatible ties do not change these
directions. RR1 is therefore closed as failed and 1/4/8 remains unauthorized.

### CPR-v1 execution status (2026-08-03)

The cpr2 adapter completed its single authorized training run. Its primary q1 qualification
scored all 96 external conditional-route pairs and all 1,536 CRD pairs, then failed before
publishing any result artifact because the classifier requested the obsolete
`tie_sensitivity.*.accuracy_interval` field. The paired-audit producer has always emitted
`all_ties_incorrect_accuracy` and `all_ties_compatible_accuracy`; the protocol already requires
the former pessimistic endpoint. This is an analysis-schema implementation failure, not an
experimental outcome, and no q1 decision exists.

One q2 infrastructure recovery is allowed only with an external authorization bound to the
immutable q1 claim, the unchanged cpr2 adapter, the corrected analysis-code hash, and a new
independent output. It repeats the same locked OFF→ON panel, cannot train or scan mappings, and
does not relax any gate. The dedicated recovery Colab is
[`notebooks/p0d2h_cpr_qualification_recovery/p0d2h_cpr_qualification_recovery_colab.ipynb`](../notebooks/p0d2h_cpr_qualification_recovery/p0d2h_cpr_qualification_recovery_colab.ipynb).
Until q2 completes and is reviewed, CPR remains unqualified and 1/4/8 remains unauthorized.

## Question and intervention

CPR-v1 asks whether one fresh LoRA can improve conditional routing without damaging retrieval
or route→retrieve→action composition. It starts from the exact frozen Qwen2.5-1.5B base and
uses the RR1 carrier budget (all q_proj/v_proj layers, rank 8, alpha 16). It never resumes RR1.

The fixed synthetic bank contains 48 train groups (1,152 rows) and 12 disjoint dev groups
(288 rows). Each group has eight rows for each task:

- `route_only`: apply marker→slot and return only the slot;
- `retrieval_only`: receive an explicit selected slot and copy its action;
- `combined`: apply marker→slot, retrieve the selected record, and copy its action.

Within every task and group, target slot, slot presentation order, and answer-panel order are
fully crossed. Frozen lesson strings are prohibited. The group-disjoint dev bank is compiled
and reserved for structural/leakage auditing only; CPR-v1 does not score it and it cannot select
a checkpoint. CPR-v1 uses SFT rehearsal only: adding candidate-logit KL would require choosing a
new loss weight, so it is deferred to a separately preregistered protocol rather than hidden in
this fixed pilot.

Training is one seed (`20260802`), one natural epoch (72 optimizer steps with effective batch
16), and one final checkpoint. No retries, seed sweep, hyperparameter scan, KL-weight scan, or
mapping scan are allowed.

## Locked same-runtime evidence contract

The qualification loads one base-model object, attaches the new adapter once, and scores OFF
then ON for every prompt. It reuses the source-provenance-checked 96 external
conditional-route decisions and all 1,536 CRD decisions.

| Claim | Frozen check |
|---|---|
| usable routing | conditional-route adapter tie-robust lower accuracy ≥ 0.75 |
| route improvement | route-only adapter lower accuracy > 0.625 and paired CI lower > 0 |
| retrieval preservation | retrieval-only adapter lower accuracy = 1.0 |
| composition preservation | combined paired CI lower ≥ −0.02 and tie-worst ON−OFF ≥ −0.02 |
| runtime integrity | one base object, OFF replay sentinel pass, only expected-compatible ties |

All uncertainty-bearing comparisons use a 10,000-sample lesson-cluster bootstrap. McNemar is
secondary. Ties are retained as intervals; a tie is admissible only when the expected candidate
is in the exact top tie. The gate uses the pessimistic end of every accuracy interval.

Passing all checks yields only `reading_qualification_candidate_review_required`. It does not
change the historical gate or authorize 1/4/8. A failed result closes CPR-v1; any new loss,
weight, seed, or curriculum requires a new protocol and authorization.

## Lifecycle

```bash
uv run plasticity-p0d2hcpr plan ...
uv run plasticity-p0d2hcpr authorize --output ... --authorization /outside/approved.json
uv run plasticity-p0d2hcpr train --output ...
uv run plasticity-p0d2hcpr qualify --output ... --analysis-output /new/independent/path
```

The one-time cpr2 q1 recovery additionally requires
`--recovery-authorization /outside/recovery-authorization.json`. This flag is not a general
retry mechanism: it is accepted only when the original claim exists, its output has no result
artifacts, no prior recovery claim exists, and every bound recovery field matches.

`plan` verifies the complete RR1 artifact graph and same-runtime failure source, compiles the
new bank, and writes a preregistration plus a pending authorization template. Authorization
must be created outside the experiment output and bind its exact preregistration hash. Training
and qualification each have a hard limit of one.

## Result template

Report the four OFF/ON accuracies, ON−OFF lesson-cluster intervals, tie intervals, transition
counts, sentinel maximum score delta, every gate check, and the final status. Do not describe
the adapter as qualified unless the independent review and later formal gate recheck also pass.
