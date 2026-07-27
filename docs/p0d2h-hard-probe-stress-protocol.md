# P0-D2H Hard-Probe Stress Protocol

## 1. Status and question

- **Protocol status:** frozen design / implementation complete
- **Results status:** TBD
- **Source:** one complete verified P0-D2 budget-match run with `late-matched`
  in `eligible_locus_conditions`
- **Primary question:** Does the P0-D2 late-band advantage survive evaluation that
  requires stronger contextual binding, conflict rejection, conditional routing,
  and retrieval through longer distractor context?

P0-D2H is an evaluation-only diagnostic. It does not retrain adapters and therefore
does not by itself test training-complexity scaling. Its purpose is to decide whether
a separate multi-mapping training experiment is warranted.

## 2. Read-only source and independent outputs

The source P0-D2 manifest, compiled lessons, adapters, raw rows, aggregate, and
environment records are read-only. P0-D2H writes only beneath its own stage root:

```text
/content/drive/MyDrive/plasticity-p0d/hard-probe/v1/pipelines/
```

The P0-D2H run identity covers:

- source P0-D2 manifest SHA-256 and run ID;
- source P0-D2 aggregate SHA-256;
- selected lesson IDs and training seeds;
- the four source condition definitions and source adapter hashes;
- hard-probe compiler version and compiled-bank hashes;
- model name and immutable revision;
- inference length limits and current code hash.

Any source adapter, manifest, source aggregate, compiled artifact, probe bank, or
configuration change requires a new attempt.

## 3. Frozen conditions and scale

Only the equal-budget theory comparison is carried forward:

| Condition | Source placement | Source rank/alpha | Trainable parameters |
| --- | --- | ---: | ---: |
| `full-base` | all 24 layers | 4/8 | 270,336 |
| `early-matched` | layers 0–7 | 12/24 | 270,336 |
| `middle-matched` | layers 8–15 | 12/24 | 270,336 |
| `late-matched` | layers 16–23 | 12/24 | 270,336 |

`no_write` and `external` are evaluated once per lesson as calibration anchors.
The adapter matrix contains:

```text
4 conditions × 24 lessons × 3 seeds = 288 read-only adapter units
```

Each lesson receives 16 hard probes. The complete result matrix contains:

```text
24 × 2 base arms × 16 probes
+ 288 adapter units × 16 probes
= 5,376 rows
```

P0-D2H does not evaluate the three 90,112-parameter base bands because they do not
answer the parameter-matched shortcut question.

## 4. Frozen hard-probe suite

Every category has four deterministic variants. Action-choice order is balanced
within category. All probes retain the original lesson target, include the target
context and condition, and do not state the target action before the output-choice
line.

| Category | Difficulty mechanism | Failure interpretation |
| --- | --- | --- |
| `binding_decoys` | Select the active target record among three archived records carrying distractor actions | weak context–condition binding |
| `conflict_stack` | Reject three mutually inconsistent, explicitly unverified target-context suggestions | surface-level action steering |
| `conditional_route` | Apply a simple routing rule before retrieving the selected target mapping | failure under two-step composition |
| `long_context` | Retrieve the target after a longer audit trail containing all distractor actions | context dilution / recency shortcut |

The compiler audit rejects:

- duplicate IDs or missing categories/variants;
- any target action stated in the challenge body;
- missing target context or condition;
- fewer than two distinct distractor actions;
- exact equality with a training prompt;
- unbalanced action-choice positions;
- non-deterministic output or an unexpected probe count.

## 5. Metrics and estimands

Primary metrics:

1. strict hard-probe exact-action accuracy;
2. invalid-action rate;
3. category-specific hard accuracy;
4. per-seed and per-lesson stability;
5. hard-minus-original target-generalization degradation.

For condition \(c\) and lesson \(i\):

\[
D_{c,i}=A^{hard}_{c,i}-TG^{original}_{c,i}.
\]

The primary locus contrast is:

\[
\Delta^{hard}_{late,full}
=A^{hard}_{late\text{-}matched}-A^{hard}_{full\text{-}base}.
\]

The primary shortcut diagnostic is the difference in degradation:

\[
\Delta^{resilience}_{late,full}
=D_{late\text{-}matched}-D_{full\text{-}base}.
\]

Positive resilience means late-matched loses no more accuracy under stress than
full-base. Confidence intervals use lesson-clustered paired bootstrap after averaging
the three seeds within each lesson.

Secondary diagnostics report:

- late versus early/middle matched;
- lesson type;
- reserve versus original confirmatory lessons;
- desired action, including `fact_mapping × act_k2`;
- wins/ties/losses and leave-largest-paired-effect-out means;
- input tokens and latency.

## 6. Frozen decision rules

All gates require a complete provenance-valid run.

`hard_locus_robust` requires:

- late-minus-full hard-accuracy 95% CI lower bound above 0;
- late-minus-full resilience 95% CI lower bound at least `−0.05`;
- positive late-minus-full hard differences in at least 2/3 seeds;
- positive differences in both lesson types;
- positive leave-largest-effect-out mean.

`shortcut_hypothesis_supported` is emitted when either:

- late-minus-full hard-accuracy 95% CI upper bound is at most 0; or
- late-minus-full resilience 95% CI upper bound is below `−0.05`.

All other valid outcomes are `mixed_or_inconclusive`.

These are diagnostic labels, not universal mechanism claims. P0-D2H never
automatically selects layers, starts a narrow scan, or starts new training.

## 7. Entry to genuinely harder training

A separate multi-mapping training experiment may be frozen only after P0-D2H is
complete and reviewed. Before that experiment:

1. the hard suite must avoid a common floor and retain a working external anchor;
2. the number of mappings per adapter must be independently varied, such as
   1/4/8 mappings;
3. model, LoRA budget, update/token budget, action balance, and evaluation budget
   must remain comparable;
4. training and evaluation templates must have zero exact overlap;
5. the new experiment must use another package, manifest schema, Colab directory,
   and Drive namespace.

No P0-D2H result currently exists. All result fields remain TBD until a verified
aggregate is produced.
