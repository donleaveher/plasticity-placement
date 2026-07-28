# P0-D2H-R Hard-Probe Stress Protocol

## 1. Status and question

- **Protocol status:** repaired design / implementation complete
- **Results status:** verified aggregate complete / suite-quality review complete
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
- inference length limits and current code hash;
- source-training and repaired evaluation length limits separately;
- prompt-token audit hash and external prompt-rendering version.

Any source adapter, manifest, source aggregate, compiled artifact, probe bank, or
configuration change requires a new attempt.

### Colab execution blocks

The notebook keeps setup costs and GPU work explicit:

1. checkout/install verifies `torch.cuda.is_available()` and records the GPU;
2. metadata preflight validates the frozen matrix without scanning adapter files;
3. a CPU tokenizer audit checks every plain/external prompt without truncation,
   verifies the strict output suffix, and freezes `prompt_token_audit.json`;
4. formal run performs exactly one complete 504-adapter source-integrity scan on
   CPU/Drive, with progress every 25 units;
5. one base model/tokenizer remains resident on CUDA while the 288 read-only
   adapters are activated and removed sequentially;
6. manifest verification and aggregate run in separate CPU/Drive blocks.

The repaired run preserves the source training limit (`256`) as provenance but
uses an independent default evaluation limit (`512`). Any prompt longer than the
evaluation limit fails before inference; silent truncation is not permitted.
Verified external memory is rendered immediately before the strict action-choice
suffix so long-context distractors cannot displace the answer instruction.

Source training precision and current evaluation precision are distinct provenance
fields. A valid adapter trained with `nf4-bfloat16` may therefore be evaluated with
`nf4-float16` on a different Colab GPU without being rejected, while all result rows
within one unit must still share the recorded evaluation precision.

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
| `binding_decoys` | Select the explicitly status-labelled active target among three archived-invalid records carrying distractor actions | weak context–condition binding |
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

The repaired aggregate also emits a separate suite-quality status. Next-stage
entry requires all of the following:

- zero truncated result rows and complete preservation of every output instruction;
- external hard accuracy at least `0.75`;
- external invalid rate at most `0.05`;
- external accuracy above chance plus `0.05` in every category;
- no hard category where every parameter-matched condition remains at or below
  chance plus `0.05`.

A provenance-valid run may retain its frozen locus label while still receiving
`suite_repair_required`; in that case no later-stage experiment is eligible.

## 7. Verified P0-D2H-R results

### 7.1 Run identity and completeness

The repaired aggregate was produced under:

```text
/content/drive/MyDrive/plasticity-p0d/hard-probe/v1/pipelines/
  pipeline-r1/runs/code-00236805a5_source-2482ac01a0/
  hard_probe-r1/
```

It contains the complete frozen matrix:

- 24 lessons and three training seeds;
- 288 read-only adapter units;
- 5,376 hard-probe result rows;
- four parameter-matched LoRA conditions plus `no_write` and `external`;
- zero truncated rows;
- output-instruction preservation rate of `1.0` in every condition;
- invalid-action rate of `0.0` in every condition.

The prompt repair therefore removed the evaluation-integrity failure observed in
the first diagnostic attempt. External invalid rate fell from `0.3333` to `0.0`,
and external `long_context` accuracy increased from `0.0` to `0.3229`. Because the
hard-probe compiler, binding wording, evaluation length, and external rendering
changed, the two attempts are diagnostic comparisons and must not be pooled.

### 7.2 Main condition results

Confidence intervals are 95% lesson-clustered bootstrap intervals.

| Condition | Hard accuracy [95% CI] | Hard − original TG [95% CI] | Invalid | Truncated |
| --- | ---: | ---: | ---: | ---: |
| `late-matched` | 0.4905 [0.4227, 0.5616] | −0.4488 [−0.5191, −0.3802] | 0.0000 | 0.0000 |
| `full-base` | 0.3828 [0.3290, 0.4410] | −0.4835 [−0.5694, −0.3872] | 0.0000 | 0.0000 |
| `middle-matched` | 0.3003 [0.2526, 0.3498] | −0.5174 [−0.6076, −0.4149] | 0.0000 | 0.0000 |
| `early-matched` | 0.2865 [0.2483, 0.3273] | −0.3698 [−0.4601, −0.2778] | 0.0000 | 0.0000 |
| `external` | 0.4297 [0.3698, 0.4974] | −0.4714 [−0.5312, −0.4089] | 0.0000 | 0.0000 |
| `no_write` | 0.2526 [0.2161, 0.2917] | 0.0078 [−0.0365, 0.0547] | 0.0000 | 0.0000 |

The primary paired late-minus-full contrast was `+0.1076`, with 95% CI
`[0.0781, 0.1380]`. Late-matched won on 20 lessons, tied on four, and lost on
none. The mean remained `+0.1014` after removing the largest paired lesson
effect. All three seed-specific differences were positive (`+0.0990`, `+0.1224`,
and `+0.1016`), as were both lesson-type differences (`fact_mapping: +0.0990`;
`procedure_recovery: +0.1163`).

The late-minus-full resilience difference was `+0.0347`, with 95% CI
`[−0.0347, 0.0938]`. It satisfies the frozen `−0.05` non-inferiority margin, but
the interval includes zero; this supports non-inferiority, not a claim that late
placement is strictly more resilient.

### 7.3 Category results

| Category | Late | Full | Late − full [95% CI] | External | No-write |
| --- | ---: | ---: | ---: | ---: | ---: |
| `binding_decoys` | 0.2222 | 0.1979 | +0.0243 [0.0000, 0.0521] | 0.5208 | 0.1354 |
| `conditional_route` | 0.5104 | 0.4028 | +0.1076 [0.0521, 0.1736] | 0.3958 | 0.2812 |
| `conflict_stack` | 0.7153 | 0.5729 | +0.1424 [0.0799, 0.2083] | 0.4792 | 0.3333 |
| `long_context` | 0.5139 | 0.3576 | +0.1562 [0.0868, 0.2326] | 0.3229 | 0.2604 |

The late-placement advantage is therefore supported on conditional routing,
conflict rejection, and long-context retrieval. `binding_decoys` is not a usable
confirmatory locus category: every parameter-matched condition remained below
the frozen common-floor limit of `0.30`, and the late-minus-full interval touched
zero.

### 7.4 Frozen locus decision versus suite-quality decision

The paired locus safeguards all passed:

- late-minus-full hard-accuracy CI lower bound above zero;
- resilience CI lower bound above the frozen non-inferiority margin;
- positive effects in all three seeds;
- positive effects in both lesson types;
- positive leave-largest-effect-out mean.

The frozen diagnostic label is therefore:

```text
decision_status = hard_locus_robust
```

This label is bounded to the verified 0.5B model, fixed cohort, fixed LoRA budget,
and three non-floor hard categories. It is not a universal late-layer mechanism
claim and does not imply absolute hard-task mastery: late-matched still lost
`0.4488` accuracy relative to its original target-generalization score.

The separate suite-quality gate did not pass:

```text
suite_quality_status = suite_repair_required
next_stage_status = suite_repair_required
```

The prompt-integrity checks passed, but two quality checks failed:

1. external accuracy was `0.4297`, below the frozen `0.75` threshold, although it
   exceeded no-write by `0.1771` and produced no invalid outputs;
2. `binding_decoys` remained a common parametric floor.

The repaired result rules out silent truncation as the explanation for the
late-minus-full advantage. At the same time, the weak external anchor shows that
the 0.5B model often fails to use an explicitly supplied correct mapping under
hard distractor context. This is consistent with an instruction-following,
conflict-suppression, or model-capacity bottleneck, but it does not by itself
identify which mechanism is responsible.

### 7.5 Supported and unsupported claims

The verified result supports:

- at equal LoRA parameter budget, late placement outperforms full, early, and
  middle placement on the fixed P0-D2 cohort;
- the late-minus-full advantage survives repaired, untruncated conditional-route,
  conflict-stack, and long-context probes;
- the relative placement effect cannot be explained solely by the original
  `max_length=256` evaluation bug;
- late placement is non-inferior to full placement in hard-minus-original
  degradation under the frozen margin.

The verified result does not support:

- absolute hard-task robustness or mastery;
- a late-placement advantage on a valid non-floor binding-decoy test;
- a working near-ceiling external anchor;
- generalization to larger or deeper models;
- the claim that small model size or shallow depth is the proven causal
  explanation;
- entry into a narrow layer scan or multi-mapping training experiment under the
  frozen suite-quality rule.

### 7.6 Completed calibration follow-up and next action

The current `pipeline-r1/hard_probe-r1` result should be frozen and archived rather
than overwritten or repeatedly tuned. The next experiment should be a separate,
base-only oracle calibration with no adapter loading:

1. `no_write`;
2. the current verified-memory `external`;
3. an answer-copy oracle that places the correct action immediately before
   `Action:`.

This can distinguish failure to copy/follow the output instruction from failure to
use verified memory under distractors. A same-family larger-model base canary may
then test whether the weak anchor is specific to the 0.5B model. Thresholds and
decision rules must be frozen before that run. The existing external threshold
must not be relaxed retrospectively.

This follow-up was implemented as
[`P0-D2H-CAL`](p0d2hc-oracle-calibration-protocol.md). The 0.5B source failed its
answer-copy oracle (`0.7891`), while the 1.5B canary passed the oracle (`0.9948`)
but failed the frozen strict external gate (`0.6901` accuracy, `0.2318` invalid).
The resulting status was `mixed_scale_result`, with no eligible model.

The post-hoc
[`invalid-output audit`](p0d2hc-invalid-output-audit.md) then showed that `87/89`
of the 1.5B external invalids contained the single correct action plus extra text.
Its conservative semantic external accuracy was `0.9167`, but
`conditional_route` remained at `0.6667`. This localizes most of the canary's
overall failure to formatting while retaining a category-specific semantic
deficit. Neither result changes the P0-D2H-R locus conclusion or authorizes
adapter training.

The next diagnostic should therefore use independent base-only forced-choice
candidate scoring. Its prospectively frozen gate must include a per-category
external floor before any 1/4/8 mappings-per-adapter design can be reviewed.

## 8. Entry to genuinely harder training

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

The verified P0-D2H-R result does not yet satisfy item 1. No narrow scan or
multi-mapping training experiment is currently eligible under the frozen protocol.
