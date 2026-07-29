# P0-D2H-CAL-FC forced-choice calibration results

## 1. Evidence status

This report records the formal aggregate supplied after the
P0-D2H-CAL-FC run:

```text
source_run_id = p0d2hfc-forced-choice-f604efe6bf
summary_path =
  /content/drive/MyDrive/plasticity-p0d/hard-probe-forced-choice/v1/
  pipelines/pipeline-f1/runs/code-a478007273_source-6c694ef6a2/
  forced_choice-f1/results/aggregate/summary.json
```

The supplied table and gate output are internally consistent with the frozen
thresholds. A later copy of the formal forced-choice output tree was locally
revalidated at the unit, row, candidate-token, score, ranking, session, and
aggregate levels. The copy does not include the upstream P0-D2H-CAL source tree,
so the original cross-stage source files could not be independently rehashed.
The detailed verification boundary is recorded in the
[`conditional-route post-hoc audit`](p0d2hfc-conditional-route-posthoc-audit.md).

## 2. Primary results

The primary endpoint is four-candidate full-string conditional scoring with
candidate `sum_logprob`. Confidence intervals are the frozen
lesson-clustered 95% bootstrap intervals.

| Model | Arm | Forced choice [95% CI] | Linked strict | Margin | Sum/mean disagree | Tie/error/non-finite |
|---|---|---:|---:|---:|---:|---:|
| 1.5B scale canary | no-write | 0.1745 [0.1276, 0.2240] | 0.1276 | 1.2243 | 0.0000 | 0.0000 |
| 1.5B scale canary | external | 0.9089 [0.8828, 0.9349] | 0.6901 | 4.2143 | 0.0000 | 0.0000 |
| 1.5B scale canary | answer-copy oracle | 0.9948 [0.9870, 1.0000] | 0.9948 | 4.3763 | 0.0000 | 0.0000 |
| 0.5B source | no-write | 0.2500 [0.2109, 0.2917] | 0.2526 | 1.2856 | 0.0000 | 0.0000 |
| 0.5B source | external | 0.4349 [0.3724, 0.5052] | 0.4297 | 1.2750 | 0.0000 | 0.0000 |
| 0.5B source | answer-copy oracle | 0.7812 [0.7214, 0.8411] | 0.7891 | 1.5975 | 0.0000 | 0.0000 |

The 1.5B external forced-choice endpoint is 0.2188 above the linked strict
accuracy as a descriptive matched-row difference. The corresponding 0.5B
difference is only 0.0052. These descriptive differences do not replace the
paired bootstrap intervals in the canonical summary.

## 3. Category results

| Model | Arm | Binding decoys | Conflict stack | Conditional route | Long context |
|---|---|---:|---:|---:|---:|
| 1.5B | no-write | 0.1146 | 0.3542 | 0.0000 | 0.2292 |
| 1.5B | external | 1.0000 | 1.0000 | **0.6354** | 1.0000 |
| 1.5B | answer-copy oracle | 0.9896 | 1.0000 | 0.9896 | 1.0000 |
| 0.5B | no-write | 0.1354 | 0.3125 | 0.2812 | 0.2708 |
| 0.5B | external | 0.5104 | 0.4896 | 0.4167 | 0.3229 |
| 0.5B | answer-copy oracle | 0.6979 | 0.8333 | 0.9688 | 0.6250 |

Each category contains 96 decisions. For the 1.5B external arm,
`conditional_route` has 61 correct and 35 incorrect decisions. The other three
external categories are at 96/96, so the 0.9089 overall result must not conceal
the routing failure. The 1.5B oracle reaches 95/96 on the same category, which
shows that the frozen candidate endpoint and answer-copy interface themselves
can express the correct action.

The 1.5B long-context result rises from linked strict 0.1875 to forced-choice
1.0000. In contrast, `conditional_route` rises only from 0.5729 to 0.6354.
This separates the previous long-context format failure from a remaining local
routing/composition deficit.

All four 0.5B external categories remain below the frozen 0.75 floor. Its oracle
also fails overall and in binding-decoy, conflict-stack, and long-context
categories. Removing free-generation formatting therefore does not calibrate
the smaller model.

## 4. Paired contrasts

| Model | Contrast | Difference [95% CI] |
|---|---|---:|
| 0.5B source | external minus no-write | +0.1849 [+0.1354, +0.2396] |
| 0.5B source | oracle minus external | +0.3464 [+0.2917, +0.4036] |
| 1.5B scale canary | external minus no-write | +0.7344 [+0.6875, +0.7786] |
| 1.5B scale canary | oracle minus external | +0.0859 [+0.0625, +0.1094] |

External memory helps both models relative to no-write, with strictly positive
paired lower confidence bounds. That improvement is necessary but not
sufficient for calibration: the absolute oracle, overall external, and
every-category floors still apply.

## 5. Frozen gate decision

| Model | Passed checks | Failed checks | Status |
|---|---|---|---|
| 1.5B scale canary | provenance/token audit; zero scoring invalids; oracle overall/categories; external overall; no-write ceiling; external-minus-no-write contrast | external every-category floor (`conditional_route=0.6354 < 0.75`) | `category_calibration_failed` |
| 0.5B source | provenance/token audit; zero scoring invalids; no-write ceiling; external-minus-no-write contrast | oracle overall; oracle every-category; external overall; external every-category | `oracle_failed` |

The scoring-integrity diagnostics do not indicate a format-stability
implementation failure: both models have zero tie/error/non-finite rows, zero
sum/mean prediction disagreement, and a passing prompt/provenance/token audit.

The immutable cross-scale outcome is:

```text
cross_scale_status = scale_improvement_without_full_calibration
next_stage_status = calibration_followup_required
eligible_model_ids = []
automatic_training_started = false
automatic_narrow_scan_started = false
```

No model is eligible for a 1/4/8 mappings-per-adapter training-complexity
review. These results must not start adapter training, a layer scan, GRPO/RLVR,
or a router experiment.

## 6. Bounded interpretation

The result supports three claims:

1. Free-generation format compliance materially underestimated the 1.5B
   model's external-memory action selection, especially on long context.
2. The 1.5B model nevertheless retains a category-specific
   `conditional_route` deficit after output generation is removed.
3. The 0.5B model's weak oracle and external results are not explained by the
   strict parser or by candidate-length scoring.

The result does not establish model scale as the unique causal mechanism, show
that the hard suite is fully calibrated, validate any LoRA placement, or permit
training.

## 7. Completed follow-up

The required read-only audit of all 96 supplied 1.5B external
`conditional_route` rows is complete. The copied forced-choice tree is
internally consistent: 48/48 unit hashes, 2,304 decision rows, 2,304
candidate-audit records, and 9,216 score sequences pass local verification.

The audit finds that 34 of the 35 external errors are correct under the paired
oracle arm. Errors occur in 20/24 lessons and are enriched in a confounded
`procedure_recovery × variant 3` intersection, with accompanying action and
candidate-position preferences. The final-action endpoint cannot distinguish a
route-selection error from a subsequent retrieval error.

Full integrity evidence, factor tables, confusion matrix, margin analysis, and
the next-stage boundary are recorded in
[`conditional-route post-hoc audit`](p0d2hfc-conditional-route-posthoc-audit.md).
The audit does not change the frozen gate or eligibility.
