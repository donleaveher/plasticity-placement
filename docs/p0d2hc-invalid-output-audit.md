# P0-D2H-CAL Invalid-Output Audit

## Status and scope

- **Analysis type:** post-hoc supplementary
- **Source:** one complete, verified P0-D2H-CAL run
- **Results status:** complete
- **Audit run ID:** `p0d2hc-invalid-audit-95ae30ec5e`
- **Compute:** CPU only; no model loading or inference
- **Primary gates:** unchanged
- **Next-stage eligibility:** unchanged

The audit separates strict output-contract failure from conservatively recoverable
action selection. It validates every source row through the canonical
P0-D2H-CAL provenance checks before deriving any new value.

## Strict versus semantic estimands

The frozen primary parser remains unchanged: after trimming whitespace and
case-folding, the entire generated string must equal one of the four allowed
action tokens.

The supplementary semantic parser searches `generated_text` for boundary-delimited
allowed action tokens. A row is semantically recovered only when:

1. exactly one distinct allowed action occurs anywhere in the output; and
2. that action equals the expected action.

Outputs that mention the expected action together with any distractor remain
ambiguous and are not recovered.

Reported summaries include:

- strict accuracy and invalid rate with lesson-clustered bootstrap intervals;
- accuracy among strict-valid rows;
- conservative semantic accuracy with lesson-clustered bootstrap intervals;
- semantic-minus-strict recovery gain;
- expected-action mention rate;
- recovery rate among invalid rows;
- generation-limit rate among invalid rows;
- model × arm and model × arm × hard-category breakdowns.

These supplementary metrics do not replace the strict endpoint.

## Mutually exclusive invalid taxonomy

| Class | Definition |
| --- | --- |
| `empty_output` | empty or whitespace-only output |
| `expected_action_with_extra_text` | exactly one distinct action occurs, it is correct, but the whole output is not a strict token |
| `wrong_action_with_extra_text` | exactly one distinct action occurs and it is wrong |
| `multiple_actions_including_expected` | multiple distinct actions occur, including the expected action |
| `multiple_actions_excluding_expected` | multiple distinct actions occur and none is expected |
| `no_allowed_action_at_generation_limit` | no complete allowed action occurs and generated tokens reach the frozen limit |
| `no_allowed_action_other` | no complete allowed action occurs without reaching the limit |

Reaching the generation limit is a diagnostic flag, not proof that the output was
truncated mid-action.

## Independent artifacts

The source P0-D2H-CAL run remains read-only. The audit writes to:

```text
/content/drive/MyDrive/plasticity-p0d/
  hard-probe-calibration-analysis/v1/pipelines/
```

Each audit stage contains:

```text
invalid_audit-<attempt>/
├── invalid_output_audit.json
├── invalid_output_audit.md
└── invalid_output_records.jsonl
```

The JSON report records source manifest hash, source raw-result tree hash, source
experiment code hash, analysis code hash, taxonomy version, and bootstrap budget.
Artifacts are immutable within an attempt. A source or analysis change requires a
new attempt.

The classified JSONL contains only source rows that were strict-invalid. Raw source
files and the source manifest are hashed before and after the notebook command; a
change aborts the audit.

## Command and Colab

```bash
uv run plasticity-p0d2hc audit-invalid \
  --output <verified-p0d2hc-stage> \
  --audit-output <independent-audit-stage> \
  --bootstrap-samples 10000
```

[Open the CPU-only audit in Colab](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/p0d2h_invalid_audit/p0d2h_invalid_output_audit_colab.ipynb)

The default notebook reads the completed `pipeline-c1/oracle_calibration-c1`
source and writes to `pipeline-a1/invalid_audit-a1`.

## Verified results

The audit covered all `2,304` calibration rows. It found `183` strict-invalid
rows, all from the 1.5B scale canary. No invalid row was empty, ambiguous between
multiple allowed actions, missing an allowed action, or at the generation limit.

The verified source aggregate was:

```text
/content/drive/MyDrive/plasticity-p0d/hard-probe-calibration/v1/pipelines/
  pipeline-c1/runs/code-43e278fd23_source-4fbec74d8c/
  oracle_calibration-c1/results/aggregate/summary.json
```

### Overall strict and semantic results

| Model | Arm | Strict | Invalid | Semantic | Semantic − strict | Valid-only |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1.5B scale canary | `no_write` | 0.1276 | 0.2448 | 0.1901 | +0.0625 | 0.1690 |
| 1.5B scale canary | `external` | 0.6901 | 0.2318 | 0.9167 | +0.2266 | 0.8983 |
| 1.5B scale canary | `answer_copy_oracle` | 0.9948 | 0.0000 | 0.9948 | +0.0000 | 0.9948 |
| 0.5B source | `no_write` | 0.2526 | 0.0000 | 0.2526 | +0.0000 | 0.2526 |
| 0.5B source | `external` | 0.4297 | 0.0000 | 0.4297 | +0.0000 | 0.4297 |
| 0.5B source | `answer_copy_oracle` | 0.7891 | 0.0000 | 0.7891 | +0.0000 | 0.7891 |

For the 1.5B `external` arm, the 95% lesson-clustered intervals were:

- strict accuracy `0.6901 [0.6562, 0.7240]`;
- invalid rate `0.2318 [0.1979, 0.2630]`;
- semantic accuracy `0.9167 [0.8906, 0.9427]`;
- semantic recovery gain `+0.2266 [0.1953, 0.2552]`.

The corresponding 1.5B `no_write` semantic accuracy was
`0.1901 [0.1406, 0.2422]`, and its recovery gain was
`+0.0625 [0.0443, 0.0833]`. The semantic parser therefore did not simply inflate
every canary arm toward high accuracy.

### Invalid-output taxonomy

| Model | Arm | Class | Count | Share of arm invalids |
| --- | --- | --- | ---: | ---: |
| 1.5B scale canary | `no_write` | `expected_action_with_extra_text` | 24 | 0.2553 |
| 1.5B scale canary | `no_write` | `wrong_action_with_extra_text` | 70 | 0.7447 |
| 1.5B scale canary | `external` | `expected_action_with_extra_text` | 87 | 0.9775 |
| 1.5B scale canary | `external` | `wrong_action_with_extra_text` | 2 | 0.0225 |

Thus, `87/89` canary `external` invalids contained exactly one allowed action and
that action was correct. In contrast, only `24/94` canary `no_write` invalids were
recoverable; most contained a single wrong action.

### Category localization

| 1.5B arm | Category | Strict | Invalid | Semantic | Gain |
| --- | --- | ---: | ---: | ---: | ---: |
| `external` | `binding_decoys` | 1.0000 | 0.0000 | 1.0000 | +0.0000 |
| `external` | `conflict_stack` | 1.0000 | 0.0000 | 1.0000 | +0.0000 |
| `external` | `conditional_route` | 0.5729 | 0.1146 | 0.6667 | +0.0938 |
| `external` | `long_context` | 0.1875 | 0.8125 | 1.0000 | +0.8125 |
| `answer_copy_oracle` | `binding_decoys` | 0.9896 | 0.0000 | 0.9896 | +0.0000 |
| `answer_copy_oracle` | `conflict_stack` | 1.0000 | 0.0000 | 1.0000 | +0.0000 |
| `answer_copy_oracle` | `conditional_route` | 0.9896 | 0.0000 | 0.9896 | +0.0000 |
| `answer_copy_oracle` | `long_context` | 1.0000 | 0.0000 | 1.0000 | +0.0000 |

The canary `long_context` deficit is therefore almost entirely an exact-output
format problem under this recovery rule. `conditional_route` remains a semantic
selection problem: recovery raises accuracy only to `0.6667`, despite the
category oracle reaching `0.9896`.

The 0.5B model produced no strict-invalid rows, so semantic recovery does not
alter its result. Its answer-copy oracle remained weak in `binding_decoys`
(`0.7396`) and `long_context` (`0.6146`), consistent with an interface/capacity
limitation rather than a formatting-only explanation.

## Frozen decision and claim boundary

The post-hoc audit does not alter the original calibration decision:

```text
cross_scale_status = mixed_scale_result
next_stage_status = calibration_followup_required
eligible_model_ids = []
automatic_training_started = false
automatic_narrow_scan_started = false
```

The verified evidence supports the bounded claim that the 1.5B model can select
the correct externally supplied action on most hard probes, while its unconstrained
generation often violates the exact-token contract. It also localizes a remaining
semantic/compositional weakness to `conditional_route`.

It does not support:

- retrospectively passing the frozen strict external gate;
- treating all canary invalids as correct;
- claiming that the complete hard suite is solved;
- starting LoRA parameter conditions, a narrow layer scan, or 1/4/8
  mappings-per-adapter training;
- generalizing the result beyond the two frozen model revisions and 24-lesson
  cohort.

## Next experiment

The next experiment should be an independent, base-only, format-stable
forced-choice calibration. It should score each of the four allowed action strings
under the frozen prompt and select the highest conditional-likelihood candidate,
while retaining the old strict-generation result only as a referenced secondary
format metric.

This follow-up must use a new package, manifest/schema, notebook directory, and
Drive namespace. It must not load adapters or expose training. Entry into a later
training-complexity design should require both a strong overall external score and
a prospectively frozen per-category floor, so that the current
`conditional_route` deficit cannot be hidden by ceiling categories.

The implementation handoff is recorded in
[`P0-D2H format-stable calibration next-session prompt`](p0d2h-format-stable-calibration-next-session-prompt.md).

## Interpretation rule applied

The observed 1.5B `external` invalids are overwhelmingly
`expected_action_with_extra_text`, so they support a format-adherence bottleneck
under the conservative recovery definition. The remaining
`conditional_route` error persists after recovery and therefore cannot be
explained by exact-output formatting. Both findings must be reported beside the
original strict result and frozen `next_stage_status`.
