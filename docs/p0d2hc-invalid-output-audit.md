# P0-D2H-CAL Invalid-Output Audit

## Status and scope

- **Analysis type:** post-hoc supplementary
- **Source:** one complete, verified P0-D2H-CAL run
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

## Interpretation boundary

If most invalid external rows are `expected_action_with_extra_text`, the evidence
supports a format-adherence bottleneck conditional on the conservative recovery
definition. It does not retrospectively pass the external gate.

If invalid rows are dominated by wrong, multiple, or no-action classes, semantic
memory use remains unresolved. Either outcome must be reported alongside the
original strict result and frozen `next_stage_status`.
