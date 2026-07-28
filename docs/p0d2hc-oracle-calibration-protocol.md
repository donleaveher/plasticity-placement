# P0-D2H-CAL Base-Only Oracle Calibration Protocol

## 1. Purpose and status

- **Implementation status:** complete
- **Results status:** complete; supplementary invalid-output audit pending
- **Source:** one complete, verified, run-valid P0-D2H-R attempt
- **Question:** Is the weak hard-suite external anchor caused by failure to follow
  the output interface, failure to use a supplied mapping under distractors, or
  insufficient model scale?

This is a no-training calibration. It does not load P0-D2H-R adapters and does not
start a layer scan or a multi-mapping experiment.

The completed run found:

- source 0.5B oracle accuracy `0.7891`, yielding `oracle_failed`;
- scale-canary 1.5B oracle accuracy `0.9948`, passing every oracle check;
- scale-canary external accuracy `0.6901` and invalid rate `0.2318`, failing both
  frozen external checks;
- `mixed_scale_result`, with no model eligible for multi-mapping training.

The separate invalid-output audit determines whether the canary invalids contain
conservatively recoverable correct action tokens; it cannot change these strict
results.

## 2. Frozen matrix

The source cohort is reused read-only:

```text
24 lessons × 16 hard probes × 3 arms = 1,152 rows/model
```

The required source model is the immutable model revision recorded by P0-D2H-R.
One optional same-family scale canary adds another 1,152 rows.

| Arm | Prompt content | Diagnostic role |
| --- | --- | --- |
| `no_write` | frozen hard probe only | chance/background anchor |
| `external` | current verified-memory rendering | mapping-use anchor |
| `answer_copy_oracle` | explicit correct action immediately before the strict output contract | instruction/output-copy upper anchor |

All arms use deterministic greedy generation and the same frozen hard-probe bank.
The oracle renderer changes only the oracle copy of each prompt.

## 3. Immutable source checks

Before any inference, the runner requires:

- a `p0d2h-manifest-v1` with `p0d2h-config-v2`;
- 24 selected lessons, 288 verified adapter units, and 24 verified base units;
- a `p0d2h-summary-v2` with 5,376 rows and `run_valid=true`;
- a valid P0-D2H-R prompt audit with zero truncation and no missing output
  instruction;
- unchanged P0-D2 source manifest, compiled lesson hashes, hard-probe hashes,
  source run IDs, model name, and immutable model revision.

P0-D2H-CAL reads no source adapter bundle. Source manifest and summary hashes are
carried into every calibration run identity.

## 4. Prompt-token and inference requirements

The `audit` command loads each tokenizer on CPU and freezes one audit record for
every model, lesson, arm, and probe. Formal inference is refused unless:

- every prompt fits within `evaluation_max_length=512`;
- truncation count is zero;
- the strict output instruction remains present;
- the prompt hash and retained token count match the later result row.

The `run` command requires CUDA. Each base model is loaded once, evaluates all 24
lessons and three arms, and is then released. No adapter activation is possible
through this command.

## 5. Frozen gates

For each model, the answer-copy oracle passes only if:

- overall accuracy is at least `0.90`;
- accuracy in every hard category is at least `0.80`;
- invalid-action rate is at most `0.01`.

The external anchor passes only if:

- overall accuracy is at least `0.75`;
- invalid-action rate is at most `0.05`.

Model-level outcomes are:

| Status | Meaning |
| --- | --- |
| `oracle_failed` | the model cannot reliably copy an explicit answer under the hard interface |
| `oracle_pass_external_failed` | output copying works, but supplied verified-memory use under distractors does not |
| `calibrated` | both oracle and external anchors pass |

With a scale canary:

- source failure plus canary calibration emits `scale_bottleneck_supported`;
- both calibrated emits `both_models_calibrated`;
- both oracle failures emit `shared_interface_or_output_bottleneck`;
- both oracle-pass/external-fail outcomes emit
  `shared_verified_memory_bottleneck`;
- all other combinations emit `mixed_scale_result`.

At least one `calibrated` model is required for
`training_complexity_design_eligible`. This label only permits a separately
reviewed and frozen 1/4/8 mappings-per-adapter design; no training starts
automatically.

## 6. Outputs and recovery

The independent Drive root is:

```text
/content/drive/MyDrive/plasticity-p0d/
  hard-probe-calibration/v1/pipelines/
```

Each run writes:

```text
oracle_calibration-<attempt>/
├── manifest.json
├── notebook_context.json
├── source_sessions.json
├── preflight/prompt_token_audit.json
├── environment/
├── results/raw/<model_id>/<lesson_id>.jsonl
└── results/aggregate/
    ├── summary.json
    ├── main_table.md
    └── next_stage_decision.json
```

Each lesson-model unit is written atomically. A normal Colab interruption may be
resumed in the same attempt: verified rows are hash/provenance checked and skipped,
while an `evaluating` unit without a complete raw file is rerun. A recorded
`failed` unit is immutable; preserve it for audit and use a new calibration
attempt after fixing the cause.

Strict-invalid outputs may be examined later through the separate, read-only
[`P0-D2H-CAL invalid-output audit`](p0d2hc-invalid-output-audit.md). That analysis
writes outside this source stage and cannot change the frozen gates.

## 7. Colab execution

[Open P0-D2H-CAL in Colab](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/p0d2h_calibration/p0d2h_calibration_colab.ipynb)

Run the notebook in order. Checkout/install and the prompt-token audit are mostly
CPU work, so zero GPU memory during those cells is expected. GPU use begins in
the explicitly labelled **Formal base-only evaluation (GPU)** cell.

The default notebook evaluates the P0-D2H-R source model and
`Qwen/Qwen2.5-1.5B-Instruct` as the same-family scale canary. The planning step
resolves and freezes the canary to an immutable model commit.
