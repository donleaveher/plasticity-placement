# P0-D2H-CAL-FC Base-Only Format-Stable Forced-Choice Calibration

## 1. Status and question

- **Protocol status:** prospectively frozen
- **Implementation status:** in repository
- **Results status:** formal aggregate completed; gate failed
- **Direct source:** one complete, verified P0-D2H-CAL run containing both frozen
  model revisions and all 2,304 strict-generation rows
- **Question:** Can each base model select the correct action from the four frozen
  choices when output-format generation is removed from the primary endpoint?

P0-D2H-CAL-FC is an evaluation-only, base-only diagnostic. It neither discovers
nor loads an adapter, and it exposes no training, layer scan, GRPO/RLVR, or
1/4/8-mapping execution path. A passing result permits only a separate human
review of a future frozen training-complexity design.

The completed aggregate and bounded interpretation are recorded in
[`P0-D2H-CAL-FC results`](p0d2h-format-stable-calibration-results.md).

## 2. Read-only source and independent namespace

The complete P0-D2H-CAL source manifest, prompt audit, raw rows, summary, and its
transitive P0-D2H-R/P0-D2 artifacts remain read-only. The new stage writes only
beneath:

```text
/content/drive/MyDrive/plasticity-p0d/
  hard-probe-forced-choice/v1/pipelines/
```

The fixed matrix is:

```text
2 models × 24 lessons × 3 arms × 16 probes = 2,304 decision rows
```

The models are exactly the source 0.5B revision and 1.5B scale-canary revision
recorded by P0-D2H-CAL. The arms remain `no_write`, `external`, and
`answer_copy_oracle`. The hard-probe order, category, expected action, ordered
allowed actions, renderer, and formatted prompt hash must match the source row
and frozen probe bank.

Before deriving any row, the implementation runs the canonical P0-D2H-CAL source
validator. It then records and verifies:

- source run, manifest, summary, prompt-audit, and raw-tree SHA-256 values;
- each source raw-file path and SHA-256;
- a stable source row key and canonical row SHA-256;
- the transitive P0-D2H run and hard-probe bank identities;
- both model names and immutable revisions.

## 3. Primary endpoint

The primary endpoint never asks the model to generate an answer. For each frozen
formatted chat prompt, it scores all four ordered actions as teacher-forced
continuations.

The canonical continuation is:

```text
<empty leading whitespace><exact action token>
```

The empty leading whitespace is frozen because the Qwen chat template already
ends with the assistant-turn newline. The CPU audit must show that tokenizing the
formatted prompt plus continuation preserves every prompt token and that the
remaining IDs exactly equal standalone continuation tokenization.

For prompt token count \(P\), full input IDs \(x_0,\ldots,x_{T-1}\), and causal
logits \(z\), only candidate positions are accumulated:

\[
S(c)=\sum_{t=P}^{T-1}\log p(x_t\mid x_{<t})
    =\sum_{t=P}^{T-1}\log\operatorname{softmax}(z_{t-1})_{x_t}.
\]

The frozen primary score is `sum_logprob`. `mean_logprob` is retained only as a
length/tokenizer sensitivity diagnostic. The unique highest sum score is the
primary prediction. Exact highest-score ties, non-finite scores, and any runtime
prompt/candidate token mismatch produce an explicit invalid/error row; they are
never tie-broken silently.

Every decision row records:

- source manifest/run/raw-tree/file/row identities and hashes;
- model ID, role, name, immutable revision, and evaluation precision;
- lesson, pair, lesson type, probe, category, arm, expected action, and ordered
  allowed actions;
- formatted prompt hash, untruncated prompt tokens, and full sequence limit;
- for every candidate: continuation, token IDs/count, sum/mean score, and rank;
- sum and mean predictions, disagreement flag, top-1/top-2 sum margin;
- correct, tie, mean-tie, non-finite, error status/message, and latency;
- environment fingerprint and execution-session identity;
- source strict-generation correctness/invalid status as a linked secondary
  format metric.

Old strict accuracy is neither overwritten nor redefined.

## 4. Preflight and formal scoring

The CPU-only `audit` command must complete before formal scoring. It freezes one
record per decision row and verifies:

- the canonical source validator accepts all 2,304 source rows;
- prompt rendering reproduces the source formatted prompt SHA-256;
- no prompt or prompt-plus-candidate sequence exceeds 512 tokens;
- no tokenizer truncation is used;
- every candidate preserves the prompt token prefix;
- standalone and concatenated candidate token IDs match exactly;
- decoded continuation, candidate token count, ordered actions, expected action,
  source row key, and source hashes are exact;
- each tokenizer chat-template hash and candidate-scoring version are recorded.

The manifest identity covers the full source identity, model revisions, selected
lessons, arms/categories/actions, renderers, chat-template audit, candidate-token
audit hash, scoring version, canonical whitespace, `sum_logprob` definition,
thresholds, evaluation limit, and experiment code hash.

Formal scoring requires CUDA only after the existing audit has been loaded and
verified. Each model is loaded once and remains resident while all lessons are
processed. The four candidates for one lesson/arm/probe are evaluated as a batch.
Progress heartbeats identify the model, lesson, and arm.

## 5. Metrics

Confidence intervals use the deterministic lesson-clustered bootstrap with
10,000 samples by default. Reports include:

- forced-choice accuracy and 95% interval;
- model × arm × category accuracy;
- paired `external - no_write` difference and 95% interval;
- paired `oracle - external` difference and 95% interval;
- top-1/top-2 sum-score margin;
- sum-score versus mean-score prediction disagreement rate;
- tie, error, and non-finite rates;
- paired forced-choice minus linked strict-generation accuracy as a secondary
  metric.

## 6. Prospectively frozen gates

Every model is evaluated against all of these checks:

- complete prompt/provenance/candidate-token audit and zero silent truncation;
- tie/error/non-finite rate exactly `0`;
- oracle overall accuracy at least `0.95`;
- oracle accuracy in every category at least `0.90`;
- external overall accuracy at least `0.85`;
- external accuracy in every category at least `0.75`;
- no-write overall accuracy at most `0.35`;
- paired `external - no_write` 95% interval lower bound strictly above `0`.

Only a model passing every check receives
`training_complexity_review_eligible`. This status means only that a separate
1/4/8 mappings-per-adapter design may be reviewed manually.

If external overall accuracy reaches `0.85` but `conditional_route` is below
`0.75`, the model status is explicitly `category_calibration_failed`, even if
the other three categories are at ceiling.

All result and next-stage artifacts must retain:

```text
automatic_training_started = false
automatic_narrow_scan_started = false
```

No later stage starts automatically.

## 7. Outputs, recovery, and failure semantics

Each stage writes:

```text
forced_choice-<attempt>/
├── manifest.json
├── notebook_context.json
├── source_sessions.json
├── environment.json
├── environment_sessions.json
├── preflight/candidate_token_audit.json
├── results/raw/<model_id>/<lesson_id>.jsonl
└── results/aggregate/
    ├── summary.json
    ├── main_table.md
    └── next_stage_decision.json
```

Each model × lesson unit is written atomically. A complete orphan raw file may be
verified and adopted after a normal interruption. A verified unit is immutable.
An incomplete or provenance-conflicting artifact fails verification. A recorded
`failed` unit is terminal: preserve that attempt and use a new attempt after a
fix; never delete the failure and overwrite it in place.

Before aggregation, all source hashes and the complete 48-unit/2,304-row matrix
are revalidated. Duplicate, missing, reordered, or mismatched rows are rejected.

## 8. Colab

[Open P0-D2H-CAL-FC in Colab](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/p0d2h_forced_choice/p0d2h_forced_choice_colab.ipynb)

Run cells in order. Checkout, source validation, planning, and the complete
candidate-token audit are CPU preflight. The first cell allowed to load a model
onto CUDA is explicitly titled **Formal full-string candidate scoring (GPU)**.
Aggregation is a separate CPU/Drive step after all raw units verify.

## 9. Claim boundary

A passing forced-choice result supports format-stable action selection only for
the two frozen model revisions, fixed 24-lesson cohort, three prompt arms, and
four hard categories. It does not establish a universal capacity mechanism,
change the prior strict-generation gate, validate LoRA placement, or demonstrate
multi-mapping training. Any 1/4/8 experiment requires a new protocol, package,
manifest, notebook, namespace, and explicit human approval.
