# P0-D2H-CRD Base-Only Route/Retrieval Decomposition

## 1. Status and scientific question

- **Protocol status:** prospectively frozen
- **Implementation status:** complete and locally verified
- **Results status:** not run
- **Training status:** forbidden

P0-D2H-CRD asks where the 1.5B canary's P0-D2H-CAL-FC
`conditional_route` errors occur:

1. can the model apply an explicit marker-to-slot routing table;
2. can it retrieve the correct action when the selected live record is supplied;
3. can it compose both operations in one prompt?

This is a new base-only diagnostic. It does not change the completed
P0-D2H-CAL-FC endpoint, gate, status, or raw rows and cannot make a model
training-eligible.

## 2. Immutable source and model

The direct source is the complete formal P0-D2H-CAL-FC run:

```text
run_id = p0d2hfc-forced-choice-f604efe6bf
manifest_sha256 =
  89ba71ea0fde052089329e630d23c1a86652bc47a129de1921786c8e0b4dd214
candidate_token_audit_sha256 =
  d140cfb2ac0c57b4819d3793b59f5f7c767fe428336be856fc9299460731b230
raw_tree_sha256 =
  dfe9b9014966f484e988a2f6d9f8e18d4ee7a1b5f06d671ae2e973cef8a0b24a
summary_sha256 =
  24b1fe51bf400e5c1e0b15aa68d67b0db913591cce082aeac13b9396854d22e1
```

The CRD frozen-source validator must verify the exact P0-D2H-CAL-FC manifest,
summary, candidate-token audit, all 48 raw-file hashes, the complete 2,304-row
matrix, and the raw-tree hash. It then follows the stored source paths and
hashes to verify the full P0-D2H-CAL raw tree and prompt audit, the P0-D2H-R
manifest and hard-probe bank, and the P0-D2 compiled bank before deriving any
prompt. This independent hash validator is required because the old code
identity covered the entire source tree and necessarily changes when the new
CRD package is added.

Only the frozen 1.5B scale canary is scored:

```text
model_name = Qwen/Qwen2.5-1.5B-Instruct
model_revision = 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
model_id = scale_canary
precision = source-compatible 4-bit evaluation
```

The 0.5B model is excluded because its forced-choice answer-copy oracle already
failed; CRD is specifically a localization of the canary's remaining external
conditional-routing deficit.

## 3. Independent probe bank and namespace

CRD reuses the frozen 24 lesson identities and external notes read-only but
compiles a new prompt bank. It writes only beneath:

```text
/content/drive/MyDrive/plasticity-p0d/
  hard-probe-route-decomposition/v1/pipelines/
```

For every lesson, the bank contains:

| Endpoint | Factor crossing | Rows/lesson | Candidates/row |
|---|---|---:|---:|
| `route_only` | 4 route variants × 2 slot-candidate orders × 2 slot-content orders | 16 | 2 |
| `retrieval_only` | 4 selected-record variants × 4 action-panel rotations | 16 | 4 |
| `combined` | 4 route variants × 4 action-panel rotations × 2 slot-content orders | 32 | 4 |

The frozen full matrix is:

```text
1 model × 24 lessons × 64 rows/lesson = 1,536 decision rows
384 route-only rows × 2 candidates
  + 1,152 retrieval/combined rows × 4 candidates
  = 5,376 candidate sequences
24 atomic model×lesson units
```

Each CRD row links to the corresponding immutable 1.5B external
`conditional_route` source row for the same lesson and route variant. That link
is provenance only; the old prompt and score are not reused as the new
prediction.

## 4. Frozen endpoint prompts

### 4.1 Shared routing table

Every route-bearing cell uses two deterministic opaque markers and an explicit
table:

```text
<marker-X> -> slot_a
<marker-Y> -> slot_b
Current marker: <one of the two markers>
```

Variants 1 and 3 select `slot_a`; variants 2 and 4 select `slot_b`. Marker
strings are deterministic hashes of the lesson and variant. Slot-content order
is independently counterbalanced.

### 4.2 `route_only`

The prompt includes only the routing table and two opaque slot records. It
contains no external note, workspace/condition mapping, or action token. The
candidates are exactly:

```text
slot_a
slot_b
```

Their displayed order is independently crossed with route variant and slot
content order. This endpoint tests routing without retrieval.

### 4.3 `retrieval_only`

The prompt states that routing has already been completed and supplies the
selected live record's workspace and condition. It contains the verified
external note but no routing table, current-marker decision, or archived slot.
The four canonical action candidates are scored. This endpoint tests retrieval
without routing.

### 4.4 `combined`

The prompt contains the routing table, current marker, one live slot, one
archived decoy slot, and the verified external note. The model must select the
slot and then retrieve the action. The four canonical action candidates are
scored.

The archived slot contains all three distractor actions and an unrelated
deterministic workspace. Status labels explicitly mark it non-executable.

## 5. Counterbalancing and leakage audit

Before tokenizer loading, the deterministic bank audit must prove:

- exactly 24 lessons and the 1,536-row/5,376-candidate matrix;
- six lessons per expected action;
- every lesson receives each action-panel position equally in
  `retrieval_only` and within every combined route/content-order cell;
- route target slots A/B are balanced;
- route candidate positions and slot-content presentation orders are balanced;
- `route_only` contains none of the four action tokens, external notes,
  workspace IDs, conditions, or desired actions;
- `retrieval_only` contains no routing table or marker-selection instruction;
- `combined` contains exactly one live record and one archived record;
- every row maps to the expected immutable source conditional-route row;
- probe IDs, prompt hashes, bank-record hashes, factor IDs, and orderings are
  unique and deterministic.

Any imbalance, leakage, duplicate, or source mismatch forbids formal scoring.

## 6. Candidate-only full-string scoring

All candidates use the same audited chat template and canonical empty leading
whitespace as P0-D2H-CAL-FC. Only continuation tokens contribute to:

```text
sum_logprob   # frozen primary score
mean_logprob  # tokenizer/length diagnostic only
```

The unique maximum `sum_logprob` is the prediction. Exact ties, non-finite
scores, prompt/candidate concatenation failures, silent truncation, and stored
ranking inconsistencies are explicit invalid/error rows. Candidate count is
endpoint-specific and frozen at two or four.

The CPU preflight audits all 1,536 prompts and 5,376 continuations before any
model may load onto CUDA.

## 7. Metrics and prospective diagnostic statuses

All confidence intervals use the deterministic lesson-clustered bootstrap with
10,000 samples by default. Report:

- endpoint accuracy and 95% interval;
- accuracy by lesson type, lesson side, route variant, target slot, expected
  action, candidate position, and slot-content order where applicable;
- top-1/top-2 margin and sum/mean disagreement;
- tie/error/non-finite rate;
- predicted-action and predicted-position distributions;
- paired lesson-level `combined - retrieval_only`;
- composition gap `min(route_only, retrieval_only) - combined`;
- the linked old combined forced-choice accuracy as a secondary metric only.

The prospectively frozen diagnostic floors are:

```text
route_only accuracy >= 0.90
retrieval_only accuracy >= 0.90
combined accuracy >= 0.75
tie/error/non-finite rate = 0
all bank/provenance/token audits pass
```

Statuses are:

- `scoring_integrity_failed`: any integrity check fails;
- `route_bottleneck_supported`: route fails and retrieval passes;
- `retrieval_bottleneck_supported`: retrieval fails and route passes;
- `shared_component_failure`: route and retrieval both fail;
- `composition_bottleneck_supported`: both components pass but combined fails;
- `no_component_bottleneck_detected`: all three floors pass.

These are diagnostic labels, not training gates. Every output must retain:

```text
training_complexity_review_eligible = false
automatic_training_started = false
automatic_narrow_scan_started = false
```

## 8. Provenance, recovery, and outputs

The manifest identity covers the exact source hashes, model revision,
decomposition-bank hash, prompt renderers, factor design, candidate-token audit,
score definition, thresholds, evaluation limit, and code hash.

Outputs are independent:

```text
route_decomposition-<attempt>/
├── manifest.json
├── notebook_context.json
├── source_sessions.json
├── environment.json
├── environment_sessions.json
├── preflight/
│   ├── decomposition_bank.json
│   └── candidate_token_audit.json
└── results/
    ├── raw/scale_canary/<lesson_id>.jsonl
    └── aggregate/
        ├── summary.json
        ├── main_table.md
        └── next_stage_decision.json
```

Each model×lesson unit is atomic. Verified units are immutable and may be
adopted after a normal interruption only after full verification. A failed unit
is terminal for that attempt; fixes require a new attempt.

The model loads once, remains resident for all 24 lessons, and is released after
scoring. No adapter discovery/loading, LoRA training, layer scan, GRPO/RLVR, or
automatic next-stage action exists.

When a completed run contains an exact tie or another explicit scoring error,
the CPU-only supplementary audit may be written to an independent directory:

```bash
uv run plasticity-p0d2hcrd audit-integrity \
  --output <verified-crd-stage> \
  --audit-output <independent-audit-stage>
```

The audit independently recomputes candidate/token/score/ranking invariants,
emits every anomaly row with its complete candidate evidence, reports tie
concentration and all-ties-correct sensitivity, and hashes the immutable source
before and after reading it. It does not modify the source attempt, change the
zero-tie gate, or authorize a next stage.

[Open the CPU-only scoring-integrity audit in Colab](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/p0d2h_route_decomposition_integrity_audit/p0d2h_route_decomposition_integrity_audit_colab.ipynb)

The prospectively frozen follow-up is the
[P0-D2H-CRD BF16 precision retry](p0d2h-route-retrieval-decomposition-bf16-retry-protocol.md).

## 9. Colab and acceptance

The independent notebook is:

```text
notebooks/p0d2h_route_decomposition/
  p0d2h_route_decomposition_colab.ipynb
```

[Open P0-D2H-CRD in Colab](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/p0d2h_route_decomposition/p0d2h_route_decomposition_colab.ipynb)

Checkout, exact-source verification, bank compilation, balance/leakage audit,
and candidate-token audit are CPU preflight. The first CUDA model load appears
only in the explicitly titled formal-scoring cell.

Acceptance requires focused tests, the full repository suite, Ruff, notebook
regeneration, real-tokenizer CPU audit where the cached immutable model is
available, Python compilation, and `git diff --check`.

## 10. Claim boundary

CRD can distinguish performance of an isolated textual routing step, an
isolated external-memory retrieval step, and their composition on one fixed
model and lesson cohort. It cannot identify internal neural circuitry, prove
model scale as the unique cause, retroactively pass P0-D2H-CAL-FC, validate a
LoRA placement, or authorize training.
