# P0-D2H Route-Remediation LoRA Pilot

## 1. Status and scope

- **Protocol status:** implementation-ready; must be frozen by `plan` before use.
- **Results status:** not run.
- **Training status:** forbidden until a separately authored authorization is
  adopted.
- **Scientific purpose:** repair marker-to-slot/action routing in the frozen
  1.5B scale canary.
- **Out of scope:** 1/4/8 mappings-per-adapter comparisons, hyperparameter
  search, narrow scan, GRPO, and RLVR.

The pilot follows the completed scoring-integrity investigation. Strict-FP32
resolved all five finite exact ties, while the conservative interval gave the
same `route_bottleneck_supported` diagnosis at both bounds. The pilot does not
reinterpret that diagnostic as training authorization; it creates a new,
independent authorization boundary.

## 2. Immutable source

The source must pass the exact frozen-source validator already used by
P0-D2H-CRD:

```text
P0-D2H-CAL-FC run_id =
  p0d2hfc-forced-choice-f604efe6bf
manifest SHA-256 =
  89ba71ea0fde052089329e630d23c1a86652bc47a129de1921786c8e0b4dd214
model =
  Qwen/Qwen2.5-1.5B-Instruct
revision =
  989aa7980e4cf806f80c7fef2b1adb7bc71aa306
```

The validator follows the transitive P0-D2H-CAL, P0-D2H-R, and compiled-lesson
sources and verifies their manifests, hashes, matrices, and raw results. The
pilot never modifies any source artifact.

## 3. Synthetic route-remediation data

The data generator is
`p0d2hrr-counterfactual-route-bank-v1`. It creates:

| Split | Groups | Rows/group | Rows |
|---|---:|---:|---:|
| Train | 60 | 8 | 480 |
| Dev | 12 | 8 | 96 |

Every group crosses:

```text
task ∈ {route_only, route_and_copy}
target_slot ∈ {slot_a, slot_b}
slot_content_order ∈ {1, 2}
```

`route_only` asks for the selected slot. `route_and_copy` places an explicit,
balanced action token in each synthetic slot and asks the model to copy the
action from the routed slot. It does not require hidden mapping retrieval.

The audit requires:

- disjoint train/dev group IDs and marker namespaces;
- unique prompt and example IDs;
- complete eight-cell crossing in every group;
- balanced tasks, target slots, display orders, templates, and target actions;
- no frozen lesson ID, workspace ID, condition, or external note in the
  synthetic prompts.

## 4. Frozen intervention

Exactly one adapter is permitted:

```text
layers                  = full
target_modules          = q_proj, v_proj
rank                    = 8
alpha                   = 16
dropout                 = 0
learning_rate           = 2e-4
batch_size              = 4
gradient_accumulation   = 4
optimizer_steps         = 30
epochs                  = 1
training_seed           = 20260729
model loading           = NF4
checkpoint selection    = fixed final step
```

With 480 rows, batch size 4, and accumulation 4, 30 optimizer steps form one
complete deterministic data pass. There is no checkpoint, seed, or
hyperparameter selection.

## 5. Authorization boundary

`plan` writes:

```text
manifest.json
preregistration.json
authorization.template.json
preflight/
  route_train.jsonl
  route_dev.jsonl
  route_data_audit.json
  dev_token_audit.json
  forced_choice_token_audit.json
  crd_bank_audit.json
  crd_token_audit.json
```

The template has `decision = pending` and is not an approval. A reviewer must
create an approved authorization outside the output directory. The accepted
file must:

- bind the exact `preregistration_sha256`;
- name the approver and timestamp;
- allow exactly one training run and one locked external evaluation;
- explicitly disallow hyperparameter search, 1/4/8 scans, narrow scans, and
  RLVR.

The lifecycle is one-way:

```text
planned → authorized → training → trained
        → evaluating → evaluated → verified
```

Any training/evaluation failure is terminal for the attempt.

## 6. Locked evaluation

The adapter is evaluated once under the exact source-compatible precision. The
runtime rejects a GPU whose resulting precision string differs from the frozen
source precision.

The complete panels are:

1. 96 group-disjoint synthetic dev decisions, reported descriptively and never
   used for checkpoint selection.
2. 1,152 scale-canary P0-D2H-CAL-FC decisions:
   24 lessons × 3 arms × 16 probes.
3. 1,536 P0-D2H-CRD decisions:
   384 route-only, 384 retrieval-only, and 768 combined.

Both panels reuse the frozen prompt renderers, chat template, candidate-only
full-string sum-logprob scorer, candidate token audits, and deterministic
ordering. No subset-only FP32 replacement is supported. A later full-set FP32
audit would require a separate protocol and authorization.

## 7. Decision gate

The adapter first has to pass the complete original P0-D2H-CAL-FC model gate,
including oracle, external overall/category, no-write, contrast, and
scoring-integrity checks.

It must then satisfy all route-remediation guardrails:

```text
external conditional_route >= 0.75
CRD route_only             >  0.625
CRD retrieval_only         >= 1.0
CRD combined               >= 0.9688
forced-choice tie/error/non-finite rate = 0
CRD tie/error/non-finite rate           = 0
```

Possible terminal decisions:

- `route_remediation_passed_review_required`;
- `original_calibration_gate_failed`;
- `route_guardrail_failed`.

A pass creates eligibility only for human review of a separately frozen 1/4/8
experiment. The package never starts that experiment automatically.

## 8. Commands

Colab 入口：
[`notebooks/p0d2h_route_remediation/p0d2h_route_remediation_colab.ipynb`](../notebooks/p0d2h_route_remediation/p0d2h_route_remediation_colab.ipynb)。
Notebook 复用其他 P0-D2H Colab 的 Drive namespace、`code_revision.txt`
commit lock、detached checkout、流式日志和 source before/after hash 检查。
默认只有 `RUN_PLAN = True`；授权采纳、训练、locked evaluation 和 aggregate
默认均关闭，并且一次 notebook pass 最多只允许启用其中一个阶段。

Install the training dependencies and freeze the plan:

```bash
uv sync --extra train --extra colab

uv run plasticity-p0d2hrr plan \
  --output /path/to/route-remediation-attempt \
  --source-manifest /path/to/p0d2h-forced-choice/manifest.json \
  --spec configs/p0d2hrr-route-remediation-pilot-v1.json
```

After independent approval:

```bash
uv run plasticity-p0d2hrr authorize \
  --output /path/to/route-remediation-attempt \
  --authorization /independent/path/authorization.approved.json

uv run plasticity-p0d2hrr train \
  --output /path/to/route-remediation-attempt

uv run plasticity-p0d2hrr evaluate \
  --output /path/to/route-remediation-attempt

uv run plasticity-p0d2hrr aggregate \
  --output /path/to/route-remediation-attempt
```

Read-only status:

```bash
uv run plasticity-p0d2hrr status \
  --output /path/to/route-remediation-attempt
```

No experimental result has been generated by this implementation. All result
claims must come from the complete verified aggregate.
