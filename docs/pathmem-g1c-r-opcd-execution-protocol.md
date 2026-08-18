# PathMem G1-C R-OPCD execution protocol

This protocol freezes and executes the parameter-consolidation qualification
planned by G0-v2. The immutable planning bundle remains read-only. Execution
uses a separate namespace, authorization artifact, manifest, checkpoint tree,
evaluation rows, summary, and report.

## Frozen recipe

- Base model: `Qwen/Qwen2.5-1.5B-Instruct` at revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- Precision: NF4 with double quantization and BF16 compute; CUDA and native BF16
  support are mandatory.
- Side memory: one isolated full-layer `q_proj`/`v_proj` LoRA module for each
  `(item, terminal_state)` unit; rank 16, alpha 32, dropout 0, no bias.
- Optimization: AdamW at `1e-4`, betas `(0.9, 0.999)`, epsilon `1e-8`, zero
  weight decay, constant schedule, no warmup, and gradient norm cap 1.0.
- Exposure: 48 optimizer steps per unit. The four teacher/student prompt pairs
  are used round-robin with one on-policy student rollout per step.
- Rollout: sampling at temperature 0.8 and top-p 0.95, with 1–8 new tokens.
- Loss: equal-weight forward KL on the on-policy response tokens and forward KL
  over the exact four forced-choice action sequence scores, both at temperature
  1.0.
- Context limit: 512 tokens for training and evaluation.
- Persistence: immutable safetensors-plus-optimizer checkpoints every eight
  steps; the fixed step-48 adapter is final. No outcome-based selection occurs.

The 24 modules are each trained exactly once. A retention schedule does not
retrain a module: it applies the planned 12 subsequent modules through an exact
key router, records all 288 applications, and evaluates the anchor afterward.
If two applications address the same non-anchor key, the later application
replaces the earlier route while both applications remain in the audit record.

## Authorization boundary

All CLI commands require the verified G0-v2 bundle and frozen G0-v1 parent. The
safe `inspect` command writes nothing and performs no model load. Before GPU
work, an external human approval must exactly match the generated template and
is immutably adopted into the output root. It binds:

- source manifest, source plan, and parent identities and hashes;
- the complete recipe and recipe hash;
- Git revision and all execution/source/dependency file hashes; and
- the absolute output directory.

The only true permissions are G1-C training and G1-C GPU inference. P0, path
contrast, kill/reserve access, RL control, and automatic hyperparameter search
are explicitly false. A mismatch or added field rejects the approval.

## Lifecycle

Use the generated notebook in `notebooks/pathmem_g1c_execution`, or invoke the
same CLI directly:

```bash
uv run python -m plasticity_placement.pathmem_consolidation_exec inspect \
  --bundle /path/to/g0-v2/bundle \
  --g0-v1-manifest notebooks/pathmem_g1_p0/protocol_snapshot/experiments/g0/manifest.json \
  --output /path/to/g1c-run

uv run python -m plasticity_placement.pathmem_consolidation_exec \
  authorization-template \
  --bundle /path/to/g0-v2/bundle \
  --g0-v1-manifest notebooks/pathmem_g1_p0/protocol_snapshot/experiments/g0/manifest.json \
  --output /path/to/g1c-run
```

Complete the template with `decision=approved`, a responsible human identifier
in `approved_by`, and a timezone-aware ISO-8601 `approved_at`, then run, in
order, `adopt-authorization`, `preflight`, repeated bounded `train`, repeated
bounded `evaluate`, `aggregate`, and `verify` commands. Training/evaluation
accept `--max-units`; `--unit-id` selects one planned unit for recovery.

Preflight resolves and records the exact model, tokenizer, package, GPU, prompt,
candidate-tokenization, rollout-headroom, and privileged-teacher identities.
Training cannot start unless the teacher gate passes. An interrupted unit
resumes from its latest verified checkpoint. Evaluation cannot start until all
24 final adapters verify.

## Evaluation and stopping

Each unit produces 62 rows, for an exact 1,488-row run:

- 4 base-immediate and 4 activated-immediate rows;
- 4 retained and 14 retained-core rows after the 12 applications;
- 12 base-control and 12 routed-control locality rows;
- 4 access-off and 4 restored-base rollback rows; and
- 4 wrong-swap rows.

Aggregation rejects missing, duplicate, non-finite, misrouted, or unexpected
rows before calculating the frozen G1-C metrics and gate. `verify` is read-only:
it rechecks source, authorization, environment, prompts, every adapter and row,
then regenerates both the summary and Markdown report byte-for-byte. Passing
G1-C does not authorize P0; P0 requires a separate implementation and review.
