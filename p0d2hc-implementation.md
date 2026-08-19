# P0-D2H-CAL Implementation Handoff

## Delivered

- `plasticity_placement.p0d2hc` package with frozen request/config schemas;
- independent `p0d2hc-manifest-v1` lesson-model recovery manifest;
- deterministic `answer_copy_oracle` prompt renderer;
- strict verified P0-D2H-R source and hard-probe provenance checks;
- CPU prompt-token audit bound into the run identity;
- one-load-per-model, CUDA-only, base-only evaluator;
- atomic per-lesson raw rows and interrupted-run recovery;
- lesson-clustered condition summaries and paired contrasts;
- model and cross-scale decision gates;
- `plasticity-p0d2hc` environment/plan/audit/run/aggregate CLI;
- generated standalone Colab notebook and independent Drive namespace;
- protocol, project-layout, notebook-index, and P0-D2H-R follow-up links.

No code path in the new CLI trains LoRA, discovers or activates an adapter, starts
a narrow scan, or starts a multi-mapping experiment.

## Frozen execution size

```text
source model only: 24 × 16 × 3 = 1,152 rows
source + canary:    2 × 24 × 16 × 3 = 2,304 rows
```

Raw files are grouped by model and lesson:

```text
results/raw/<source_model|scale_canary>/<lesson_id>.jsonl
```

## Commands

```bash
uv run plasticity-p0d2hc plan \
  --output <new-calibration-dir> \
  --source-manifest <verified-p0d2h-r-manifest> \
  --canary-model-name Qwen/Qwen2.5-1.5B-Instruct

uv run plasticity-p0d2hc audit ...
uv run plasticity-p0d2hc run ...
uv run plasticity-p0d2hc aggregate \
  --output <new-calibration-dir>
```

The Colab notebook assembles these commands and freezes exact code, source, and
model revisions.

## Recovery behavior

- complete verified units are revalidated and skipped;
- a complete raw lesson file written before interruption is verified and promoted
  to `verified` without reloading the model;
- an `evaluating` unit without a complete raw file is rerun;
- an explicit `failed` unit is immutable and requires a new attempt.

## Validation evidence

- focused P0-D2H-CAL tests: `10 passed`;
- full repository tests: `115 passed`;
- repository Ruff lint: passed;
- notebook generator equality and Python syntax: passed;
- whitespace/error diff check: passed.

The end-to-end fake backend covers the complete 2,304-row two-model matrix,
resumption without recomputation, result-tamper rejection, and the
`scale_bottleneck_supported` decision path.
