# P0-D2H-CRD BF16 Precision-Retry Protocol

## 1. Status and purpose

- **Analysis type:** prospective precision intervention
- **Execution status:** not yet run
- **Primary question:** do the three structurally valid CRD exact ties disappear
  when the same immutable model revision is loaded from its original BF16
  weights instead of NF4?
- **Training status:** forbidden

The completed NF4 attempt produced three exact score ties, all in `combined`.
The independent scoring-integrity audit verified that the candidate strings,
continuations, token sequences, token log-probabilities, sums, means, ranks, and
stored statuses were internally consistent. This protocol therefore changes
model precision only; it does not repair or redefine scoring.

## 2. Frozen comparison

The BF16 attempt must preserve:

- model `Qwen/Qwen2.5-1.5B-Instruct`;
- revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`;
- exact completed P0-D2H-CAL-FC source
  `p0d2hfc-forced-choice-f604efe6bf`;
- its manifest, aggregate, candidate-token-audit, and raw-tree hashes;
- the same 24 lessons, 1,536 decisions, and 5,376 candidate sequences;
- the same decomposition bank, prompt renderers, chat template, canonical
  continuation strings, and candidate-only sum-logprob score;
- the same lesson-clustered bootstrap and diagnostic thresholds;
- exact ties as invalid rows and a zero tie/error/non-finite gate.

The only intervention is:

```text
NF4 weights with BF16 compute  ->  original BF16 weights
use_4bit = true                ->  use_4bit = false
```

The runtime must record every verified unit as `evaluation_precision=bfloat16`.
Any fallback to FP32, FP16, NF4, CPU, or mixed device placement invalidates the
attempt.

## 3. Prospective interpretation

After the complete matrix is aggregated:

- zero invalid rows plus route failure and retrieval success yields the existing
  `route_bottleneck_supported` diagnostic;
- any exact tie or other scoring error retains `scoring_integrity_failed`;
- a material change beyond tie removal is reported as a precision-sensitive
  result rather than silently substituted for the NF4 estimate;
- the original NF4 attempt and its audit remain immutable and are never
  retrospectively passed.

The retry remains diagnostic-only under every outcome:

```text
training_complexity_review_eligible = false
automatic_training_started = false
automatic_narrow_scan_started = false
```

## 4. Independent namespace and recovery

The BF16 run writes only to:

```text
/content/drive/MyDrive/plasticity-p0d/
  hard-probe-route-decomposition-bf16/v1/pipelines/
    pipeline-crd-bf16-1/
```

Its code revision is locked independently. The attempt name is
`route_decomposition-crd-bf16-1`. Verified lesson units remain immutable and
may be resumed only under the identical code, source, environment, model
revision, and BF16 configuration.

Use an L4, A100, or another CUDA GPU with BF16 support. The 1.5B model fits in
BF16 on an L4; no adapter or training dependencies are loaded into the model.

## 5. Colab and acceptance

[Open the P0-D2H-CRD BF16 precision retry in Colab](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/p0d2h_route_decomposition_bf16/p0d2h_route_decomposition_bf16_colab.ipynb)

Acceptance requires:

- source identity and raw-tree verification before scoring;
- the complete CPU bank and candidate-token audits;
- a CUDA BF16 capability check before model loading;
- 24 verified BF16 units, 1,536 unique rows, and 5,376 candidates;
- aggregate and next-stage artifacts with all no-training fields false;
- focused and full repository tests, Ruff, notebook regeneration, Python
  compilation, and `git diff --check`.
