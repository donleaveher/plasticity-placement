# P0-D2H-CRD Strict-FP32 Audit and Conservative Tie Policy

## 1. Status and scope

- **Analysis type:** post-hoc mechanistic precision audit plus supplementary
  conservative tie-policy analysis.
- **Source runs:** the immutable completed NF4 run
  `p0d2hcrd-route-decomposition-8527b64f36` and BF16 run
  `p0d2hcrd-route-decomposition-fd812a9314`.
- **Scoring scope:** only the five structurally valid finite exact-tie rows are
  rescored in strict FP32; the completed 1,536-row source matrices are never
  rewritten.
- **Training status:** forbidden.

The NF4 and BF16 scoring-integrity audits independently verified distinct
candidate strings, continuations, token sequences, persisted token
log-probabilities, sums, means, ranks, and statuses. Their three and two exact
ties have no overlapping probe ID, but all occur in the combined
procedure-recovery endpoint with target slot B and slot-content order 1.

## 2. Strict-FP32 mechanistic intervention

The audit reconstructs each prompt from the frozen decomposition bank and
requires the reconstructed prompt hash, candidate order, expected candidate,
and token IDs to match its persisted source row. It then loads the exact model
and revision:

```text
Qwen/Qwen2.5-1.5B-Instruct
989aa7980e4cf806f80c7fef2b1adb7bc71aa306
```

The audit requires:

- all model parameters and forward logits to be `float32`;
- CUDA execution with TF32 disabled for CUDA matmul and CuDNN;
- PyTorch float32 matmul precision `highest`;
- no autocast, quantization, adapter, generation, or candidate-order tie-break;
- the unchanged candidate-only sum-logprob definition.

This is a mechanistic audit of whether BF16 forward rounding creates the
observed score plateaus. It is not a new confirmatory accuracy estimate.

## 3. Conservative exact-tie interval

The independent `aggregate-v2` analysis never changes the frozen v1 aggregate.
For every endpoint it reports:

```text
lower accuracy = correct unique-top decisions / all decisions
upper accuracy = (correct unique-top decisions
                  + finite ties containing the expected candidate) / all decisions
```

Finite exact ties are valid ambiguity records only when every candidate/token/
score/rank/status structural invariant passes. Candidate-token mismatch,
non-finite scores, structural inconsistency, or another explicit scoring error
remains fatal. No deterministic or candidate-order tie-break is used.

A diagnostic status is robust only when the lower and upper endpoint
accuracies imply the same existing component diagnosis. Otherwise the result is
`tie_interval_ambiguous`.

For the completed BF16 source, this analysis is explicitly supplementary and
post hoc. A future confirmatory use must freeze this policy and analysis code
before evaluating a new immutable held-out or counterbalanced attempt.

## 4. Governance

Both outputs must retain:

```text
source_gate_status_changed = false
training_complexity_review_eligible = false
automatic_training_started = false
automatic_narrow_scan_started = false
```

Neither a resolved FP32 tie nor a bound-robust route diagnosis directly creates
eligible model IDs. Training still requires a separate, prospectively frozen
training-complexity review and experiment protocol.

## 5. Colab

[Open the strict-FP32 audit and conservative tie analysis in Colab](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/p0d2h_route_decomposition_precision_audit/p0d2h_route_decomposition_precision_audit_colab.ipynb)

Use an L4, A100, or another CUDA GPU with enough memory for the 1.5B model in
FP32. The notebook uses a separate checkout, code lock, and Drive namespace,
verifies both source trees before and after analysis, loads the FP32 model once,
and contains no training execution path.
