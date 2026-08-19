# P0-D2H-RR same-runtime adapter OFF/ON qualification audit

## 1. Question and boundary

This inference-only experiment tests whether the observed combined-task drop is
reproduced when the fixed route-remediation adapter is the only manipulated
state. It does not train a model, alter the completed route-remediation run, or
authorize the 1/4/8 mappings-per-adapter experiment.

The causal contrast is:

```text
same process + same loaded base object + same prompt + same scorer
adapter OFF → adapter ON
```

Historical base and adapter rows were generated in separate sessions and are
therefore descriptive comparators only. This audit replaces that runtime
confound with within-object paired scoring.

## 2. Frozen source and matrix

- Model: `Qwen/Qwen2.5-1.5B-Instruct` at the immutable revision recorded by the
  verified route-remediation run.
- Adapter: the single verified route-remediation LoRA bundle and exact hash.
- Precision: the source-compatible NF4 precision recorded by the frozen run.
- Forced-choice: all 96 `external × conditional_route` rows.
- CRD: all 1,536 rows: 384 route-only, 384 retrieval-only, and 768 combined.
- Decoder/scorer: the frozen candidate-only full-string sum-logprob scorer.
- Bootstrap: 10,000 resamples, clustered by the 24 lesson IDs, seed 20260801.

The other 1,056 forced-choice rows are excluded because they do not enter the
current causal question. A later formal requalification must still rerun the
complete original calibration gate.

## 3. Runtime intervention

The base model is loaded exactly once. The adapter is attached exactly once as
a read-only PEFT adapter. For every locked prompt:

1. enter PEFT's `disable_adapter()` context and score adapter OFF;
2. leave the context and immediately score adapter ON;
3. retain both candidate score vectors and all row metadata.

No generation sampling, training, optimizer, checkpoint selection, or model
reload occurs between the two scores. The evaluator records the Python object
identities of the PEFT wrapper and wrapped base model. A fixed OFF sentinel is
rescored before and after the full matrix; its candidate set, prediction,
status, and summed log-probabilities must agree within `1e-5`.

## 4. Estimands and decision rules

For external conditional-route and each CRD endpoint, report:

- OFF and ON accuracy;
- ON-minus-OFF accuracy with a 95% lesson-cluster bootstrap interval;
- correct→wrong and wrong→correct counts;
- signed expected-candidate margin shift;
- exact McNemar test as a secondary row-level sensitivity analysis;
- finite-score tie bounds and the existing component-cell taxonomy.

The combined non-inferiority margin is frozen at 2 percentage points.

- `composition_interference_confirmed`: route-only CI is wholly above zero and
  combined CI is wholly below -0.02.
- `material_combined_regression_confirmed`: combined CI is wholly below -0.02,
  without a confirmed route-only improvement.
- `same_runtime_reading_qualification_candidate`: combined CI lower bound is
  at least -0.02, adapter conditional-route is at least 0.75, route-only is
  strictly above 0.625, retrieval-only is 1.0, and primary endpoints contain
  no ties.
- Otherwise the result is non-confirmatory or inconclusive and must be reviewed
  before another intervention.

Even the qualification-candidate outcome does not pass the historical gate:
the complete original forced-choice matrix and frozen guardrails remain a
separate required requalification after any remediation decision.

## 5. Artifacts and next action

The evaluator writes immutable OFF/ON raw rows, paired records, component
failure records, a JSON summary, a Markdown report, environment identity, and
an artifact-hash manifest to a new independent Drive directory. It hashes the
route-remediation and frozen forced-choice source trees before and after the
run and refuses publication if either changes. Raw artifacts are staged first;
`audit_manifest.json` is written last as the sole completion marker after the
final source check. Colab reuse verifies that marker, the exact requested
settings/source/code identity, every artifact hash, and the current source-tree
snapshot.

If `composition_interference_confirmed`, the next experiment is a separately
designed composition-preserving remediation. If not, inspect the paired score
shifts and runtime integrity before deciding whether remediation is warranted.
No branch automatically starts training or 1/4/8.
