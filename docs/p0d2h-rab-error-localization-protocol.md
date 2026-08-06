# P0-D2H RAB error-localization reader protocol

## Scope

This CPU-only reader accepts the exact completed RAB paired error-topology audit. It reads the
published summary, factor slices, 192 cell records, and 48 unit records. It does not load a model,
run inference, train, reclassify RAB/RABD, or authorize the 1/4/8 experiment.

## Frozen source

The required source is run `p0d2hrabd-a82d35757b`, with status
`descriptive_error_topology_complete`, unchanged historical RAB decision, and all inference,
training, and 1/4/8 authorization flags false. Planning invokes the canonical RABD verifier,
revalidates preregistration and the frozen analysis plan, verifies every audit-manifest artifact
digest, and freezes the complete source tree hash.

## Frozen analyses

The reader publishes four exhaustive descriptive views:

1. **Cell-transition localization.** For every observed level of `pair_id`, `lesson_type`,
   `route_variant`, `orientation`, `receipt`, `expected_action`, `candidate_position`, and
   `expected_token_count`, report `C→C`, `C→W`, `W→C`, and `W→W` counts/rates plus adverse and
   rescue shares.
2. **Taxonomy migration.** Report the complete OFF→ON unit-taxonomy transition matrix, with
   dedicated rows for all base `single_action_locked` units and all adapter
   `receipt_invariant`/`partial_mixed` units.
3. **Margin-transition profiles.** Within each correctness transition, report OFF, ON, and ON−OFF
   mean/median selected-minus-counterfactual margin, positive/zero/negative margin counts, and a
   10,000-sample pair-cluster bootstrap interval for the mean margin change.
4. **Unit hotspots.** Publish all 48 units ranked first by total adverse-cell count
   (`C→W + W→W`), then by `C→W`, `W→W`, and rescue count, followed by taxonomy transition and
   per-cell margin changes. Ranking is descriptive and never drops units.

Factor levels are exhaustive and fixed by source rows. No multiplicity-adjusted subgroup claim or
new qualification threshold is introduced.

## Lifecycle

```bash
uv run plasticity-p0d2hrabdl plan \
  --output /new/rab-error-localization-rabdl1 \
  --rabd-output /completed/rab-error-topology-rabd1

uv run plasticity-p0d2hrabdl run --output /new/rab-error-localization-rabdl1
uv run plasticity-p0d2hrabdl verify --output /new/rab-error-localization-rabdl1
```

Planning and execution are separate. No approval artifact is needed because execution is a
deterministic transformation of immutable published rows.

[Open the RAB error-localization reader in Google Colab](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/p0d2h_rab_error_localization/p0d2h_rab_error_localization_colab.ipynb).

## Interpretation boundary

The reader may nominate the most concentrated descriptive barrier for a future preregistered
intervention. It cannot establish a causal subgroup effect, modify a historical decision, start
training, or authorize 1/4/8.
