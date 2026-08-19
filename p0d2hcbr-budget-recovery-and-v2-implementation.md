# P0-D2H-CBR budget recovery and corrected placement implementation

## Implemented recovery

- The frozen 1% gate is unchanged.
- `budget-deviation-template` records the observed 7.142857% mismatch and binds an
  external approval to the original preregistration and recovery code hash.
- `authorize-budget-deviation` allows the 12 existing adapters to receive their
  locked evaluations without authorizing any new training.
- Unit OFF/ON results and within-placement curriculum comparisons remain available.
  Cross-placement contrasts are labeled `exploratory_parameter_count_confounded`.
- The recovered aggregate can only report
  `budget_deviation_full_matrix_complete`, never a causal placement decision.

## Implemented corrected experiment

- CBR-v2 requires the recovered CBR-v1 aggregate to be complete and verified.
- Six full-depth controls are imported by immutable adapter and metadata identities.
- Six new late adapters target explicit layers 20–27 at rank 28 and alpha 56.
- The budget identity is exact: `28 × 8 = 8 × 28 = 224` rank-layer units.
- Data, seeds, optimizer, evaluation panels, gates, and prohibition on 1/4/8 remain
  frozen.

## Colab handoff

- Builder:
  `notebooks/p0d2h_cbr_budget_recovery_corrected/build_p0d2hcbr_recovery_notebook.py`
- Generated notebook:
  `notebooks/p0d2h_cbr_budget_recovery_corrected/p0d2h_cbr_budget_recovery_corrected_colab.ipynb`
- Original experiment revision: `f4a71efe6bc717d345843b025c95fceeb6071954`.
- Recovery/corrected implementation revision:
  `04437d33de7af53d6bdd408665a3be941c440965`.

## Verification

- 393 repository tests passed.
- Focused recovery-notebook tests passed, including AST parsing, safe controls,
  lifecycle ordering, revision locks, and builder parity.
- Full Ruff, Python compileall, CLI help, and `git diff --check` passed.

This file documents implementation only. It is not an authorization artifact.
