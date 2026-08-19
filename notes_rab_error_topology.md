# Notes: RAB paired error topology

## Bound aggregate result

- RAB decision: `binding_not_supported`, with valid scoring integrity.
- OFF selected-binding accuracy: `0.5781` (111/192).
- ON selected-binding accuracy: `0.6406` (123/192).
- ON−OFF: `+0.0625 [0.0208, 0.1042]` (net 12 rows).
- ON causal specificity: `0.2812 [0.2083, 0.3542]`, passed.
- Canonical/swapped ON accuracy: `0.6354 / 0.6458`.
- Adapter noninferiority passed; training and 1/4/8 remain unauthorized.

## Analysis constraints

- Do not infer row topology from aggregate values.
- Freeze mutually exclusive unit taxonomy before running against real paired rows.
- Keep inferential claims limited to the already preregistered RAB analysis; new factor slices are
  descriptive.
- Preserve sum-logprob candidate scoring and pair-level clustering for overall margin summaries.

## Verification record

- Focused RAB diagnostic and notebook tests: 18 passed.
- Repository-wide Ruff: passed.
- Repository-wide pytest: passed.
- CLI smoke test for `plan/run/verify/status`: passed.

## Independent review changes

- Complete verification now rechecks both preregistration and the self-hashed analysis plan.
- Every paired/OFF/ON row is anchored to the preregistered binding probe and an exact
  12-pair × 4-variant × 4-cell balance.
- Candidate token counts must be positive integers; consistent result-only mutations are rejected.
- Reviewer recheck: approved with no remaining findings.
- Post-fix focused suite: 22 passed; post-fix repository-wide Ruff and pytest passed.
