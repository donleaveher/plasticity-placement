# P0 route-remediation paired audit

This Colab runs the independent CPU-only P0 follow-up audit for the completed
route-remediation pilot.

It:

- verifies preregistration, authorization, code-revision, adapter, raw-result,
  aggregate, frozen forced-choice, NF4 base-CRD, and both historical experiment
  code-lock identities;
- joins all 1,152 forced-choice and 1,536 CRD rows base-to-adapter;
- reports correctness transitions, expected-candidate signed-margin shifts,
  lesson-clustered bootstrap intervals, exact McNemar summaries, tie bounds,
  endpoint-scoped factor strata, expected-vs-distractor score shifts,
  panel-position diagnostics, and matched route/retrieval/combined associations;
- writes only to an independent immutable analysis directory.

It does not load a model, change the completed run, change any gate, or
authorize training.

Because the base and adapter scores were produced in separate historical runs,
the notebook reports paired differences rather than causal adapter effects.
Same-session causal rescoring would require a new preregistered inference-only
protocol. The current artifact graph also cannot establish historical attempt
uniqueness; the recorded preregistration-hash mismatch keeps
`paper_evidence_ready=false`.

Regenerate the notebook with:

```bash
uv run python notebooks/p0d2h_route_remediation_p0_audit/build_p0d2hrr_p0_audit_notebook.py
```
