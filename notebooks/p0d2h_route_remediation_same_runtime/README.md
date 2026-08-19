# Same-runtime route-remediation qualification audit

This Colab performs the inference-only adapter OFF/ON experiment required
before deciding whether to design composition-preserving remediation.

It loads one base-model runtime, attaches the verified adapter once, and scores
96 external conditional-route plus 1,536 CRD prompts as adjacent OFF→ON pairs.
It writes immutable raw rows, paired analyses, source/runtime integrity checks,
and a decision report to an independent Drive directory.
`audit_manifest.json` is the last-written completion marker; reruns verify its
identity, every artifact hash, and the current frozen-source snapshot before
reusing an existing result.

The notebook performs no training, changes no historical gate, and never starts
the 1/4/8 mappings-per-adapter experiment.

Regenerate it with:

```bash
uv run python \
  notebooks/p0d2h_route_remediation_same_runtime/build_p0d2hrr_same_runtime_notebook.py
```
