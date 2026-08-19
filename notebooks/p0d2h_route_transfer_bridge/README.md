# P0-D2H route-transfer bridge audit

[Open directly in Google Colab](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/p0d2h_route_transfer_bridge/p0d2h_route_transfer_bridge_colab.ipynb)

This notebook runs one preregistered, inference-only diagnostic on the immutable CPR-cpr2
adapter. It crosses route grammar, slot lexicon, and payload in a 2×2×2 slot-readout matrix and
adds actual-action and forced-slot-action endpoints. The training-like corner reuses exact frozen
CRD route-only prompts, and the forced-slot endpoint preserves the exact external payload. It
cannot train, retry qualification, or authorize 1/4/8.

The user-editable controls and `APPROVER` are isolated in one dedicated Colab cell. Safe defaults
run planning only; planning, authorization, zero-artifact recovery, and the single GPU audit must
be separate passes.

For the stale `running` attempt with no result artifacts, the notebook now supports a formal
one-time recovery. A safe pass first verifies eligibility and emits an external approval template.
After human inspection, enable only `RUN_RECOVER_ZERO_ARTIFACT`; this restores the same manifest to
`authorized` without scoring. The same controls cell requires a named `APPROVER` and explicit
`ORIGINAL_RUNTIME_TERMINATED = True` attestation. Then enable only `RUN_AUDIT` in a later pass. The recovery cannot
change the frozen bank, adapter, prompts, thresholds, original experiment revision, or authorize
training/1/4/8.

Regenerate with:

```bash
uv run python notebooks/p0d2h_route_transfer_bridge/build_p0d2hrtb_notebook.py
```
