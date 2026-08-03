# P0-D2H route-transfer bridge audit

[Open directly in Google Colab](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/p0d2h_route_transfer_bridge/p0d2h_route_transfer_bridge_colab.ipynb)

This notebook runs one preregistered, inference-only diagnostic on the immutable CPR-cpr2
adapter. It crosses route grammar, slot lexicon, and payload in a 2×2×2 slot-readout matrix and
adds actual-action and forced-slot-action endpoints. The training-like corner reuses exact frozen
CRD route-only prompts, and the forced-slot endpoint preserves the exact external payload. It
cannot train, retry qualification, or authorize 1/4/8.

Safe defaults run planning only. Authorization and the single GPU audit are separate passes.

Regenerate with:

```bash
uv run python notebooks/p0d2h_route_transfer_bridge/build_p0d2hrtb_notebook.py
```
