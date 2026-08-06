# P0-D2H frozen RAB counterbalancing audit

[Open directly in Google Colab](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/p0d2h_rab_counterbalancing/p0d2h_rab_counterbalancing_colab.ipynb)

This notebook runs the authorized 1536-condition frozen-adapter audit described in
[`docs/p0d2h-rab-counterbalancing-protocol.md`](../../docs/p0d2h-rab-counterbalancing-protocol.md).
It counterbalances receipt/mapping, slot display order, and candidate order in one same-runtime
OFF→ON pass. It propagates ties without using candidate order and does not reclassify RAB,
authorize training, or authorize 1/4/8.

Only edit the dedicated controls cell. Run planning first, inspect all preflight artifacts, adopt an
external human authorization in a separate pass, and only then run the GPU audit. Safe defaults do
not authorize or execute inference.

Regenerate the checked-in notebook with:

```bash
uv run python notebooks/p0d2h_rab_counterbalancing/build_p0d2hrabc_notebook.py
```
