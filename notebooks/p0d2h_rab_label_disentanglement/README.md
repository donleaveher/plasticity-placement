# P0-D2H frozen RAB receipt/slot-label disentanglement audit

[Open directly in Google Colab](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/p0d2h_rab_label_disentanglement/p0d2h_rab_label_disentanglement_colab.ipynb)

This notebook runs the inference-only audit described in
[`docs/p0d2h-rab-label-disentanglement-protocol.md`](../../docs/p0d2h-rab-label-disentanglement-protocol.md).
It uses an explicit canonical/crossed receipt-to-slot codebook to separate receipt-token A/B from
selected Slot A/B while retaining display-order and candidate-order controls.

Only edit the dedicated controls cell. Planning is the safe default. Authorization and GPU
inference require separate passes and remain disabled by default. The audit never trains,
reclassifies RAB/RABC, or authorizes the 1/4/8 experiment.

Regenerate the checked-in notebook with:

```bash
uv run python notebooks/p0d2h_rab_label_disentanglement/build_p0d2hrabx_notebook.py
```
