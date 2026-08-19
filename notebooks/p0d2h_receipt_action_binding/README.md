# P0-D2H receipt/action binding audit

[Open directly in Google Colab](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/p0d2h_receipt_action_binding/p0d2h_receipt_action_binding_colab.ipynb)

This notebook first derives the descriptive RSH oracle/wrong-receipt four-quadrant table without
GPU inference. It then freezes a two-valid-slot factorial crossing receipt A/B with
canonical/swapped action binding and runs one inference-only same-runtime OFF→ON audit.

All user-editable `RUN_*` controls and `APPROVER` are isolated in one dedicated cell. Planning,
authorization, and GPU execution are separate passes. The notebook cannot reclassify RSH or
authorize training/1/4/8.

Regenerate with:

```bash
uv run python notebooks/p0d2h_receipt_action_binding/build_p0d2hrab_notebook.py
```
