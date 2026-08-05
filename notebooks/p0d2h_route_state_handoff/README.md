# P0-D2H route-state handoff audit

[Open directly in Google Colab](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/p0d2h_route_state_handoff/p0d2h_route_state_handoff_colab.ipynb)

This notebook performs a CPU-only RTB/q2 replay qualification, freezes four prompts per source
row, and then runs one inference-only same-runtime OFF→ON audit. It derives direct action,
predicted-slot chain, oracle-slot action, and wrong-slot target-action endpoints without training
or using the expected slot in the predicted chain.

All user-editable `RUN_*` controls and `APPROVER` are isolated in one dedicated cell. Planning,
authorization, and GPU execution are separate passes. The notebook cannot reclassify RTB or
authorize training/1/4/8.

Regenerate with:

```bash
uv run python notebooks/p0d2h_route_state_handoff/build_p0d2hrsh_notebook.py
```
