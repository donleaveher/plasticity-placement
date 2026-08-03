# P0-D2H-CPR q1 qualification recovery Colab

[Open directly in Google Colab](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/p0d2h_cpr_qualification_recovery/p0d2h_cpr_qualification_recovery_colab.ipynb)

This GPU notebook is only for the cpr2 q1 failure that occurred after all locked-panel
inference completed but before any result artifact was published. It reuses the exact immutable
cpr2 adapter and permits one externally approved q2 qualification under the corrected analysis
schema. It does not plan, authorize, or train an adapter and never authorizes 1/4/8.

The default is inspection-only. Set a real `APPROVER` and `RUN_RECOVERY = True` to run q2 once.

Regenerate the notebook with:

```bash
uv run python notebooks/p0d2h_cpr_qualification_recovery/build_p0d2hcpr_recovery_notebook.py
```
