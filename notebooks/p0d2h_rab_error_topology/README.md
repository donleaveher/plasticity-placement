# P0-D2H RAB paired error-topology diagnostic

[Open directly in Google Colab](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/p0d2h_rab_error_topology/p0d2h_rab_error_topology_colab.ipynb)

This notebook performs a deterministic CPU-only analysis of the exact completed RAB OFF/ON
records. It reports cell transitions, four-cell unit error categories, paired score margins,
factor slices, and action bias. It never reloads the model, runs inference, trains, reclassifies
RAB, or authorizes 1/4/8.

The user-editable `RUN_PLAN` and `RUN_DIAGNOSTIC` switches are isolated in one dedicated cell.
Planning is the safe default, and execution is a separate pass. This post-hoc diagnostic does not
need an approval identity or authorization artifact.

Regenerate with:

```bash
uv run python notebooks/p0d2h_rab_error_topology/build_p0d2hrabd_notebook.py
```
