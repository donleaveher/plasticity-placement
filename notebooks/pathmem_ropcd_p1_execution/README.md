# PathMem R-OPCD P1 Colab

[Open directly in Google Colab](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/pathmem_ropcd_p1_execution/pathmem_ropcd_p1_execution_colab.ipynb)

This notebook consumes the immutable G0-v2 plan, passing G1-C run, immutable
R-OPCD P0 run, and separately repaired passing G2 aggregate. It then enforces
two distinct human approvals:

1. a four-item `hardware_dev` resource benchmark that cannot access `kill`; and
2. formal 12-item, three-seed P1 execution bound to the verified resource profile.

The safe default is CPU inspection. Edit only the dedicated control cell and
enable exactly one lifecycle flag per pass. Leave `TARGET_UNIT_ID` empty during
normal topological execution. A recovery ID has the full form
`p1r:pmv1-kill-01:seed-41:ABA`.

Formal P1 contains 396 trained artifacts, 19,008 optimizer steps, and 18,840
evaluation rows. Evaluation intentionally refuses to start until all 396 units
are trained. Neither a passing integrity gate nor a meaningful primary result
automatically authorizes P1b, P2, `reserve`, learned routing, RL, or HPO.

Regenerate the notebook with:

```bash
uv run python notebooks/pathmem_ropcd_p1_execution/build_notebook.py
```
