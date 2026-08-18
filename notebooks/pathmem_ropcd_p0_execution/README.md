# PathMem R-OPCD-specific P0 Colab

[Open directly in Google Colab](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/pathmem_ropcd_p0_execution/pathmem_ropcd_p0_execution_colab.ipynb)

This notebook consumes the immutable G0-v2 bundle and the already passing
`g1c-r-opcd-r1` run. It first writes and verifies a separate CPU plan, then
requires a new exact human authorization before any P0 GPU work.

Run nine separate passes by changing only the dedicated control cell:

1. `RUN_INSPECT=True`;
2. `RUN_PREPARE_PLAN=True`;
3. `RUN_VERIFY_PLAN=True`;
4. `RUN_AUTHORIZE=True` with a non-empty human `APPROVER`;
5. `RUN_PREFLIGHT=True` on an NVIDIA GPU with native BF16;
6. repeat `RUN_TRAIN=True` until all 44 units are trained;
7. repeat `RUN_EVALUATE=True` until all 44 units are verified;
8. `RUN_AGGREGATE=True`; and
9. `RUN_VERIFY=True`.

Keep `USE_GOOGLE_DRIVE=True`, `G1C_RUN_LABEL="g1c-r-opcd-r1"`, and one fresh
P0 `RUN_LABEL` throughout. Leave `TARGET_UNIT_ID` empty during normal ordered
execution. A recovery selector is an exact plan ID such as
`p0r:pmv1-smoke-01:seed-41:ABA`.

The new authorization covers only the four-item engineering smoke, its R-OPCD
training/inference, and the two declared smoke path comparisons. It never
authorizes P1, kill/reserve access, learned routing, RL, or HPO.

Regenerate the notebook with:

```bash
uv run python notebooks/pathmem_ropcd_p0_execution/build_notebook.py
```
