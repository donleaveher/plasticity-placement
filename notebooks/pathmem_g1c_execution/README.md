# PathMem G1-C R-OPCD execution Colab

[Open directly in Google Colab](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/pathmem_g1c_execution/pathmem_g1c_execution_colab.ipynb)

This is the authorization-gated CUDA entry point for the frozen G1-C R-OPCD
qualification. It consumes the immutable planning bundle produced by
`notebooks/pathmem_g1c_consolidation`, trains the 24 planned side-memory units,
runs the exact routing/locality/rollback matrix, aggregates G1-C, and verifies
the complete run.

Run it as seven separate passes by editing only the dedicated control cell:

1. `RUN_INSPECT=True` (safe CPU default);
2. `RUN_AUTHORIZE=True` with a human `APPROVER`;
3. `RUN_PREFLIGHT=True` on a native-BF16 NVIDIA GPU;
4. repeat `RUN_TRAIN=True` until all 24 units are trained;
5. repeat `RUN_EVALUATE=True` until all 24 units are verified;
6. `RUN_AGGREGATE=True`; and
7. `RUN_VERIFY=True`.

Use `USE_GOOGLE_DRIVE=True` to consume the planning bundle at
`/content/drive/MyDrive/pathmem/g0-v2-r-opcd-plans/v1/g0-v2-r-opcd-plan-v1/bundle`
and persist the execution. Keep the same `RUN_LABEL` throughout. Training and
evaluation are bounded by `MAX_UNITS`; `TARGET_UNIT_ID` is an optional recovery
selector. Leave it empty for normal ordered execution. A targeted recovery uses
an exact frozen ID such as `g1c:pmv1-interface_dev-04:A`.

Authorization is exact and immutable: it binds the source manifest and plan,
recipe, source hashes, Git revision, and resolved output directory. It permits
only G1-C training and inference. P0, path contrast, kill/reserve access, RL,
and automatic hyperparameter search remain false and have no notebook control.

Regenerate the checked-in notebook with:

```bash
uv run python notebooks/pathmem_g1c_execution/build_notebook.py
```
