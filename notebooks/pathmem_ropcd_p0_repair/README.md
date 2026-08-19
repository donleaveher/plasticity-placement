# R-OPCD P0 analysis repair Colab

[Open directly in Google Colab](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/pathmem_ropcd_p0_repair/pathmem_ropcd_p0_analysis_repair_colab.ipynb)

This CPU-only notebook repairs the aggregate-time handling of the 64 immutable
qualification rows whose frozen schema correctly records
`obsolete_action=null`. It recovers the unique current/obsolete label pair from
the matching 14-row parametric core panel for each `item × path`, without using
probabilities to choose labels.

Run four passes:

1. `RUN_INSPECT=True`;
2. `RUN_AUTHORIZE=True` with a non-empty human `APPROVER`;
3. `RUN_AGGREGATE=True`;
4. `RUN_VERIFY=True`.

Use a CPU Colab runtime for all four passes. Keep
`SOURCE_RUN_LABEL="ropcd-p0-r1"`, choose one fresh `REPAIR_LABEL`, and change
only the four stage flags between passes. The source run, its authorization,
manifest, adapters, checkpoints, and 2,168 result rows remain untouched. The
corrected aggregate is written under
`ropcd-p0-analysis-repairs/v1/<REPAIR_LABEL>` and is bound to a separate CPU
analysis-repair authorization and code revision.

Success is the final JSON object with `"passed": true`. Its `g2_gate` is the
recomputed engineering gate; it still does not authorize P1 or establish a
scientific path result.

Regenerate the notebook with:

```bash
uv run python notebooks/pathmem_ropcd_p0_repair/build_notebook.py
```
