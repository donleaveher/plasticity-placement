# PathMem G0-v2 / G1-C R-OPCD planning Colab

[Open directly in Google Colab](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/pathmem_g1c_consolidation/pathmem_g1c_consolidation_colab.ipynb)

This notebook is the Colab entry point for the first, CPU-only R-OPCD
implementation slice. It can:

1. inspect the frozen G0-v1 parent and deterministic G1-C plan without writing;
2. prepare the immutable `contract.json`, `g1c_plan.json`, and `manifest.json`
   bundle locally or on Google Drive; and
3. detach the checkout at the recorded revision and verify every parent, source,
   artifact, identity, and regeneration hash.

The default pass is `RUN_INSPECT=True`. Edit only the dedicated user-control
cell, enable exactly one lifecycle stage, and run all cells from the top. Use the
same `RUN_LABEL` and storage choice for the prepare and verify passes.

This workflow does not install training dependencies, request a GPU, load a
model, execute consolidation, authorize G1-C/P0, expose kill/reserve examples,
or calculate a path contrast. Its immutable output is the source for the
separately gated CUDA runner in `notebooks/pathmem_g1c_execution`; preparing a
bundle here never authorizes that runner.

Regenerate the checked-in notebook with:

```bash
uv run python notebooks/pathmem_g1c_consolidation/build_notebook.py
```
