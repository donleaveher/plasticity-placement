# P0-D2H-CBR budget recovery and corrected placement Colab

[Open directly in Google Colab](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/p0d2h_cbr_budget_recovery_corrected/p0d2h_cbr_budget_recovery_corrected_colab.ipynb)

The notebook has two ordered phases:

1. externally authorize and run the complete descriptive evaluation of the 12 existing CBR-v1
   adapters, with cross-placement claims explicitly marked parameter-count-confounded;
2. create CBR-v2, import six immutable full-depth controls, train six corrected late adapters on
   layers 20–27 at rank 28, and run the matched-budget evaluation matrix.

For new CBR-v2 runs, use the dedicated
[`p0d2h_cbr_v2` notebook](../p0d2h_cbr_v2/p0d2h_cbr_v2_colab.ipynb), which includes
resumable shard evaluation and deliberately omits the legacy monolithic `RUN_EVALUATE_V2` gate.
[Open that standalone CBR-v2 notebook directly in Google Colab](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/p0d2h_cbr_v2/p0d2h_cbr_v2_colab.ipynb).

The default gate only prints the deviation authorization template. Edit only the dedicated
controls cell and enable one stage per pass. Do not start a new CBR-v2 evaluation from this
combined notebook.

Regenerate the notebook with:

```bash
uv run python \
  notebooks/p0d2h_cbr_budget_recovery_corrected/build_p0d2hcbr_recovery_notebook.py
```

No stage automatically authorizes further training, hyperparameter search, checkpoint selection,
or 1/4/8 mappings.
