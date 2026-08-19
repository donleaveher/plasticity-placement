# P0-D2H-CBR-v2 corrected matched-budget Colab

[Open the standalone CBR-v2 notebook directly in Google Colab](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/p0d2h_cbr_v2/p0d2h_cbr_v2_colab.ipynb)

Use this notebook only after CBR-v1 `verify` succeeds and returns
`aggregate/summary.json`. It contains only the CBR-v2 lifecycle:

1. plan the exact parameter-budget-matched experiment and import six immutable full-depth
   controls;
2. authorize the frozen plan;
3. train only the six corrected `late_matched` adapters;
4. inspect all 12 training-unit states;
5. inspect and authorize resumable evaluation;
6. evaluate all 12 units with immutable 64-prompt shards;
7. aggregate and verify the complete CBR-v2 artifacts.

The notebook deliberately does not expose the legacy monolithic `RUN_EVALUATE_V2` gate. A Colab
reconnect can rerun `RUN_EVALUATE_RESUMABLE_V2=True`; validated completed shards are skipped.

The corrected experiment uses `pipeline-cbr2r1` and pins plan/train to the resumable-aware code
revision. The earlier `pipeline-cbr2` attempt may contain only a revision lock from a failed plan;
leave it untouched. No CBR-v1 stage needs to be rerun.

Edit only the dedicated controls cell and enable exactly one gate per pass. The default is
`RUN_PLAN_V2=True`; all later state-changing gates are disabled. `APPROVER` is a human or
responsible-party identifier, never a credential.

Regenerate the checked-in notebook with:

```bash
uv run python notebooks/p0d2h_cbr_v2/build_p0d2hcbr_v2_notebook.py
```

No gate authorizes additional training, checkpoint selection, hyperparameter search, or the
1/4/8 mappings-per-adapter experiment.
