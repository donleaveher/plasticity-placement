# P0-D2H-CBR resumable evaluation Colab

[Open directly in Google Colab](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/p0d2h_cbr_resumable_evaluation/p0d2h_cbr_resumable_evaluation_colab.ipynb)

This notebook reuses the existing trained CBR-v1 adapters and replaces monolithic
unit evaluation with immutable prompt shards. A reconnect reruns the same
`evaluate-resumable` command; validated shards are skipped automatically.

Run one gate per pass:

1. `RUN_INSPECT_RESUME=True`.
2. `RUN_AUTHORIZE_RESUME=True`, with `APPROVER` set.
3. `RUN_EVALUATE_RESUMABLE=True`. Optionally select one exact unit. Rerun this same
   pass after any disconnect.
4. `RUN_AGGREGATE=True` after all 12 claims are complete.
5. `RUN_VERIFY=True`.

The default shard size is 64 prompts. Do not change it after authorization. The
workflow does not authorize training, checkpoint selection, narrow scans, or the
1/4/8 mappings-per-adapter experiment.

Regenerate the checked-in notebook with:

```bash
uv run python \
  notebooks/p0d2h_cbr_resumable_evaluation/build_p0d2hcbr_resumable_notebook.py
```
