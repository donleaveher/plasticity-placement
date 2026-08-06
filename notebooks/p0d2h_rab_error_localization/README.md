# P0-D2H RAB error-localization reader

[Open directly in Google Colab](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/p0d2h_rab_error_localization/p0d2h_rab_error_localization_colab.ipynb)

This notebook performs a deterministic CPU-only localization of the exact completed RAB topology
result. It reports exhaustive factor-level transition counts, unit-taxonomy migrations,
transition-conditioned score margins, and all 48 ranked unit records. It never reloads a model,
runs inference, trains, reclassifies a historical result, or authorizes 1/4/8.

The user-editable `RUN_PLAN` and `RUN_LOCALIZATION` switches are isolated in one dedicated cell.
Planning is the safe default and localization is a separate pass. No approval identity or
authorization artifact is required.

Regenerate with:

```bash
uv run python notebooks/p0d2h_rab_error_localization/build_p0d2hrabdl_notebook.py
```
