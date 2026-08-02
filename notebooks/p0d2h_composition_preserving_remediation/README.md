# P0-D2H-CPR-v1 Colab

Open `p0d2h_composition_preserving_remediation_colab.ipynb` in a GPU Colab runtime.
The default pass performs checkout, provenance validation, data compilation, and
preregistration only. Inspect the artifacts, then enable exactly one of authorization,
training, or qualification per later pass.

Regenerate the notebook with:

```bash
uv run python \
  notebooks/p0d2h_composition_preserving_remediation/build_p0d2hcpr_notebook.py
```

The notebook binds code, RR1, the completed same-runtime failure summary, the frozen spec,
training data, authorization, adapter, and qualification artifacts. It never automatically
authorizes 1/4/8 mappings-per-adapter.
