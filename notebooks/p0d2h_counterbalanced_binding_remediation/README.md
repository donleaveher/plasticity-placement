# P0-D2H-CBR-v1 Colab

[Open directly in Google Colab](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/p0d2h_counterbalanced_binding_remediation/p0d2h_counterbalanced_binding_remediation_colab.ipynb)

The notebook defaults to the preregistration-only plan pass. Inspect the frozen banks and
authorization template, then enable exactly one later lifecycle stage in the dedicated controls
cell. Training and evaluation can run the complete matrix or one explicit unit. A completed unit
is reused; an incomplete claimed unit is never silently retried.

Regenerate the notebook with:

```bash
uv run python \
  notebooks/p0d2h_counterbalanced_binding_remediation/build_p0d2hcbr_notebook.py
```

No CBR outcome automatically authorizes additional training or 1/4/8 mappings.

The original notebook remains the immutable CBR-v1 training workflow. The separately generated
budget-recovery/corrected notebook first authorizes a descriptive evaluation of the observed 7.14%
budget deviation, then creates CBR-v2 with six imported full-depth controls and six corrected late
runs on explicit layers 20–27 at rank 28.
