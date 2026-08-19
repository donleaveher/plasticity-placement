# P0-D2H-CAL Invalid-Output Audit

[Open in Colab](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/p0d2h_invalid_audit/p0d2h_invalid_output_audit_colab.ipynb)

This CPU-only notebook reads the complete verified P0-D2H-CAL `pipeline-c1`
results and writes deterministic supplementary analysis under
`plasticity-p0d/hard-probe-calibration-analysis/v1/`.

It does not rerun inference, mutate source raw rows, replace strict accuracy, or
change frozen next-stage eligibility.
