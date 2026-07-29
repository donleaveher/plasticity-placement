# P0-D2H-CRD Colab

[Open the notebook in Colab](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/p0d2h_route_decomposition/p0d2h_route_decomposition_colab.ipynb)

This notebook runs the independent base-only route/retrieval decomposition. It
reads the exact completed P0-D2H-CAL-FC result and its transitive source chain
without modification, then writes only below
`plasticity-p0d/hard-probe-route-decomposition/v1/`.

The frozen matrix is 24 lessons × 64 decisions: 384 two-candidate route-only
rows, 384 four-candidate retrieval-only rows, and 768 four-candidate combined
rows. That gives 1,536 decisions and 5,376 candidate sequences.

Cells 1–5 are source, bank, leakage, and candidate-token preflight. The first
model load onto CUDA occurs in the explicit formal-scoring cell. The result is
diagnostic only; no adapter, training, narrow-scan, or automatic next-stage path
is present.
