# P0-D2H-CAL-FC Colab

[Open the notebook in Colab](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/p0d2h_forced_choice/p0d2h_forced_choice_colab.ipynb)

This notebook performs the independent base-only full-string forced-choice
calibration. It reads the complete verified P0-D2H-CAL run without modifying it
and writes only under `plasticity-p0d/hard-probe-forced-choice/v1/`.

Cells 1–5 are CPU/source preflight. The first model load onto CUDA occurs in the
explicit formal-scoring cell. No adapter, training, narrow-scan, or automatic
next-stage path is present.
