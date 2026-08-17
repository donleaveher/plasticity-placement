# PathMem G0-v2 / G1-C R-OPCD protocol

This is the first, CPU-only implementation slice of the parameter-consolidation
branch. It binds an additive G0-v2 contract to the immutable G0-v1 manifest and
compiles the exact `interface_dev` teacher/student prompts, evaluation panels,
12-subsequent-consolidation retention schedules, and routing swap controls.

It does not load a model, start training, run GPU inference, authorize G1-C or
P0, open `kill`/`reserve`, or compute a path contrast. The frozen G0-v1 snapshot
under `notebooks/pathmem_g1_p0/protocol_snapshot` is read and verified but never
modified.

Inspect the plan without writing artifacts:

```bash
uv run python -m plasticity_placement.pathmem_consolidation inspect \
  --g0-v1-manifest \
  notebooks/pathmem_g1_p0/protocol_snapshot/experiments/g0/manifest.json
```

Write and verify an immutable G0-v2 planning bundle:

```bash
uv run python -m plasticity_placement.pathmem_consolidation prepare \
  --g0-v1-manifest \
  notebooks/pathmem_g1_p0/protocol_snapshot/experiments/g0/manifest.json \
  --output artifacts/pathmem-g0-v2-r-opcd

uv run python -m plasticity_placement.pathmem_consolidation verify \
  --g0-v1-manifest \
  notebooks/pathmem_g1_p0/protocol_snapshot/experiments/g0/manifest.json \
  --output artifacts/pathmem-g0-v2-r-opcd
```

## Colab workflow

The generated CPU notebook at
`notebooks/pathmem_g1c_consolidation/pathmem_g1c_consolidation_colab.ipynb`
provides the same three lifecycle stages. Its safe default is read-only
`inspect`; `prepare` and `verify` must be selected explicitly in the sole user
control cell. Optional Google Drive persistence records the exact checkout
revision beside, but outside, the immutable three-file bundle. Reusing a run
label with another revision is rejected.

The notebook installs only the base project environment. It does not request a
GPU or expose any training, G1-C execution, P0, kill/reserve, RL-controller, or
path-contrast switch.

The next implementation slice must freeze the CUDA rollout, distillation loss,
optimizer, side-memory parameterization, and persistence recipe before formal
G1-C execution. Passing G1-C would authorize only a separately reviewed
R-OPCD-specific P0 implementation.
