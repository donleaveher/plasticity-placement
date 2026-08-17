# Notes: G1-C R-OPCD Colab runner

## Repository findings

- The first-part package exposes only `inspect`, `prepare`, and `verify`; all three are CPU-only.
- The frozen parent is `notebooks/pathmem_g1_p0/protocol_snapshot/experiments/g0/manifest.json`.
- The current implementation intentionally has no `pyproject.toml` entry point because the old
  PathMem implementation identity is frozen; Colab must invoke the module with `python -m`.
- Repository notebook policy requires one user-control cell, safe defaults, fixed lifecycle order,
  generated-notebook parity, and no hidden control assignments.
- The current branch is `agent/add-lora-evaluation`; its published Colab URL uses the encoded branch.

## Frozen Colab contract

- CPU runtime only; the notebook must not install `train`/`colab` extras or call `nvidia-smi`.
- The only lifecycle stages are `inspect`, `prepare`, and `verify`, with `inspect` enabled by default.
- `prepare` writes the existing three-file immutable bundle; it grants no execution authorization.
- Optional Drive persistence uses a validated `RUN_LABEL` beneath a dedicated G0-v2 namespace.
- Checkout revision is recorded and displayed; the bundle itself locks all implementation sources.
- Existing G0-v1/G1/P0 trees are read-only and the notebook exposes no kill/reserve or path split.

## Verification evidence

- Focused G1-C package and notebook tests: 10 passed.
- Focused Ruff check: passed.
- Generated notebook was rebuilt from the checked-in builder; parity test passed.
- Real CLI `inspect -> prepare -> verify` lifecycle passed in a fresh temporary directory.
- The lifecycle reproduced plan ID
  `9eea51b86368aaf1b96deadd9883ddaa033c14f2f700cc8336484bd21bc6911e`.
- Full repository: 457 tests passed in 103.07 seconds.
- Full Ruff, Python compileall, and `git diff --check`: passed.
- Frozen G0-v1 snapshot has no working-tree modification.
