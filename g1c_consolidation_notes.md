# G1-C consolidation notes

## Fixed decisions

- Operator: Routed On-Policy Context Distillation (R-OPCD).
- Memory lifecycle: external episodic write first, selective parametric
  consolidation second, routed side-memory activation at inference.
- RLHF/RLVR is reserved for a later controller/policy gate; it is not the
  storage mechanism and is not part of G1-C.
- G1-C remains `interface_dev` qualification only.
- No path-contrast claim is authorized by G1-C.

## Target G1-C checks

- Teacher: current top-1 >= 0.95; obsolete intrusion <= 0.05; parser validity
  == 1.0.
- Immediate write: current top-1 >= 0.80; per-action minimum >= 0.70; no
  zero-performing action.
- Retention: after 12 consolidations top-1 >= 0.75; aggregate drop <= 0.05;
  per-action drop <= 0.10.
- Locality: unrelated regression <= 5 percentage points, with distributional
  diagnostics reported.
- Routing: miss rate <= 0.05; false activation <= 0.02; wrong-swap sensitivity
  must be measured.
- Rollback/isolation: access-off and restore maximum candidate deltas <= 1e-5;
  hashes, lineage, prompts, model revision, and precision are locked.

## Existing evidence

- G1 Recipe A failed parametric current accuracy, per-action accuracy, and
  unrelated locality despite passing integrity.
- G1 Recipe B also failed; increasing reset-LoRA capacity/training did not fix
  action coverage and worsened unrelated regression.
- Therefore Recipe A/B are retained as negative evidence and no Recipe C on
  the same shared reset-LoRA operator is authorized.

## Working interpretation of "first part"

Implement the protocol/plan layer before model training. This provides an
auditable contract and deterministic training/evaluation schedule that a later
Colab GPU runner can consume without silently moving gates or leaking P0 into
qualification.

## Implementation decisions

- The new namespace is `plasticity_placement.pathmem_consolidation`, outside
  `plasticity_placement.pathmem`, so the G0-v1 implementation identity remains
  unchanged.
- No `pyproject.toml` entry point was added because that file is part of the
  frozen G0-v1 identity. The module is invoked with `python -m`.
- The plan has 24 units (12 items x terminal states A/B), four privileged/
  student prompt pairs per unit, and balanced six-unit coverage per action.
- Every retention anchor gets a deterministic 12-update cross-key distractor
  schedule; every unit gets a discriminating wrong-swap module control.
- The immutable G0-v2 bundle contains only `contract.json`, `g1c_plan.json`,
  and `manifest.json` and remains planning-only.

## Verification log

- Initial focused test exposed tuple-versus-list validation before JSON
  serialization. Fixed by making unit/prompt payloads explicitly JSON-native.
- Focused tests: 6 passed.
- Focused Ruff check: passed.
- One reporting-only shell command failed because bare `python` is unavailable;
  reran it through the repository's `uv run python` environment.
- CPU `inspect`: passed; final plan ID
  `9eea51b86368aaf1b96deadd9883ddaa033c14f2f700cc8336484bd21bc6911e`.
- Full repository suite: 453 passed in 101.08 seconds.
- Full Ruff check and `git diff --check`: passed.

## Colab packaging continuation

- Added a generated CPU notebook for `inspect`, immutable `prepare`, and read-only `verify`.
- Safe default remains non-writing inspection; no training/GPU/P0/path control exists.
- Optional Drive runs lock the exact checkout revision outside the three-file bundle.
- Focused package/notebook suite: 10 passed.
- Final full repository suite after Colab packaging: 457 passed in 103.07 seconds.
- Full Ruff, compileall, notebook parity, CLI lifecycle, and diff checks passed.
