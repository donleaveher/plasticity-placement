# G1-C consolidation implementation plan

## Goal

Write the routed on-policy context distillation (R-OPCD) decision into the
authoritative PathMem documents and implement the first, CPU-verifiable slice:
the versioned G0-v2 contract plus deterministic G1-C consolidation-plan
compilation. The existing G0-v1 protocol snapshot remains immutable.

## Scope

- Document the fast external memory -> on-policy consolidation -> routed
  parametric side-memory architecture.
- Separate consolidation (R-OPCD) from any later RLHF/RLVR controller.
- Register the G1-C qualification thresholds and stopping rule.
- Implement typed/versioned G0-v2 and G1-C contracts.
- Compile and verify a deterministic, CPU-only consolidation plan.
- Add tests and user-facing CLI documentation.

## Out of scope

- GPU model loading or optimization.
- Running OPD, RLHF, DPO, or LoRA training.
- P0 path-contrast, ABA/BAA comparisons, or kill-split evaluation.
- Editing `notebooks/pathmem_g1_p0/protocol_snapshot` or rewriting G0-v1.

## Steps

- [completed] Audit authoritative documents and existing execution code.
- [completed] Update the mainline research documents with the R-OPCD/G1-C design.
- [completed] Implement G0-v2 contract and deterministic G1-C plan compiler.
- [completed] Add CLI surface, generated artifact checks, and unit tests.
- [completed] Run focused and full verification; review the diff.

## Acceptance criteria

- G0-v1 files and frozen snapshot are byte-for-byte untouched by the diff.
- G0-v2 names the new operator and contains all G1-C threshold families.
- The plan compiler is deterministic and rejects invalid/unsafe contracts.
- Generated plans explicitly prohibit P0/path contrast before G1-C passes.
- Documentation and code agree on gate names and threshold values.
- Relevant tests pass.
