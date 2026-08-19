# PathMem R-OPCD P1 investment/kill protocol

## Purpose and source boundary

This runner implements the G3/P1 investment test for the exact R-OPCD operator
that passed G1-C and the repaired G2 engineering gate. The immutable inputs are:

- the G0-v2 R-OPCD planning bundle and frozen G0-v1 parent manifest;
- the verified 24-unit G1-C run;
- the immutable 44-unit R-OPCD P0 plan and execution;
- the separately authorized, non-mutating G2 analysis repair.

`inspect`, `prepare-plan`, and `verify-plan` reverify that complete chain. The
P0 source run and repair are read-only. The G2 handoff records
`p1_authorized=false`; it establishes implementation-review eligibility only.

## Frozen P1 design

The CPU plan contains 12 `kill` items and training seeds `41`, `42`, and `43`.
Each `item × seed` cell has ten scientific DAG nodes (`A1`, `B1`, `AB`, `BA`,
`ABA`, `BAA`, `BAB`, `ABB`, `A2`, `B2`) plus one technical duplicate. The
totals are:

- 36 isolated zero-delta root adapters;
- 360 scientific nodes and 36 technical duplicates (396 trained artifacts);
- 19,008 optimizer steps and 1,584 privileged-teacher/student prompt pairs;
- 18,840 frozen evaluation rows.

Every semantic event resumes the exact parent LoRA weights and resets AdamW and
the constant scheduler. Training seeds are nested within item; the panel,
bootstrap, and global topological-order seeds remain separate. Comparator paths
and technical duplicates bind identical final-event exposure hashes.

## Two authorization gates

The first approval is limited to four `hardware_dev` nodes. It records the exact
runtime, one-node training time, adapter bytes, peak CUDA allocation, base-model
load overhead, and back-to-back `P`/`Both` scoring latency. It cannot read
`kill` or compute a scientific path contrast. `benchmark-aggregate` freezes
`resource_profile.json`; `benchmark-verify` regenerates it.

The second approval is the formal P1 approval. Its identity includes the
verified resource-profile hash, plan, implementation, output directory, and
qualified recipe. Only this approval enables `kill` training, GPU inference,
and the frozen P1 contrast. It still denies `reserve`, P1b, P2, learned routing,
RL control, and automatic hyperparameter search.

## Formal execution order

Use the generated Colab notebook at
[`pathmem_ropcd_p1_execution_colab.ipynb`](../notebooks/pathmem_ropcd_p1_execution/pathmem_ropcd_p1_execution_colab.ipynb).
Edit only its dedicated control cell and enable one stage per pass:

1. `RUN_INSPECT`, `RUN_PREPARE_PLAN`, `RUN_VERIFY_PLAN`;
2. `RUN_BENCHMARK_AUTHORIZE`, then repeat `RUN_BENCHMARK` until 4 units verify;
3. `RUN_BENCHMARK_AGGREGATE`, `RUN_BENCHMARK_VERIFY`;
4. `RUN_AUTHORIZE`, `RUN_PREFLIGHT`;
5. repeat `RUN_TRAIN` until all 396 units are trained;
6. repeat `RUN_EVALUATE` until all 396 units are verified;
7. `RUN_AGGREGATE`, then `RUN_VERIFY`.

Evaluation deliberately fails before every planned unit has trained. During
normal execution, leave `TARGET_UNIT_ID` empty. For exact recovery, supply a
full planned ID such as `p1r:pmv1-kill-01:seed-41:ABA`; never invent a short ID.
Use a new `RUN_LABEL` whenever the locked revision, approval, GPU environment,
or output identity changes.

## Frozen analysis

For each `item × training_seed`, the runner averages 14-probe Jensen–Shannon
defects for `ABA` versus `BAA` (`D_A`) and `BAB` versus `ABB` (`D_B`), then
forms `D=(D_A+D_B)/2`. Seeds are averaged within item before the item-clustered
10,000-resample bootstrap (`seed=20260813`). The three-way rule is:

- lower 95% bound above `0.02` nats: meaningful violation;
- upper 95% bound below `0.02` nats: approximate consistency;
- otherwise: inconclusive.

The consequence endpoint `C` is reported separately and is consequential only
when a valid meaningful violation also has a lower 95% bound above `0.10`.
Access-on/off utility, usefulness asymmetry, current/stale behavior, unrelated
locality, external/ICL/hybrid controls, same-adapter rescoring, adapter disable,
write qualification, and duplicate noise remain typed secondary or integrity
outputs. Failed paths are not filtered from the intention-to-audit analysis.

The G3 summary may mark P1b implementation review as eligible only for a valid
meaningful violation. It never writes downstream authorization. P1b and P2
require new frozen plans and explicit human decisions.
