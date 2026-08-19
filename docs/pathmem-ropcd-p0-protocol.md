# PathMem R-OPCD-specific P0 protocol

This is the separately frozen engineering smoke permitted for implementation
review by the passing G1-C R-OPCD qualification. It does not reuse the legacy
reset-LoRA P0 authorization or artifacts.

## Operator and DAG

- Model and recipe are exactly the G1-C-qualified Qwen2.5-1.5B revision,
  NF4/BF16 precision, rank-16 full-layer q/v LoRA, dual forward-KL objective,
  and 48 optimizer steps per consolidation.
- Four `smoke` items use one seed and the endpoint-equivalent `ABA/BAA` and
  `BAB/ABB` histories.
- Each item has its own deterministically seeded immutable LoRA root, shared by
  all paths for that item. Each item then has ten core DAG nodes plus one
  balanced technical duplicate: four roots, 44 isolated side-memory units, and
  2,112 total optimizer steps.
- A child starts from the exact immutable parent adapter. AdamW and its scheduler
  reset at every semantic consolidation boundary. Each node has its own
  namespace; deterministic routing selects only the evaluated final module.
- Matched occurrences of the same event block use the same step seed namespace
  and 48-step prompt exposure. On-policy responses remain dependent on the
  current parent state.
- Probe-panel order and the dependency-respecting unit execution order are
  separately frozen by `panel_seed=20260814` and
  `execution_order_seed=20260817`; neither changes training-pair order.

## Planning and authorization

The CPU plan transitively verifies the old G1-C run from its recorded Git
revision, source-file hashes, authorization, source bundle, preflight, 24
adapters, 1,488 evaluation rows, regenerated summary, and report. It never
rewrites or reauthorizes that run.

The immutable P0 plan has every execution permission false. A separate external
human approval binds the plan, G1-C handoff, exact recipe, implementation hashes,
Git revision, and output directory. Its only positive permissions are P0 smoke
training, P0 smoke inference, and the two declared smoke path comparisons. P1,
kill/reserve access, learned routing, RL control, and automatic search are false.

## Evaluation and G2 gate

The frozen matrix contains 2,168 rows:

- the 1,848 no-memory, P-latest, base, parametric, rescore, access-off,
  external-latest, ICL-history, Both, and technical-duplicate rows;
- 256 paired base/parametric unrelated rows; and
- 64 separate path-qualification rows.

Endpoint Jensen–Shannon defects use only the original 14 core probes. The four
qualification probes report every path write decision and all eight joint path
passes without filtering failed paths. Unrelated rows report the separate
locality axis.

G2 passes only when provenance and the complete matrix verify, external and
technical-duplicate upper 95% defect bounds are below 0.002 nats, rescore and
adapter-disable deltas are at most `1e-5`, external final-state prompts are byte
identical, all rows are finite/valid/untied/untruncated, and no order/cache or
lineage contamination is detected. P0 output is engineering evidence only; it
cannot establish a scientific path result, change `epsilon_JS`, or authorize P1
without another implementation review and human decision.

## Post-execution qualification-label amendment

The first aggregate implementation treated the intentional
`path_qualification.obsolete_action=null` sentinel as the candidate string
`"None"`. This is an analysis defect, not a missing score or incomplete unit.
The frozen recovery rule requires exactly four qualification rows and fourteen
matching `parametric_path` core rows per `item × path`, recovers the single
unambiguous current/obsolete label pair from those core rows, and checks that
both labels occur in every qualification candidate set. It never uses outcome
probabilities to choose labels and never filters a failed path.

An already completed, commit-bound run is repaired only through the separate
CPU analysis-repair lifecycle in
`notebooks/pathmem_ropcd_p0_repair/pathmem_ropcd_p0_analysis_repair_colab.ipynb`.
That lifecycle revalidates the recorded Git objects and all source artifacts,
requires its own exact human approval, and writes a new repair manifest,
summary, and report outside the source run. It cannot train, load a model,
rescore candidates, authorize P1, or modify the original authorization,
manifest, adapters, checkpoints, or 2,168 result rows. G2 remains undecided
until the separate repaired aggregate passes regeneration verification.
