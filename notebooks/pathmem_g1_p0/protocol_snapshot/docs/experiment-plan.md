# Experiment Plan

Status: preregistration draft v0.3

Authorization is sequential: G0 research-contract freeze → G1 interface qualification → P0 engineering smoke → P1 outcome-bearing investment test. P2 requires a meaningful P1a violation, completion/freeze of the required P1b diagnostics, and untouched confirmatory data. All numerical result cells are `TBD`.

## 1. Claim boundary

The primary experiment is an **intention-to-audit, within-operator metamorphic test**: two traces branch from the same full parent state, declare the same final memory, contain the same semantic-event and example multiset, and use the same final event and update budget. Every randomized unit remains in the primary analysis, including paths that fail to learn the terminal target. Every evaluation episode starts with volatile conversation state cleared; only the named persistent backend may carry information across episodes.

The experiment does not identify a generic “memory substrate effect.” Each optimizer/update implementation is a separate operator. Cross-backend defect magnitudes are descriptive because external notes, ICL, LoRA, and textual consolidation have different write/read interfaces and exposure costs. Memory usefulness, endpoint consistency, and lifecycle evidence are separate axes; none is substituted for another or averaged into one mechanism score.

## 2. Claim–evidence matrix

| Claim | Required evidence | Phase | Status |
| --- | --- | --- | --- |
| C0: the audit is valid | shared full parent hash, logical reducer, equal event/example multiset, same final block, common block-level minibatches, fresh-session isolation, exact token/step audit, frozen backend observability profile | P0–P1 | planned |
| C1: a named operator violates terminal consistency | non-negative paired endpoint defect with lower CI above the practical margin | P1–P2 | TBD |
| C2: violations have a new reproducible structure | semantic conflict/independence features add held-out prediction beyond update-count, terminal-loss, gradient-cosine, optimizer-kernel, and Lie-bracket baselines | P1b–P2 | TBD |
| C3: the problem matters beyond synthetic LoRA | different-family robustness plus a realistic task auditing both an existing parametric/sequential updater and a textual consolidator; at least one consequential failure is missed by standard evaluation | P2 | TBD |
| C4: a new repair is needed | simple restore, refit, modular, and existing editing baselines leave a resource-constrained gap | P3 | blocked on C3 |
| C5: a proposed repair helps | improved consistency–accuracy–compute–storage frontier on untouched data | P3 | blocked on C4 |

P1 is an investment gate, not a submission-level result. An ICLR-level claim requires C1 plus C2 or C3; a new-method paper additionally requires C4–C5.

## 3. Dataset: PathMem-v1

### 3.1 History-family unit definition

Each semantic item contains one or more frozen **history families**. A history family is the compiled protocol object below; it does not increase the number of independent observations.

```text
semantic item
  └── history family
        ├── shared logical fixture and physical root
        ├── endpoint-equivalent histories
        ├── qualification, evaluation, and control roles
        └── operator × training-seed executions
```

Every family manifest includes:

```yaml
history_family_id:
content_domain:
contract_family:
initial_fixture:
logical_reducer:
histories:
endpoint_hash:
event_multiset:
episode_roles:
controls:
observability_profile:
trace_expectation:
false_positive_pattern:
must_demonstrate:
```

Terminology is fixed: `content_domain` means fact/procedure/etc.; `history_family` means a set of controlled histories; `task_family` means a realistic scenario family; and `model_lineage` distinguishes Qwen, Llama, or another model family. None changes the semantic item as the statistical unit.

The initial PathMem history families contain:

- a nonce `context_id` and condition;
- lesson type in `{fact_mapping, procedure_recovery}`;
- mutually exclusive logical states `A` and `B`;
- counterbalanced action labels;
- two frozen four-example paraphrase blocks per state, `A1/A2` and `B1/B2`;
- a canonical external note per state; and
- qualification and audit probes that never reveal path identity.

The initial action strings are `act_n7`, `act_p3`, `act_v9`, and `act_k2`. The compiler rejects any comparator with unequal candidate-token length, prompt truncation, or mismatched example/token multisets.

### 3.2 Frozen splits

Use generator seed `20260813` and write every split before any outcome-bearing run.

| Split | Items | Balance | Permitted use |
| --- | ---: | --- | --- |
| `interface_dev` | 12 | all directed action pairs; balanced types | G1 and write-recipe calibration only; no path contrasts |
| `smoke` | 4 | 2 fact + 2 procedure; balanced cycle | engineering only |
| `kill` | 12 | all 12 directed `A→B` pairs; 6 + 6 types | P1 investment decision |
| `confirmatory` | 24 | `2 types × 12 directed pairs` | locked P2 result |
| `hardware_dev` | 4 | disjoint nonce items | runtime, memory, and API compatibility benchmarks only; never outcome evidence |
| `reserve` | 12 | all directed pairs; alternating types | untouched second model-lineage P2 slice only |

Model/optimizer recipes may be selected on `interface_dev` using only current-only write success, locality, and cost. Path effects are never computed on that split. Smoke outcomes may not change item wording, schedules, or the practical effect margin.

### 3.3 Primary paths

| Path | Event position 1 | Event position 2 | Event position 3 | Logical endpoint | Event/example multiset |
| --- | --- | --- | --- | --- | --- |
| `ABA` | `A1` | `B1` | `A2` | `A` | `A1,B1,A2` |
| `BAA` | `B1` | `A1` | `A2` | `A` | `A1,B1,A2` |
| `BAB` | `B1` | `A1` | `B2` | `B` | `B1,A1,B2` |
| `ABB` | `A1` | `B1` | `B2` | `B` | `B1,A1,B2` |

Within each pair, ordered examples inside each named block, loss-bearing tokens, total forward tokens, optimizer steps, root adapter, and final block are identical. Only the order of the first two semantic blocks differs.

Secondary four-block paths (`AABB`/`ABAB` and label swap) and independent-key paths (`ijR`/`jiR`) are authorized only in P1b/P2.

### 3.4 Randomness and common-state contract

- One immutable root LoRA initialization is created for every `item × training_seed × operator` and shared by every path in that cell.
- `root_adapter_seed = H(experiment_seed, item_id, training_seed, operator_id)`; the serialized root adapter hash is recorded.
- Ordered minibatches for an event block use `block_seed = H(training_seed, event_block_hash, occurrence_id)`. The seed does not include path name or event position.
- Probe order and path execution order use separate seeds and cannot affect training data order.
- Training seed, panel seed, bootstrap seed, and execution-order seed are distinct manifest fields.

This common-random-number design prevents a block moving from event position 1 to event position 2 from silently receiving a different internal shuffle.

### 3.5 Episode roles and controls

Each history family compiles into explicit episode roles inspired by longitudinal persistence evaluation, while retaining this project's multi-history estimand:

| Role | Purpose | Required isolation |
| --- | --- | --- |
| `root/cold` | measure the shared pre-update behavior and headroom | no history-specific persistent state |
| `update-path` | execute `h` or `h'` from the identical full root | immutable parent and event hashes |
| `qualification` | measure current-write success on disjoint probes | never used to remove primary units |
| `eval-near` | exact and paraphrase current-state audit | fresh session |
| `eval-far` | transfer, compositional, and future-suffix audit | fresh session |
| `control` | no-state, stale-only, canonical-current, wrong-key/channel, and technical duplicate | same probe, renderer, model, seed, and tool budget |

`root/cold` is not the matched memory-off baseline. For the secondary utility estimand, the same final state is evaluated with persistent-state access enabled and disabled while all other inference inputs remain fixed.

## 4. Probes and terminal qualification

Compile 30 probes per terminal state.

### 4.1 Disjoint qualification panel: 4 probes

These probes measure whether a path learned the declared current target. A path is `write-qualified` when all four top-1 decisions are current and its mean current-minus-obsolete margin is at least `ln(2)` nats. A pair is `jointly qualified` only when both paths qualify.

Qualification is a post-update outcome. The primary analysis keeps all units. Joint qualification rate and a qualified-subset sensitivity analysis are reported separately; the latter is descriptive and not called a causal estimand.

### 4.2 Frozen audit panel: 26 probes

- 2 exact held-out queries;
- 4 paraphrases;
- 4 compositional/transfer probes;
- 4 obsolete-rule conflict probes;
- 4 near-neighbor controls;
- 8 unrelated matched controls.

The 14 exact/paraphrase/transfer/conflict probes form the **core endpoint panel**. Near-neighbor and unrelated probes measure isolation/locality. Generation exact match is secondary to forced-choice distributional scoring.

P1b may develop one future-suffix family on the `kill` split after a primary violation, then freezes its generator and metric before P2. P2 applies the locked short suffix to both current endpoints on untouched data and reports the confirmatory operational-equivalence result. Future-suffix defect is always analyzed separately from current-state equivalence.

## 5. Interface qualification (G1)

Before any path outcome is computed, the selected model/operator must pass on `interface_dev`:

| Check | Frozen threshold |
| --- | ---: |
| Answer-copy top-1 | `≥ 98%` |
| Parser/finite-score validity | `100%` |
| `E-latest` current target top-1 | `≥ 95%` |
| `E-latest` obsolete intrusion | `≤ 5%` |
| `P-latest` current target top-1 | `≥ 80%` overall and `≥ 70%` per action label |
| `P-latest` unrelated accuracy regression from base | `≤ 5 pp` |
| Adapter-disable/base score agreement | maximum absolute candidate-score delta `≤ 1e-5` in one runtime |

At most two predeclared write recipes may be compared on `interface_dev`; selection uses only these qualification/cost metrics. If the 1.5B-class model fails, it may be replaced before P1 without inspecting any `kill` path output. The replacement and a different-family P2 model must be frozen in the manifest before P1 begins.

## 6. Models and update operators

### 6.1 P0 engineering model

```yaml
model_name: Qwen/Qwen2.5-0.5B-Instruct
model_revision: 7ae557604adf67be50417f59c2c2f167def9a775
role: engineering_smoke_only
```

### 6.2 P1 primary model

```yaml
model_name: Qwen/Qwen2.5-1.5B-Instruct
model_revision: 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
role: first_outcome_bearing_operator_audit
```

Resolved precision, quantization, tokenizer hash, and library lock hash are frozen per phase. `bfloat16_when_supported` is not a valid manifest value.

### 6.3 Reset-at-event LoRA operator

Initial recipe, subject only to G1 qualification:

```yaml
quantization: nf4_double_quant
compute_dtype: bf16
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.0
target_modules: [q_proj, v_proj]
layer_band: full
learning_rate: 0.0002
optimizer: AdamW
weight_decay: 0.0
batch_size: 1
gradient_accumulation_steps: 1
optimizer_steps_per_event: 16
training_max_length: 256
evaluation_max_length: 512
checkpoint_selection: fixed_final_step
evaluation_sampling: false
```

LoRA weights resume from the exact parent node, while optimizer/scheduler state resets at every semantic event. This is a reset-at-event operator, not continuous online learning.

### 6.4 Conditional P1b operators

If P1 finds a meaningful reset-operator violation, audit:

1. **carried AdamW:** LoRA, optimizer moments, scheduler, scaler, RNG, and step counter resume from the parent;
2. **memoryless SGD:** no momentum; its learning rate is frozen on `interface_dev` before any P1b path output.

These are separate treatments. If carried AdamW violates while reset AdamW does not, that is an optimizer-state contribution—not evidence that no contract failure occurred under carried AdamW.

## 7. Conditions and baselines

| Condition | Implementation | Role | Interpretation boundary |
| --- | --- | --- | --- |
| `N` | frozen base/root | reference | no added persistent write, not “no memory” |
| `access-off(path)` | evaluate the exact final path state with its persistent backend disabled | matched utility/integrity control | adapter disable for modular P; store removal for E; backend-specific |
| `E-latest` | versioned store returns only canonical current state | deterministic negative control | same-final prompt must be byte-identical |
| `ICL-history` | event demos in chronological prompt order | prompt-order control | within-ICL path contrast only |
| `P-latest` | root trained only on terminal block | write-success anchor | not exposure-matched to full path |
| `P-path` | sequential LoRA path | primary P1 treatment | infer within operator |
| `Both-path` | exact P adapter plus final external note | stale-state suppression diagnostic | no second adapter training |
| `P-bag` | one 48-step fit to the unordered example bag | segmentation/reset control; P2 | not a pure order comparator |
| `canonical-refit` | train current ledger from the same root | simple repair; P1b/P2 | report matched compute curve |
| `checkpoint-restore` | load prior full state | physical rollback upper bound | report storage/latency |
| `adapter-disable` | deactivate/switch delta | modular rollback | not semantic inverse |
| `technical duplicate` | identical parent, event, ordered minibatches, seed, and config | noise/integrity | distinct run ID only |
| `geometry diagnostics` | terminal loss, gradient cosine, bracket-based features | mechanism characterization | prediction failure does not prove semantics |

P2 additionally requires one runnable existing sequential editor/parametric-memory updater and one reproducible LLM-managed textual-consolidation updater. Selection is subject to a pre-P2 code/license audit and is frozen before confirmatory data are opened.

### 7.1 Contract × lifecycle × evidence applicability

Before execution, each `operator × contract` cell declares every update-lifecycle locus as `observable`, `intervention-accessible`, or `N/A`. Missing telemetry is never scored as zero.

| Update-lifecycle locus | Versioned/LLM-managed textual memory | Modular parametric memory | Candidate diagnostic |
| --- | --- | --- | --- |
| commit/encode | stored artifact, version diff, write trace | event data hash, loss trace, effective `ΔW`, child hash | canonical store; current-ledger refit is a whole-operator repair, not a pure LoRA commit cut |
| supersede/resolve | active/superseded/expiry metadata and resolver output | stale/current behavior only; internal resolution usually unobservable | stale purge or canonical-current store; no fictitious parametric retrieval cut |
| retrieve/surface | retrieval key/result and byte-level rendered evidence | `N/A` for explicit retrieval; adapter activation is separately logged | forced canonical retrieval/render; wrong-key/channel control |
| apply/infer | action distribution after rendered evidence | action distribution with expected adapter active | artifact delete/swap necessity perturbation; adapter access-off; current-state external override |
| maintain/future update | fresh-session survival, rollback, common suffix | parent-state lineage, rollback, common suffix | restore/rollback control; cache/index rebuild rescue; identical-future-suffix equivalence probe |

The mechanism report is a sparse vector, not an equal-weight score:

```text
commit_or_encode_ok
supersede_or_stale_suppression
retrieve_ok          # explicit textual backends; N/A when no retrieval event exists
activate_ok          # modular parametric backends; N/A when no activation event exists
apply_ok
locality_ok
future_equivalence
```

`N/A` fields remain visible in tables. A trace association is only pathway evidence; a causal-stage statement additionally requires a pre-registered intervention that changes the defect under declared held-fixed conditions.

### Null-event semantics for secondary contracts

`IDLE` performs no update and does not advance the update clock. `CLOCK_NULL` executes the frozen number of zero-loss/zero-gradient steps and advances optimizer/scheduler/RNG state. They are different operators. Latest-write and idempotence tests report both; neither is used in the primary equal-multiset contrast.

## 8. Primary endpoint and secondary metrics

For probe `q` and path `p`, obtain full-string summed log scores for the four equal-token-length actions, then normalize with softmax:

\[
\pi_{q,p}(y)=\frac{\exp s(y\mid q,p)}{\sum_{y'}\exp s(y'\mid q,p)}.
\]

Jensen–Shannon divergence uses natural logarithms:

\[
JS(\pi,\pi')=\tfrac12 KL(\pi\|m)+\tfrac12 KL(\pi'\|m),\quad m=(\pi+\pi')/2.
\]

For item `i`, average across the 14 core probes:

\[
D_{A,i}=\operatorname{mean}_q JS(\pi_{q,ABA},\pi_{q,BAA}),\qquad
D_{B,i}=\operatorname{mean}_q JS(\pi_{q,BAB},\pi_{q,ABB}),
\]

\[
D_i=(D_{A,i}+D_{B,i})/2.
\]

`D` is the sole primary endpoint and is non-negative. The practical contract tolerance is frozen at

\[
\epsilon_{JS}=0.02\ \text{nats}.
\]

Technical-duplicate and deterministic-external mean defects must have an upper 95% confidence bound below `0.002` nats; otherwise the pipeline fails. The practical margin is never changed after inspecting P0 or P1 path outcomes.

### 8.1 Secondary utility and typed diagnostics

For PathMem path `h`, let the frozen task score be mean probability assigned to the current target across the same 14 core endpoint probes:

\[
S_{i,h}^{x}=\operatorname{mean}_{q\in\mathcal P_i^{\mathrm{core}}}
\pi_{q,h}^{x}(\mathrm{current}),\qquad x\in\{\mathrm{on},\mathrm{off}\}.
\]

For realistic P2 task families, `S` is one task-native utility measure frozen before any family outcome is viewed; the default is paired downstream success rate. Define within operator

\[
U_{i,h}=S_{i,h}^{on}-S_{i,h}^{off},
\qquad
A_{\mathrm{path},i}=U_{i,h}-U_{i,h'}.
\]

`U` tests whether memory contributes useful behavior; `A_path` tests whether endpoint-equivalent histories differ in usable value. Both are signed secondary estimands and cannot replace `D`. Access-off is implemented by adapter disable for modular parametric memory and store removal for external memory, with byte-identical probes and otherwise matched inference state. Cross-backend `U` magnitudes are descriptive unless their inference interfaces and costs are explicitly matched.

After a P1 violation, a frozen **canonicalization defect-reduction** intervention at observable textual-memory locus `j` may be evaluated through

\[
R_{j,i}=D_{i,\mathrm{native}}-D_{i,\operatorname{do}(j=\mathrm{canonical})}.
\]

Each intervention declares its target locus, backend, randomization unit, held-fixed state, resource delta, and artifact count. `R_j` values are reported separately with item-clustered intervals; they are neither additive nor interpreted as proportions of the defect. `R_j>0` is defect reduction, not by itself a full rescue. Trace evidence without a positive defect-reduction contrast supports association only.

Artifact deletion uses the separate necessity contrast `N_j = S_native - S_delete(j)` with a frozen task-utility threshold; swap perturbations use a pre-registered expected target/logit shift. The future-suffix endpoint defect `D_suffix` applies the same frozen suffix to both endpoints and tests operational equivalence. Neither `N_j` nor `D_suffix` is interpreted with the canonicalization-rescue gate.

### 8.2 Practical calibration of `epsilon_JS`

`0.02` is not justified by being larger than numerical noise. Before any path run, freeze `experiments/configs/js-calibration-v1.csv`, generated analytically from four-action distributions. Representative rows are:

| Path distribution 1 `(current, obsolete, other, other)` | Path distribution 2 | JS (nats) | Current-probability shift | Current/obsolete log-odds shift | Top-1 flip? |
| --- | --- | ---: | ---: | ---: | --- |
| `(0.70,0.10,0.10,0.10)` | `(0.55,0.25,0.10,0.10)` | 0.0211 | 0.15 | 1.157 | no |
| `(0.60,0.30,0.05,0.05)` | `(0.40,0.50,0.05,0.05)` | 0.0227 | 0.20 | 0.916 | yes |
| `(0.55,0.35,0.05,0.05)` | `(0.35,0.55,0.05,0.05)` | 0.0224 | 0.20 | 0.904 | yes |

The table shows the intended scale: around `0.02` can correspond to a 15–20 percentage-point current-probability change or a current/obsolete decision reversal in simple calibrated cases. It is not a universal equivalence between JS and decisions.

Define the PathMem consequence endpoint

\[
C_i=\operatorname{mean}_{q,\,r\in\{A,B\}}
\left|\pi_{q,h_r}(\text{current})-\pi_{q,h'_r}(\text{current})\right|.
\]

A **distributional violation** requires the JS rule in Section 9. A **behaviorally consequential PathMem violation** additionally requires the lower 95% confidence bound of mean `C` to exceed `0.10`. Realistic P2 tasks freeze one task-native consequence before outcomes—default: a paired downstream success-rate gap whose lower confidence bound exceeds 5 percentage points. These are hierarchical gates, not interchangeable endpoints selected after results.

Secondary metrics:

- signed current-minus-obsolete margins `H_A` and `H_B` for directional diagnosis;
- current-target top-1 accuracy and margin;
- obsolete intrusion;
- paraphrase/transfer behavior;
- near-neighbor and unrelated locality regression;
- joint write-qualification rate;
- absolute current-target probability shift `C` and top-1 endpoint disagreement rate;
- future-suffix operational defect;
- effective `ΔW` diagnostics, never raw LoRA-factor distance;
- write/rollback time, optimizer steps, read/input tokens, peak allocation, artifact bytes, and scoring latency;
- per-path persistence utility `U`, signed usefulness asymmetry `A_path`, canonicalization defect reduction `R_j`, artifact necessity `N_j`, and future-suffix defect `D_suffix`;
- a utility–consistency report that keeps current-task utility, endpoint defect, and cost as separate columns/axes.

## 9. Statistics and three-way decision

1. Average probe scores within `item × training_seed × path`.
2. Form `D_A`, `D_B`, and `D` within `item × training_seed`.
3. Average training seeds within item.
4. Treat semantic item—not probe or seed—as the independent unit.
5. Use an item-clustered bootstrap with 10,000 resamples and seed `20260813`, retaining all paths/seeds/probes for every resampled item.
6. Report two-sided 95% percentile intervals, per-item/per-seed values, task/action strata, and leave-one-item-out stability.
7. Keep every randomized item in the primary intention-to-audit analysis.

Let `[L_D,U_D]` be the 95% interval for mean `D`:

- `L_D > 0.02`: evidence of a practically meaningful violation;
- `U_D < 0.02`: evidence of approximate consistency at the frozen tolerance;
- otherwise: inconclusive.

Statistical significance against zero is neither necessary nor sufficient for this decision. Qualified-subset, signed-margin, locality, and individual-contract analyses are secondary and explicitly labeled.

## 10. Phase P0 — engineering smoke

```text
4 smoke items × 1 training seed × reset-at-event LoRA
paths: ABA, BAA, BAB, ABB
controls: N, E-latest, ICL-history, P-latest, Both, adapter-disable
```

Per item, cache 10 trained nodes (`A1`, `B1`, `AB`, `BA`, four final paths, `A2`, `B2`) plus one independently repeated final-event duplicate: 44 trained artifacts total.

P0 passes only if:

- planned units are complete and no extra units exist;
- every root/parent/config/data/tokenizer/library hash verifies;
- event/example/token/step matching is exact and no prompt truncates;
- the same event block receives byte-identical ordered minibatches across comparator paths;
- same-adapter rescoring differs by at most `1e-5` per candidate score;
- external prompts are byte-identical within final state;
- external and technical-duplicate defect upper bounds are below `0.002` nats;
- adapter-disable restores base scores within `1e-5` in the same runtime;
- invalid, tie, error, and nonfinite rates are zero.

P0 outputs cannot establish a scientific result or change `epsilon_JS`.

## 11. Phase P1 — 12-item investment gate

### P1a: reset-at-event operator

```text
12 kill items × 3 training seeds × 10 LoRA core-DAG nodes = 360 core artifacts
+ N, matched access-off, E-latest, ICL-history, Both, restore/disable, qualification and audit panels
model: qualified Qwen2.5-1.5B or pre-P1 documented replacement
```

P1 also trains one duplicate final node per `item × seed` for the mandatory noise audit: `12 × 3 = 36` additional artifacts. Thus P1a has 396 trained artifacts in total; 360 refers only to the scientific core DAG.

Apply the three-way decision in Section 9:

- meaningful violation → run P1b and plan locked P2 confirmation;
- approximate consistency → stop the reset-LoRA phenomenon line;
- inconclusive → no claim; authorize at most one pre-specified precision expansion if its projected sample size is within the resource cap.

There is no `H_A/H_B` same-sign requirement because contract violation is direction-free. Signed contrasts are descriptive.

### P1b: explanation and semantic-structure investment

Only after a P1a violation:

- repeat the primary DAG for carried AdamW and memoryless SGD;
- add matched conflicting, same-state/paraphrase, and independent-key schedules;
- run the final-write dose curve in Section 11.1;
- record terminal loss and gradient statistics at shared parent nodes;
- freeze and run only backend-valid typed diagnostics: canonical store/render rescue and artifact delete/swap necessity tests for explicit textual state; adapter access-off/restore and canonical-refit baselines for parametric state; common future suffix as a separate operational-equivalence probe;
- freeze any proposed defect predictor before P2.

Carried AdamW and SGD require `2 × 360 = 720` additional artifacts for the primary paths if run on the full P1 grid. Conflict/no-conflict, independent-key, and canonical-refit nodes are extra and must receive their own exact count and hardware benchmark in the frozen P1b manifest. Their outcomes classify which operators violate; they do not retroactively erase P1a.

Geometry diagnostics use cross-validation within P1 only for development. The pre-registered baseline predictor uses update counts, final dose, terminal-loss difference, gradient cosine, optimizer impulse/kernel features, and Lie-bracket norm/projection. The augmented predictor adds semantic relations such as conflicting rewrite, same-state paraphrase, and independent key/scope. Before P2, freeze the model classes and the primary prediction metric. The default C2 gate is at least 10% relative held-out MAE reduction by the augmented predictor with an item-bootstrap 95% CI excluding zero improvement. Failure of a baseline predictor is reported as limited predictive coverage, not proof of a semantic mechanism.

Lifecycle diagnostics remain conditional rather than a substitute for C2/C3. A canonicalized locus may be described as rescuing the defect only when `R_j` has a 95% lower bound above zero, the intervened defect's 95% upper bound falls below `epsilon_JS`, and current-task utility/locality do not regress by more than the pre-registered task-family tolerance. Artifact necessity requires its own utility/behavior threshold and expected-direction test. A common suffix is reported only as operational equivalence. If a trace correlates with failure but its corresponding typed diagnostic fails, stop that causal interpretation.

### 11.1 Final-write dose robustness

The primary 16-step result defines one fixed-budget operator. P1b repeats only the final blocks at `{8,16,32,64}` optimizer steps while reusing verified `AB` and `BA` prefixes; both paths in every comparator receive the same final dose. The existing 16-step nodes are reused, so the three additional doses require

```text
12 items × 3 seeds × 4 final paths × 3 new doses = 432 artifacts.
```

Report `D`, `C`, write qualification, terminal loss, and cost as a dose curve. If the defect falls below the practical margin at 32 or 64 steps while write qualification improves, the claim is limited to a transient fixed-budget operator effect. A stable contract-failure claim requires a meaningful defect at two adjacent qualified doses, including at least one of 32 or 64 steps. Non-monotonicity is reportable but must replicate on P2 before mechanism claims.

## 12. Phase P2 — confirmation and relevance

P2 is authorized only after a meaningful P1a violation, the required P1b explanation/dose work, and frozen hypotheses/configuration. Approximate consistency or an unresolved P1 interval does not authorize P2 by default.

### 12.1 Untouched same-model confirmation

- 24 confirmatory items;
- 3 training seeds;
- the decisive operator(s) and paths frozen after P1;
- `P-bag` and future-suffix tests;
- no tuning of templates, thresholds, items, rank, steps, or scoring.

For one parametric operator, `24 × 3 × 12 = 864` core trained artifacts correspond to 8 primary DAG nodes, 2 current-only anchors, and 2 bag/segmentation controls per item/seed. Add one duplicate final per `item × seed`: 72 artifacts, for 936 trained artifacts before future-suffix or realistic-operator work.

The exact future-suffix count is `2 × number_of_endpoint_nodes × number_of_frozen_suffixes` and depends on the suffix family frozen after P1; it must be enumerated in the P2 manifest before authorization. Existing-updater and textual-consolidator counts depend on their APIs and are mandatory separately benchmarked P2 workloads. Therefore 864/936 are same-model parametric core/with-duplicate counts, not full-P2 totals.

### 12.2 Different model-lineage replication

Use 12 untouched `reserve` items, three seeds, and the 8-node primary DAG on a model lineage frozen before P1. Before any reserve output is viewed, the second-lineage model/operator must pass all applicable functional G1 thresholds on `interface_dev`; `hardware_dev` is used only for runtime, memory, and API compatibility benchmarking. Scaling Qwen2.5 from 1.5B to another size is a scale check, not the required family replication.

### 12.3 Realistic update operators

At least one confirmatory slice must use:

- a runnable existing sequential editor or parametric-memory updater; and
- a reproducible LLM-managed textual-consolidation updater with frozen prompts/model.

Use repeated factual edits (for example, a suitable WikiBigEdit/serial-edit subset) or executable procedure/API changes. PAST-Bench's Update families provide task-structure prototypes for fact correction, global or scoped rule migration, temporary-exception expiry, SOP patches, and recall-then-modify; use its [official implementation](https://github.com/Gen-Verse/PAST-Bench) when protocol-compatible, or create independently worded endpoint-equivalent variants with explicit provenance. Do not copy its prompts and then present them as a new benchmark. Historical-provenance questions are not endpoint-equivalent to current-state questions and are labeled separately.

### 12.4 ICLR-level positive gate

In addition to replicating C1, meeting the task-family consequence threshold, and passing the dose-robustness scope rule, P2 must establish at least one:

1. the frozen semantic-augmented predictor improves held-out error over the full pre-registered geometry/optimizer baseline stack by the C2 margin and exposes a stable conflict/independence interaction; or
2. an existing realistic updater shows a failure above its frozen task-native consequence threshold that its standard accuracy/forgetting evaluation misses.

Without one of these, the project remains an operator audit/benchmark result even if `D>epsilon_JS` replicates. Different-family replication is required robustness evidence but does not satisfy this novelty gate by itself.

## 13. Phase P3 — optional mitigation

Authorize only if P2 verifies a deployment-relevant gap after simple baselines.

Candidate classes include versioned/modular deltas, conflict-aware composition, selective canonical refit/replay, or explicit provenance with restore planning. Compare against:

- full checkpoint/snapshot restore;
- adapter disable/switch;
- current-ledger refit from the root;
- independent per-item adapters;
- standard sequential LoRA;
- the closest runnable sequential editor.

A method succeeds only if it improves the consistency–current-accuracy–locality–compute–storage–rollback-latency frontier. If simple restore/refit/modularity dominates, C1 may still stand but C4–C5 fail.

## 14. Manifest and lineage contract

Freeze and hash:

- experiment, phase, schema, and logical-reducer versions;
- history-family specification, episode-role registry, observability profile, and trace-expectation schema;
- event schema and full generated split bank;
- git revision plus dirty-tree patch hash;
- model/tokenizer revisions and tokenizer files;
- resolved precision/quantization and CUDA/library lock;
- root-adapter initialization and all parent artifacts;
- optimizer implementation/config and, when carried, optimizer/scheduler/scaler/RNG states;
- renderer, prompt template, scorer, probe bank, and external note;
- all named seeds and planned node/result sets.

### Node identity

```text
node_id = SHA256(
  experiment_schema || history_family_sha256 || logical_reducer || event_schema ||
  base_model_revision || root_adapter_sha256 || parent_state_sha256 ||
  event_block_sha256 || ordered_minibatch_sha256 || occurrence_id ||
  trainer_and_optimizer_impl_sha256 || trainer_config_sha256 ||
  tokenizer_sha256 || library_lock_sha256 || code_and_patch_sha256 ||
  resolved_precision
)
```

For carried optimizers, `parent_state_sha256` covers LoRA weights, optimizer, scheduler, scaler, RNG, and step counter. Cache only on the full identity; never cache by final state, item ID, or path label.

### Result identity

```text
result_id = SHA256(
  node_id || probe_bank_sha256 || renderer_and_prompt_sha256 ||
  external_state_sha256 || access_mode || lifecycle_trace_sha256 ||
  scorer_sha256 || evaluation_precision
)
```

Lifecycle: `planned → training → trained → scoring → verified`, with terminal `failed`. Verified artifacts are immutable; writes are atomic and resumable.

## 15. Compute and storage budget

Core-DAG counts are exact for their stated scope. Full phase totals require the mandatory duplicate, dose, future-suffix, and realistic-updater rows below; wall time is never extrapolated from 0.5B to 1.5B.

| Phase | Trained artifacts | Authorization |
| --- | ---: | --- |
| P0 | 44 | after G0 and G1 pass |
| P1a scientific core | 360 | after P0/G2 integrity pass |
| P1a technical duplicates | 36 | mandatory; total P1a trained artifacts = 396 |
| P1b carry + SGD primary paths | 720 additional | only after P1a violation; added schedules budgeted separately |
| P1b final-dose robustness | 432 additional final nodes | only after P1a violation; reuses prefixes and 16-step finals |
| P2 same-model parametric core | 864 | after meaningful P1a violation, required P1b completion, and locked hypothesis |
| P2 same-model technical duplicates | 72 | same authorization; mandatory, giving core + duplicates = 936 |
| P2 second model-lineage primary DAG | 288 | same authorization plus second-lineage G1 on `interface_dev` and four-item `hardware_dev` benchmark |
| P2 future suffix | exact formula in frozen manifest | same authorization; mandatory and based on the suffix frozen after P1b |
| P2 realistic parametric + textual updaters | exact per-adapter/API count in frozen manifest | same authorization; both mandatory and benchmarked before execution |
| Typed lifecycle diagnostics | normally 0 trained artifacts; exact scored-prompt count in manifest | conditional after a violation; canonical refit and future suffix are budgeted separately |

The earlier project measured a 0.5B/16-step adapter median of 5.38 seconds, 0.902 GiB peak CUDA allocation, and 1.044 MiB per adapter; see the local source [`p0c-confirmatory-results-and-next-experiments.md`](../../plasticity-placement/docs/p0c-confirmatory-results-and-next-experiments.md). Those values are sanity proxies only and do not price 1.5B training or forced-choice scoring.

Before P1/P2 authorization, benchmark the four `hardware_dev` items and freeze:

```text
wall_time = N_nodes × measured_seconds_per_node
          + N_scored_prompts × measured_seconds_per_prompt
          + measured_load_and_checkpoint_overhead.
```

Run one node at a time on one GPU, load each final adapter once, score `P` and `Both` back-to-back, and reuse exact prefixes. Raw verified rows, manifests, logs, and artifact hashes are retained; aggregate tables alone are insufficient.

## 16. Implementation boundary

Safe to reuse from `plasticity-placement`:

- nonce/action vocabulary and template patterns;
- completion-only loss/collator;
- LoRA parent resumption;
- adapter activation/deactivation;
- full-string candidate scorer;
- provenance and item-cluster bootstrap patterns.

Must be replaced or extended:

- history-family compiler, episode-role validator, backend observability adapter, and lifecycle-trace schema;
- single-state lesson schema and one-note compiler;
- one-adapter-per-lesson runtime;
- linear `lesson::seed` manifest;
- fixed N/E/P/B analysis schema;
- one-shot external injector;
- generation-only evaluator;
- parent resumption without full state/config verification.

Use a separate path-aware namespace. Frozen artifacts from the former project remain unchanged.

## 17. Result templates

### Primary endpoint

| Phase / operator | Items | `D_A` | `D_B` | mean `D` [95% CI] | mean `C` [95% CI] | JS decision | Consequential? | Joint qualification | Complete? |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |
| P1 / reset AdamW | 12 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| P1b / carried AdamW | 12 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| P1b / SGD | 12 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| P2 / confirmed operator | 24 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| P2 / second model lineage | 12 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### Utility–consistency profile

| Phase / operator | Path | score on | score off | `U` [95% CI] | `A_path` [95% CI] | endpoint `D` [95% CI] | Cost | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| P1 / reset AdamW | `h` | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| P1 / reset AdamW | `h'` | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| P2 / textual consolidator | `h` | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| P2 / textual consolidator | `h'` | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### Direction and consequences

| Operator | `H_A` | `H_B` | Current top-1 | Obsolete intrusion ↓ | Transfer | Locality regression ↓ | Future-suffix defect |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E-latest | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ICL-history | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Reset AdamW LoRA | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Carried AdamW LoRA | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SGD LoRA | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Text consolidator | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### Contract profile

| Operator | Content domain | Supersession | Repetition | Composition | Reversibility | Worst-family decision |
| --- | --- | --- | --- | --- | --- | --- |
| Reset AdamW LoRA | fact | TBD | TBD | TBD | TBD | TBD |
| Reset AdamW LoRA | procedure | TBD | TBD | TBD | TBD | TBD |
| Text consolidator | realistic update slice | TBD | TBD | TBD | TBD | TBD |

Each contract cell records `violation`, `approximate consistency`, `inconclusive`, or `not run`; no overall mean can overwrite the per-contract decision.

### Lifecycle evidence and typed diagnostics

| Operator | Contract | Lifecycle locus | Observable? | Trace signal | Diagnostic type | Operation | Estimand | Native value | Diagnostic value | Estimate [95% CI] | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| Text consolidator | TBD | commit/supersede/retrieve/render/apply/maintain | TBD | TBD | canonicalization defect reduction / artifact necessity | TBD | `R_j` / `N_j` | TBD | TBD | TBD | TBD |
| Parametric updater | TBD | encode/optimize/persist/activate/infer/maintain | TBD | TBD | access control / repair / suffix equivalence | TBD | `U` / repair frontier / `D_suffix` | TBD | TBD | TBD | TBD |

### Integrity

| Phase | Root/parent hashes | Block RNG match | Token/step match | External `D` UCB | Duplicate `D` UCB | Rollback tolerance | Invalid/nonfinite | Complete DAG |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |
| P0 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| P1 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| P2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

No experimental result has been generated here. Every `TBD` must be filled from verified artifacts or protocol-compatible public results.

### Planned main figures

1. **Utility–consistency frontier:** current-task utility on one axis, endpoint defect on the other, with cost encoded separately. This exposes the degenerate “consistent because both paths failed” region.
2. **Contract × content heatmap:** per-cell three-way decisions rather than a pooled overall score.
3. **Lifecycle rescue heatmap:** item-paired `R_j` intervals for valid backend-specific interventions; `N/A` remains visible.
4. **History-family path DAG and trace case:** shared root, endpoint-equivalent branches, identical endpoint hash, lifecycle artifacts, current/stale distributions, and a counterfactual cut.
