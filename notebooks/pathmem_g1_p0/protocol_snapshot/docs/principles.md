# Principles and Formalization

## 1. Memory as a state-transition system

For backend `a`, define physical system state

\[
S^{(a)} = (\theta, m^{(a)}, o^{(a)}, r^{(a)}),
\]

where `θ` is the fixed base model, `m` is the writable memory state, `o` is optional optimizer/update-rule state, and `r` is backend-specific read/routing state. An event `e` induces an update

\[
S_{t+1}^{(a)} = U_{e_t}^{(a)}(S_t^{(a)}).
\]

The same semantic event may therefore induce different physical transitions in a versioned text store, an ICL history, a LoRA adapter, or a hybrid system.

## 2. Logical state versus physical state

A deterministic logical reducer `G` maps a trace `h=(e_1,…,e_T)` to its intended current memory:

\[
z_h = G(h).
\]

For example, `G` may implement latest-valid-write per key, preserve independent keys, and remove a key after a valid delete. Two histories are **logically endpoint-equivalent** when

\[
G(h)=G(h').
\]

This does not imply that their physical states are equal. The project tests whether they are equivalent in behavior.

A **history family** is the complete frozen specification that groups a shared logical fixture, reducer, endpoint relation, endpoint-equivalent histories, evaluation roles, controls, and backend observability profile. It is a compilation and provenance object, not a new statistical unit: the semantic item remains the independent unit and histories, probes, operators, and seeds are nested within it.

## 3. Behavioral endpoint equivalence

Let `P_i` be a frozen probe panel for unit `i`. For probe `q`, let `pi_q(S)` be the normalized four-candidate output distribution, and let

\[
B_{\mathcal P_i}(S)
\]

denote the complete audit record of candidate distributions, target margins, and task decisions produced from state `S`. The primary behavior distance `D` uses only the `pi_q(S)` distributions: it is mean Jensen–Shannon divergence across the frozen core probes. Signed target-margin differences and discrete decisions are secondary diagnostics, not inputs to JS.

Approximate endpoint consistency for a specified update operator requires

\[
G(h)=G(h'),\; C(h)=C(h')
\quad\Longrightarrow\quad
D_{\mathcal P_i}(S_h,S_{h'}) \le \varepsilon,
\]

where `C` is a vector of pre-treatment/design controls: root state, event multiset, final update block, semantic payload, ordered minibatches per event, loss-bearing tokens, total forward tokens, optimizer steps, precision, and randomization schedule.

Terminal loss and write success are outcomes of the update path. They are reported on a disjoint qualification panel but are not included in `C` and are not used to remove units from the primary analysis. The tolerance `ε` is fixed before outcome-bearing paths; technical duplicates must fall well below it or the pipeline fails.

## 4. Potential-outcome view

For every item `i`, backend `a`, and history `h`, define the potential candidate-distribution record

\[
Y_i^{(a)}(h)=\{\pi_q(S_{i,h}^{(a)}):q\in\mathcal P_i\}.
\]

Because every frozen item is executed under every scheduled history from the same base and root-adapter hash, the primary contract statistic is paired:

\[
\Delta^{(a)}(h,h')=
\mathbb E_i\left[D_{\mathrm{JS}}\!\left(Y_i^{(a)}(h),Y_i^{(a)}(h')\right)\right].
\]

For the symmetric final-`A` and final-`B` comparators, the item-level primary endpoint averages their two non-negative defects, as frozen in the experiment plan. Signed current-versus-stale margin contrasts are secondary mechanism descriptors. Randomization concerns label assignment, schedule execution order, training seed, and probe order. Within a comparator, the ordered minibatches for a given event block are held common using a seed derived from the block hash rather than its event position. The semantic item—not the training seed—is the main statistical unit.

### 4.1 Usefulness, consistency, and pathway evidence

Endpoint consistency is not task utility. Let `T_i(S)` be a frozen current-task utility measure, such as current-target success or task-native reward. For a history `h`, define a within-operator access contrast

\[
U_i^{(a)}(h)=T_i(S_{i,h}^{(a)},\text{memory on})-T_i(S_{i,h}^{(a)},\text{memory off}).
\]

For endpoint-equivalent histories, the signed secondary contrast

\[
A_{\mathrm{path}}^{(a)}(h,h')=\mathbb E_i[U_i^{(a)}(h)-U_i^{(a)}(h')]
\]

asks whether the two histories produce different usable memory value. It does not replace the non-negative endpoint defect: signed usefulness can cancel across items, and high consistency can arise when both paths are equally unusable. `Memory off` is backend-specific—adapter disable for a modular parametric state, store-access removal for an external state—and must leave the prompt, probe, base state, and inference configuration otherwise fixed.

Pathway evidence is reported separately from both `U` and the endpoint defect. A trace can show that an expected artifact was written, selected, or rendered, but trace alignment alone does not establish that the artifact was necessary for behavior.

## 5. Update contracts

These are approximate behavioral contracts, not claims of exact equality in parameter space.

For reporting, the contracts are grouped without changing their definitions:

- **Supersession:** terminal consistency and latest-write-wins;
- **Repetition:** idempotence;
- **Composition:** independent-update commutativity and isolation/locality;
- **Reversibility:** rollback/inverse.

This grouping supports contract-level macro summaries, but every contract keeps its own decision and uncertainty interval. An overall mean cannot override a severe family-specific failure.

### 5.1 Terminal consistency

Endpoint-equivalent and control-matched histories should produce behaviorally equivalent current states.

This is the umbrella contract and the primary audit target.

### 5.2 Latest-write-wins

For conflicting writes `k←A` followed by `k←B`, the final behavior should match a canonical `k←B` endpoint and suppress stale `A`:

\[
B(U_{k\leftarrow B}U_{k\leftarrow A}S)
\approx
B(U_{k\leftarrow B}S).
\]

Current-target write success is reported alongside this comparison. It is not a post-treatment inclusion rule.

### 5.3 Idempotence

Repeating an already-committed write should not materially change behavior once the current state is already usable:

\[
B(U_eU_eS)\approx B(U_eS).
\]

Because repeated training also changes exposure, this contract requires compute/exposure-matched sham or split-budget controls and is secondary rather than the first decisive experiment.

### 5.4 Independent-update commutativity

For events on independent keys or disjoint scopes,

\[
B(U_iU_jS)\approx B(U_jU_iS).
\]

The preferred test appends the same joint refresh block to both histories (`ijR` versus `jiR`) so that their final block and final logical state match.

### 5.5 Isolation/locality

An update scoped to key or skill `k` should preserve behavior on a pre-registered unrelated panel:

\[
D_{\mathcal P_{\neg k}}(U_kS,S)\le\varepsilon_{\mathrm{local}}.
\]

### 5.6 Rollback/inverse

For a declared rollback operator `V_e`, applying and reverting an update should restore prior behavior:

\[
B(V_eU_eS)\approx B(S).
\]

Three different notions must not be conflated:

- **physical restore:** load an earlier checkpoint/snapshot;
- **modular disable:** deactivate an adapter or memory version;
- **semantic correction:** write a countervailing statement or train on a replacement target.

Only the first two are expected to approximate a true inverse without additional assumptions.

## 6. Primary path contrasts

For mutually exclusive semantic states `A` and `B`:

| Final state | History 1 | History 2 | Matched property |
| --- | --- | --- | --- |
| `A` | `ABA` | `BAA` | `A×2, B×1`, final block `A` |
| `B` | `BAB` | `ABB` | `B×2, A×1`, final block `B` |
| `B` | `AABB` | `ABAB` | `A×2, B×2`, final block `B` |
| `A` | `BBAA` | `BABA` | `B×2, A×2`, final block `A` |

The two label directions are counterbalanced so that a token prior cannot masquerade as a memory effect.

## 7. Diagnostic decomposition

Diagnosis uses three different axes that must not be conflated:

1. **Contract:** which expected relation was violated;
2. **Update-lifecycle locus:** where the mismatch first becomes observable;
3. **Evidence kind:** integrity check, utility, endpoint defect, trace, counterfactual rescue, transfer, or downstream consequence.

Candidate causes are then tested rather than inferred from a locus. An observed endpoint defect may arise from:

1. backend implementation or lineage contamination;
2. unequal semantic exposure or tokenization;
3. prompt/recency sensitivity;
4. optimizer-state memory;
5. ordinary non-commuting gradient fields;
6. incomplete semantic overwrite or stale representation;
7. retrieval/index staleness;
8. interaction between parametric and external state.

External/textual and parametric operators expose different lifecycle loci:

| Operator class | Observable update lifecycle | Important boundary |
| --- | --- | --- |
| Versioned or LLM-managed textual memory | commit → supersede/resolve → retrieve → render → apply → future update | Traceable retrieval is not proof of behavioral necessity |
| Modular parametric memory | encode → optimize → persist/activate → infer/apply → future update | There is no explicit retrieval event; missing retrieval telemetry is `N/A`, not failure |

Every backend declares which loci are observable, intervention-accessible, or `N/A` before execution. Positions 1/2/3 in a history are called **event positions**; `update-lifecycle locus` is reserved for commit/retrieve/apply-style diagnosis.

For a pre-registered canonicalization intervention at an observable lifecycle locus `j`, define the paired **defect-reduction contrast**

\[
R_j=D_{\mathrm{native}}-D_{\operatorname{do}(j=\mathrm{canonical})}.
\]

Valid examples are canonical-store replacement and byte-identical current-state rendering for an explicit textual backend. `R_j>0` shows only that this canonicalization reduces the observed defect under its declared held-fixed conditions. The term **rescue** is reserved for the stricter gate requiring a positive lower confidence bound, post-intervention defect below tolerance, and preserved utility/locality. `R_j` is not “the percentage explained,” different values do not add, and a whole-operator canonical refit is a repair baseline rather than a single-locus intervention.

Artifact delete/swap is a different diagnostic: it tests whether behavior depends on a candidate artifact, not whether canonicalizing a lifecycle locus rescues endpoint consistency. For deletion, a secondary necessity contrast may be reported as

\[
N_j=T_{\mathrm{native}}-T_{\mathrm{delete}(j)},
\]

with the expected direction and practical utility threshold frozen in advance. Swap perturbations report the pre-registered target/logit shift; they are not evaluated by requiring endpoint `D` to fall below the consistency tolerance.

A common future suffix is an operational-equivalence probe:

\[
D_{\mathrm{suffix}}=D\!\left(B(U_gS_h),B(U_gS_{h'})\right),
\]

where both endpoints receive the same frozen suffix `g`. It is not a rescue cut. Adapter disable/restore is an access or rollback control; canonical refit is an alternative operator/repair.

The experiments diagnose candidate causes through nested controls and clearly typed diagnostics. They do not assume that components add linearly, and failure of one predictor does not by itself establish a new semantic mechanism. A trace association without the appropriate rescue or necessity evidence terminates the corresponding causal interpretation.

## 8. Required controls

- Byte-identical terminal prompt for external-overwrite histories.
- Technical duplicate paths that must produce identical hashes or bounded numerical noise.
- Label and event-order counterbalancing.
- Same loss-bearing tokens and forward-token counts.
- AdamW with carried state, AdamW reset per event, and memoryless SGD as separately named update operators.
- A disjoint terminal-write qualification panel, with no post-treatment deletion from the primary analysis.
- Current-target, stale-target, paraphrase, transfer, and unrelated probes.
- Effective `ΔW` and behavior-space comparisons; never raw LoRA factor distance alone.
- Conflict, paraphrase/no-conflict, and independent-key schedules plus held-out update-geometry prediction.
- Checkpoint restore, adapter disable, and canonical refit baselines.

## 9. Scope of any conclusion

A violation establishes a property of the named update operator, even when optimizer state explains it. A broader memory-science claim requires a reproducible semantic structure or a consequential failure in a real memory updater. A new-method claim additionally requires that restore, refit, or modular solutions leave a meaningful resource-constrained gap.
