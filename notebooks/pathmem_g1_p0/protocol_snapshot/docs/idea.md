# Idea and Positioning

## 1. Normalized idea card

**Task.** Audit whether an online LLM memory implementation realizes stable update semantics across logically equivalent update histories.

**Audience.** Researchers working on test-time learning, model editing, continual learning, LLM agents, and memory-augmented language models.

**Gap.** Current work measures recall, edit success, forgetting, retrieval, or final downstream reward, but usually does not hold the declared final memory ledger fixed while varying the update path and recording the complete update state.

**Root challenge.** A memory backend is a dynamical system. The same declared logical state can be reached through different physical trajectories, and neural or LLM-managed update operators need not be path-independent. A naive path comparison is ambiguous because it can mix event order with optimizer state, minibatch order, root-adapter initialization, terminal underfitting, and unequal exposure.

**Core insight.** Treat memory operations as an interface contract and test approximate algebraic laws in behavior space using endpoint-controlled randomized interventions.

**Proposed mechanism.** A provenance-locked history-family compiler and path DAG generate within-operator matched update histories from the same full parent hash. Backends consume the same canonical semantic ledger through backend-specific write/read interfaces; effect magnitudes are not treated as exposure-matched across backends. Fresh-session evaluation separately measures memory usefulness, endpoint defects, stale leakage, locality, and rollback. Backend-specific lifecycle traces, canonicalization defect-reduction interventions, artifact-necessity perturbations, and common-suffix equivalence probes localize failures without assuming an additive mechanism score. Optimizer reset/carry, memoryless SGD, external overwrite, technical duplicates, canonical refit, and gradient diagnostics characterize competing mechanisms.

**Contribution type.** Primary: evaluation/causal protocol plus empirical diagnostic analysis. Secondary: formal operationalization. Optional: method, only if a deployment-relevant failure leaves a gap after simple repairs.

**Expected evidence.** Paired effects with item-clustered uncertainty, contract-specific stress tests, cross-model confirmation, compute/storage accounting, and explicit negative controls.

**Why now.** Recent memory architectures make model state writable during inference, while surveys identify reversible, diagnosable update rules and controlled multi-dimensional evaluation as open problems. Recent order-geometry and optimizer-memory work also provides strong null explanations that a credible audit must test explicitly.

**Main risk.** The result may reduce entirely to known optimizer or sequential-learning order effects. That is a kill condition, not a finding to rebrand.

## 2. Problem statement

Let an online memory system advertise a current logical state such as “for key `k`, the valid value is `B`.” A user reasonably expects the model to behave according to `B` regardless of whether the system reached that state by `A→B`, `B→A→B`, or a logically equivalent series of independent updates. In practice, parameters, optimizer buffers, summaries, retrieval indices, and cached states may retain information about the path.

The scientific problem is not merely to detect order sensitivity. It is to determine whether a specified update operator violates its logical-state contract under a controlled budget, and then determine which state component or semantic relation predicts that violation.

## 3. What changes relative to the previous project

The old four arms are reinterpreted as backends:

| Old label | New interpretation | Role in this project |
| --- | --- | --- |
| No-write | No additional persistent write; base/context memory still exists | Baseline and interface qualification |
| External | Agent/system-level addressable store with explicit overwrite | Deterministic negative control and realistic retrieval variant |
| Parametric | Explicit writable parameter module, initially LoRA | Main online neural-memory backend |
| Both | External current state plus parametric residual state | Diagnostic: can a clean current-state note suppress stale parametric behavior? |

This avoids claiming that the four arms are four fundamental memory types.

## 4. Proposed artifact

The minimum viable artifact is an audit harness with:

1. a history-family schema defining shared fixtures, endpoint relations, episode roles, controls, and observability;
2. a canonical event schema (`WRITE`, `REVISE`, `DELETE`, `RESTORE`, `NOOP`);
3. a logical reducer that maps an event trace to the intended current state;
4. a path-DAG executor with immutable parent hashes;
5. backend adapters for external overwrite, ICL history, sequential LoRA, and hybrid use;
6. behavior-space scorers and contract-specific probe panels;
7. sparse backend lifecycle traces and valid counterfactual intervention adapters; and
8. an analysis layer that rejects unmatched histories before estimating effects.

### 4.1 Evidence decomposition

The audit uses a `Contract × Lifecycle × Evidence` design:

- **Contract** asks which relational law is under test: supersession, repetition, composition, or reversibility.
- **Lifecycle** asks where a mismatch first becomes observable. Textual memory exposes commit, resolve, retrieve, render, apply, and future-update loci; parametric memory exposes encode, optimize, persist/activate, infer, and future-update loci.
- **Evidence** distinguishes integrity checks, task utility, endpoint defect, trace alignment, counterfactual rescue, transfer, and downstream consequence.

The primary endpoint remains a non-negative behavioral defect. Memory utility is a separate access-on/off axis, so “both paths failed equally” cannot be presented as a healthy memory. A trace is pathway evidence, not causal attribution. Canonical store/render can test defect rescue; artifact delete/swap tests behavioral dependence on an artifact; a common future suffix tests operational equivalence. These diagnostics answer different questions and are never pooled or added.

## 5. Honest innovation claims

### Claimable if implemented

- A common endpoint-controlled protocol applied within each named memory-update operator.
- Equal-multiset, same-final-block interventions with explicit optimizer controls and disjoint terminal-write diagnostics.
- An operational bridge between external memory management and parametric online memory without conflating their taxonomy.

### Claimable only if observed

- A held-out, semantic-structure-dependent law of endpoint defects beyond generic update statistics.
- A systematic difference in which contracts fail across backends.
- A predictor of contract failure that generalizes across unseen histories or items.

### Not currently claimable

- A new memory architecture.
- A universally superior update mechanism.
- A new theory of neural memory.
- Robust rollback for arbitrary LLM knowledge.

## 6. Alternatives considered

### Continue carrier-selection/value-surface work

Rejected as the primary direction because hybrid routing and recurrence-conditioned consolidation are crowded, and a recurrence crossover can collapse into simple cost amortization.

### Study raw “semantic hysteresis”

Rejected as the main claim because sequential editing, online consolidation, and update-order geometry already establish path effects. The project retains path interventions only as a controlled instrument.

### Build a rollback method immediately

Deferred. Checkpoint restore, adapter disable, canonical refit, and versioned stores are strong simple baselines. A new method is warranted only if they leave a meaningful deployment gap under matched budgets.

## 7. Reviewer-facing risk register

| Risk | Type | Mitigation or decision rule |
| --- | --- | --- |
| “This is just sequential fine-tuning order sensitivity.” | Likely pivot | Treat each optimizer as its own operator; add conflict/independence controls and held-out prediction |
| “The histories do not have the same intended endpoint.” | Design-fixable | Same event multiset, same final block, canonical logical reducer; report write success separately |
| “LoRA factor distances are not meaningful.” | Design-fixable | Use behavior/logit distances and effective `ΔW`, not raw factor coordinates |
| “External memory wins trivially because overwrite is exact.” | Writing/design-fixable | Use it as a negative control, not as a SOTA contest; add realistic retrieval only later |
| “Synthetic nonce tasks do not establish relevance.” | Evidence-fixable | Add repeated factual edits and a procedure/API-update slice after P1 |
| “A simple restore/refit solves everything.” | Requires-new-result | Include restore/refit/modular adapter baselines; kill the method claim if they dominate |
| “Small-model interface failures create the effect.” | Evidence-fixable | Pre-registered answer-copy/external-use gate, 1.5B P1, and different-family confirmation |
| “The benchmark is the only contribution.” | Venue-mismatch risk | Tie the protocol to a held-out predictive law or consequential real-updater failure; otherwise position as an evaluation resource |
| “PAST-Bench already separates memory gains from pathway evidence.” | High prior-art overlap | Treat fresh-session on/off attribution, update/stale families, and trace diagnosis as prior art; retain endpoint-equivalent multi-history confluence and hidden-state control as the core delta |

## 8. Paper storyline if the gates pass

1. Writable memory implies update semantics, not merely recall.
2. Existing evaluation cannot distinguish logical-state violations from optimization history.
3. Endpoint-controlled interventions make the distinction measurable.
4. Current backends are audited under a common set of contracts.
5. Utility, endpoint consistency, and lifecycle evidence are reported separately; causal cuts localize only those failures that survive the primary gate.
6. Optimizer, geometry, semantic conflict, and terminal-write diagnostics delimit what can be concluded.
7. A mitigation is introduced only if the audit exposes a deployment gap that simple restore/refit methods do not close.
