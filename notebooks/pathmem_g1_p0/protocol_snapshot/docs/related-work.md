# Related Work and Novelty Subtraction

Updated: 2026-08-14

This document positions the project against the closest primary sources. It is deliberately written as a **claim subtraction**: each subsection first states what prior work already establishes, then identifies the narrower question that remains.

## 1. Memory taxonomy: coordinates, not four boxes

[Memory for Large Language Models](https://arxiv.org/abs/2607.25380) organizes model-level memory along three approximately orthogonal axes: implicit versus explicit representation, offline versus online update dynamics, and short- versus long-term persistence. This corrects several tempting but inaccurate shortcuts:

- external does not mean explicit in every setting;
- parametric does not necessarily mean implicit;
- inference-time reading does not imply online writing; and
- update speed does not determine retention horizon.

Accordingly, this project does not present `no-write/external/parametric/both` as a memory taxonomy. They are treatment backends. A switchable per-lesson LoRA is an independently managed parametric state; an agent-side key-value store is an external system component outside the survey's model-level scope; and `both` is a composed treatment rather than a new memory class.

The survey also identifies reversibility, delayed relevance, interference, stable online updating, and controlled multidimensional evaluation as open challenges. Those challenges motivate this project, but merely restating them is not a contribution.

## 2. Writable and self-updating model memory

[MEMORYLLM](https://proceedings.mlr.press/v235/wang24s.html) and [Self-Adapting Language Models](https://papers.neurips.cc/paper_files/paper/2025/hash/6b41e04c41726e2a60e456d0a2b961ab-Abstract-Conference.html) establish that persistent model state can be updated online or through model-generated training directives. Their central questions are retention, adaptation, and downstream performance. They do not test whether logically equivalent histories converge to behaviorally equivalent memory states.

Therefore, this project cannot claim the first writable, self-updating, or long-lived parametric memory. It asks whether the update interface implements stable semantics once such a memory exists.

## 3. Lifelong and sequential model editing

Large editing sequences are already known to cause attenuation and forgetting. [Model Editing at Scale Leads to Gradual and Catastrophic Forgetting](https://aclanthology.org/2024.findings-acl.902/) studies degradation under many edits, while [Can We Continually Edit Language Models?](https://aclanthology.org/2024.findings-acl.323/) studies knowledge attenuation and proposes a mitigation. [SimIE](https://proceedings.mlr.press/v267/guo25c.html), [WISE](https://papers.nips.cc/paper_files/paper/2024/hash/60960ad78868fce5c165295fbd895060-Abstract-Conference.html), and [RLEdit](https://proceedings.mlr.press/v267/li25an.html) improve lifelong editing through recursive update approximations, sharded memories/routing, or sequence-level optimization. [Serial Lifelong Editing via Mixture of Knowledge Experts](https://aclanthology.org/2025.acl-long.1492/) directly evaluates repeated revisions of the same concept and latest-answer behavior.

These works occupy the claims that repeated edits interfere, old knowledge can persist, and latest-write correctness is difficult. Their usual unit is a fixed editing stream or the average performance after a sequence. The remaining distinction here is the paired comparison of multiple histories that begin from the same full state and declare the same final knowledge ledger, while exposure, update budget, final block, and within-block randomness are controlled. Terminal write success is reported as an outcome rather than used for post-treatment matching.

## 4. Direct path dependence and training-state memory

This is the most dangerous overlap.

[Forgetting Is Not a Fix: Path Dependence in Sequential Engram Editing](https://arxiv.org/abs/2607.24805) directly reports path dependence when the same concept-level intervention set is applied in different orders. [AI Engram](https://arxiv.org/abs/2606.14997) supplies the compositional/erasure setting and a commutative-manifold hypothesis. A paper whose only result is “update order changes the final model” would therefore have no defensible novelty.

Generic optimization work narrows the gap further. [Training Memory in Deep Neural Networks](https://arxiv.org/abs/2601.21624) treats training state as an augmented object containing parameters, optimizer moments, scheduler, random state, and sampler state. [Process-Tensor Tomography of SGD](https://arxiv.org/abs/2601.16563) uses interventions and resets to expose non-Markovian training memory. [Optimizer Memory Makes Shuffle Order a First-Order Source of Fine-Tuning Noise](https://arxiv.org/abs/2606.29554) shows why equal-multiset reorderings can have a first-order AdamW effect. [First-Order Predictable but Pairwise Fragile](https://arxiv.org/abs/2607.16821) and [The Geometry of Sequential Learning](https://arxiv.org/abs/2606.24993) connect two-update order effects to Lie-bracket terms and use them predictively across models or training tasks. Reproducing their held-out predictability is therefore a baseline result, not a new law.

Consequently, optimizer carry/reset, memoryless SGD, fixed-clock controls, final-write dose/convergence, gradient cosine, optimizer-kernel features, and a Lie-bracket predictor are not optional ablations. They are novelty gates. A new predictive result must improve on this full baseline stack through a pre-registered memory-operation interaction; otherwise the result is ordinary sequential optimization.

## 5. External textual memory also has update-path failures

Path problems are not confined to parameters. [Useful Memories Become Faulty When Continuously Updated by LLMs](https://arxiv.org/abs/2605.12978) reports that the same underlying trajectories can yield qualitatively different consolidated textual memories under different update schedules. This rules out a claim that schedule-dependent memory degradation is unique to LoRA or parametric storage.

Our deterministic `E-latest` backend therefore serves as a semantic negative control: a correctly implemented versioned overwrite store should expose the same final state for endpoint-equivalent histories. LLM-generated summarization or consolidation is a separate, lossy update operator and may fail the same contracts. The protocol can compare both, but the distinction must be explicit.

## 6. Reversibility and versioned rollback

[SoLA](https://arxiv.org/abs/2603.11239) supports removal through semantically routed, modular LoRA edits. [ChronoMem](https://arxiv.org/abs/2607.27773) supports external-memory snapshots and version selection. These are strong rollback baselines, not evidence that learned counter-updates form an algebraic inverse.

The project separates:

- physical restore of an earlier snapshot;
- modular disable/switch of an isolated delta; and
- semantic correction through another write or gradient update.

A new rollback method is warranted only if the first two are infeasible or dominated under a stated deployment constraint. Otherwise they are the correct simple solution.

## 7. Benchmarks and evaluation axes

[WikiBigEdit](https://proceedings.mlr.press/v267/thede25a.html) provides large-scale real knowledge changes and comparisons among editing, retrieval, and continual fine-tuning. [ForgetBench](https://arxiv.org/abs/2607.26455) evaluates temporal forgetting under continuous parametric updates. [MQuAKE](https://aclanthology.org/2023.emnlp-main.971/) tests whether edits propagate through multihop semantic consequences.

These resources can supply realistic confirmatory slices, but none is by itself an endpoint-equivalent path intervention. `PathMem-v1` is therefore a controlled diagnostic generator first; realistic benchmark subsets are confirmation, not substitutes for the causal design.

## 8. PAST-Bench and trajectory-level memory attribution

[PAST-Bench](https://arxiv.org/abs/2608.04003) evaluates ordered fresh-session task families under matched persistence-on/off conditions. It separates downstream task gains from pathway evidence and covers memory, procedural reuse, information gathering, and update. Its Update families include factual correction, global and scoped rule migration, temporary-exception handling, SOP patches, and recall-then-modify. Hermes+ then maps observed failures to Plan, Render, Route, Gate, and Close interventions. The authors present the on/off pairing as a strong attribution control rather than complete causal proof. The [official repository](https://github.com/Gen-Verse/PAST-Bench) provides the executable benchmark and framework modifications.

This occupies several claims that this project must not make: first longitudinal fresh-session agent-memory audit, first matched attribution of later-task gains to persistent state, first separation of task outcome from write/retrieve/update evidence, or first trace-backed stale-update diagnosis. Typed current/superseded bindings, retrieval-before-action gating, and overwrite-and-flush closeout are prior mechanisms and must be treated as baselines if a textual-memory repair is proposed.

The estimands differ. PAST-Bench asks whether access to retained state improves a later episode along one task-family trajectory:

\[
\Delta_f=S_f^{\mathrm{on}}-S_f^{\mathrm{off}}.
\]

This project asks whether multiple histories with the same full root, semantic-event multiset, final event, and declared logical endpoint produce behaviorally equivalent memory states. PAST-Bench's pathway score is trace-alignment evidence, not an additive causal mediation analysis. We therefore borrow its fresh-session episode roles, outcome/pathway separation, family-level macro reporting, and diagnosis-driven interventions. The core novelty delta remains endpoint-equivalent multi-history randomization, augmented-state controls, and relational update laws. Backend-specific lifecycle traces and canonicalize/delete/swap diagnostics support interpretation but are not novel merely because they are implemented.

PAST-Bench's Update families are suitable P2 task-structure prototypes, not substitutes for the primary causal generator. Any adapted family must preserve the endpoint-equivalent path relation and distinguish current-state questions from historical-provenance questions.

## 9. Composable interventions and metamorphic testing

[Composable Interventions for Language Models](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7f5f9a88c6516469c83d074c6f2976fb-Abstract-Conference.html) supplies a unified framework for composing editing, compression, and unlearning interventions and reports that their order can matter. [A3E](https://papers.neurips.cc/paper_files/paper/2025/file/3d4c0a618d0acd7921493e4f30395c22-Paper-Conference.pdf) studies whether multiple edits can be used compositionally. These works further rule out novelty claims based on intervention composition or order alone; their aim is multi-intervention compatibility, not confluence to a declared current-memory ledger.

More generally, [metamorphic testing for machine-learning applications](https://www.cs.columbia.edu/wp-content/uploads/sites/7/2016/08/Murphy-SEKE2008.pdf) evaluates relations between transformed inputs and expected output relations when a pointwise oracle is unavailable. Endpoint-equivalent histories are a domain-specific metamorphic relation. This project therefore does not claim to invent metamorphic testing. Its prospective contribution is the memory-specific state model, update-contract suite, full-state lineage, and statistical decision protocol.

## 10. What prior work already owns

The project must not claim any of the following:

| Occupied claim | Representative prior work |
| --- | --- |
| Hybrid or adaptive memory is new | Memory survey and existing hybrid-memory systems |
| Sequential updates are order-sensitive | Engram path-dependence and sequential-learning geometry work |
| Optimizer state carries history | Training-memory, process-tensor, and optimizer-memory work |
| Repeated edits cause forgetting or stale behavior | Lifelong/sequential model-editing literature |
| Latest-write-wins is a new evaluation goal | Serial lifelong editing |
| Textual consolidation can degrade with schedule | Continuously updated textual-memory study |
| Rollback or versioning is new | SoLA and ChronoMem |
| A new memory benchmark alone establishes memory science | WikiBigEdit, ForgetBench, MQuAKE and related suites |
| Fresh-session retained-experience attribution and outcome/pathway separation are new | PAST-Bench |
| Typed current/superseded rendering or overwrite-and-flush closeout is a new textual-memory repair | PAST-Bench / Hermes+ |
| Testing expected relations between transformed executions is new | Metamorphic-testing literature |
| Sequential LLM interventions can interact or be order-sensitive | Composable Interventions and A3E |

## 11. Residual question this project can defend

The remaining question is not whether history matters. It is:

> Given the same fully specified pre-update state and the same declared active-memory endpoint, which named online memory operators satisfy terminal consistency, latest-write-wins, idempotence, independent-update commutativity, isolation, and rollback in behavior space, and what predicts their violations?

The proposed delta is the combination of:

1. a deterministic logical reducer defining the intended current memory;
2. endpoint-equivalent, equal-exposure path interventions from a shared parent state;
3. explicit control of hidden update state, including optimizer, scheduler, RNG, router/index, and cache lineage;
4. a common behavioral contract suite spanning external and parametric update operators; and
5. current-state, locality, semantic-transfer, and future-update consequences rather than weight distance alone.

No single paper in the current search set implements all five. That does not make the contribution automatic: the experiments must reveal a held-out predictive law, a consequential failure in a real updater, a method-relevant deployment gap, or a useful and explicitly scoped negative audit. A raw `AB` versus `BA` effect or a result confined to one nonce task is insufficient.

## 12. Submission-time audit rule

The literature search is current through 2026-08-14. The 2026 preprints and their code/venue status must be rechecked before any submission. If a later work contains the same shared-initial-state, endpoint-equivalent, hidden-state-controlled, multi-contract audit, the novelty claim must be narrowed or the project must pivot.
