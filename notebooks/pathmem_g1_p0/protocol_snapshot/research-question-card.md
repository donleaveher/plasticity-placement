# Research Question Card

Updated: 2026-08-14

Target venue: ICLR 2027

Stage: pre-experiment, kill-test design

## Working title

**Do LLM Memories Obey Update Contracts? Endpoint-Controlled Causal Audits of Online Memory Operators**

## Research question

For a fixed base model and memory item, suppose two update histories:

1. begin from the same pre-update state;
2. contain the same multiset of semantic events and matched exposure/compute;
3. end in the same intended logical memory state.

Under a fixed update budget, do concrete online-memory operators produce behaviorally equivalent endpoints on current, stale, paraphrase, transfer, and unrelated probes? Terminal write success is measured on a disjoint qualification panel and never used to delete units from the primary analysis.

## Why this matters

Persistent LLM memory is increasingly treated as a writable state. A useful memory system therefore needs more than recall: users and downstream systems implicitly expect overwrite, repetition, independent updates, deletion, and rollback to have stable semantics. Existing evaluations mostly measure final task accuracy, edit success, retrieval, or forgetting. They rarely ask whether two logically equivalent update traces are behaviorally equivalent after controlling update history.

## Gap after closest work

The project does **not** claim that update order matters for the first time. Sequential knowledge editing, continual fine-tuning, online textual consolidation, and optimizer analyses already establish order effects and overwrite failures. The narrower gap is an endpoint-controlled audit that:

- fixes the initial state and final logical memory;
- uses paired, equal-multiset update paths;
- treats optimizer carry/reset and memoryless updating as distinct audited operators;
- checks a small algebra of update contracts; and
- applies common contracts to parametric and textual update operators while keeping inference within each backend.

[PAST-Bench](https://arxiv.org/abs/2608.04003) further occupies fresh-session task families, matched persistence-on/off controls, separation of downstream outcome from pathway evidence, and trace-backed diagnosis of write/retrieve/update failures. This project therefore does not claim the first longitudinal or mechanism-aware agent-memory benchmark. Its remaining core distinction is the comparison of multiple equal-multiset histories with the same declared endpoint, explicit control of augmented update state, and a suite of relational update laws. Counterfactual lifecycle interventions are conditional diagnostics, not part of the novelty claim by themselves.

## Core insight

Memory should be evaluated as a **state-transition interface**, not only as a storage location or an accuracy score. The critical question is whether its update operators implement the logical contract advertised by the memory state.

## Primary estimand

For update operator `a`, unit `i`, and two endpoint-equivalent histories `h` and `h'`, estimate the non-negative paired behavioral endpoint defect

\[
\Delta^{(a)}_{i}(h,h') = D_{\mathcal{P}_i}\!\left(B(S^{(a)}_{i,h}), B(S^{(a)}_{i,h'})\right),
\]

where `B` is the four-candidate behavior distribution on a frozen probe panel and `D` is mean Jensen–Shannon divergence in natural-log units. Histories are matched by design on event multiset, final event, ordered minibatches per event, loss-bearing tokens, and optimizer steps. Terminal fit is a post-update diagnostic, not a matching variable.

For the symmetric primary schedules, let `Delta_A,i` denote `ABA` versus `BAA` and `Delta_B,i` denote `BAB` versus `ABB`. The phase-level sole primary endpoint is

\[
D_i=(\Delta_{A,i}+\Delta_{B,i})/2,
\]

followed by the item-level paired aggregation and confidence interval specified in the experiment plan. Signed current-versus-stale margins remain secondary.

`D` answers whether the operator is endpoint-consistent. A separate, secondary persistence-on/off contrast answers whether the resulting memory is useful at all. These axes are never collapsed: low defect can mean either a healthy memory or two paths that both failed to learn.

## Proposed contributions

1. **Evaluation/identification:** an endpoint-controlled causal protocol for online-memory update operators.
2. **Formalization:** operational contracts for terminal consistency, latest-write-wins, idempotence, independent-update commutativity, isolation, and rollback.
3. **Diagnostic evidence:** outcome, pathway trace, and pre-registered counterfactual-rescue evidence that distinguish external-state bugs, prompt order, optimizer state, ordinary gradient non-commutativity, incomplete overwrite, and unrelated interference without assuming an additive mechanism decomposition.
4. **Optional method:** only after a deployment-relevant failure and simple-baseline gap are established, a contract-aware update/rollback mechanism under matched resource budgets.

## Non-goals

- A new implicit/explicit/parametric/external taxonomy.
- A claim that hybrid memory or adaptive routing is novel.
- A first demonstration of path dependence, sequential-edit interference, or optimizer order sensitivity.
- A universal ranking of external versus parametric memory.
- Cognitive claims about human memory or “hysteresis.”
- A new mitigation before the failure survives the pre-registered controls.

## Decisive experiment

The primary paired schedules are:

- final `A`: `ABA` versus `BAA`;
- final `B`: `BAB` versus `ABB`.

Each pair contains the same event multiset, uses the same final update, and receives the same training-token and optimizer-step budget. Secondary four-block contrasts are `AABB` versus `ABAB` and the label-swapped counterpart `BBAA` versus `BABA`.

## Success condition

A publishable phenomenon claim first requires:

- the lower confidence bound of the behavior-space endpoint defect exceeds a pre-registered practical margin;
- the effect appears across semantic items and training seeds;
- the intention-to-audit analysis includes every randomized unit;
- it appears on held-out paraphrase/transfer probes, not only exact label recall;
- it reproduces on a different model family;
- it is tested on a realistic repeated-update task with both an existing parametric/sequential updater and a reproducible textual-consolidation updater.

For an ICLR-level novelty claim, at least one further condition is required: memory-operation features must add pre-registered held-out predictive value beyond update count, terminal loss, gradient cosine, optimizer-kernel, and Lie-bracket baselines; the audit must expose a behaviorally consequential failure missed by an existing realistic updater's standard evaluation; or a repair must improve the resource-matched frontier over simple restore/refit/modular baselines. A second model family establishes robustness but is not novelty by itself.

## Kill condition

Demote the broad or semantic-mechanism claim if the apparent effect:

- exists only in parameter coordinates but not behavior;
- fails to replicate outside one nonce template, one label mapping, or one seed; or
- shows no conflict- or memory-operation-specific structure beyond generic sequential-update geometry.

Separately, checkpoint restore, canonical refit, or independent adapters dominating at equal/lower cost kills the **new-method need**, not the factual finding that the audited operator violates a contract. If the upper confidence bound lies below the practical margin, the correct result is evidence of approximate consistency for that operator. A confidence interval crossing the margin is inconclusive rather than a negative result.

## Resource boundary

One GPU, PEFT updates, cached path DAGs, and no training from scratch. Learned mitigation, additional operators, and larger backbones are gated on the P1 investment test.
