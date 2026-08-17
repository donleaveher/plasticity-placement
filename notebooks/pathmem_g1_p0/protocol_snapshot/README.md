# Memory Update Consistency

Working title: **Do LLM Memories Obey Update Contracts? Endpoint-Controlled Causal Audits of Online Memory Operators**

Target: ICLR 2027, empirical/diagnostic track within machine learning.

Status: G0 passed; the CPU-only protocol implementation, deterministic PathMem-v1 bank, immutable G0 manifest, lineage identities, observability matrix, and integrity tests are frozen. G1 interface qualification has not started, and no central empirical claim has been established.

## One-sentence idea

We branch endpoint-equivalent update traces from the same full parent state, match their semantic events and update budget, and test whether a concrete online-memory operator reaches behaviorally equivalent endpoints.

## Why this is a separate project

This project is not a continuation of the old claim that an experience should be assigned to `no-write`, `external`, `parametric`, or `both` depending on future recurrence. Those four arms may still be reused as experimental backends, but they are treatment bundles rather than a memory taxonomy. The new object of study is the **semantics of memory updates**: overwrite, repeat, reorder, isolate, and rollback.

## Central question

> If two controlled update traces represent the same final logical memory state, do they produce the same model behavior? If not, is the discrepancy caused by ordinary optimizer memory, non-commuting gradient updates, incomplete semantic overwrite, or a carrier-specific memory failure?

## Project map

- [`research-question-card.md`](research-question-card.md): compact research contract and claim boundaries.
- [`docs/idea.md`](docs/idea.md): problem, gap, insight, contribution plan, and non-goals.
- [`docs/principles.md`](docs/principles.md): formal state-transition view and update contracts.
- [`docs/goals-and-gates.md`](docs/goals-and-gates.md): milestones, go/no-go rules, and resource boundaries.
- [`docs/experiment-plan.md`](docs/experiment-plan.md): phased experimental design, controls, metrics, and statistics.
- [`docs/related-work.md`](docs/related-work.md): narrative positioning against the closest work.
- [`literature-search-20260813-memory-update-consistency/`](literature-search-20260813-memory-update-consistency/): reusable search record and paper table.
- [`experiments/README.md`](experiments/README.md): implementation contract and artifact layout.
- [`experiments/g0/manifest.json`](experiments/g0/manifest.json): immutable G0 bundle identity and gate result.
- [`../plasticity-placement/src/plasticity_placement/pathmem/`](../plasticity-placement/src/plasticity_placement/pathmem/): executable CPU-only G0 implementation.
- [`task_plan.md`](task_plan.md): current execution queue.
- [`ccfa.yaml`](ccfa.yaml): project state for the CCFA workflow.

## Current thesis, stated conservatively

The intended contribution is an **endpoint-controlled metamorphic/causal audit**, not another hybrid-memory architecture, longitudinal memory benchmark, or demonstration that sequential fine-tuning is order-sensitive. It reports memory usefulness, endpoint consistency, and lifecycle evidence as distinct axes. A contract violation is defined for each concrete update operator; optimizer state and gradient geometry are explanations to characterize, not nuisances that can simply be “subtracted.” An ICLR-level paper additionally needs a held-out predictive law, a consequential failure in a real memory updater, or a repair that beats simple restore/refit baselines.

## Resource assumptions

- One commodity datacenter/Colab GPU at a time.
- Qwen2.5-0.5B for engineering smoke tests only.
- Qwen2.5-1.5B for the first outcome-bearing kill test, subject to interface qualification.
- A different model family—not merely another Qwen size—for confirmation.
- PEFT/LoRA updates; no pretraining from scratch.
- Paired designs and cached update DAGs to avoid repeated training.

## Evidence policy

All result cells remain `TBD` until filled by executed experiments. Public-paper claims are linked to primary sources. Proposed novelty is explicitly conditional on the kill tests in this repository.
