# Goals, Gates, and Decision Rules

## North-star objective

Determine whether named online LLM memory-update operators satisfy approximate behavioral contracts, and whether any violations have a reproducible semantic structure or deployment consequence beyond a generic order effect.

## Success is staged

The project does not assume that a new method is necessary. Each stage earns authority for the next.

## G0 — Research-contract freeze

Deliverables:

- frozen terminology and non-goals;
- frozen history-family schema, fresh-session episode roles, and machine-readable family compiler contract;
- logical event schema and reducer;
- primary histories and control vector;
- sparse `operator × contract × update-lifecycle locus` observability/intervention matrix, with unavailable loci marked `N/A`;
- claim–evidence matrix;
- related-work audit;
- immutable experiment manifest schema.

Pass gate:

- every central claim maps to a falsifying experiment;
- “first path dependence” and “new hybrid” claims are absent;
- histories are logically endpoint-equivalent and exposure/compute-matchable;
- no experiment requires large-scale pretraining.

Current status: **passed on 2026-08-14**. The executable implementation is in `../../plasticity-placement/src/plasticity_placement/pathmem/`; the deterministic bundle and gate result are frozen in `../experiments/g0/manifest.json`. G1/P0 remain unauthorized, and no training or GPU inference was started by G0.

## G1 — Interface qualification

Purpose: show that the model can use the memory interface before interpreting update failures.

Required checks:

- answer-copy and canonical-note use;
- current-versus-stale forced-choice separation;
- parser/format validity;
- adapter on/off exactness;
- external overwrite byte identity;
- lineage and cache hash verification;
- explicit-store write/version/retrieval/render traces and parametric parent/optimizer-state lineage traces;
- fresh-session isolation and matched persistent-state access on/off behavior.

Pass gate:

- answer-copy top-1 accuracy is at least 98% with zero invalid outputs;
- canonical external-state use is at least 95% top-1 with obsolete intrusion at most 5%;
- current-only parametric writes reach at least 80% top-1 overall and at least 70% for every action label on the disjoint qualification bank;
- duplicate runs agree within the numerical tolerance;
- adapter disable restores the expected baseline within tolerance;
- every planned trace field is present or predeclared `N/A`; missing telemetry is never silently scored as failure.

Fail action: repair the interface or change the pre-P1 model without inspecting kill-set path outputs. Do not interpret operator or contract effects.

## G2 — P0 smoke test

Scope:

- 4 counterbalanced semantic pairs;
- one training seed;
- both primary contrasts: `ABA` versus `BAA` and `BAB` versus `ABB`;
- external overwrite, ICL history, LoRA–AdamW-reset;
- current/stale margin plus unrelated probes.

Goal: validate the path DAG, token/compute matching, forced-choice scorer, and paired analysis—not establish a scientific result.

Pass gate:

- complete provenance for every node;
- external overwrite path defect is within tolerance;
- path write-qualification and joint-pass rates are reported without deleting failed paths;
- no execution-order or cache contamination is detected.

## G3 — P1 investment/kill test

Scope:

- 12 frozen semantic pairs across fact and procedure families;
- 3 nested training seeds;
- symmetric 3-block endpoint-equivalent histories under reset-at-event AdamW;
- external overwrite, ICL history, sequential LoRA, hybrid, and restore/disable integrity controls;
- matched path-specific access-on/off evaluation for the secondary memory-utility axis;
- current, stale, paraphrase, transfer, and unrelated probes.

Decision rule for the mean Jensen–Shannon endpoint defect `D`, with practical margin `ε=0.02` nats:

- **meaningful violation:** lower 95% confidence bound is above `ε`;
- **approximate consistency:** upper 95% confidence bound is below `ε`;
- **inconclusive:** the interval intersects `ε`.

Only the first result authorizes P1b mechanism work by default. P1b audits carried-state AdamW and memoryless SGD as separate operators and adds conflict/no-conflict and independent-key schedules. An inconclusive result may authorize one pre-specified sample expansion only when a blinded precision calculation shows it can resolve the margin within the resource cap.

P1b may also run pre-registered backend-valid lifecycle diagnostics. Trace correlation alone is not causal localization. A positive `R_j` is called defect reduction; a canonicalized locus is considered to **rescue** a defect only when `R_j` has a 95% lower bound above zero, the intervened defect's upper bound falls below the contract tolerance, and current-task utility/locality remain within their frozen task-family tolerance. Artifact delete/swap uses a separate necessity or expected-behavior test, while a common future suffix is an operational-equivalence probe developed in P1b and frozen for untouched P2 confirmation. These quantities are reported separately and never summed into a mechanism score.

Kill/demote condition:

- the upper confidence bound is below the practical margin for the audited operator;
- technical contamination invalidates the run;
- only parameter distance changes.

Allowed outcomes:

1. **Meaningful violation:** continue to explanation and confirmation.
2. **Violation is fully generic:** retain the operator-level finding but stop the broad semantic-memory claim unless a real updater consequence is demonstrated.
3. **Approximate consistency:** stop that operator line rather than adding complexity.
4. **Inconclusive:** make no claim; expand only under the pre-specified precision rule.

## G4 — P2 confirmation

Required before an ICLR-level empirical claim:

- a different model family, not only a larger Qwen checkpoint;
- the different-lineage model/operator passes all applicable G1 interface thresholds on `interface_dev` before any `reserve` result is viewed; `hardware_dev` is compatibility-only;
- at least one realistic repeated-update dataset/task;
- held-out histories or longer trace lengths;
- at least one runnable existing sequential editor/parametric-memory updater and one reproducible LLM-managed textual-consolidation updater;
- realistic update families covering a frozen subset of factual correction, scoped rule migration, temporary-exception expiry, SOP patching, or recall-then-modify;
- pre-registered final-write dose robustness for the decisive parametric operator;
- effect-size and cost scaling.

Pass gate:

- the same contract-level violation reproduces with its endpoint-defect lower bound above the frozen practical margin;
- the effect is not confined to nonce exact match;
- either semantic memory-operation features reduce held-out prediction error by the frozen practical margin beyond update-count, terminal-loss, gradient-cosine, optimizer-kernel, and Lie-bracket baselines, or an existing realistic updater exhibits a behaviorally consequential failure missed by its standard evaluation;
- the distributional violation is paired with the pre-registered consequence threshold for that task family.

Fail action: narrow the claim to an operator-specific audit or pivot. A nonzero order effect, a second-family replication, or replication of a known Lie-bracket predictor is insufficient for an ICLR-level novelty claim.

## G5 — Optional mitigation

Only begin after G4. Simple restore/refit/modular baselines determine whether a new method is needed; they do not erase a verified failure of the original operator.

Candidate mechanism classes:

- versioned/modular parameter deltas;
- update projection or conflict-aware composition;
- canonical refit with selective replay;
- explicit provenance plus restore planning;
- hybrid current-state override with stale-residual suppression.

Required baselines:

- full checkpoint restore;
- adapter disable/switch;
- retrain current canonical memory from the base;
- independent per-item adapters;
- standard sequential LoRA;
- closest runnable sequential-edit method.

Method pass gate:

- reduces contract violations under matched update compute, inference cost, and storage;
- preserves editability, current-target accuracy, locality, and unrelated abilities;
- improves over the simple restore/refit/modular baselines rather than only standard LoRA.

## Resource guardrails

- One GPU at a time.
- P0 uses 0.5B only for pipeline validation.
- P1 uses a qualified 1.5B-class model, cached prefix adapters, and no retraining of shared prefixes.
- P2 confirms on untouched items and a separately frozen second model family.
- No learned router, broad layer sweep, or large HPO before G3.
- Separate `training_seed`, `panel_seed`, and `execution_order_seed` in every manifest.

## Timeline target

| Window | Deliverable | Decision |
| --- | --- | --- |
| Days 1–3 | G0 freeze and lineage tests | Is the question operationally coherent? |
| Days 4–6 | G1 interface qualification | Can the model use the memory interface? |
| Days 7–10 | G2 smoke | Is the harness trustworthy? |
| Weeks 3–4 | G3 investment test | Is there a meaningful contract violation worth explaining? |
| Weeks 5–7 | G4 confirmation | Is the result broad enough for a paper claim? |
| Later | G5 mitigation | Is a new method necessary and competitive? |

The dates are planning targets, not evidence of completion.
