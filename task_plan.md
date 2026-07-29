# Task Plan: P0-D2H-CAL-FC format-stable forced-choice calibration

## Goal
Implement and verify an independent, base-only forced-choice calibration that
scores the four frozen actions by candidate-only conditional log-likelihood,
without changing prior experiments or starting training.

## Phases
- [x] Phase 0: Commit the pre-task working tree as an explicit baseline
- [x] Phase 1: Read the canonical protocols, layout, notebooks, package, and tests
- [x] Phase 2: Freeze the package architecture, schemas, manifest identity, gates,
  provenance, and output contracts in the new protocol
- [x] Phase 3: Implement candidate scoring, source audit, atomic persistence,
  aggregation, gates, CLI, and report generation in `src/plasticity_placement/p0d2hfc/`
- [x] Phase 4: Add the independent notebook generator, generated Colab, README,
  project documentation, and CLI entry point
- [x] Phase 5: Add focused tests for scoring, provenance, matrix integrity, gates,
  persistence semantics, notebook structure, and absence of training paths
- [x] Phase 6: Run focused tests, full tests, Ruff, notebook regeneration,
  real-tokenizer CPU audit where available, and `git diff --check`
- [x] Phase 7: Review scientific boundaries, repository status, and handoff

## Key Questions
1. How does the existing calibration render prompts and identify immutable source
   rows, models, lessons, probes, and actions?
2. How can prompt/candidate concatenation be audited without accidentally scoring
   prompt tokens or silently accepting tokenizer boundary changes?
3. Which persisted identities are required for safe resume and immutable attempts?
4. How should overall and category gates distinguish eligibility from
   `category_calibration_failed` without triggering downstream work?
5. Can the generated Colab keep all preflight work CPU-only and make the first
   CUDA-loading cell explicit?

## Decisions Made
- Baseline commit: `db185e5` (`docs: prepare format-stable calibration handoff`).
- Preserve all P0-D2H-R, P0-D2H-CAL, and invalid-audit artifacts as read-only.
- Use `sum_logprob` as the frozen primary prediction score and `mean_logprob`
  only as a tokenizer/length-sensitivity diagnostic.
- Do not use subagents, adapter discovery/loading, training, narrow scans, RLVR,
  or automatic next-stage actions.
- The user separately requested a commit; the implementation is recorded in
  `f147fe9`. Do not push or commit later result-review changes without a new
  request.
- Use the complete verified P0-D2H-CAL run as the direct source and call its
  canonical validator before deriving the forced-choice matrix.
- Freeze the canonical continuation prefix to the empty string: the real Qwen
  chat template already ends with the assistant-turn newline, and both prompt
  prefix and standalone candidate token IDs remain exact without added whitespace.
- Refuse formal scoring unless an immutable full candidate-token audit already
  exists; formal scoring re-tokenizes and checks every frozen ID before inference.
- Store explicit non-finite/tie/error rows so aggregation can report and gate them,
  while infrastructure/model failures retain immutable failed-unit semantics.

## Errors Encountered
- Initial focused Ruff check reported three unused imports in the new runtime;
  removed them immediately. Package syntax compilation already passed.
- Raw non-finite floats would have serialized as non-standard JSON `NaN`/`Inf`;
  normalized them to JSON `null` with an explicit `score_finite=false` flag.
- First end-to-end fake-backend audit used a character tokenizer and correctly
  failed 1,920/2,304 decisions at the frozen 512-token full-sequence limit.
  Replaced the unrealistic fixture with deterministic word-level tokenization;
  no production threshold or fallback was changed.
- Result-review verification initially invoked `python -m pytest`, but this
  workstation exposes no `python` command. This is an interpreter-entry issue,
  not a test failure; verification was retried with the available repository
  runtime.
- Two initial `jq` projections treated `source_sessions.json` and
  `environment_sessions.json` as objects; both are valid one-record arrays.
  The inspection query was corrected without changing either artifact.
- A broad home-directory `find` for the copied formal artifact did not return
  promptly and was interrupted. The known scoped path
  `/Users/tourbillion/Downloads/forced_choice-f1/` is used instead.
- The first CRD runtime-integrity test imported a sibling test as
  `tests.test_p0d2hcrd_probes`, but `tests/` is intentionally not a Python
  package. The shared 24-ID fixture was made local to the test module.
- The first piece-tokenizer fixture ended the fake chat template immediately
  after `>`, so its permissive token regex merged the assistant marker with the
  continuation and correctly failed all 1,536 prefix audits. The fixture now
  mirrors the real template's trailing newline; production token rules were not
  relaxed.
- The first notebook namespace assertion could not see the Drive path because
  its source literal was split across two adjacent strings. The generated cell
  now keeps the frozen namespace in one inspectable literal.
- While adding the aggregate fake-backend test, its first patch landed before
  the preceding assertion's closing lines. Manual inspection caught the
  malformed test structure before execution; the assertion/tamper block was
  restored ahead of the new test.
- A combined plan/notes verification patch used a notes-only heading as context
  in `task_plan.md` and was rejected without changing either file. The two
  updates were applied against their actual section boundaries.
- The first Colab `plan` exposed a legacy-schema mismatch in the CRD transitive
  validator: P0-D2H-CAL units predate `result_sha256`, while the new validator
  treated it as required. The validator now relies on the already-frozen exact
  48-file raw-tree hash for legacy units and applies per-unit hashes only when
  the optional field exists.

## Status
**P0-D2H-CRD implementation verified; formal run not started** - The independent
base-only decomposition, generated Colab, source-chain validation, and tests are
complete. Existing P0-D2H-CAL-FC result-review changes are preserved and the old
experiment remains immutable. No GPU inference, training, or narrow scan was
run.

## Formal-result review

- [x] Transcribe the supplied aggregate without redefining old strict metrics.
- [x] Check every prospective model gate and the cross-scale decision.
- [x] Separate scoring-integrity checks from model-capability failures.
- [x] Record bounded claims and the prohibition on training/narrow scans.
- [x] Specify the next read-only conditional-route audit.
- [x] Run the full 148-test suite, Ruff, and `git diff --check`.

## Conditional-route raw-row audit

- [x] Phase A: Inventory the supplied formal output tree and revalidate immutable
  manifest, session, audit, aggregate, unit, and raw-file identities.
- [x] Phase B: Reproduce the complete 2,304-row aggregate and isolate the 96
  1.5B external `conditional_route` rows without modifying source artifacts.
- [x] Phase C: Analyze confusion, lesson/type/variant/slot/action factors,
  paired arms, and correct-versus-error score margins.
- [x] Phase D: Write the bounded post-hoc audit into the result report and
  project notes without redefining the frozen gate.
- [x] Phase E: Run verification and report repository status.

### Formal-result decisions

- 1.5B is `category_calibration_failed`: external overall passes, but
  `conditional_route=0.6354 < 0.75`.
- 0.5B is `oracle_failed`: forced choice does not remove its calibration deficit.
- Zero tie/error/non-finite and zero sum/mean disagreement, together with the
  passing provenance/token audit, do not indicate a scoring-integrity failure.
- `eligible_model_ids=[]`; the 1/4/8 training-complexity review must not start.
- The copied forced-choice tree is locally available and internally verified.
  The upstream P0-D2H-CAL source tree was not included, so only the stored
  cross-stage identities—not the original source files—could be revalidated.

## P0-D2H-CRD route/retrieval decomposition

### Goal

Implement and verify a separately frozen, base-only diagnostic that isolates
slot routing from external-memory retrieval while retaining a combined endpoint
for comparison.

### Phases

- [x] Phase 1: Read the complete forced-choice config/scoring/runtime/analysis,
  probe/compiler, notebook, and test contracts.
- [x] Phase 2: Freeze the CRD protocol, endpoint prompts/candidates,
  counterbalanced bank, source identity, metrics, gates, and output namespace.
- [x] Phase 3: Implement the independent package, schemas, candidate-token audit,
  scoring runtime, immutable persistence, aggregation, and CLI.
- [x] Phase 4: Add focused tests for decomposition semantics, balancing,
  candidate-only scoring, provenance, matrix integrity, recovery, and gates.
- [x] Phase 5: Add an independent generated Colab, README, and project-layout
  documentation.
- [x] Phase 6: Run focused tests, full tests, Ruff, notebook regeneration, CPU
  token audits, compile checks, and `git diff --check`.
- [x] Phase 7: Review scientific boundaries and hand off without running formal
  GPU inference, training, or a narrow scan.

### Design questions

1. Should CRD reuse only the 96 canary external conditional-route rows or build
   a new held-out, fully counterbalanced probe bank?
2. How should route-only A/B candidates and retrieval-only four-action candidates
   share source identities without leaking the expected answer?
3. Which metrics can diagnose route versus retrieval without retroactively
   changing P0-D2H-CAL-FC eligibility?
4. What prospective diagnostic gates are scientifically useful while remaining
   explicitly non-training-authorizing?

### Verification

- Focused CRD tests: 24 passed, including legacy P0-D2H-CAL manifest
  compatibility and raw-tree tamper rejection.
- Full repository tests with train extras installed: 172 passed.
- Full-repository Ruff, compileall, generated-notebook equality, CLI help, and
  `git diff --check`: passed.
- The cached frozen 1.5B tokenizer passed all 1,536 prompt and 5,376
  continuation audits. Maximum prompt/full lengths were 264/267 tokens;
  action candidates use three tokens and slot candidates use two.
- The complete fake backend persisted and revalidated 1,536 rows / 5,376
  candidates, aggregated them, and kept every training/scan flag false.

### Final scientific boundary

- The split implementation and CPU audits are complete; no formal CRD GPU score
  or empirical route/retrieval conclusion has been generated.
- The downloaded forced-choice copy contains the complete direct tree but not
  its upstream P0-D2H-CAL/P0-D2H-R/P0-D2 files. Formal `plan/audit/run` therefore
  remains intentionally gated on the full Drive source chain used by the
  generated Colab.
- CRD cannot change the old P0-D2H-CAL-FC category failure or create an eligible
  model. Every prospective CRD status remains diagnostic only.
- No commit or push was made because the user did not request one for this
  implementation turn.
