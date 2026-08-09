# Notes: P0-D2H-R evaluation repair

## Observed run

- `late-matched` beat `full-base` by 10.07 percentage points on hard accuracy.
- `binding_decoys` was at a common 25% floor for every parametric/no-write arm.
- `external` had 33.33% invalid outputs and 0% `long_context` accuracy.
- External prompts averaged about 30 more input tokens than other arms.

## Initial code evidence

- External memory is prepended to the probe.
- Tokenization currently uses truncation with the source run's `max_length`.
- The frozen locus gate consumes only the late-vs-full contrast and does not express
  suite-quality readiness.

## Implementation constraints

- Existing source manifests, adapters, aggregates, and hard-probe outputs are
  read-only.
- Configuration/probe changes require a new attempt namespace.
- Any new quality thresholds must be explicit in provenance and aggregate output.

## Design selected

- Configuration schema v2 records both `source_training_max_length` and
  `evaluation_max_length`.
- Default repaired evaluation length is 512 tokens.
- A deterministic prompt-token audit is written before inference and hashed into
  the manifest.
- Raw rows record untruncated length, truncation count, instruction preservation,
  prompt variant, and evaluation length.
- External memory is inserted after the challenge body and immediately before the
  strict output instruction.
- The frozen locus label remains intact. A separate suite-quality result checks:
  no truncation, a working external anchor, and absence of common parametric
  category floors.
- The repaired notebook uses `pipeline-r1/hard_probe-r1`.

## Verification evidence

- 105 repository tests pass.
- Ruff lint and `git diff --check` pass.
- Real `Qwen/Qwen2.5-0.5B-Instruct` tokenizer audit over 768 prompt variants:
  - maximum plain prompt: 259 tokens;
  - maximum repaired external prompt: 303 tokens;
  - prompts truncated at 512: 0;
  - output instructions missing at 512: 0.
- This confirms that the old 256-token evaluation limit could truncate some plain
  hard prompts as well as the longer external prompts.

---

# Notes: P0-D2H-CAL oracle calibration

## Fixed scope

- Base models only; no adapter discovery, activation, training, or rollback.
- Three arms: `no_write`, `external`, and `answer_copy_oracle`.
- Reuse the verified P0-D2H-R 24-lesson, 16-probe hard suite read-only.
- Support a primary 0.5B model and an optional same-family larger-model canary.
- Independent manifest schema, raw rows, aggregate schema, Colab directory, and
  Drive namespace.

## Frozen default gates

- Answer-copy oracle: overall accuracy at least 0.90, every category at least
  0.80, invalid rate at most 0.01.
- External: overall accuracy at least 0.75 and invalid rate at most 0.05.
- Every prompt must be untruncated and preserve the strict output instruction.

## Required decisions

- Oracle fails: output-copy/instruction interface bottleneck.
- Oracle passes but external fails: verified-memory use under distractors fails.
- 0.5B fails while larger model passes: model-scale canary supports a capacity
  bottleneck.
- Oracle and external pass: anchor calibration is eligible for a separately
  frozen training-complexity experiment.

## Implementation evidence

- New package and CLI: `plasticity_placement.p0d2hc` /
  `plasticity-p0d2hc`.
- Independent Colab and Drive root:
  `notebooks/p0d2h_calibration/` and
  `plasticity-p0d/hard-probe-calibration/v1/`.
- Source-only matrix: 1,152 rows; source plus canary: 2,304 rows.
- Atomic lesson-model outputs recover normally after Colab interruption.
- No adapter path, training command, narrow scan, or automatic next-stage action
  is exposed by the CLI or notebook.
- Focused tests: 10 passed. Full repository: 115 passed. Ruff and diff checks
  passed.

---

# Notes: P0-D2H-CAL invalid-output audit

## Observed calibration result

- Source 0.5B: oracle `0.7891`, invalid `0.0`; frozen status `oracle_failed`.
- Scale canary 1.5B: oracle `0.9948`, invalid `0.0`; oracle gate passes.
- Scale canary external: strict accuracy `0.6901`, invalid `0.2318`.
- Scale canary no-write: strict accuracy `0.1276`, invalid `0.2448`.
- The strict parser accepts only a generated string exactly equal to one frozen
  action token; invalid does not by itself imply that no correct action was
  mentioned.

## Audit constraints

- No new inference.
- No raw-row or manifest mutation.
- No retrospective gate or threshold change.
- Supplementary semantic recovery must remain visibly separate from strict
  exact-action accuracy.

## Frozen taxonomy

- `empty_output`
- `expected_action_with_extra_text`
- `wrong_action_with_extra_text`
- `multiple_actions_including_expected`
- `multiple_actions_excluding_expected`
- `no_allowed_action_at_generation_limit`
- `no_allowed_action_other`

Semantic recovery is conservative: exactly one distinct allowed action must occur,
and it must equal the expected action. Mentioning the expected action alongside a
distractor is not counted as semantically correct.

## Implemented artifacts

- CLI: `plasticity-p0d2hc audit-invalid`.
- Source module: `p0d2hc/invalid_audit.py`.
- Independent outputs: summary JSON, Markdown report, and classified invalid
  JSONL.
- Summaries: model × arm and model × arm × category, including lesson-clustered
  strict/semantic intervals and paired recovery-gain intervals.
- Independent CPU-only Colab:
  `notebooks/p0d2h_invalid_audit/p0d2h_invalid_output_audit_colab.ipynb`.
- Source manifest and raw tree are hashed before and after the Colab audit.
- Focused P0-D2H-CAL tests: 24 passed.
- Full repository validation: 129 passed; Ruff, notebook regeneration, and diff
  checks passed.

---

# Notes: P0-D2H-CAL verified audit interpretation and next stage

## Verified audit evidence

- Audit run ID: `p0d2hc-invalid-audit-95ae30ec5e`.
- Complete source: 2 models × 24 lessons × 3 arms × 16 probes = 2,304 rows.
- Strict-invalid rows: 183, all from the 1.5B scale canary.
- Canary external:
  - strict `0.6901 [0.6562, 0.7240]`;
  - invalid `0.2318 [0.1979, 0.2630]`;
  - conservative semantic `0.9167 [0.8906, 0.9427]`;
  - paired semantic gain `+0.2266 [0.1953, 0.2552]`;
  - 87 of 89 invalids were the unique correct action plus extra text.
- Canary no-write:
  - semantic `0.1901 [0.1406, 0.2422]`;
  - only 24 of 94 invalids were recoverable and 70 contained a wrong action.
- Canary external by category:
  - binding `1.0000` semantic;
  - conflict `1.0000` semantic;
  - conditional `0.6667` semantic;
  - long context `1.0000` semantic versus `0.1875` strict.
- No invalid output reached the generation limit.
- The 0.5B model had no formatting recovery; its oracle remained `0.7891`.

## Bounded interpretation

- Most of the 1.5B overall external strict failure is attributable to extra-text
  format noncompliance.
- The 1.5B `conditional_route` deficit remains semantic/compositional after
  recovery.
- The 0.5B result is not explained by invalid formatting.
- The frozen result remains `mixed_scale_result` /
  `calibration_followup_required`; no model is eligible for training.
- P0-D2H-CAL and its audit contain external/no-write/oracle only, not LoRA
  parameter conditions.

## Next-stage design decision

- Implement a new base-only format-stable forced-choice calibration.
- Score all four exact action continuations with teacher-forced conditional
  log-likelihood; use sequence sum as primary and token mean as sensitivity.
- Preserve the same 24-lesson suite, three arms, and two immutable model revisions.
- Require an external per-category floor so ceiling binding/conflict/long-context
  results cannot hide conditional-route failure.
- Keep all training, adapter loading, narrow scans, and automatic next-stage
  actions outside this experiment.

---

# Notes: P0-D2H-CAL-FC implementation

## Baseline

- Branch: `agent/add-lora-evaluation`.
- Required ancestor `ba54377` is present.
- Pre-task documentation state committed as `db185e5`.
- The user later requested a commit; the completed implementation is commit
  `f147fe9`. It has not been pushed.

## Frozen implementation constraints

- New independent package, schema, manifest, CLI, notebook, outputs, and Drive
  namespace.
- Read-only reuse of the verified hard-probe bank and complete calibration run.
- Exactly 2 models × 24 lessons × 3 arms × 16 probes = 2,304 decision rows.
- Four ordered frozen candidates; candidate `sum_logprob` is primary.
- Prompt tokens must not contribute to candidate score.
- Ties, non-finite scores, prompt/candidate tokenization mismatches, missing rows,
  duplicates, provenance mismatches, and silent truncation are explicit failures.
- Gate thresholds and automatic-action false flags must be manifest identity.
- No adapters, training, narrow scan, GRPO/RLVR, or 1/4/8 experiment execution.

## Research log

- P0-D2H-CAL already validates the complete P0-D2H-R source, loads the frozen hard
  bank, renders all three arms through `render_calibration_probe`, and stores
  48 atomic model×lesson files for two models.
- Its canonical chat path is `chat_prompt(tokenizer, prompt)` with
  `add_generation_prompt=True`, followed by tokenization with
  `add_special_tokens=False`.
- The direct P0-D2H-CAL source has a stable row key
  `(calibration_model_id, lesson_id, arm, probe_id)` and stores the exact formatted
  prompt hash needed for source pairing.
- Source validation already rejects incomplete units, malformed 2,304-row
  matrices, changed source manifests/summaries/probes, prompt audit mismatches,
  model revision mismatches, and adapter-bearing rows.
- Existing recovery semantics make `verified` and `failed` terminal. A complete
  raw file left by interruption can be verified and adopted; incomplete or
  conflicting artifacts are rejected.
- Existing lesson-clustered bootstrap uses deterministic seed `20260726`.

## Token-boundary audit

- Real cached `Qwen/Qwen2.5-0.5B-Instruct` tokenizer was audited on CPU.
- The formatted prompt ends with
  `"<|im_end|>\n<|im_start|>assistant\n"`.
- Appending `act_n7` preserves the complete prompt-token prefix and yields the
  same candidate IDs as standalone tokenization: `[531, 1089, 22]`.
- Appending `" act_n7"` is also prefix-stable but changes the first candidate
  token. Because the source generation endpoint starts immediately after the
  assistant-turn newline, the new endpoint freezes canonical leading whitespace
  as the empty string.
- A leading newline does not preserve the separately tokenized prompt prefix and
  is therefore invalid.

## Frozen P0-D2H-CAL-FC architecture

- Direct source: complete verified `p0d2hc-manifest-v1` plus its 2,304 raw rows.
- New schemas: `p0d2hfc-config-v1`, `p0d2hfc-candidate-token-audit-v1`,
  `p0d2hfc-manifest-v1`, row v1, summary v1, and next-stage v1.
- New package modules: config, scoring, manifest, runtime, analysis, and CLI.
- CPU `audit` freezes prompt IDs, four candidate IDs, concatenation invariants,
  source-row hashes, chat-template hashes, and full-sequence length checks.
- GPU `run` requires the audit, loads each frozen base model once, and batches the
  four candidates per prompt.
- Primary candidate score is the sum of candidate-token log probabilities; mean
  is diagnostic only.
- Output rows link source manifest/run/raw-tree/file/row identities and old strict
  correctness without redefining the strict endpoint.

## Verification evidence

- Focused P0-D2H-CAL-FC tests: 19 passed.
- Focused Ruff: passed.
- End-to-end fake-backend exercise built a complete verified P0-D2H-CAL source,
  then validated:
  - 2,304 direct source rows;
  - 2,304 forced-choice decisions and 9,216 candidate sequences;
  - 48 model×lesson atomic verified units with raw-file hashes;
  - one model load per model on the initial run;
  - zero model reloads and zero raw rewrites on resume;
  - aggregation, category gates, and
    `scale_capacity_bottleneck_supported` routing.
- Full real-tokenizer CPU audit over both Qwen2.5 Instruct tokenizers:
  - 1,152 decisions / 4,608 candidates per model;
  - zero prompt-prefix, standalone-ID, decode, or length failures;
  - maximum prompt length 303 tokens;
  - maximum prompt-plus-candidate length 306 tokens;
  - every action continuation uses three tokens;
  - both chat-template SHA-256 values are
    `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`.
- Final full repository tests: 148 passed.
- Final full-repository Ruff: passed.
- Generated notebook equals `build_p0d2hfc_notebook.py`; all code cells parse.
- CLI entry point and help output execute successfully.
- Notebook JSON validation, Python compileall, `git diff --check`, and new-file
  trailing-whitespace scan passed.

## Final scientific boundary

- Implementation and CPU audits are complete; no formal 0.5B/1.5B GPU scores
  have been generated in this workspace.
- The old P0-D2H-CAL strict gate and invalid-output audit remain unchanged.
- Any future `training_complexity_review_eligible` result permits manual design
  review only. Training and narrow-scan flags remain false in config, summary,
  decision output, CLI plan, protocol, and notebook.
- The implementation was committed as `f147fe9` after the user explicitly
  requested the commit. It has not been pushed.

---

# Notes: P0-D2H-CAL-FC formal result

## Supplied aggregate

- Run ID: `p0d2hfc-forced-choice-f604efe6bf`.
- 1.5B external: `0.9089 [0.8828, 0.9349]`; oracle: `0.9948`; no-write:
  `0.1745`.
- 1.5B external categories: binding `1.0000`, conflict `1.0000`,
  conditional route `0.6354`, long context `1.0000`.
- 0.5B external: `0.4349 [0.3724, 0.5052]`; oracle: `0.7812`; no-write:
  `0.2500`.
- External-minus-no-write paired lower confidence bounds are positive for both
  models.
- Tie/error/non-finite and sum/mean disagreement rates are zero for every arm.

## Gate interpretation

- 1.5B passes external overall but fails the prospective every-category floor;
  status is `category_calibration_failed`.
- 0.5B fails oracle overall/categories and external overall/categories; status
  is `oracle_failed`.
- Cross-scale status is `scale_improvement_without_full_calibration`.
- No model is eligible; training and narrow scans remain prohibited.

## Next action

- Preserve the outcome as negative calibration evidence.
- The formal output tree was later supplied and all 96 1.5B external
  `conditional_route` rows were audited descriptively.
- Do not alter the old gate, omit failures, or promote a post-hoc subgroup.

## Documentation verification

- Full repository tests: 148 passed.
- Full-repository Ruff: passed.
- `git diff --check`: passed.

---

# Notes: P0-D2H-CAL-FC conditional-route post-hoc audit

## Local integrity

- Supplied root: `/Users/tourbillion/Downloads/forced_choice-f1/`.
- Manifest SHA-256:
  `89ba71ea0fde052089329e630d23c1a86652bc47a129de1921786c8e0b4dd214`.
- Candidate audit SHA-256:
  `d140cfb2ac0c57b4819d3793b59f5f7c767fe428336be856fc9299460731b230`;
  it matches the manifest.
- Local forced-choice raw-tree SHA-256:
  `dfe9b9014966f484e988a2f6d9f8e18d4ee7a1b5f06d671ae2e973cef8a0b24a`.
- Verified 48/48 unit hashes, 2,304 unique rows, 2,304 unique audit records,
  9,216 candidate token/score sequences, all rankings/margins, session links,
  and aggregate accuracies with zero inconsistency.
- The upstream P0-D2H-CAL tree was not included, so original cross-stage source
  files could not be independently rehashed; their stored identities agree
  throughout the copied tree.

## Target findings

- 1.5B external `conditional_route`: 61/96 correct, 35 errors.
- Paired `(no_write, external, oracle)` outcome counts:
  `(0,0,0)=1`, `(0,0,1)=34`, `(0,1,1)=61`.
- Thus 34/35 external errors pass oracle; the dominant deficit is external
  record use/routing, not the final four-candidate interface.
- Linked strict versus forced choice:
  both wrong 32, forced-only correct 9, strict-only correct 3, both correct 52.
- Errors occur in 20/24 lessons; only four lessons are 4/4.
- Exploratory enrichment:
  `procedure_recovery × variant 3 = 2/12`, versus 59/84 elsewhere.
- Expected actions are balanced, but predictions over-select `act_v9`
  (35 predictions) and under-select `act_p3` (14 predictions).
- Expected option positions are balanced, but position 2 is predicted 36 times.
- Wrong decisions have median margin 0.3751 versus 1.1250 when correct, although
  eight wrong decisions still have margin at least 1.0.

## Scientific boundary

- Action identity, option position, route variant, target slot, lesson side,
  and lesson type are observationally confounded in the frozen bank.
- The endpoint observes only final action scores, so it cannot distinguish
  route selection from post-route retrieval.
- Any continuation requires a separately frozen base-only route-only versus
  retrieval-only diagnostic on a new or held-out, fully counterbalanced bank.
- No eligibility, training, or narrow-scan decision changes.

## Final verification

- Full repository tests: 148 passed.
- Full-repository Ruff: passed.
- `git diff --check` and local documentation-link checks: passed.

---

# Notes: P0-D2H-CRD route/retrieval decomposition

## Frozen design

- New package/stage: `p0d2hcrd` / `hard_probe_route_decomposition`.
- Direct source: exact completed P0-D2H-CAL-FC run
  `p0d2hfc-forced-choice-f604efe6bf`, transitively validated by the CRD
  exact-hash source-chain loader.
- Exact source hashes are frozen in the CRD protocol.
- Model: only the frozen 1.5B `scale_canary`; the 0.5B answer-copy oracle failed
  and is outside this localization question.
- New namespace:
  `plasticity-p0d/hard-probe-route-decomposition/v1/pipelines/`.
- Endpoints:
  - `route_only`: routing table and opaque slots, candidates
    `slot_a/slot_b`, no memory or action leakage;
  - `retrieval_only`: selected live record plus verified memory, no routing
    decision, four action candidates;
  - `combined`: routing table, live/archived slots, verified memory, four action
    candidates.

## Counterbalanced matrix

- `route_only`: 4 route variants × 2 slot-candidate orders ×
  2 slot-content orders = 16 rows/lesson.
- `retrieval_only`: 4 record variants × 4 action-panel rotations =
  16 rows/lesson.
- `combined`: 4 route variants × 4 action-panel rotations ×
  2 slot-content orders = 32 rows/lesson.
- Total: 24 lessons × 64 = 1,536 decisions and 5,376 candidate sequences in
  24 atomic units.
- Action-panel position is fully crossed with combined route/content factors;
  target slot, route candidate position, and slot presentation order are
  balanced.

## Diagnostic boundary

- Prospective floors: route ≥0.90, retrieval ≥0.90, combined ≥0.75, and zero
  scoring invalids.
- Status distinguishes route, retrieval, shared-component, and composition
  bottlenecks.
- All statuses are diagnostic. Training-review eligibility and all automatic
  action flags remain false.

## Implemented CRD surface

- Package modules: `config`, `probes`, `scoring`, `manifest`, `runtime`,
  `analysis`, and `cli` under `p0d2hcrd`.
- CLI: `plasticity-p0d2hcrd environment|plan|audit|run|aggregate`.
- Source loader uses exact immutable P0-D2H-CAL-FC hashes and follows its
  stored paths to verify P0-D2H-CAL raw/prompt artifacts, P0-D2H-R hard probes,
  and the P0-D2 compiled bank. It does not invoke the old global-code-hash
  validator after the new package changes the repository code identity.
- CPU preflight writes immutable decomposition-bank and variable-candidate token
  audits; the GPU runtime requires both before loading the one 1.5B base model.
- Aggregate reports lesson-clustered endpoint intervals, factor summaries,
  source-linked old accuracy, paired composition contrasts, and one of six
  diagnostic-only statuses.
- Generated Colab and CLI contain no adapter discovery, training, narrow scan,
  or automatic next-stage path.

## CRD verification evidence

- Focused CRD tests: 24 passed, including legacy P0-D2H-CAL manifest
  compatibility and raw-tree tamper rejection.
- Final full repository tests with train extras installed: 172 passed.
- Full-repository Ruff, Python compileall, notebook regeneration/equality, CLI
  help, and `git diff --check`: passed.
- Complete cached real-tokenizer CPU audit for frozen
  `Qwen/Qwen2.5-1.5B-Instruct` revision:
  - 1,536 decisions and 5,376 candidate continuations;
  - endpoint counts 384 route-only, 384 retrieval-only, 768 combined;
  - zero prefix, standalone-ID, decode, overflow, or bank failures;
  - maximum formatted prompt 264 tokens and maximum full input 267 tokens;
  - action candidates use three tokens and slot candidates use two;
  - chat-template SHA-256
    `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`.
- Complete fake-backend exercise persisted and revalidated all 1,536 rows /
  5,376 candidate sequences, produced the aggregate, and confirmed all
  training/scan flags remain false.

## CRD handoff boundary

- No formal CRD GPU inference has been run, so there is no empirical routing,
  retrieval, or composition result yet.
- The local copied forced-choice tree omits its upstream source directories;
  the exact CRD validator correctly requires those full trees. The generated
  Colab points at the original Drive namespaces where the stored absolute
  source paths resolve.
- The old P0-D2H-CAL-FC gate remains unchanged and still has no eligible model.
- No CRD status can authorize training or a narrow scan.

---

# Notes: CBR budget mismatch recovery and corrected experiment

## Confirmed failure

- Frozen model has 28 transformer layers.
- A q/v LoRA rank-layer unit contributes 4,864 trainable parameters.
- `full_depth`: `28 * 8 * 4,864 = 1,089,536`.
- Current `late_matched`: `10 * 24 * 4,864 = 1,167,360`.
- Relative excess is exactly `16 / 224 = 0.07142857142857142`.
- Evaluation fails before `_claim_evaluation`, so the shown first-unit failure does
  not consume an evaluation claim or create a valid result artifact.

## Scientific scope

- Existing adapters remain valid trained artifacts and can support descriptive
  full-matrix evaluation.
- The original parameter-matched placement claim is invalid for CBR-v1.
- Curriculum comparisons within the same placement retain equal parameter budgets.
- Per-adapter OFF/ON remediation and preservation results remain interpretable.
- Cross-placement effect and drift differences must be labeled exploratory and
  parameter-count-confounded.

## Corrected budget

- Use explicit layers 20 through 27 (last eight layers), rank 28, alpha 56.
- Corrected late budget: `8 * 28 * 4,864 = 1,089,536`, exactly equal to full depth.
- A fresh experiment identity must freeze the source CBR-v1 run, imported full-depth
  trees, explicit layers, and new late-only authorization before any replacement
  training.

## Implemented contracts

- `budget-deviation-template` produces a read-only template bound to the original
  preregistration, observed budget audit, and current recovery code hash.
- `authorize-budget-deviation` requires an external human-authored approval and writes
  an immutable adopted authorization; it leaves all training flags false.
- Recovered aggregate status is forced to `budget_deviation_full_matrix_complete` and
  placement scope to `exploratory_parameter_count_confounded`.
- `plan-corrected` requires the verified completed recovered source, copies frozen
  preflight banks, imports six full-depth controls by exact metadata/adapter identities,
  and leaves exactly six corrected late units pending.
- Corrected runs use the ordinary authorize/train/evaluate/aggregate/verify lifecycle;
  authorization permits six training runs and twelve locked evaluations.

## Verification checkpoint

- Focused CBR tests: 19 passed.
- Full repository pytest: passed.
- Full Ruff, compileall, and `git diff --check`: passed.
