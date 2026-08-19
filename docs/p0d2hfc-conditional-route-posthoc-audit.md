# P0-D2H-CAL-FC conditional-route post-hoc audit

## 1. Status and scope

- **Analysis status:** post-hoc, supplementary, read-only
- **Source run:** `p0d2hfc-forced-choice-f604efe6bf`
- **Source directory inspected:**
  `/Users/tourbillion/Downloads/forced_choice-f1/`
- **Target rows:** 1.5B scale canary × `external` ×
  `conditional_route`
- **Target row count:** 96
- **Frozen gate changed:** no
- **Eligibility changed:** no
- **Source artifacts modified:** no

This audit implements the follow-up required by the failed prospective
P0-D2H-CAL-FC category gate. It analyzes every target row and does not select a
favorable subgroup.

## 2. Local integrity verification

The copied forced-choice output is internally self-consistent:

| Check | Observed | Status |
|---|---:|---|
| Manifest units | 48/48 `verified` | pass |
| Raw JSONL files | 48 | pass |
| Unique decision rows | 2,304 | pass |
| Unique candidate-audit records | 2,304 | pass |
| Candidate sequences | 9,216 | pass |
| Candidate-audit SHA-256 matches manifest | yes | pass |
| Per-unit raw SHA-256 matches manifest | 48/48 | pass |
| Prompt/action/token linkage | 2,304/2,304 | pass |
| Candidate sum/mean recomputation | 9,216/9,216 | pass |
| Stored ranking, prediction, and margin recomputation | 2,304/2,304 | pass |
| Execution-session/environment linkage | 2,304/2,304 | pass |
| Aggregate arm/category accuracy recomputation | exact | pass |
| Tie/error/non-finite rows | 0 | pass |

The copied artifact identities are:

```text
manifest_sha256 =
  89ba71ea0fde052089329e630d23c1a86652bc47a129de1921786c8e0b4dd214
candidate_token_audit_sha256 =
  d140cfb2ac0c57b4819d3793b59f5f7c767fe428336be856fc9299460731b230
forced_choice_raw_tree_sha256 =
  dfe9b9014966f484e988a2f6d9f8e18d4ee7a1b5f06d671ae2e973cef8a0b24a
summary_sha256 =
  24b1fe51bf400e5c1e0b15aa68d67b0db913591cce082aeac13b9396854d22e1
next_stage_decision_sha256 =
  789166b715c53179f64733d1215e1dcd30188e09670ac10ea45489ceafa6040e
```

The recorded formal scoring environment used an NVIDIA L4 with
`nf4-bfloat16`, CUDA 13.0, Python 3.12.13, PyTorch 2.13.0, and Transformers
5.14.1. The recorded Git commit is `f147fe943c309bfb30ec9c6d6720d3fb9dd946cb`
and the experiment code SHA-256 is
`a4780072732eea0a9218d59db121c6e63b1e15df6227fc3122121417e1bc5ade`.

The copied directory does not include the upstream P0-D2H-CAL source tree.
Consequently, the local audit verified every stored cross-stage source identity
for internal agreement but could not independently rehash the original source
manifest and source raw files. That transitive limitation does not affect the
row-level computations below, but it remains part of the provenance boundary.

## 3. Paired endpoint localization

The target endpoint has 61/96 correct decisions:

```text
forced-choice external conditional_route = 0.6354
errors                                      = 35/96
```

The same 96 probes pair as follows:

| No-write correct | External correct | Oracle correct | Rows |
|---:|---:|---:|---:|
| no | no | no | 1 |
| no | no | yes | 34 |
| no | yes | yes | 61 |

No-write is wrong on all 96 rows, whereas answer-copy oracle is correct on
95/96. In particular, 34 of the 35 external errors are correct under the
paired oracle arm. This rules out a general four-candidate scoring or
answer-label interface failure for those 34 rows and localizes the deficit to
using the external record together with the routing composition.

The single row that also fails oracle is:

```text
lesson_id       = P_pair02_a
probe_id        = P_pair02_a_conditional_route_03
expected        = act_p3
oracle predicts = act_v9
oracle margin   = 0.5000
```

## 4. Relation to strict generation

| Linked strict | Forced choice | Rows |
|---:|---:|---:|
| wrong | wrong | 32 |
| wrong | correct | 9 |
| correct | wrong | 3 |
| correct | correct | 52 |

Forced choice therefore produces a net gain of 6/96 over linked strict
generation: 0.6354 versus 0.5729. Eleven strict rows are invalid; forced choice
recovers eight of them and remains wrong on three. Format removal helps, but it
does not account for the remaining 35 forced-choice errors.

## 5. Exploratory factor breakdown

All breakdowns in this section are descriptive and post-hoc. They have no
prospectively frozen subgroup thresholds and must not be used to change
eligibility.

| Factor | Group | Correct / rows | Accuracy |
|---|---|---:|---:|
| Lesson type | fact mapping | 34/48 | 0.7083 |
| Lesson type | procedure recovery | 27/48 | 0.5625 |
| Variant | 1 | 17/24 | 0.7083 |
| Variant | 2 | 16/24 | 0.6667 |
| Variant | 3 | 10/24 | 0.4167 |
| Variant | 4 | 18/24 | 0.7500 |
| Selected slot | A | 27/48 | 0.5625 |
| Selected slot | B | 34/48 | 0.7083 |
| Lesson side | `a` | 26/48 | 0.5417 |
| Lesson side | `b` | 35/48 | 0.7292 |

Slot A is used by odd variants. Its lower marginal result is therefore
confounded with variant 3: variant 1 also selects A but reaches 0.7083, whereas
variant 3 reaches 0.4167. The most pronounced observed intersection is:

```text
procedure_recovery × variant 3 = 2/12 = 0.1667
all other target rows          = 59/84 = 0.7024
```

This intersection contributes 10 of the 35 errors, but errors are not confined
to it. Twenty of 24 lessons contain at least one error; only four lessons are
4/4 correct. The deficit is therefore distributed rather than caused by one or
two corrupted lessons.

## 6. Action and option-order behavior

Expected actions are exactly balanced at 24 rows each:

| Expected action | Correct / rows | Accuracy | False negatives | False positives |
|---|---:|---:|---:|---:|
| `act_k2` | 16/24 | 0.6667 | 8 | 7 |
| `act_n7` | 14/24 | 0.5833 | 10 | 10 |
| `act_p3` | 11/24 | 0.4583 | 13 | 3 |
| `act_v9` | 20/24 | 0.8333 | 4 | 15 |

The model predicts `act_v9` 35 times and `act_p3` only 14 times, despite 24
expected instances of each. The complete confusion matrix is:

| Expected \ predicted | `act_n7` | `act_p3` | `act_v9` | `act_k2` |
|---|---:|---:|---:|---:|
| `act_n7` | 14 | 2 | 6 | 2 |
| `act_p3` | 4 | 11 | 6 | 3 |
| `act_v9` | 2 | 0 | 20 | 2 |
| `act_k2` | 4 | 1 | 3 | 16 |

Expected candidate positions are also exactly balanced at 24 each, but predicted
positions are not:

| Candidate-list position | Expected count | Predicted count | Accuracy when expected |
|---:|---:|---:|---:|
| 1 | 24 | 25 | 0.5417 |
| 2 | 24 | 36 | 0.8333 |
| 3 | 24 | 16 | 0.6250 |
| 4 | 24 | 19 | 0.5417 |

The `procedure_recovery × variant 3` subset is itself imbalanced by the frozen
action-order schedule: its expected action is in position 1 on five rows,
position 2 on one row, and position 4 on six rows; the model predicts position
2 on six of the 12 rows. This makes route difficulty, action identity, and option
position observationally entangled. The post-hoc data support an action/order
preference diagnosis, but not a unique causal attribution.

## 7. Margin structure

| Outcome | Rows | Mean margin | Median margin | Q1 | Q3 |
|---|---:|---:|---:|---:|---:|
| Correct | 61 | 1.3525 | 1.1250 | 0.5000 | 1.8750 |
| Error | 35 | 0.6464 | 0.3751 | 0.1875 | 1.0000 |

For the 35 errors, the expected action ranks second on 21 rows, third on 11,
and fourth on three. Errors are generally lower-margin than correct decisions,
but they are not all near-ties: eight errors have margin at least 1.0, three
have margin at least 1.5, and one exceeds 2.0.

## 8. Bounded conclusion

The audit supports the following localization:

1. The copied forced-choice artifact is internally intact; no row-level
   tokenization, score arithmetic, ranking, session, unit-hash, or aggregate
   inconsistency was found.
2. Output formatting is not the primary remaining cause. Forced choice improves
   linked strict accuracy by only 6/96 on this category.
3. Thirty-four of 35 external errors pass the paired oracle arm, so the dominant
   deficit lies in using/routing the external record rather than expressing the
   final action through the four-candidate endpoint.
4. Errors are broad but enriched in `procedure_recovery × variant 3`, lesson side
   `a`, under-predicted `act_p3`, over-predicted `act_v9`, and candidate position
   2. These factors are confounded by the frozen probe schedule and cannot be
   interpreted as separate causal mechanisms.

The current rows expose only the final action likelihood. They cannot determine
whether a wrong answer arose while selecting slot A/B or while retrieving the
action after selecting the correct slot.

## 9. Next-stage boundary

No training or narrow scan should start. If the project continues, the next
experiment should first be a separately frozen, base-only diagnostic with:

1. a route-only forced-choice endpoint that selects slot A/B;
2. a retrieval-only four-action endpoint in which the correct slot is supplied;
3. the original combined route-plus-retrieval endpoint;
4. exact counterbalancing of action identity and option position within lesson
   type, route variant, target slot, and lesson side;
5. a new probe bank or held-out cohort so the post-hoc pattern is not converted
   into an in-sample gate.

Its protocol, thresholds, namespace, and approval must be independent.
P0-D2H-CAL-FC remains `category_calibration_failed`, and
`eligible_model_ids=[]`.
