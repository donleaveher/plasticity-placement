# P0-D2H route-transfer bridge audit protocol

## Status and scope

CPR-cpr2 is closed and failed. In its single q2 qualification, adapter ON changed route-only
from `0.5052` to `0.9922` but external conditional-route from `0.6667` to `0.5625`
(`95%` lesson-cluster CI for ON−OFF `[-0.1875, -0.0208]`). Retrieval-only remained `1.0000`.
One base combined row had a top tie that excluded the expected candidate; this isolated scoring
integrity violation does not alter routing correctness or rescue qualification.

This bridge audit is a post-failure mechanism diagnostic on the immutable cpr2 adapter. It is
not a CPR retry, training experiment, prompt search, or capacity experiment. Every output keeps
the historical gate unchanged and sets training and 1/4/8 authorization to false.

## Question

Did cpr2 learn a route operator that fails to transfer because of one or more prompt-format
axes, or does routing remain intact under the external prompt while the route-to-action stage
fails?

The three primary format axes are:

1. route grammar: explicit two-entry `map` versus one-marker/default `default` rule;
2. slot lexicon: training-like `slot_a/slot_b` versus external-like `A/B`;
3. payload: opaque slot records versus live/archive records plus verified external memory.

The external payload level reuses the frozen hard probe's exact live/archive slot clauses,
including archive identifiers and rotated distractor order, plus the exact verified-memory note.
Only the preregistered slot-line order may be counterbalanced for the slot-readout matrix.

## Frozen bridge bank

The primary matrix is a complete `2×2×2` factorial. Each of the eight cells contains the same
24 frozen lessons and four conditional-route variants, for 96 decisions per cell. Target slot
and candidate position are balanced. Every prompt asks for a slot token, so route correctness is
observed without requiring action retrieval.

The `map + snake + opaque` cell is not newly paraphrased: its 96 rows are exact frozen CRD
route-only prompts. Each corresponding hybrid row inherits that anchor's current marker,
slot-content order, and candidate order. Across rows, the current marker points to the first and
second routing-table entries equally often. This prevents a first-mentioned-slot shortcut from
being mistaken for a route-grammar rescue. Preflight checks each formatted anchor hash against
the immutable q2 `adapter_off_crd.jsonl` artifact bound by its qualification manifest. It also
binds the complete static scoring identity: expected candidate, candidate order, target slot,
current marker, counterbalance orders, route variant, source probe, and source row.

Two 96-row action cells are added:

- `external_action`: the exact frozen external conditional-route action prompt;
- `forced_slot_action`: the exact same slot records and verified-memory payload with an
  independently supplied correct slot; only the routing clause and routing instruction change.

The complete bank therefore contains 960 prompts. Each is scored adapter OFF and then ON in one
loaded base-model object. Candidate-only forced-choice scoring, token audits, one OFF replay
sentinel, immutable source snapshots, and exact adapter hashing are required.

All 768 primary factorial rows require `error_status == ok` and zero top-score ties in both
adapter states. Any primary tie makes every barrier claim ineligible and yields
`scoring_integrity_failed`; it is not scored as an ordinary error. The two secondary action cells
allow only expected-compatible ties for diagnostic reporting.

Immediately after inference, the raw OFF and ON rows and their hashes are written as an immutable
checkpoint before any pairing, bootstrap, or classification code runs. If post-inference analysis
fails, those raw scores remain available, the run is marked failed, and no automatic recovery or
rerun is authorized. Any read-only recovery requires a separate explicit protocol.

## Primary estimands and multiplicity

The fully external slot-readout cell is `default + natural + external`. For each factor, the
primary rescue contrast reverts only that factor while holding the other two external:

```text
rescue = (ON−OFF correctness with factor reverted)
       − (ON−OFF correctness in fully external slot readout)
```

This difference-in-differences controls for changes in base prompt difficulty. Uncertainty uses
a 10,000-sample bootstrap clustered by `lesson_id`. The three primary two-sided intervals use
98.333% coverage (Bonferroni family-wise control at 0.05). A factor is called a supported
material barrier only if the rescue estimate is at least `0.10` and the adjusted interval lower
bound is above zero. Margins, cell accuracies, and the action triangle are secondary diagnostics
with ordinary 95% cluster intervals.

Multiple factors may be supported. A null result means only that no single preregistered factor
met the material rescue rule; it does not establish that prompt format is irrelevant.

## Lifecycle and stop rule

```bash
uv run plasticity-p0d2hrtb plan \
  --output /new/route-transfer-rtb1 \
  --cpr-output /frozen/composition-remediation-cpr2 \
  --qualification-summary /frozen/same-runtime-q2/summary.json

uv run plasticity-p0d2hrtb authorize \
  --output /new/route-transfer-rtb1 \
  --authorization /outside/authorization-rtb1.json

uv run plasticity-p0d2hrtb run --output /new/route-transfer-rtb1
```

Planning freezes and token-audits the bank before authorization. Exactly one audit run is
allowed. No cell, threshold, prompt, or adapter may change after results. A supported barrier is
only a candidate for a separately preregistered intervention; it does not authorize training.

The Colab entry point is
[`notebooks/p0d2h_route_transfer_bridge/p0d2h_route_transfer_bridge_colab.ipynb`](../notebooks/p0d2h_route_transfer_bridge/p0d2h_route_transfer_bridge_colab.ipynb).

## Result template

| Cell/contrast | N | OFF | ON | ON−OFF/rescue | Cluster interval | Status |
|---|---:|---:|---:|---:|---:|---|
| Eight factorial cells | 96 each | TBD | TBD | TBD | TBD | diagnostic |
| Grammar rescue | 96 paired units | — | — | TBD | TBD | supported/not supported |
| Lexicon rescue | 96 paired units | — | — | TBD | TBD | supported/not supported |
| Payload rescue | 96 paired units | — | — | TBD | TBD | supported/not supported |
| Exact external action | 96 | TBD | TBD | TBD | TBD | diagnostic |
| Forced-slot external action | 96 | TBD | TBD | TBD | TBD | diagnostic |

No bridge result has been generated at protocol freeze time.
