# CBR resumable evaluation handoff

## Outcome

CBR now has a separately authorized `evaluate-resumable` lifecycle. It preserves
the existing 3,360-prompt OFF/ON unit matrix and writes one immutable checkpoint
per prompt shard. Re-running the same unit validates and skips complete shards,
adopts a valid shard written immediately before a manifest interruption, and
stops on missing or changed previously recorded artifacts.

The implementation does not retrain or modify adapters. Legacy zero-artifact
claims are archived only during an externally approved migration. Partial legacy
artifacts are rejected.

## Runtime semantics

- Default shard size: 64 prompts (53 shards per unit).
- Adapter OFF is scored immediately before ON for every prompt.
- Every shard records model-object identity and an OFF sentinel before/after the
  shard.
- A reconnect creates a new runtime segment. The final summary reports the exact
  segment count and does not claim one process across a resumed unit.
- Final standard CBR artifacts are reconstructed deterministically from validated
  shards.
- Aggregation resolves each unit from its complete claim, permitting mixed legacy
  and resumed evaluation roots.

## Revisions and entry point

- Core implementation: `99462a4a7742b82f0cee395d4da891d1ed6e48db`.
- Colab: `notebooks/p0d2h_cbr_resumable_evaluation/`
- CLI stages: `resumable-evaluation-template`,
  `authorize-resumable-evaluation`, and `evaluate-resumable`.

## Verification

- Full repository: 400 tests passed.
- Ruff: passed.
- Python compilation: passed.
- Generated-notebook AST and parity: passed.
- `git diff --check`: passed.
