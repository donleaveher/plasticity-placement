# Notes: Receipt/action binding audit

## Supplied RSH result

- Decision: `handoff_not_supported`, with valid scoring integrity.
- Conservative handoff rescue: `0.1458 [0.0417, 0.2396]`, supported.
- Adapter-ON predicted chain equals oracle action at `0.9271`.
- Adapter-ON oracle-minus-wrong target rate: `0.1354 [0.0729, 0.2083]`, below the
  preregistered `0.20` material threshold.
- Training and 1/4/8 remain unauthorized.

## Source-structure finding

- In RSH, the target slot contains a valid query and verified mapping.
- The opposite slot is an archived advisory decoy listing distractor labels; it does not contain
  a uniquely defined alternative query-to-action mapping.
- Consequently, high target-action persistence under the wrong receipt is evidence of weak
  receipt specificity in that panel, but it is not a clean test that the model can select between
  two simultaneously valid receipt-bound actions.

## Proposed correction

- Pair the frozen `a` and `b` lessons sharing one action panel.
- Put both valid queries and both verified mappings in the same prompt.
- Cross receipt A/B with canonical/swapped slot assignment.
- Require the selected action to change with both receipt intervention and binding swap.
- Preserve every negative authorization boundary from RSH.

## Frozen implementation

- 12 lesson pairs × 4 route variants = 48 units.
- 2 binding orientations × 2 receipts = 192 prompts and 384 OFF→ON raw decisions.
- Both verified-memory statements remain in fixed `a,b` order; only query-to-slot assignment is
  swapped.
- Each unit inherits the lesson-a four-action candidate order, which balances every expected
  candidate position 12 times per factorial cell.
- Primary gates use conservative compatible-tie bounds and pair-cluster bootstrap intervals.
- Non-finite and malformed scorer outcomes are retained as maximally conservative bounds and
  produce `scoring_integrity_failed`, rather than crashing classification.

## Verification

- Focused RAB package/notebook tests after review fixes: 19 passed.
- Full repository suite after review fixes: 303 passed.
- Ruff, compileall, CLI help, notebook parse/parity, and diff checks passed before review.
