# Notes: Route-state handoff audit

## Supplied RTB evidence

- RTB completed with `scoring_integrity_failed` because the primary slot factorial required zero
  ties.
- All 26 primary violations were expected-compatible top ties with zero signed margin.
- Tie counts were 18 for adapter OFF and 8 for adapter ON across all eight slot cells.
- All eight slot-readout ON-OFF gains remain positive under conservative compatible-tie bounds.
- Exact external action changed from 61/96 to 53/96, while forced-correct-slot action changed from
  86/96 to 89/96.
- The point-estimate forced-slot rescue of adapter effect is 11/96 = 0.1145833; it does not yet
  have its own preregistered clustered interval.

## Integrity boundary

- No RTB threshold or tie rule may be changed after results.
- A new audit may prospectively use expected-compatible tie bounds, but must retain the old RTB
  decision and artifact hashes.
- The model-predicted chain must use the first-stage predicted slot, never the expected slot.
- Oracle-slot and wrong-slot endpoints are controls, not substitutes for the predicted chain.
