# Notes: RTB zero-artifact recovery

## Supplied evidence

- Stale manifest state: `running`.
- No manifest errors.
- No `adapter_off`, `adapter_on`, raw checkpoint, pairs, summary, or audit manifest.
- The interruption therefore produced no persisted score or interpretable result.

## Integrity boundary

- The original invocation occurred, so a retry cannot be silently granted.
- No result was available for selection, so a separately authorized identical retry can be
  scientifically defensible if all frozen identities remain unchanged.

## Existing contracts

- RTB moves `authorized -> running` before model loading and writes raw OFF/ON rows only after all
  960 OFF/ON pairs and the OFF sentinel complete.
- Therefore a killed Colab process can leave `running` with no files even though no result was
  persisted; normal Python exception handling cannot catch a runtime termination.
- The original `run_audit` verifies the preregistered source-code hash before inference. Recovery
  can reset governance state using separately locked code, then the retry must still run under the
  original experiment commit.
- CPR recovery already establishes the useful pattern: external authorization, adopted immutable
  authorization, immutable claim, one permitted recovery, and idempotent completion after a
  governance-write interruption.

## Frozen recovery checks

- Manifest is `running`, error list is empty, and its update time is stale by at least six hours;
  the external approver also confirms that the original runtime has terminated.
- No raw score, result, paired record, report, summary, or audit manifest exists.
- Preregistration, preflight hashes, original authorization, adapter/source snapshot, run ID, and
  experiment code-revision lock remain unchanged.
- Authorization binds the prior running-manifest hash, preregistration hash, experiment revision,
  recovery-code hash, exact output, and one identical retry only.
