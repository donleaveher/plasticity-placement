# P0-D2H RABX zero-artifact recovery protocol

## Scope

This protocol permits one externally authorized identical retry when the frozen RABX audit is
stale in `running` state after a terminated Colab session and no inference or analysis artifact
was persisted. It is infrastructure recovery, not a second experiment, prompt change, threshold
change, training run, or 1/4/8 scan.

## Eligibility

Recovery is allowed only when all checks pass:

- the manifest is `running`, has no errors, and has been stale for at least six hours;
- the original Colab process is confirmed terminated;
- no file exists under `results/`, and no raw checkpoint, paired records, summary, report, or audit
  manifest exists;
- the immutable preregistration, preflight hashes, original authorization, RABC/RAB source trees,
  adapter identity, and original experiment revision remain unchanged;
- no prior zero-artifact recovery claim was consumed.

## Two-revision execution

The experiment remains locked to commit `58eaa8c4dde431e393925662b6c1e913bba02611`. A separate
recovery checkout supplies the recovery command without changing the preregistered experiment
code. The external recovery authorization binds both revisions, the output path, run ID,
preregistration hash, stale manifest hash, source evidence, and the one identical retry.

After authorization, recovery writes immutable authorization and claim files and changes only the
same manifest from `running` back to `authorized`. The subsequent audit executes the original
experiment commit and retains the same run ID and output directory.

## Colab passes

Do not delete or edit the running manifest. After six hours from its `updated_at` timestamp:

1. In the dedicated controls cell, disable all stages except `RUN_RECOVERY_INSPECTION` and rerun
   the combined lifecycle cell. Inspect the external recovery template.
2. In a separate pass, disable inspection, set `APPROVER`, set
   `ORIGINAL_RUNTIME_TERMINATED = True`, enable only `RUN_RECOVER_ZERO_ARTIFACT`, and rerun the
   controls and lifecycle cells.
3. Confirm the manifest is `authorized`. Disable recovery, enable only `RUN_AUDIT`, and rerun the
   controls and lifecycle cells.

If inspection reports any persisted result artifact, changed provenance, or a non-stale attempt,
stop. Do not remove the reported file or manually change the manifest.

## Interpretation boundary

The recovered result remains the same single frozen RABX audit. Recovery cannot reclassify RAB or
RABC, authorize remediation training, or authorize the 1/4/8 experiment.
