# Project memory

## Colab authorization controls

For every Colab workflow that requires a person to change authorization or execution state:

- Put all user-editable controls in one dedicated code cell. This includes `RUN_*` flags,
  `APPROVER`, authorization toggles, and similar per-pass values.
- Do not mix those controls with fixed paths, imports, repository checkout, environment setup,
  identity derivation, authorization adoption, or experiment execution.
- Use safe defaults that perform planning or inspection only. Training, qualification, audit,
  publication, and other state-changing stages must default to disabled.
- Make the control cell the only cell the user edits between passes. Clearly document which
  control values to use for each pass and which downstream cell(s) to rerun.
- Keep setup, identity derivation, and lifecycle execution ordered and atomic where a later step
  depends on variables created by an earlier step. Do not rely on an optional intermediate cell
  having been run.
- Treat `APPROVER` as a human/responsible-party identifier, never as a secret, password, API key,
  or model-generated approval.
- Reuse an existing valid authorization artifact idempotently. Do not regenerate timestamps or
  silently replace an adopted authorization during a retry.
- Add notebook-generation tests that verify the dedicated control cell, safe defaults, execution
  order, and generated-notebook parity with its builder.

Apply this convention to new notebooks and whenever an existing authorization workflow is
modified.
