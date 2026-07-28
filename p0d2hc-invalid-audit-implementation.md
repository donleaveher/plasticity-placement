# P0-D2H-CAL Invalid-Output Audit Handoff

## Delivered

- deterministic boundary-aware action mention extraction;
- conservative semantic recovery distinct from strict exact-action parsing;
- seven mutually exclusive invalid-output classes;
- lesson-clustered bootstrap intervals for strict accuracy, invalid rate,
  semantic accuracy, and paired semantic-recovery gain;
- model × arm and model × arm × category summaries;
- classified JSONL for every strict-invalid source row;
- immutable JSON/Markdown/JSONL artifacts in an independent output directory;
- source manifest, prompt audit, raw tree, experiment code, and analysis code
  provenance;
- `plasticity-p0d2hc audit-invalid` CPU-only CLI;
- standalone generated Colab notebook for the completed `pipeline-c1` source;
- protocol, notebook index, and project-layout documentation.

## Scientific boundary

The output is marked `post_hoc_supplementary`. It explicitly records:

```text
strict_gate_status_changed = false
next_stage_eligibility_changed = false
raw_rows_modified = false
manifest_modified = false
```

A row is semantically recovered only when exactly one distinct allowed action
occurs and it equals the expected action. Multiple-action responses remain
ambiguous.

## Command

```bash
uv run plasticity-p0d2hc audit-invalid \
  --output <verified-p0d2hc-source-stage> \
  --audit-output <independent-invalid-audit-stage> \
  --bootstrap-samples 10000
```

## Outputs

```text
invalid_output_audit.json
invalid_output_audit.md
invalid_output_records.jsonl
```

The command rejects an audit directory that overlaps the source run. Existing
artifacts are reusable only when their bytes match exactly.

## Verification

- focused P0-D2H-CAL suite: 24 passed;
- full repository validation: 129 passed;
- Ruff, both notebook regenerations, and diff checks: passed.
