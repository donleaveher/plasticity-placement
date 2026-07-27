# P0-D2H-R Implementation Handoff

## Outcome

P0-D2H-R is an evaluation-only repair. It reuses the frozen P0-D2 adapters and
writes to `pipeline-r1/hard_probe-r1`; it does not modify or resume the previous
`pipeline-a2/hard_probe-a1` result.

## Implemented safeguards

- Separate source-training (`256`) and evaluation (`512`) context lengths.
- CPU-only pre-inference audit of all 768 plain/external prompt variants.
- Immutable `preflight/prompt_token_audit.json` hashed into the manifest.
- Immediate failure before inference if any prompt is truncated or loses its output
  instruction.
- External memory rendered immediately before the strict action-choice suffix.
- Revised binding-decoy wording with authoritative ACTIVE/ARCHIVED status labels.
- Raw-row provenance for complete/truncated token counts, prompt variant, rendering
  version, instruction preservation, and both context lengths.
- Separate frozen locus status and suite-quality/next-stage status.
- Explicit external-anchor and common-floor thresholds.

## Colab execution

Run the regenerated `notebooks/p0d2_hard_probe/p0d2_hard_probe_colab.ipynb` from
top to bottom. The new Block 5 token audit must report:

- `all_prompts_fit: true`
- `input_truncated_count: 0`
- `output_instruction_missing_count: 0`

Only then should Block 6 start the formal GPU evaluation.

## Verification

- 105 repository tests passed.
- Ruff lint passed.
- `git diff --check` passed.
- Real Qwen tokenizer audit: 768 prompts, maximum 303 tokens, zero truncation and
  zero missing output instructions at the 512-token evaluation limit.
