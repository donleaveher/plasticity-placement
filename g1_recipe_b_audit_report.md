# G1 Recipe B Audit Report

## Conclusion

Recipe A's failure is not explained by a train/eval continuation mismatch. On the pinned Qwen tokenizer, all four actions have equal three-token continuations, the chat prompt IDs are identical under the training/scoring special-token modes, and the real completion dataset exposes the same action-token prefix used by forced-choice scoring. Answer-copy/external controls and exact rescore/disable agreement also rule out a general parser, scorer, or adapter-activation failure.

The evidence is instead consistent with action-asymmetric under-capacity or optimization: Recipe A achieved `0.4896` current top-1, `0` on `act_n7`, `1.0` on `act_p3`, and caused `9.375 pp` unrelated regression. This does not identify a unique mechanism, so the implementation adds a read-only diagnostic rather than asserting a causal root cause.

## Training-code findings and fixes

1. The old resume fast path verified only a subset of lineage fields. It now re-derives node identity and verifies adapter, parent, root, data, trainer, precision, rank/alpha, target modules, optimizer-step, scheduler, and metadata identities before scoring.
2. The non-duplicate stored node-identity payload previously retained its derived `node_id`, while the canonical node hash is defined over the payload without that field. New artifacts store and verify the canonical preimage.
3. G1/P0 did not require the recipe step budget to equal frozen event-block exposure. This is now a hard preflight check. It caught and rejected the initial 32-step Recipe B draft before GPU execution.
4. Token parity now executes `CompletionDataset` itself, records the EOS loss target, requires equal candidate lengths, and checks prompt boundary reconstruction.
5. Verified score files and controls are now hash-anchored and rechecked on resume. Unanchored crash remnants are deterministically rescored.

## Locked next recipe

Recipe B is a capacity-only change:

```text
rank = 16
alpha = 32
learning_rate = 2e-4
optimizer_steps_per_event = 16
alpha/r = 2
```

All other Recipe A settings remain unchanged. Because B was specified after seeing Recipe A's qualification failure, it must be reported as adaptive interface-development calibration, not as an originally preregistered recipe. It is locked before any Recipe B outcome or path contrast is observed.

P0 accepts only a passing, integrity-complete Recipe B authorization. A failed B run remains immutable and does not authorize P0.

## Colab execution

Use a new `RUN_LABEL` such as `pathmem-g1-recipe-b`; do not reuse or delete the failed `pathmem-v1` directory. Set `USE_GOOGLE_DRIVE=True`, `RUN_G1=True`, `RUN_P0=False`, `G1_RECIPE="B"`, and fill `APPROVER`. After a genuinely passing G1 authorization, rerun with `RUN_G1=False`, `RUN_P0=True`, keeping the same label, recipe, and approver.

The results cell automatically runs the CPU-only `diagnose-g1` command and writes `g1_diagnostic.json`. The diagnostic is descriptive and cannot change a gate or create authorization.

## Verification boundary

All local CPU tests and lint pass (`447 passed`). Formal NF4/BF16 behavior was not executed on this Mac; Recipe B qualification must be run on an NVIDIA CUDA runtime with native BF16 support.
