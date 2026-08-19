# Notes: G1 training audit and Recipe B

## Evidence from Recipe A

- All 24 current-only anchors and all 960 score rows completed.
- Every integrity check passed; same-adapter rescore and adapter-disable deltas were exactly zero.
- Answer-copy and external-latest accuracy were 1.0, ruling out a general parser or scorer failure.
- Parametric current accuracy was 0.4896, per-action minimum was 0, and unrelated regression was 9.375 percentage points.
- Per-action top-1 was highly asymmetric: `act_n7=0`, `act_p3=1`, `act_v9=0.4583`, `act_k2=0.5`.

## Audit Findings

- The frozen scorer uses an empty canonical leading whitespace. Training optimizes `act_*` and qualification scores the exact same continuation bytes.
- On the pinned Qwen2.5-1.5B tokenizer, every action continuation has exactly three tokens. For all four actions, separately tokenized completion IDs equal the suffix IDs from tokenizing `formatted_prompt + completion`.
- Qwen's chat-formatted prompt has identical IDs under `add_special_tokens=True` and `False`, so the generic completion dataset does not introduce an extra BOS mismatch.
- Parent adapter resumption is trainable, optimizer/scheduler objects are recreated for every semantic event, and weight decay is explicitly zero. Adapter disable/rescore evidence also supports correct activation lifecycle.
- Recipe A therefore appears capacity/generalization limited rather than affected by a train/eval text mismatch. Its r=8, alpha=16 update learns the model-favored label perfectly but fails completely on `act_n7`, while producing 9.375 pp unrelated regression.
- The first Recipe B draft (r=16, alpha=16, learning rate 1e-4, 32 steps) conflicts with the frozen G0 event blocks, which require 16 optimizer steps per event. It must not be run.
- Corrected Recipe B changes only conditional capacity: r=16, alpha=32, learning rate 2e-4, 16 steps. This doubles rank while preserving alpha/r=2, the learning rate, scheduler, and the frozen per-event exposure.
- G1 resume verification is incomplete: the fast path checks only adapter/data/parent/root hashes and does not fully re-derive node identity, trainer config, precision, step count, or result-file hashes before authorization. This is a blocking integrity issue independent of the observed functional failure.
- Preflight checks fit/validity but does not persist an explicit loss-token/scoring-token parity check or require equal candidate token lengths. The current tokenizer passes both, but the invariant should be executable.

## Implemented corrections

- Recipe B is `r=16`, `alpha=32`, learning rate `2e-4`, 16 optimizer steps. It preserves `alpha/r=2` and all frozen exposure variables while doubling conditional LoRA rank.
- G1/P0 preflight now executes the real `CompletionDataset` tokenization path and requires the action loss-token prefix to equal formal forced-choice continuation tokens; the EOS target is recorded separately.
- G1/P0 now require every compiled event block's `optimizer_steps` to match the selected recipe.
- Resume verification re-derives node identity, adapter/parent/data hashes, trainer config, precision, optimizer steps, rank/alpha/targets, scheduler inputs, and training metadata before scoring or authorization.
- Verified result files and base-control files are hash-checked. A crash between writing an unanchored control file and saving its hash causes a deterministic rescore instead of silent adoption.
- `diagnose-g1` is CPU-only and reads a completed Recipe A or B run without changing gates. It emits action confusion, expected-probability margins, base-to-adapter paired shifts, unrelated correctness transitions, and final-loss summaries.

## Verification

- Focused G1/P0 and notebook tests passed.
- Frozen PathMem compiler/schema/manifest tests passed.
- Full repository suite, including the resume-integrity unit test, passed: 447 tests.
- No formal CUDA training was run locally; functional qualification still requires Colab or another supported NVIDIA BF16 runtime.
