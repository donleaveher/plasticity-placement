# Experiments

This directory is reserved for the new path-aware audit. It must not import frozen P0-C manifests or write into the old experiment artifact directories.

## Intended layout

```text
experiments/
  README.md
  g0/               # immutable CPU-only protocol bundle and G0 manifest
  configs/          # frozen phase configurations, history families, and manifests
  results/          # verified tables linked to immutable raw manifests and rows
```

The G0 implementation now uses the separate `plasticity_placement.pathmem` source namespace, with modules for:

- event/domain schema;
- history-family compiler and episode-role validator;
- logical reducer;
- compiler and exposure audit;
- path-DAG manifest and cache;
- versioned external store and ICL renderer;
- backend observability adapters and lifecycle trace schema;
- matched access-on/off, canonicalization-rescue, artifact-necessity, and future-suffix diagnostic interfaces;
- resumed-LoRA lineage wrapper;
- candidate scorer;
- contract metrics and paired statistics;
- CLI and verification tests.

G0 preparation and verification are CPU-only:

```bash
uv run plasticity-pathmem prepare \
  --output ../memory-update-consistency/experiments/g0 \
  --protocol-root ../memory-update-consistency

uv run plasticity-pathmem verify \
  --output ../memory-update-consistency/experiments/g0 \
  --protocol-root ../memory-update-consistency
```

These commands compile or verify frozen protocol artifacts only. They do not load a model, start training, or authorize G1/P0.

## Authorization rule

The authorization order is `G0 → G1 → P0 → P1 → P1b → P2 → P3`. P0 is not executable until G0 and G1 pass. No P1/P2 configuration is executable until:

1. the preregistration and config hashes are frozen;
2. P0 integrity gates pass;
3. a separate run directory is selected;
4. result-bearing item splits remain untouched.

## Artifact identity

Every child adapter must name and verify its exact parent hash. Every result must name the exact node, probe bank, renderer, scorer, precision, and environment. A terminal-state label is never sufficient as a cache key.

Every compiled history family must also record its logical fixture, reducer, endpoint relation, event multiset, episode roles, controls, observability profile, and trace expectations. Lifecycle fields that do not exist for a backend are serialized as `N/A`; they are not omitted or converted to numeric zero.

## Result policy

No placeholder or simulated number may be placed in `results/` as if it were experimental evidence. Synthetic fixtures for unit tests must live under the eventual test directory and be visibly labeled.

Every aggregate must record the raw manifest URI/path, raw-row SHA-256, artifact-index SHA-256, retention policy, and analysis code hash. Raw rows and lineage records are retained for recomputation; an aggregate-only result is not considered verified.
