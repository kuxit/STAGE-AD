# Experiment Snapshot Before Compute Shutdown

This snapshot records reproducible code and small audit metadata only. It does not contain raw datasets, checkpoints, embeddings, score arrays, or large logs.

## Frozen formal experiment

- Formal main commit: `74fd2b0b7c35672210bad023eb4305ca975fd481`
- Fourteen original baselines: 2702/2702 valid units
- DPAD and DNE: 386/386 valid evaluation units
- Sixteen-baseline total: 3088/3088, error=0
- STAGE v1 Tuning: 528/528
- STAGE v1 seed-2026 Eval: 193/193
- Formal v1 comparison: 5/6 global metric wins and 4/12 track-metric wins; therefore the all-cell confirmatory gate did not pass.

## Dataset-specific Tuning lock

- All ten datasets have dataset-specific training and head selections from official TSB-AD Tuning.
- Evaluation feedback used for selection: false
- Track-level fallback: none
- Lock fingerprint: `aaea353206be38b569759cd442f8d79a755aa6f86de2259b211ea5b109c37ca9`
- A fresh Eval for this new lock had not been run at shutdown, so its per-dataset SOTA status is unknown.

## Tuning-only mechanism study

- Plan fingerprint: `31500128e29c241ee70d8438a085e83306bfba60d831adcf276b092ef8e9bdad`
- STAGE source SHA-256: `0d86767697a1d95fc8e4449e242e2e4fa28a0b1842f9000b689d491f8c0a2284`
- Runner SHA-256: `07b6a32aace4bf4d3c0e28ef9f3bbd6b0336b54db88cf39698132690a6275853`
- Planned size: 110 physical units / 154 logical records
- Strict shutdown point: 14 physical units / 24 logical records
- Invalid / duplicate / error units: 0 / 0 / 0

The incomplete mechanism study must not be summarized as a finished result. Valid units are atomic and can be reused if the frozen plan is resumed.

## Validation note

The mechanism branch intentionally changes `STAGE/stage.py`. The predecessor Stage1A protocol remains bound to its original frozen source hash, so its full repository-lineage test is expected to reject this branch. The old protocol hash was not altered. All ten other v2 search/watch tests pass on this snapshot, and the mechanism runner compiles successfully.
