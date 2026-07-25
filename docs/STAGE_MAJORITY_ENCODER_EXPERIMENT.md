# STAGE majority-local encoder experiment

## Fixed scientific scope

The model family remains STAGE majority-local. Shared-timestamp and interval
alignment, adaptive intermediate granular sampling, final observed-exemplar
memory, `token_head`, and `embedding_head` are retained. Only the
timestamp-preserving encoder backbone is varied.

The experiment uses official TSB-AD Tuning labels and selects with dataset
macro VUS-PR only. Eval results never select a candidate. Baselines are
read-only and are never rerun.

## Current per-dataset deficits

The supplied main table contains 6 strict wins among 60 dataset-metric cells.

| Dataset | Strict wins / 6 | STAGE VUS-PR gap to best baseline | Worst six-metric gap |
|---|---:|---:|---:|
| UCR | 2 | -0.004 | -0.019 |
| Exathlon | 0 | -0.255 | -0.262 |
| MSL | 0 | -0.170 | -0.170 |
| SED | 0 | -0.013 | -0.038 |
| TODS | 3 | +0.051 | -0.091 |
| CATSv2 | 0 | -0.120 | -0.308 |
| GHL | 0 | -0.022 | -0.295 |
| LTDB | 0 | -0.099 | -0.156 |
| SVDB | 1 | -0.050 | -0.050 |
| TAO | 0 | -0.174 | -0.747 |

## Frozen E1 gate

- Candidates: the original dilated residual encoder, a depthwise temporal
  encoder, and a multiscale depthwise temporal encoder.
- Dataset training parameters: the ten previously frozen Tuning-only winners.
- Seed: 2026.
- Physical units: 66 (three encoders across 22 official Tuning series).
- Each trained unit evaluates the complete 20-head grid
  (`final_gb_min_split` in 4, 16, 64, 256 and `k` in 1, 3, 5, 9, 15).
- Global architecture selection: for each encoder, evaluate every dataset at
  that dataset's previously frozen head; select one encoder for all ten
  datasets by the unweighted mean of the ten macro VUS-PR values. Exact ties
  use canonical encoder type.
- The complete head grid is not used to select the global encoder. It is
  retained to freeze a new dataset-specific head after the encoder is chosen.

## Follow-up stages

1. **E2 stability:** confirm the top two E1 encoder architectures with seeds
   2027 and 2028 using the same Tuning-only rule.
2. **E3 dataset training search:** fix the single global encoder, then expand
   lightweight dataset-specific training parameters around the frozen winners.
   Rank complete coverage by macro VUS-PR; retain top three per dataset.
3. **E4 confirmation and head lock:** confirm top-three training candidates
   across three seeds, then lock one training configuration and one head for
   every dataset.
4. **Independent Eval:** run the ten frozen configurations once on official
   Eval seed 2026 and produce a full-precision 10-by-6 comparison against all
   16 read-only baselines.

Failure is diagnosed separately as low Tuning capacity, seed instability,
Tuning-to-Eval generalization, or metric trade-off. A later mechanism change
requires a new pre-frozen Tuning-only protocol.
