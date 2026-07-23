# STAGE Mechanism Validation Contract

This isolated study is governed by the paper-level contract
`STAGE_STORY_EXPERIMENT_CONTRACT.md` in the latest manuscript directory.

## Fixed thesis

Normality may become too broad around context-distorted dominant dynamics and
insufficiently supported around underrepresented ones.

The study tests three linked technical responsibilities:

1. timestamp and shared-interval alignment should reduce representation changes
   caused by shifted window contexts;
2. an adaptive intermediate partition should temper the dependence of encoder
   updates on regional patch abundance;
3. independently reconstructed final geometry should retain representative
   observed exemplars at a data-dependent granularity.

## Frozen first pass

- Split: official TSB-AD Tuning only.
- Seed: 2026.
- Targets: all ten dataset subsets and the same 22 Tuning series used by the
  frozen v2 search.
- Hyperparameters and final head: the frozen ten-dataset Tuning lock.
- Training controls: Full, context-only/no timestamp correspondence,
  timestamp-token-only, shared-interval-only, and patch-uniform sampling.
- Memory controls for the Full encoder: adaptive exemplars, uncompressed
  observed memory, and an equal-sized time-uniform observed memory.
- No checkpoints, embeddings, score arrays, or Eval feedback may be saved or
  consumed.

The predeclared representative priority set is U/UCR, U/SED, M/CATSv2, and
M/GHL.  It spans both tracks and heterogeneous application settings.  It is a
scheduling priority only; the full study remains all ten datasets, and no
dataset may be selected after inspecting outcomes.

## Interpretation gate

- Alignment is supported only if the Full objective improves paired
  shared-timestamp/interval distances and downstream Tuning metrics relative to
  the controls.
- Abundance tempering is supported only if `power < 1` changes exposure scaling
  and improves underrepresented-support diagnostics relative to `power = 1`.
- Adaptive retention is supported only if it improves coverage or metrics over
  the equal-memory observed control; comparison to uncompressed memory reports
  the compression trade-off rather than proving selection quality by itself.
- Negative or mixed evidence triggers a mechanism-local revision in a new
  frozen Tuning-only phase.  It never authorizes Eval feedback or result
  fabrication.
