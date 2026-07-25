# STAGE Majority-Local Geometry Diagnosis D1

## Purpose

The lightweight-encoder experiment showed that the original dilated residual
encoder remains the strongest uniform encoder, while the complete final-head
grid adds only a negligible Tuning VUS-PR ceiling. D1 therefore freezes the
encoder and diagnoses the three mechanism responsibilities before a larger
upgrade is attempted:

1. whether patch normalization discards useful level and scale evidence;
2. whether the current correlation objective is too indirect to preserve
   shared timestamps;
3. whether the intermediate granular partition is too fragmented or becomes
   stale during training.

## Frozen comparison

Only official TSB-AD Tuning data are used. Labels are loaded after all model
training, memory construction, and anomaly scoring. The selection metric is
dataset macro VUS-PR. Eval feedback, baseline reruns, and track-level fallback
are forbidden.

The four candidates are:

- **D0 control:** unchanged majority-local STAGE.
- **D1 statistics:** append robust patch level and log-scale descriptors to
  the normalized shape embedding.
- **D2 pairwise:** add direct cosine consistency for aligned timestamp tokens
  and their shared-interval embeddings.
- **D3 geometry:** increase the minimum intermediate-region split and rebuild
  the partition once after the representation has matured.

The diagnostic datasets are U/MSL, U/SED, M/CATSv2, and M/GHL. They cover
weak and comparatively strong prior outcomes, univariate and multivariate
tracks, and different series scales. The four candidates share identical
official Tuning series coverage.

## Required evidence for every completed unit

Each unit stores a compact post-hoc diagnostic record:

- a training-fitted PCA projection of sampled training-normal, Eval-normal,
  Eval-anomaly, and selected prototype embeddings;
- prototype support size, radius, and source timestamp;
- robustly standardized raw-series channels with anomaly intervals;
- raw RMS-z snippets for selected high- and low-support prototypes;
- normal and anomalous patch-score quantiles and their median separation.

No checkpoint, full embedding, full score array, `.npy`, `.npz`, `.pt`,
`.pth`, or `.ckpt` artifact is permitted. The visualization fields cannot
affect Tuning selection.

## Decision logic

- D1 improves score separation and VUS-PR: the shape-only representation lost
  level/scale evidence.
- D2 improves them: the prior correlation objective was too indirect or
  over-smoothed local temporal evidence.
- D3 improves them while reducing tiny regions and stabilizing memory ratio:
  the intermediate geometry was overfragmented or stale.
- None improves them: the likely bottleneck is outside these minimal axes,
  with point-level score aggregation and support-aware final memory as the next
  isolated Tuning-only candidates.

Any subsequent mechanism is frozen in a new protocol. D1 results are evidence,
not permission to alter the frozen historical STAGE results.
