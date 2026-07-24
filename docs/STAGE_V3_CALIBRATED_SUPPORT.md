# STAGE-v3: Calibrated Structural Support

## Status

This branch is a development mechanism study. It does not replace the frozen
STAGE result, and no Eval result may be used to select its parameters.

## Paper type

Method-first technique paper with a coupled-bias perspective.

## Immutable story contract

STAGE-v3 preserves the mechanisms fixed by the manuscript guidance:

1. exact timestamp-token and shared-interval alignment across shifted patches;
2. one fixed intermediate geometry that tempers abundance-driven sampling;
3. an independently reconstructed final geometry that selects observed
   exemplars for nearest-reference scoring.

The intermediate and final partitions remain outside the loss. STAGE-v3 does
not introduce negative samples, alternating partition optimization, synthetic
centroids, or a user-specified final prototype count.

## One-sentence bottleneck

The central bottleneck is not patch imbalance alone but the use of an
uncalibrated shape-only distance, which can erase level/scale evidence and
grant equally strong normal support to compact and context-dispersed regions.

## Minimal formalization

For a patch `p`, let:

- `h(p)` be the overlap-trained shape embedding;
- `a(p)` be a deterministic level/scale descriptor calibrated on the
  filename-declared training prefix;
- `beta >= 0` control how much statistical evidence enters geometry.

The v3 descriptor is

`z(p) = normalize([normalize(h(p)); beta * normalize(a(p))])`.

The final partition produces observed medoids `e_j`. For each region, let
`rho_j` be a declared quantile of member-to-medoid squared unit distance. The
calibrated candidate distance is

`d_j(q) = ||z(q) - e_j||_2^2 + lambda_r * rho_j`.

The patch anomaly score remains the mean of the `k` smallest candidate
distances. Setting `beta = 0` and `lambda_r = 0` exactly recovers the legacy
shape-only, uncalibrated scoring semantics.

## Proposition candidates

### P1: level-shift distinguishability

For two patches that differ only by a per-channel affine level shift, PatchRevIN
can map their shape pathways to the same representation. If their calibrated
statistics descriptors differ and `beta > 0`, their v3 descriptors are not
identical.

This is an implementation-level property and is covered by a deterministic
unit test.

### P2: dispersion-aware reference ordering

For two references at equal representation distance from a query, the
reference retained from the less dispersed normal region receives the smaller
calibrated candidate distance whenever `lambda_r > 0`.

This does not prove that every compact region is normal. It only prevents a
wide region from receiving the same confidence as a compact region at equal
query distance.

## Mechanism changes

### Robust level/scale evidence

Per-channel median and MAD are fitted on the label-blind training prefix.
Training-window medians and MADs then calibrate patch means and patch log
standard deviations. The resulting deviations are bounded, unit-normalized,
and concatenated with the shape embedding. They do not alter the timestamp or
interval alignment objective.

### Region-dispersion calibration

The final geometry still chooses a real observed medoid per region. It also
records a robust member-to-medoid radius. Scoring adds a non-negative,
predeclared penalty proportional to that radius before selecting the nearest
references.

## Experiment matrix

All parameter selection uses official Tuning VUS-PR only. Eval remains sealed
until the configuration is frozen.

| Claim | Controlled variants | Primary diagnostic |
|---|---|---|
| Level/scale evidence repairs shape-only blind spots | `beta in {0, 0.25, 0.5, 1.0}` with `lambda_r=0` | Tuning VUS-PR plus per-series AUC-ROC direction |
| Dispersion calibration reduces dominant-region missed detections | `lambda_r in {0, 0.1, 0.25, 0.5}` with `beta=0` | anomaly-to-reference distance and false-negative score shift |
| The mechanisms are complementary | selected non-zero `beta`, selected non-zero `lambda_r`, and their combination | interaction versus the two single mechanisms |
| Alignment remains necessary | no-alignment, token-only, interval-only, and both | identical-timestamp representation distance |
| Geometry roles remain necessary | uniform sampling and uncalibrated final memory | exposure distribution and reference-distance diagnostics |
| The repair is not GHL-only | all ten official Tuning subsets | macro VUS-PR and worst-subset delta |

The candidate grid must be narrowed in stages before server execution. GHL may
be reported as a motivating failure case but cannot be the sole selector.

## Risks and fallback variants

1. Patch statistics may dominate high-dimensional multivariate data. The
   descriptor normalizes the complete statistics branch before weighting it;
   the zero-weight ablation remains exact.
2. Radius penalties may raise scores for normal queries inside genuinely broad
   regions. Select the weight on all official Tuning subsets and report
   false-positive changes explicitly.
3. Rare normality and training contamination remain unidentifiable without
   additional assumptions. Do not claim that geometry recovers true modes.
4. If the joint mechanism is unstable, retain the stronger single mechanism
   and narrow the paper claim instead of tuning on Eval.
