# STAGE Majority-Local Order-Aware O1

## Question

The majority-local encoder aligns the same timestamps across shifted windows,
but its legacy patch embedding pools normalized tokens with only their global
mean and standard deviation. Temporal permutations can therefore map to the
same patch geometry. O1 tests whether this order-blind bottleneck, rather than
the observed-exemplar memory itself, is limiting anomaly separation.

O1 is an official TSB-AD Tuning-only diagnostic. It does not run baselines or
Eval, and its results cannot be used to make a confirmatory claim.

## Model change

The legacy moments embedding remains the base path. A complementary residual
branch can add:

1. four ordered temporal-bin residuals;
2. directional differences at relative lags near 1/8, 1/4, and 1/2;
3. lagged products at the same scales.

The branch is initialized with a learned gate of 0.1. It does not use absolute
window positions, negative samples, labels, or a second prediction task. The
hybrid candidate also aligns only directional transitions that already agree
across the two shifted views; the detached positive-agreement gate prevents
opposing or unreliable changes from being forcibly collapsed.

`patch_geometry_mode=moments` is exactly the previous majority-local model.
The production hybrid adds 46,528 parameters to 627,968 parameters
(7.409%).

## Frozen O1 screen

- Datasets: U/SED, M/LTDB, and M/GHL.
- Reason: SED and LTDB are relatively favorable sentinels on different
  tracks, while GHL showed normal/anomaly manifold overlap. A mechanism must
  preserve the first two and improve the third before spending GPU time on
  hard MSL/CATSv2 or the remaining datasets.
- Seed: 2026.
- Selection: per-dataset macro VUS-PR on complete official Tuning coverage.
- Head: fixed `final_gb_min_split=4`, `top_k=3`.
- Candidates per dataset:
  - moments control;
  - ordered pyramid;
  - order relations;
  - hybrid plus reliability-gated transition alignment.
- Expected work: four Tuning series (one SED, two GHL, one LTDB), 16
  independent GPU score units, one head
  each.

If no order-aware candidate improves GHL while preserving SED and LTDB, the next
mechanism should not add more prototype counts. The evidence would instead
support revisiting token-level learning or the definition of normal temporal
state. If one branch wins, it must next pass a separately frozen multi-seed,
all-dataset Tuning protocol before any Eval. TODS and SVDB are the next
favorable expansion set; MSL and CATSv2 are intentionally deferred until the
mechanism passes this first causal screen.

## GPU/CPU pipeline

The server command is `run-scores`. GPU workers train, build the frozen memory,
and emit only compact point-score caches. They do not run PaAno metrics.
Therefore a slow CATSv2-style CPU evaluator cannot occupy a GPU lane.
`scripts/launch_stage_majority_order_o1.sh` verifies every frozen hash and the
real-manifest plan identity before it starts a single 3090 GPU.

The local PowerShell puller copies the frozen plan and available caches, checks
their SHA-256 identities, and runs official metrics with
`C:\Users\wyyxx\.conda\envs\00gwk\python.exe`. A final unit JSON is created
atomically only after strict validation; the local transient cache is removed
by default.

No checkpoint, embedding array, or raw model state is retained. Score caches
are transient pipeline artifacts and are excluded from paper runtime claims.
