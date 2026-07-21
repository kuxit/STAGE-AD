# STAGE: Structure-Preserving Timestamp Alignment for Geometry Enhancement

This repository is the canonical implementation and evaluation package for
STAGE, a framework for debiased time-series anomaly detection. It contains the
model, frozen experiment controller, per-method adapters, protocol, local
validation evidence, and server deployment entry point.

Start here:

- [`protocol.json`](protocol.json): active no-KNN scientific protocol.
- [`docs/METHOD_IDENTITY.md`](docs/METHOD_IDENTITY.md): official method name,
  acronym semantics, and module mapping.
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md): what is present, what is
  still external, and the pre-run gate.
- [`experiment_policy.json`](experiment_policy.json) and
  [`docs/EXPERIMENT_GOVERNANCE.md`](docs/EXPERIMENT_GOVERNANCE.md): the
  machine-readable and human-readable rules for baseline settings, STAGE
  tuning, run order, precision, and timing.
- [`docs/WORKSPACE_INVENTORY.md`](docs/WORKSPACE_INVENTORY.md): local resource
  map and cleanup candidates.
- [`docs/GITHUB_AND_BACKUP_WORKFLOW.md`](docs/GITHUB_AND_BACKUP_WORKFLOW.md):
  GitHub and three-copy result-backup workflow.
- [`checksums/frozen-code.sha256`](checksums/frozen-code.sha256): active code
  and configuration hashes.
- [`scripts/verify_local_assets.ps1`](scripts/verify_local_assets.ps1): read-only
  local integrity check.
- [`scripts/snapshot_local_results.ps1`](scripts/snapshot_local_results.ps1):
  timestamped, metrics-only local result snapshot.
- [`validation/local_00gwk_smoke.json`](validation/local_00gwk_smoke.json):
  runtime-free evidence that all 15 active methods produced six finite metrics on one
  official U-series and one official M-series in the local `00gwk` environment.
- [`docs/AAAI_RECENT_BASELINES.md`](docs/AAAI_RECENT_BASELINES.md): selection
  and protocol boundary for the AAAI-24 DPAD and AAAI-25 DNE extension.
- [`validation/aaai_recent_local_00gwk_smoke.json`](validation/aaai_recent_local_00gwk_smoke.json):
  runtime-free evidence that both extension methods execute on U/M and emit all
  six metrics; this is not formal Eval evidence.

The repository should be rooted at this directory when it is pushed to
GitHub. Raw datasets, Python environments, checkpoints, score arrays, secrets,
and transient logs do not belong in Git.

The recent-AAAI extension is not yet part of the active 15-method result table.
Its configuration must first be selected on official Tuning, frozen, and
hashed before it may be evaluated.

## Scope

- Seed: `2026`
- Dataset partition: the official TSB-AD Tuning/Eval split of Liu and
  Paparrizos (2024). Tuning is used only for hyperparameter selection and Eval
  only for final reporting.
- TSB-AD-U subsets: `UCR`, `Exathlon`, `MSL`, `SED`, `TODS` (122 series)
- TSB-AD-M subsets: `CATSv2`, `GHL`, `LTDB`, `SVDB`, `TAO` (71 series)
- Every method consumes the filename-declared `tr_<N>` prefix without consulting
  labels during fitting. Labels are read only after anomaly scores have been
  produced.
- The six metrics use the frozen official evaluator with `pred=None`:
  `VUS-PR`, `VUS-ROC`, `R-based-F1`, `AUC-PR`, `AUC-ROC`, and `Standard-F1`.
- Raw per-series metrics are stored at full precision. Paper tables round only
  their rendered values to three decimals.
- No checkpoint or anomaly-score array is retained.

## Active methods

Non-deep baselines (4): `KMeansAD`, `PCA`, `IForest`, `LOF`.

Deep baselines (10): `PaAno`, `GBOC`, `MEMTO`, `PatchTST`,
`DCdetector`, `AnomalyTransformer`, `TimesNet`, `TranAD`, `USAD`,
`OmniAnomaly`.

Target method: `STAGE`.

PaAno and GBOC are the closest patch-memory / granular-ball comparators.
MEMTO adds a prototype-memory reconstruction comparator, while PatchTST adds a
patch-reconstruction comparator. KMeansAD isolates fixed-prototype behavior.

KNN is excluded from the active comparison by author decision.

The controller is resumable: a unit is skipped only when its JSON contains all
six finite metrics and no error. Final tables must not be used to change any
hyperparameter in this run.

TSB-AD-integrated baselines use `Optimal_Uni_algo_HP_dict` or
`Optimal_Multi_algo_HP_dict`, which were selected on the official Tuning split.
PaAno, GBOC, MEMTO, and DCdetector use released method configurations adapted
only to the official per-series prefix interface. Baselines receive no new
hyperparameter search. If a matching released setting is unavailable, a
documented stability default must be frozen before Eval. STAGE uses one
subset-level configuration selected on official Tuning with an equal search
budget for all ten subsets. No Eval metric is an input to configuration choice,
including for Exathlon.

The formal launcher defaults to the `baseline` phase and places PaAno first on
the GPU queue. The separate `target` phase refuses to start unless every
same-seed baseline unit is complete in the result root. `all` is available only
for a fully frozen end-to-end run. Main-run timings are diagnostic only.
Paper-ready Average Inference Time requires a separate, exclusive,
identical-hardware rerun.

On dual-GPU servers the launcher defaults to an audited heterogeneous
throughput schedule. PaAno runs first and exclusively on both GPUs. Afterwards,
one regular deep-baseline lane per GPU overlaps bounded GBOC lanes and the CPU
baseline pool. Override `STAGE_GBOC_WORKERS_PER_GPU` only after a smoke/resource
probe; this affects throughput, not method configurations or paper-eligible
runtime. MEMTO can use its separately pinned interpreter through
`STAGE_MEMTO_PYTHON`. On NUMA hosts, `STAGE_GPU_CPU_MAP` can bind each GPU lane
to its local CPU node with `taskset`; the selected map is recorded in the run
manifest.

## Current local status

- All 193 selected Eval files are present locally (122 U + 71 M), giving
  2,895 method-series units for 15 active methods.
- All ten active code/config entries listed in
  `checksums/frozen-code.sha256` match.
- Exact upstream commits for PaAno, GBOC, MEMTO, DCdetector, and TSB-AD are
  declared in `dependencies.lock.json` and can be restored by
  `scripts/bootstrap_sources.ps1`.
- The local `00gwk` portability smoke test is complete: 30/30 records valid,
  covering all 15 active methods on one official U-series and one official M-series.
  This is a runnable-interface check, not a paper efficiency benchmark.
- The separate recent-AAAI extension smoke test is complete: 4/4 records valid
  for `DPAD_AAAI24` and `DNE_AAAI25` on the same U/M interface. Formal Eval is
  still blocked until the official-Tuning configuration is frozen. If both
  extensions are admitted, the complete table will contain 17 methods and
  3,281 method-series units.

## Safety notes

- Shared-resource timings are invalid for the paper efficiency table.
- Never commit SSH keys, passwords, `.env` files, raw data, checkpoints, or
  anomaly-score arrays.

## Leakage-safe STAGE continuation

After the baseline phase is strictly complete, `scripts/stage_autopilot.py`
can run the target method without manual bookkeeping.  It uses exactly the 22
matching official TSB-AD Tuning series, applies the same deterministic
24-candidate budget to every `(track, dataset)` subset, and freezes one
configuration per subset before any Eval result is read.  The frozen manifest
is content-hashed and can be committed from `configs/stage_locked_seed2026.json`.

`scripts/launch_stage_autopilot.sh` then evaluates all 193 Eval series at seed
2026.  Seeds 2027 and 2028 are launched with the identical frozen parameters
only if STAGE strictly beats the strongest admitted baseline in all six global
metrics and all twelve U/M track-metric cells.  A failed gate produces a report
and stops; Eval feedback is never used to launch another parameter search.
