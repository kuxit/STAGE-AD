# DuoBa seed=2027 reproducibility package

This directory is the canonical entry point for the locked DuoBa comparison
requested on 2026-07-18.  It contains the frozen experiment controller,
per-method adapters, protocol, multi-server tooling, recovery assets, and the
documentation needed to audit or hand off the run.

Start here:

- [`protocol.json`](protocol.json): active no-KNN scientific protocol.
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md): what is present, what is
  still external, and the pre-run gate.
- [`docs/WORKSPACE_INVENTORY.md`](docs/WORKSPACE_INVENTORY.md): local resource
  map and cleanup candidates.
- [`docs/GITHUB_AND_BACKUP_WORKFLOW.md`](docs/GITHUB_AND_BACKUP_WORKFLOW.md):
  GitHub and three-copy result-backup workflow.
- [`checksums/frozen-code.sha256`](checksums/frozen-code.sha256): frozen code
  and recovery-asset hashes.
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

The recent-AAAI extension is not part of the active 15-method result table.
Its configuration must first be selected on official Tuning, frozen, and
hashed before it may be evaluated.

## Scope

- Seed: `2027`
- Dataset partition: the official TSB-AD Tuning/Eval split of Liu and
  Paparrizos (2024). Tuning is used only for hyperparameter selection and Eval
  only for final reporting.
- TSB-AD-U subsets: `UCR`, `Exathlon`, `MSL`, `SED`, `TODS` (122 series)
- TSB-AD-M subsets: `CATSv2`, `GHL`, `LTDB`, `SVDB`, `TAO` (71 series)
- Every method consumes the filename-declared `tr_<N>` prefix without consulting
  labels during fitting. Labels are read only after anomaly scores have been
  produced.
- The six metrics use the frozen DuoBa/PaAno evaluator with `pred=None`:
  `VUS-PR`, `VUS-ROC`, `R-based-F1`, `AUC-PR`, `AUC-ROC`, and `Standard-F1`.
- Raw per-series metrics are stored at full precision. Paper tables round only
  their rendered values to three decimals.
- No checkpoint or anomaly-score array is retained.

## Active methods

Non-deep baselines (4): `KMeansAD`, `PCA`, `IForest`, `LOF`.

Deep baselines (10): `PaAno`, `GBOC`, `MEMTO`, `PatchTST`,
`DCdetector`, `AnomalyTransformer`, `TimesNet`, `TranAD`, `USAD`,
`OmniAnomaly`.

Target method: `DuoBa` (the read-only V1 source).

PaAno and GBOC are the closest patch-memory / granular-ball comparators.
MEMTO adds a prototype-memory reconstruction comparator, while PatchTST adds a
patch-reconstruction comparator. KMeansAD isolates fixed-prototype behavior.

KNN was removed from the active comparison by author decision on 2026-07-21.
The prior 16-method protocol remains available under
`legacy/seed2027_v2_16method/` for audit only.

The controller is resumable: a unit is skipped only when its JSON contains all
six finite metrics and no error. Final tables must not be used to change any
hyperparameter in this run.

TSB-AD-integrated baselines use `Optimal_Uni_algo_HP_dict` or
`Optimal_Multi_algo_HP_dict`, which were selected on the official Tuning split.
PaAno, GBOC, MEMTO, and DCdetector use released method configurations adapted
only to the official per-series prefix interface. DuoBa uses the V1 defaults
already frozen on Tuning. No Eval metric is an input to configuration choice.

## Current local status

- All 193 selected Eval files are present locally (122 U + 71 M), giving
  2,895 method-series units for 15 active methods.
- All ten active code/config entries and both recovery assets listed in
  `checksums/frozen-code.sha256` match.
- The recovered Server B set contains 396 valid six-metric records and no bad
  record.
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

- `multiserver/merge_server_b.py` is historical and must not be used for a new
  run.
- Shared-resource timings are invalid for the paper efficiency table.
- Never commit SSH keys, passwords, `.env` files, raw data, checkpoints, or
  anomaly-score arrays.
