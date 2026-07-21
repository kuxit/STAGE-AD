# Reproducibility and pre-run gate

## Frozen experiment identity

- Protocol: `duoba-10subset-seed2027-v2`
- Seed: `2027`
- Methods: 16
- Eval series: 193 (122 univariate, 71 multivariate)
- Units: 3,088
- Metrics: VUS-PR, VUS-ROC, R-based-F1, AUC-PR, AUC-ROC,
  Standard-F1
- Fit scope: the filename-declared `tr_<N>` prefix, label blind
- Evaluation scope: the complete series
- Selection: official TSB-AD Tuning parameters or released configurations;
  Eval metrics must never affect configuration
- Storage: full-precision metrics only; no checkpoint or score array

## Local readiness matrix (2026-07-21)

| Asset | Local status | Notes |
|---|---:|---|
| Frozen controller and common utilities | Ready | Hashes match |
| DuoBa frozen source | Ready | `../GROVE-AD-V3/grove_ad.py` matches |
| Classical/deep adapters | Ready | Hashes match |
| PaAno adapter | Ready | Frozen adapter hash matches |
| GBOC adapter and released config | Ready | Frozen hashes match |
| MEMTO/DCdetector adapters | Ready | Frozen hashes match |
| Official Eval file lists | Ready | 193 selected entries |
| Selected raw CSV files | Ready | 193/193 present locally |
| Recovered Server B metrics | Ready | 396 valid, 0 invalid |
| PaAno released source tree | Present elsewhere | Located under `../phasead_pipeline/external/PaAno`; launch layout still needs normalization |
| GBOC released source tree | Partial/audit copy | Audit clone exists under `../tmp/source_audit/GBOC`; exact run layout must be recovered or rebuilt |
| MEMTO released source tree | Missing locally | Must be copied from a trusted source and pinned |
| DCdetector released source tree | Missing locally | Must be copied from a trusted source and pinned |
| Frozen server Python environments | Server-only | Recreate from an exported lock/manifest before launch |

The local workspace therefore preserves the scientific protocol, selected data,
all frozen entry points, and recovered metrics, but it is not yet a standalone
offline deployment. The missing upstream trees must be restored and verified
before calling Server B complete.

## Mandatory pre-run gate

Do not start the formal run until every item is true:

1. Run `scripts/verify_local_assets.ps1`; all checks must pass.
2. Restore the released PaAno, GBOC, MEMTO, and DCdetector source trees under a
   documented `external/` layout.
3. Record each upstream repository URL, commit, license, and directory hash.
4. Export the Python environments (`pip freeze`, Python, CUDA, PyTorch, driver)
   and store the text manifests in Git.
5. Run a no-write preflight on Server B and compare all frozen hashes.
6. Run one parity unit per external method before the formal scheduler.
7. Use a fresh result root. Never merge an old manifest into a new run.
8. Enable periodic local result pulls before the first formal unit.
9. Treat shared-resource runtime as invalid for the paper efficiency table.

## Result validity

A unit counts only when its JSON has seed 2027, `error` is null, and all six
metrics are finite. A partially written or zero-byte file does not count. The
controller's atomic rename prevents a process failure from replacing a complete
record with a partial record, but periodic off-server copies are still required.
