# Reproducibility and pre-run gate

## Frozen experiment identity

- Protocol: `duoba-10subset-seed2027-v3-no-knn`
- Seed: `2027`
- Methods: 15 (KNN excluded by author decision)
- Eval series: 193 (122 univariate, 71 multivariate)
- Units: 2,895
- Metrics: VUS-PR, VUS-ROC, R-based-F1, AUC-PR, AUC-ROC,
  Standard-F1
- Fit scope: the filename-declared `tr_<N>` prefix, label blind
- Evaluation scope: the complete series
- Selection: official TSB-AD Tuning parameters or released configurations;
  Eval metrics must never affect configuration
- Storage: full-precision metrics only; no checkpoint or score array

The superseded 16-method protocol is preserved in
`legacy/seed2027_v2_16method/`; it must not be used to schedule or summarize
the active comparison.

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
| PaAno released source tree | Ready | Exact pinned source restored under `external/PaAno` |
| GBOC released source tree | Ready | Exact pinned source restored under `external/GBOC` |
| MEMTO released source tree | Ready | Exact pinned source restored under `external/MEMTO` |
| DCdetector released source tree | Ready | Exact pinned source restored under `external/DCdetector` |
| TSB-AD/PAI source tree | Ready | Exact pinned source restored under `external/PAI` |
| Frozen server Python environments | Server-only | Recreate from an exported lock/manifest before launch |

The local workspace preserves the scientific protocol, selected data, all
active entry points, five pinned upstream trees, and recovered metrics. A
server deployment is still incomplete until its Python/CUDA environment and
all hashes pass the pre-run gate.

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
