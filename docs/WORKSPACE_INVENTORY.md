# Local workspace inventory

Audit date: 2026-07-21. Sizes are approximate and were measured under
`F:/python_project/AAAI`.

## Keep: experiment-critical assets

| Path | Approx. size | Purpose |
|---|---:|---|
| `phasead_pipeline/data` | 3,357.7 MiB | TSB-AD raw data and official file lists |
| `phasead_pipeline/external` | 176.2 MiB | PaAno source used by the pipeline |
| `DuoBa-Baseline-Seed2027` | 110.0 MiB | Frozen experiment and recovery assets |
| `GROVE-AD-V3` | 1.0 MiB | DuoBa and PaAno frozen adapters |
| `spectral_tsad_v5` | 0.4 MiB | Frozen GBOC adapter/config |
| `tmp/source_audit` | 38.7 MiB | Upstream provenance clones; keep until pinned sources are packaged |

The official Eval lists select exactly 193 existing CSV files: 122 U and 71 M.
With 16 methods this gives 3,088 units.

The validated recovery set is
`multiserver/server_b_recovery_396_20260720T1441Z`: 396 valid JSON records,
including 43 records for each of nine methods and 9 GBOC records. Its archive
hash is recorded in `checksums/frozen-code.sha256`.

## Review before cleanup

These items are not needed for a fresh accuracy run, but may contain provenance
or paper-build history. Do not remove them until the GitHub repository and a
second local backup exist.

| Candidate | Approx. reclaimable | Reason |
|---|---:|---|
| `multiserver/server_b_shutdown_snapshot_20260720T1423Z` | 108.6 MiB | Superseded 269-record snapshot; 108.3 MiB is logs |
| all `__pycache__` and `.pytest_cache` directories | 1.9 MiB | Re-creatable caches |
| `.tmp_aaai27_author_kit` | 10.8 MiB | Temporary extraction |
| `.tmp_aaai27_render` | 0.7 MiB | Temporary render |
| `.tmp_dataset_sources` | 4.4 MiB | Temporary source download |
| `_duoba_paper_build_*` | 0.8 MiB | Re-creatable paper builds |
| `DuoBa_AAAI27_Submission/tmp` | 21.7 MiB | Paper intermediate files |
| `tmp/pdfs` | 24.5 MiB | Source/reference PDF cache |

The conservative cleanup total is about 173 MiB. The two raw dataset ZIP files
(`TSB-AD-M.zip` and `TSB-AD-U.zip`) occupy another 584.9 MiB, but they should be
removed only after the extracted data has a complete hash manifest and another
copy exists.

## Do not commit to GitHub

- raw TSB-AD CSV/ZIP files;
- Python environments and package caches;
- SSH keys, passwords, tokens, or `.env` files;
- checkpoints, model weights, score arrays, or transient logs;
- shared-resource runtime measurements.

Commit code, configuration, protocol, dependency manifests, checksums,
full-precision metric JSON, final CSV/Markdown tables, and compact audit files.
