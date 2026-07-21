# Results layout

This directory is reserved for reviewed local copies of experiment outputs.

- `staging/`: mutable pull target from a running server; ignored by Git.
- `backups/`: timestamped local snapshots; ignored by Git.
- `seed2027_v3/`: final reviewed unit JSON, manifest, summaries, and checksums;
  intended for Git after validation.

Only six-metric accuracy records belong here. Checkpoints, score arrays, and
shared-resource timing claims are forbidden.
