# GitHub and result-backup workflow

## Repository boundary

Use `STAGE-AD` as the Git repository root. Keep the repository
private until upstream licenses and anonymization requirements are reviewed.
Do not initialize the whole `AAAI` workspace as one repository: it contains raw
data, unrelated experiments, paper artifacts, and audit clones.

Before the first push, package or pin the required sibling sources. Preferred
order:

1. Git submodules at exact upstream commits when the upstream repositories are
   stable and their licenses permit redistribution.
2. A `vendor/` snapshot with source URL, commit, license, and SHA256 inventory
   when an upstream repository is unstable or unavailable.
3. Never rely on an unversioned source directory that exists only on a server.

## Initial Git history

1. Run the local verification script and save its output.
2. Initialize Git in this directory with branch `main`.
3. Commit documentation, source code, configs, checksum files, and validation evidence.
4. Tag the audited protocol as `stage-seed2026-v1`.
5. Create a private GitHub repository and push `main` plus the tag.
6. Create a run branch such as `run/server-b-seed2026-v1`. Formal runs deploy
   from the frozen tag or an audited descendant with no method/config changes.

Do not place GitHub tokens on Server B. Clone with a read-only deploy key if
needed; perform result pushes from the local computer.

## Three-copy result policy

Maintain three independent copies:

1. Server B working result root.
2. Timestamped local snapshots under `results/backups/`.
3. A private GitHub result branch or GitHub Release containing reviewed metric
   JSON and checksum manifests.

Recommended cadence:

- Pull completed unit JSONs from Server B to local staging every 10 minutes.
- Create a local immutable snapshot every hour and whenever progress crosses
  another 10%, a method completes, or an error occurs.
- Push a batched result commit from the local computer at those milestones.
- At completion, commit all 2,895 active full-precision unit JSONs, the final manifest,
  full-precision and three-decimal CSV/Markdown summaries, and a SHA256
  inventory. Add a signed/tagged release such as `stage-seed2026-complete`.

Per-unit JSONs are small and suitable for Git once immutable. Do not commit a
manifest after every unit; batch commits to avoid unnecessary history growth.
Large archives belong in GitHub Releases or another object store, not normal
Git history. Git LFS is optional but is unnecessary for the metric JSONs.

## Deployment provenance

Every server deployment should record:

- Git commit and tag;
- clean/dirty worktree status;
- frozen file SHA256 values;
- upstream source URLs and commits;
- Python, CUDA, driver, PyTorch, and package-lock information;
- selected-data hash inventory;
- exact command and result root.

The scheduler should refuse to start if any identity differs from the recorded
run contract.
