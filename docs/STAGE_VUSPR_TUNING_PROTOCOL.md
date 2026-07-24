# STAGE dataset-specific VUS-PR Tuning protocol

## Purpose

This branch corrects two protocol defects found after the frozen v1/v2 runs:

1. short training series no longer collapse a declared 750--2,000 optimizer
   updates to approximately one epoch;
2. candidate and head selection uses only the full-precision macro-average
   official-Tuning VUS-PR for each dataset.

The sixteen completed baselines are read-only Eval references. No baseline is
run by this protocol.

## Non-negotiable boundaries

- Method name: STAGE.
- Architecture family: unchanged.
- Mechanisms: timestamp/interval alignment, adaptive intermediate geometry,
  and final observed-exemplar geometry remain active.
- Selection split: official TSB-AD Tuning only.
- Selection objective: per-dataset macro VUS-PR only.
- Exact ties: canonical candidate/head identifier only.
- Eval metrics are never read during selection.
- Every one of the ten datasets selects its own training and head parameters.
  Track-level fallback is forbidden.
- Checkpoints, embeddings, score arrays, NumPy arrays, raw data, credentials,
  and large logs are forbidden artifacts.

## Frozen search stages

### Stage1A: broad training screen

- 24 predeclared same-family training candidates.
- 22 official representative Tuning series across ten datasets.
- seed 2026.
- fixed provisional head: final minimum split 4 and top-k 3.
- 528 physical/logical units.
- Keep the top three distinct training candidates per dataset by macro
  Tuning VUS-PR.

The candidate set covers patch sizes 48--192, batch sizes 64--256, update
budgets 750--2,000, learning rates 5e-5--1e-3, dropout 0--0.2, geometry
activation fractions 0.1--0.5, sampling powers 0--1, compact/base/wide
encoders, and matching overlap scales. It does not change the STAGE method
type.

### Stage1B: seed confirmation

- Evaluate each dataset's frozen Stage1A top three with seeds 2027 and 2028.
- Combine seeds 2026/2027/2028.
- Select one training candidate separately for every dataset by macro Tuning
  VUS-PR.

### Stage2: final observed-exemplar head

- Freeze the selected training candidate first.
- Evaluate the 20 predeclared heads from
  `final_gb_min_split in {4,16,64,256}` and
  `top_k in {1,3,5,9,15}` with all three seeds.
- Select one head separately for every dataset by macro Tuning VUS-PR.

## Local acceptance gates

Before server deployment:

1. all Python 3.8-compatible unit tests pass in `00gwk`;
2. the actual-data plan audit reports 24 candidates, 22 series, 528 units,
   no execution deduplication, no short-series caps, and no Eval feedback;
3. two identical real-series GPU smoke runs produce exactly equal six
   metrics and final-memory size;
4. plan/resume, source/data/evaluator hashes, duplicate detection, forbidden
   artifacts, and atomic JSON behavior pass;
5. the repository contains no raw data, secrets, checkpoints, score arrays,
   or smoke outputs;
6. code, protocol, tests, documentation, and checksum manifest are committed
   and pushed before the server is started.

## Server entry point

After cloning/checking out the reviewed commit:

```bash
bash scripts/launch_stage_vuspr_stage1a.sh
```

The launcher verifies a clean worktree, the frozen checksum manifest, the
formal data/evaluator locations, two visible GPUs, and at least 15 GiB of free
space. It freezes the plan before execution and then runs three isolated
workers per GPU. The scheduler dispatches longest-estimated units first and
interleaves GPU 0/1 lanes. Existing strictly valid unit JSONs are resumed;
invalid existing units are never overwritten.

## Interpretation

This search is designed to maximize the probability of a strong Eval result
without using Eval to select parameters. No local test can logically guarantee
that every unseen Eval metric will beat every baseline. The valid engineering
target is therefore: eliminate known implementation/protocol defects, search
the declared STAGE family thoroughly on Tuning, freeze per-dataset parameters,
and run Eval exactly once after the lock.
