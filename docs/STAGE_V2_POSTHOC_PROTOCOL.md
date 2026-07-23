# STAGE-v2 Post-hoc Development and Confirmation Protocol

## 1. Status and non-retroactivity

This document is the normative scientific boundary for STAGE-v2. It is a
**post-hoc development protocol written after the frozen STAGE-v1 seed-2026
Eval result was observed and the preregistered v1 success gate failed**. It is
not part of the original v1 formal protocol, it does not amend the v1 gate, and
it cannot make v1 successful retroactively.

The v1 source, configuration lock, result JSONs, completion markers,
comparison report, and audit fingerprints are immutable evidence. They must
remain under their original versioned paths and must never be overwritten,
deleted, relabelled as v2, or merged with v2 results. The frozen v1 code commit
is `74fd2b0b7c35672210bad023eb4305ca975fd481`. The authoritative v1 result
archive, rather than this branch, remains the source for the exact v1 lock and
result fingerprints.

Observing v1 Eval contaminated the old 193-series Eval split for subsequent
method development. STAGE-v2 candidate generation, ranking, shortlisting,
stability checks, and stopping decisions therefore use only the current
official TSB-AD Tuning split. Old Eval may be retained and described only as
clearly labelled historical or post-hoc development evidence. It is not
confirmatory evidence for v2 and may not select a candidate, parameter, seed,
metric, track, subset, series, baseline, or rerun.

A confirmatory STAGE-v2 claim requires a genuinely unopened holdout as defined
in Section 8. Running a new seed on old Eval, splitting the observed 193 series
after the fact, or applying nested cross-validation to those series does not
restore independence.

The machine-readable Stage 1 registry is
`configs/stage_v2_stage1_posthoc.json`. The document and registry must agree
before execution. A mismatch is a fail-stop condition; neither authority may
be silently preferred after results exist.

The words **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, and **SHALL NOT** are
normative.

## 2. Data, metrics, and post-hoc objective

Stages 1 and 2 use all 22 current official TSB-AD Tuning series in the ten
predefined subsets:

- univariate: `UCR`, `Exathlon`, `MSL`, `SED`, and `TODS`;
- multivariate: `CATSv2`, `GHL`, `LTDB`, `SVDB`, and `TAO`.

Every candidate fits only the filename-declared `tr_<N>` prefix without
labels. Labels are read only after anomaly scores have been finalized by the
frozen evaluator. No series may be removed because it is difficult, and no
subset may receive an undeclared budget or seed exception.

All six full-precision metrics are always computed and retained:

1. `VUS-PR`;
2. `VUS-ROC`;
3. `R-based-F1`;
4. `AUC-PR`;
5. `AUC-ROC`;
6. `Standard-F1`.

For a candidate, let `mean6` be the unweighted mean of all six metrics and let
`mean_pr_f1` be the unweighted mean of `VUS-PR`, `R-based-F1`, `AUC-PR`, and
`Standard-F1`. The post-hoc Tuning score is frozen as

```text
0.5 * mean6 + 0.5 * mean_pr_f1
```

For multi-series selection, metrics are first macro-averaged over every series
in the applicable subset or track and then combined by this formula. For
multi-seed selection, the three seed-level scores receive equal weight. The
ordered tie-breakers are `VUS-PR`, `R-based-F1`, `AUC-PR`, and `Standard-F1`,
followed by the lexicographically lower candidate or head ID.

This PR/F1-weighted score is explicitly a **v2 post-hoc development
objective**. It must not be described as the v1 preregistered objective. Its
weights, metrics, aggregation order, and tie-breakers may not change after the
first Stage 1 unit starts. Old Eval values and margins to baselines are never
part of the objective.

## 3. Same-family backbone and paired-head constraint

All candidates use the same STAGE dilated-residual encoder family, timestamp
correspondence, overlap losses, optimizer family, scheduler, preprocessing,
normalization, and patch-to-timestamp aggregation. The residual topology,
`dilations=(1,2,4,8,4,2)`, and `group_norm_groups=8` are fixed.

The explicit capacity candidates `e05_compact_encoder` and
`e06_wide_encoder` scale channels and representation dimensions inside this
same architecture family. They are exploratory same-family capacity scaling,
not evidence that STAGE is backbone-agnostic. Replacing the encoder with a
Transformer, RNN, another convolutional topology, additional blocks, another
loss family, or another scoring architecture is prohibited.

All 16 training candidates use training-stage `gb_min_split=4`,
`gb_max_rounds=64`, and the fixed Stage 1 head
`(final_gb_min_split=4, top_k=3)`. The final-memory grid is not allowed to
change the intermediate training partition or training sampler.

Within Stage 2, every final-memory head for a given
`(training candidate, track, subset, series, seed)` MUST reuse the exact same
trained encoder and final embeddings. The encoder may be trained only once for
that key; a head candidate may neither retrain nor update it. This paired-head
constraint is required both for causal attribution and compute efficiency.

## 4. Stage 1A: all 16 training candidates at seed 2026

Stage 1A evaluates exactly the following 16 candidates at seed `2026` using
the fixed head `(4,3)`. The first nine candidates preserve every unique v1
Tuning winner as an anchor. The remaining seven are explicitly declared v2
exploratory points. No seventeenth candidate may be added after execution
starts.

Abbreviations in the table are: `P` = patch size, `C/T/E` =
channels/token dimension/embedding dimension, `D;t` = overlap deltas and trim,
`B` = batch size, `S` = requested steps, `rho` = activation fraction, and
`alpha` = sampling power. Parameters omitted from a row are fixed by the
machine-readable registry.

| Candidate ID | P | C/T/E | D;t | lr | drop | B | S | rho | alpha | wd | clip |
|:--|--:|:--:|:--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `v1_t00_msl_anchor` | 96 | 128/64/64 | 24,48;8 | 3e-4 | 0.1 | 256 | 1000 | 0.2 | 0.50 | 1e-4 | 1.0 |
| `v1_t06_cats_ghl_anchor` | 96 | 128/64/64 | 24,48;8 | 1e-4 | 0.0 | 256 | 750 | 0.2 | 0.25 | 1e-4 | 1.0 |
| `v1_t22_exathlon_anchor` | 96 | 128/64/64 | 24,48;8 | 1e-3 | 0.0 | 256 | 750 | 0.1 | 0.25 | 1e-4 | 1.0 |
| `v1_t17_ucr_anchor` | 128 | 128/64/64 | 32,64;8 | 1e-4 | 0.1 | 256 | 1000 | 0.4 | 0.75 | 1e-4 | 1.0 |
| `v1_t10_sed_anchor` | 64 | 128/64/64 | 16,32;4 | 1e-3 | 0.2 | 256 | 1500 | 0.1 | 0.50 | 1e-4 | 1.0 |
| `v1_t13_tods_anchor` | 64 | 128/64/64 | 16,32;4 | 1e-3 | 0.2 | 256 | 750 | 0.1 | 0.50 | 1e-4 | 1.0 |
| `v1_t12_ltdb_anchor` | 64 | 128/64/64 | 16,32;4 | 1e-3 | 0.2 | 256 | 1000 | 0.2 | 0.25 | 1e-4 | 1.0 |
| `v1_t08_svdb_anchor` | 128 | 128/64/64 | 32,64;8 | 1e-3 | 0.1 | 256 | 1500 | 0.4 | 0.25 | 1e-4 | 1.0 |
| `v1_t05_tao_anchor` | 64 | 128/64/64 | 16,32;4 | 1e-4 | 0.2 | 256 | 750 | 0.4 | 0.50 | 1e-4 | 1.0 |
| `e00_small_batch_short_series` | 96 | 128/64/64 | 24,48;8 | 1e-4 | 0.0 | 64 | 1000 | 0.2 | 0.25 | 1e-4 | 1.0 |
| `e01_batch128_no_decay` | 96 | 128/64/64 | 24,48;8 | 1e-4 | 0.0 | 128 | 1500 | 0.2 | 0.25 | 0 | 0.5 |
| `e02_low_lr_long_train` | 96 | 128/64/64 | 24,48;8 | 5e-5 | 0.0 | 256 | 2000 | 0.2 | 0.25 | 1e-5 | 1.0 |
| `e03_ball_balanced_sampling` | 96 | 128/64/64 | 24,48;8 | 1e-4 | 0.0 | 256 | 1000 | 0.2 | 0.00 | 1e-4 | 1.0 |
| `e04_patch_uniform_sampling` | 96 | 128/64/64 | 24,48;8 | 1e-4 | 0.0 | 256 | 1000 | 0.2 | 1.00 | 1e-4 | 1.0 |
| `e05_compact_encoder` | 96 | 64/32/32 | 24,48;8 | 3e-4 | 0.1 | 256 | 1000 | 0.2 | 0.50 | 1e-4 | 1.0 |
| `e06_wide_encoder` | 96 | 256/128/64 | 24,48;8 | 1e-4 | 0.1 | 128 | 1000 | 0.2 | 0.50 | 1e-4 | 1.0 |

The complete JSON objects, rather than rounded table notation, define the
executable values. Every row must pass `StageConfig.validate()` and have a
frozen source/config fingerprint before launch.

Stage 1A MUST complete all `16 x 22 = 352` candidate-series units, with no
missing, invalid, duplicate, selectively omitted, or error-bearing unit,
before any ranking is used. The fixed shortlist rule is:

- each multi-series subset retains exactly its top three candidates;
- the single-series subsets `SED`, `CATSv2`, and `LTDB` do not form a
  one-series shortlist; each uses the top-three shortlist computed over all
  Tuning series in its corresponding U or M track;
- shortlist size is always three, including after ties; the declared
  tie-breakers resolve the third position;
- a result may not change the shortlist size, replace a candidate, add a
  candidate, or return to partial Stage 1A execution.

The shortlist IDs, full-precision statistics, unit manifest, and ranking
fingerprint are frozen before Stage 1B.

## 5. Stage 1B: multi-seed training-candidate confirmation

Stage 1B evaluates only the frozen top-three shortlists at seeds `2027` and
`2028`, with the same fixed head `(4,3)`. Together with the complete seed-2026
Stage 1A records, this yields three equally weighted seeds for every shortlisted
candidate.

One training winner is selected per multi-series subset by applying the same
post-hoc score to the equal-weight mean over seeds `2026`, `2027`, and `2028`.
`SED` uses the U-track training winner; `CATSv2` and `LTDB` use the M-track
training winner. No seed-specific winner, best-seed substitution, favorable
seed subset, or seed-dependent parameter is allowed.

Stage 1B is a Tuning-only stability confirmation and final training-candidate
selection. It is not independent test evidence. After every declared
shortlist-seed unit is complete, the winner mapping and its three-seed
statistics are frozen and hashed. Stage 2 cannot switch to another training
candidate.

## 6. Stage 2: final-memory head search and stability confirmation

Only after the training winners are frozen does Stage 2 evaluate the following
20 final-memory heads:

| Head ID | Final-only `gb_min_split` | `top_k` |
|---:|---:|---:|
| H00 | 4 | 1 |
| H01 | 4 | 3 |
| H02 | 4 | 5 |
| H03 | 4 | 9 |
| H04 | 4 | 15 |
| H05 | 16 | 1 |
| H06 | 16 | 3 |
| H07 | 16 | 5 |
| H08 | 16 | 9 |
| H09 | 16 | 15 |
| H10 | 64 | 1 |
| H11 | 64 | 3 |
| H12 | 64 | 5 |
| H13 | 64 | 9 |
| H14 | 64 | 15 |
| H15 | 256 | 1 |
| H16 | 256 | 3 |
| H17 | 256 | 5 |
| H18 | 256 | 9 |
| H19 | 256 | 15 |

Every head is scored on all applicable Tuning series at seeds `2026`, `2027`,
and `2028`. The 20 heads reuse the corresponding already trained embeddings;
they do not create 20 encoder runs. Head selection uses the same post-hoc
objective after equal-weight aggregation over all three seeds. One head is
selected per multi-series subset. The single-series subsets `SED`, `CATSv2`,
and `LTDB` use the corresponding U- or M-track head selection.

`final_gb_min_split` applies only to the fresh final partition and exemplar
memory. It MUST NOT alter training-stage `gb_min_split=4`, the intermediate
partition, sampling probabilities, or encoder updates. Because the v1 code
uses a shared `gb_min_split` field, the v2 implementation must separate the
training and final values, or provide an equivalent audited interface, before
Stage 2. Setting the existing shared field globally is prohibited.

All other memory behavior remains identical to v1:

- initial partition count `K = floor(sqrt(N))`;
- the same BIC formula, split comparison, split order, and
  `gb_max_rounds=64`;
- one observed training exemplar per terminal cell, never an arithmetic
  centroid or synthetic prototype;
- the same embedding normalization, squared unit-Euclidean distance,
  nearest-neighbor reduction, and patch-to-timestamp aggregation;
- deterministic use of `min(top_k, number_of_memory_items)` when a memory has
  fewer than the requested neighbors.

The requested and effective `top_k`, final partition membership fingerprint,
and memory size must be recorded. Distinct declared heads that collapse to the
same effective partition and `top_k` remain visible in the audit; they may not
be selectively removed after their scores are known.

Stage 2 is complete only when every declared head-series-seed unit is valid and
all seed results are retained. The selected head, training winner, source,
Tuning manifests, evaluator, candidate plan, and full Stage 1/2 audit are then
frozen under one final development fingerprint. No scientific parameter may
change after this freeze.

## 7. Allowed and prohibited parameter boundaries

### Allowed only as explicitly registered before execution

- the 16 training candidates in Section 4 and their exact JSON values;
- the fixed Stage 1 head `(final_gb_min_split=4, top_k=3)`;
- the 20 final-memory heads in Section 6;
- same-family compact and wide capacity scaling in `e05` and `e06`;
- separating training and final `gb_min_split` while proving that the reference
  setting reproduces v1 computation;
- assertions, fingerprints, atomic writes, deterministic resume, and audit
  metadata that do not alter scientific outputs;
- a bug fix made before affected results are used, provided the plan is
  re-versioned and every affected unit is rerun from the beginning.

### Prohibited within STAGE-v2

- a training value, capacity, head value, loss, architecture type,
  preprocessing rule, evaluator, metric weight, or tie-breaker outside this
  document and the matching registry;
- changing the encoder or embeddings between heads of the same training run;
- changing initial `K`, BIC, observed-exemplar selection, normalization,
  distance, nearest-neighbor reduction, or timestamp aggregation;
- introducing a memory-ratio target, memory-size cap, centroid, synthetic
  prototype, or data-dependent head grid;
- using old Eval, baseline margins, or prospective holdout feedback in Stage 1
  or Stage 2;
- per-series parameters, unequal candidate budgets, Exathlon exceptions,
  manual overrides, or dataset-specific seeds;
- changing the fixed top-three shortlist size after any score is available;
- selecting the best seed, dropping a seed, averaging only favorable seeds,
  dropping a metric, suppressing a subset, or replacing a difficult series;
- adding candidates after partial results, repeatedly querying a holdout, or
  stopping when a baseline is first exceeded;
- presenting shared concurrent runtime as paper efficiency evidence.

## 8. Required unopened holdout

STAGE-v2 has no confirmatory result until it is evaluated once on a genuinely
unopened holdout. Before any holdout file, label, score, or aggregate is made
available, a holdout protocol MUST freeze and hash:

1. the data source, inclusion rule, immutable series manifest, and U/M and
   subset coverage;
2. label custody and the mechanism that prevents label or metric access during
   development;
3. the final STAGE-v2 source and complete subset-to-configuration mapping;
4. the complete baseline list and every baseline configuration;
5. all six metrics, Global/U/M aggregation, statistical test, multiplicity
   rule, missing-unit rule, and exact success criterion;
6. the seed policy and exact execution count;
7. the one-shot opening procedure and output/audit paths.

The default compute-saving final comparison uses seed `2026` once for STAGE-v2
and every stochastic comparator. If the confirmatory claim uses multiple
seeds, the identical seed set and aggregation must be frozen and applied
symmetrically to every stochastic comparator before opening the holdout. Seeds
may not be added, removed, or reassigned after a result is known.

If the paper claims superiority over sixteen baselines on all six metrics and
both tracks, all sixteen baselines and STAGE-v2 must be complete on the same
holdout, and STAGE-v2 must satisfy the frozen six Global and twelve U/M
track-metric comparisons. No baseline, metric, track, subset, series, or seed
may be excluded after scores are known. A narrower claim is permissible only
when it is written and frozen before holdout opening; it cannot be chosen from
holdout results.

If no genuinely unopened holdout is available, the study stops at post-hoc
development evidence. Old Eval, Stage 1, and Stage 2 must not be described as
an independent confirmation or an unbiased test of superiority.

## 9. Fail-stop and restart rules

The active stage stops without scientific continuation when any of the
following occurs:

- source, plan, configuration, data-manifest, evaluator, or metric hash drift;
- a missing, duplicate, non-finite, boolean, malformed, or error-bearing unit;
- incomplete candidate/head/seed coverage or ranking before required coverage;
- a shortlist other than the frozen top three;
- old Eval feedback in development or premature access to the final holdout;
- an undeclared parameter, seed, subset exception, candidate, head, or baseline
  exclusion;
- nondeterministic resume that cannot identify the exact interrupted unit;
- a scientific code change after affected results have begun.

An interrupted unit may be rerun only under the identical frozen plan when no
complete valid unit exists for its unique key. A scientific code change
invalidates every affected unit; results from old and new code may not be
merged. Premature holdout access invalidates that holdout permanently for
confirmation.

The final holdout is opened once. If the frozen success criterion fails,
STAGE-v2 fails confirmation and the result is retained and reported. Further
method development requires a separately named protocol, such as STAGE-v3,
and another genuinely unopened holdout. Continuing to adjust STAGE-v2 on the
same holdout until it exceeds a baseline is prohibited.

## 10. Reporting boundary

Every report and paper draft must keep these evidence classes separate:

1. **STAGE-v1 confirmatory evidence:** the immutable seed-2026 result and its
   failed preregistered gate;
2. **STAGE-v2 post-hoc development evidence:** the Tuning-only Stage 1/2
   search and any clearly labelled historical old-Eval diagnostics;
3. **STAGE-v2 confirmatory evidence:** the one-shot unopened-holdout result, if
   such a holdout is completed.

The paper must not state that v2 was developed without knowledge of v1 Eval,
that v1 passed its gate, that the PR/F1-weighted objective was preregistered for
v1, that capacity scaling proves backbone independence, or that old Eval is an
independent v2 test. Full-precision records, all six metrics, all declared
seeds, and all declared data groups must remain available for audit even when
the paper renders rounded or condensed summaries.
