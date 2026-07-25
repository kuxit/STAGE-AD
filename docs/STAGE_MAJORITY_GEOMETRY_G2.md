# STAGE majority-local: independent-support observed-exemplar geometry

This diagnostic keeps the majority-local encoder, timestamp/interval
alignment, intermediate granular sampling, token head, and embedding head
fixed. It changes only how the final observed-exemplar memory represents and
scores normal geometry.

## Frozen question

Does the current memory under-retain normal dynamics because overlapping
windows make temporal support look larger than it is, and because one global
distance treats broad dominant regions and compact underrepresented regions as
equally trustworthy?

## Candidates

- `G0 control`: current majority-local observed-exemplar memory.
- `G1 independent support`: choose an observed representative around a center
  weighted by timestamp-distinct support within each adaptive region.
- `G2 support radius`: keep the control representative but add a conservative
  penalty to broad, strongly supported regions.
- `G3 combined`: use both timestamp-distinct representatives and the local
  support/radius calibration.
- `G4 transition`: add a lightweight, label-blind state-transition coordinate
  to G3 so equal states with different temporal evolution can be separated.

The region partition stays adaptive and every memory item remains an observed
training exemplar. No anomaly label, Eval score, fixed prototype count, or
auxiliary negative is used.

## Efficient execution

The five candidates share exactly one encoder training and one embedding
extraction for each official TSB-AD Tuning series. Geometry and four final
heads (`min_split` in `{4,64}`, `k` in `{1,3}`) are evaluated in memory.

- 6 physical encoder-training units
- 30 logical geometry units
- 120 metric/head records
- seed 2026
- datasets: U/MSL, U/SED, M/CATSv2, M/GHL

This is a mechanism diagnostic, not a final parameter lock. A winning geometry
must pass a separately frozen multi-seed Tuning stage before any Eval.

## Evidence

For each series and candidate, the runner stores compact post-hoc diagnostics:

- raw normal/anomalous time-series intervals;
- training-normal, Eval-normal, and Eval-anomaly PCA projections;
- selected observed prototypes with source time, support, and radius;
- normal/anomaly score separation.

Labels are loaded only after all scores are produced. These figures never
participate in Tuning selection.
