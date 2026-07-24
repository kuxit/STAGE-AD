# STAGE one-hour dual-GPU runbook

This run is restricted to STAGE and the official TSB-AD Tuning split.  It
does not launch, modify, or rerun any baseline and does not consume Eval
feedback.

The frozen story contract fixes the problem and three mechanism roles, not one
particular formula: overlap-aware alignment must reduce context variation,
intermediate geometry must temper abundance-driven exposure, and final
geometry must retain supported observed exemplars adaptively. The current
24-candidate run explores mechanism-preserving implementations; complete
mechanism removals are reserved for diagnostic ablations.

## CPU work already frozen locally

`configs/stage_vuspr_stage1a_plan.json` contains the complete 528-unit plan:

- the 24 training candidates and all execution signatures;
- the 22 official Tuning files and their SHA-256 digests;
- source, protocol, manifest, data, and evaluator fingerprints;
- per-series point/channel/train-prefix metadata;
- each series' label-blind sliding-window value;
- the deterministic longest-work-first task estimates.

The server copies this plan into the result root.  It performs one parallel
22-file data preflight and one source/protocol/evaluator preflight, then every
worker trusts the fingerprinted preflight tokens.  It does not repeatedly hash
all sources and data for every one of the 528 units.

The model keeps the repeatedly sampled training prefix in a GPU window bank.
After anomaly scoring it releases model and embedding allocations before the
CPU evaluator starts.  The evaluator calls the exact official PaAno
primitives for the six frozen metrics and skips three unused metrics.  An
equality test checks that all six reported values are exactly identical to
`PaAno.utils.metrics.get_metrics`.

## Server checkout and start

```bash
cd /root/autodl-tmp
git clone --branch stage-vuspr-tuning --single-branch \
  https://github.com/kuxit/STAGE-AD.git STAGE-AD-vuspr-tuning
cd /root/autodl-tmp/STAGE-AD-vuspr-tuning
bash scripts/start_stage_vuspr_stage1a.sh
```

The launcher chooses workers per GPU from the smaller card's memory:

- below 14 GiB: 3;
- 14--21 GiB: 4;
- 22--39 GiB: 6;
- at least 40 GiB: 8.

Override only when observed memory headroom justifies it:

```bash
STAGE_WORKERS_PER_GPU=6 bash scripts/start_stage_vuspr_stage1a.sh
```

All BLAS/OpenMP pools default to one thread per worker, preventing CPU
oversubscription while the independent workers overlap GPU training, adaptive
partition construction, scoring, and official metric evaluation.

## Read-only monitoring

```bash
/root/autodl-tmp/envs/stage-py310/bin/python \
  scripts/stage_vuspr_watch.py \
  --repo /root/autodl-tmp/STAGE-AD-vuspr-tuning \
  --protocol configs/stage_vuspr_stage1a.json \
  --result-root /root/autodl-tmp/results/STAGE_vuspr_tuning/stage1a_seed2026 \
  --metrics-root /root/autodl-tmp/STAGE-AD/external \
  --pid-file /root/autodl-tmp/logs/stage_vuspr_stage1a.pid
```

Each unit JSON is atomic.  A server shutdown can lose only in-flight units;
rerunning the start script resumes strictly valid completed units.  Runtime
from this shared concurrent run is diagnostic only and is not paper evidence.
