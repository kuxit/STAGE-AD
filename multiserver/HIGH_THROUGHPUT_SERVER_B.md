# High-throughput Server B scheduler

`shard_server_b_high_throughput.py` is an **accuracy-only** scheduler for a
separate Server B result root. It does not modify any model, runner, config,
input, score, or metric. Its only experimental change is process-level
concurrency: more than one isolated process may be assigned to a physical GPU.

Do not use its wall-clock, training-time, inference-time, GPU-memory, or power
records in the paper. Concurrent units share GPU and host resources. Efficiency
experiments must still use one exclusive unit per GPU with no competing jobs.

## Safety invariants

- The scheduler uses the same `shard.lock` as the original scheduler, so two
  schedulers cannot own one result root.
- Every unit is launched through the frozen `controller.run_unit` and receives
  an independent `CUDA_VISIBLE_DEVICES` value. The unit runner still sees one
  logical CUDA device.
- Source, runner, config, interpreter, parity-report, candidate-record, and
  reference-record hashes are checked before scheduling.
- An exact job list or the verified Server B split plan fixes membership. Data
  hashes, the scheduler hash, canonical command hash, and protocol hash are
  written to the audit records.
- Resume skips only a six-metric unit record whose method, track, dataset, file,
  seed, metric ranges, and error field are valid.
- Only the scheduler's main thread writes `high_throughput_manifest.json`.
  Unit runners write disjoint unit and log paths.
- The first failed or invalid unit stops all new assignments. Units already in
  flight are allowed to finish and are recorded.
- The persistent result root is rejected if it contains `.npy`, `.npz`, `.pt`,
  `.pth`, `.ckpt`, `.safetensors`, `scores/`, or `checkpoints/`. External runners
  may create transient files in the per-unit `TemporaryDirectory`; the frozen
  `run_unit` deletes that directory and copies only the normalized metric record.

## Exact job-list schema

```json
{
  "protocol": "gboc-2-per-gpu-pilot-v1",
  "seed": 2027,
  "jobs": [
    {
      "method": "GBOC",
      "track": "M",
      "file": "138_CATSv2_id_1_Sensor_tr_16568_1st_16668.csv"
    },
    {
      "method": "GBOC",
      "track": "M",
      "file": "139_CATSv2_id_2_Sensor_tr_5592_1st_5692.csv"
    }
  ]
}
```

The list is exact: duplicates, missing files, unsupported methods, path
components, and a seed other than 2027 are rejected. The scheduler independently
hashes every referenced data file.

## Preflight a two-process-per-GPU GBOC pilot

Use a new, empty result root. Preflight performs no model run and does not write
the protocol or manifest.

```bash
/root/autodl-tmp/duoba-env/bin/python \
  /root/autodl-tmp/AAAI/DuoBa-Baseline-Seed2027/multiserver/shard_server_b_high_throughput.py \
  --root /root/autodl-tmp/AAAI \
  --result /root/autodl-tmp/AAAI/results/gboc_ht_pilot_seed2027 \
  --job-list /root/autodl-tmp/AAAI/gboc_ht_pilot_jobs.json \
  --method-plan /root/autodl-tmp/AAAI/DuoBa-Baseline-Seed2027/multiserver/server_b_method_plan.json \
  --gpus 0,1 \
  --gpu-worker-map 0=2,1=2 \
  --method-limit GBOC=4 \
  --preflight-only
```

After reviewing the three reported hashes, remove only `--preflight-only` to
run. Repeating the same command resumes the same immutable identity and skips
valid units. A different scheduler, command matrix, job list, data inventory,
worker layout, method limit, environment, or source hash is rejected.

## Use the verified split instead of an exact list

Omit `--job-list` and optionally provide `--split-plan`. Split mode defaults to
`/root/autodl-tmp/AAAI/server_b_split_plan.json` and `--methods GBOC`:

```bash
... shard_server_b_high_throughput.py \
  --result /root/autodl-tmp/AAAI/results/gboc_ht_split_seed2027 \
  --method-plan /root/autodl-tmp/AAAI/DuoBa-Baseline-Seed2027/multiserver/server_b_method_plan.json \
  --methods GBOC \
  --gpu-worker-map 0=2,1=2 \
  --method-limit GBOC=4 \
  --preflight-only
```

For a mixed-method shard, repeat `--method-limit METHOD=COUNT` to cap global
concurrency by method. `--workers-per-gpu N` gives every listed GPU the same
number of slots; `--gpu-worker-map` overrides it and must cover every GPU.

## Audit outputs

- `high_throughput_commands.json`: canonical per-job/per-GPU argv and relevant
  environment variables, with a scratch-path placeholder and command hash.
- `high_throughput_protocol.json`: immutable job, input, environment, parity,
  source, worker, and policy identity with protocol hash.
- `high_throughput_manifest.json`: single-writer progress and final status.
- `logs/` and `units/`: the unchanged runner's per-unit log and normalized
  six-metric record.

The exact runtime command (including the random temporary directory) remains in
each unit log. The canonical command file exists to make preflight and resume
identities deterministic.
