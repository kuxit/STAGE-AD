"""Audited high-throughput scheduler for an isolated Server B result root.

This scheduler changes only process-level concurrency.  It imports the frozen
unit command builder and runners from ``controller.py``; it does not alter a
model, runner, configuration, input, score, or metric.  Accuracy timings from
a concurrent run are deliberately marked as unsuitable for paper reporting.

The scheduler is Linux-only because it uses ``flock`` to guarantee that one
process owns the result root.  Use ``--preflight-only`` before every new run.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
from typing import Any, Iterable

try:
    import fcntl
except ImportError:  # pragma: no cover - the formal scheduler is Linux-only.
    fcntl = None  # type: ignore[assignment]


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
sys.path.insert(0, str(EXPERIMENT))

from common import METRICS, atomic_json, dataset_name, sha256  # noqa: E402
from controller import (  # noqa: E402
    EXTERNAL_DEEP,
    TSB_DEEP,
    command_for,
    run_unit,
    standard_path,
)
from input_integrity import verify as verify_input_integrity  # noqa: E402


SEED = 2027
ALL_METHODS = (*TSB_DEEP, *EXTERNAL_DEEP)
PARITY_FILE = "124_TAO_id_9_Environment_tr_500_1st_1.csv"
PARITY_POLICY = {
    "max_metric_absolute_difference": 0.005,
    "mean_metric_absolute_difference": 0.002,
}
EXPECTED_HASHES = {
    "common": "e1c8cb0d732dd18bde179ef552577eeea06151903e8f8821a1afe31e6e67ba59",
    "duoba": "94f31c50641c1c097b4b60cad50994d99dd6148148f2ae35cddc51579d242dd3",
    "controller": "88a394978d2547608e2ee58e30e9797a58a6bc950ca950c8c802a8b567dd8d78",
    "deep_runner": "a8bfb73293005ba5e9da9c3fc321e4f1b9cbae7db908434ac03568abd459840f",
    "classical_runner": "d97a1e4083bd1157c73afc3255b35eb871add08361c95c8e52cf8a95f90ca83c",
    "paano_one": "00437318c9a6b6ded357fd88892b336ea6cebc5fe6c7755a08bfa909d67ddf75",
    "gboc_one": "3861da207667fe4944a45712088a9072abec07fa7ce2167bbcb8751f29082e19",
    "gboc_config": "1839041de874a6a0586628274c51a14a4b6b66a049c2a7d46d38e2513d511fab",
    "memto_one": "e923c6187b0954e0c01cc3fb09eade7a037b3e66148386bab19327f3d6901371",
    "dcdetector_one": "2fb559dcfb3a7deb5b0a5c745ac5b524969f1cabf95946d2b05d1178dae1bb38",
}
METHOD_COST = {
    "GBOC": 100.0,
    "DuoBa": 8.0,
    "AnomalyTransformer": 5.0,
    "PatchTST": 4.5,
    "TimesNet": 4.0,
    "DCdetector": 3.0,
    "PaAno": 2.8,
    "USAD": 2.5,
    "TranAD": 2.4,
    "MEMTO": 2.3,
    "OmniAnomaly": 2.2,
}
FORBIDDEN_SUFFIXES = {".npy", ".npz", ".pt", ".pth", ".ckpt", ".safetensors"}
FORBIDDEN_DIRECTORY_NAMES = {"score", "scores", "checkpoint", "checkpoints"}
PROTOCOL_NAME = "duoba-server-b-high-throughput-accuracy-v1"

Job = tuple[str, str, str]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def paths(root: Path, result: Path, python: Path) -> argparse.Namespace:
    experiment = root / "DuoBa-Baseline-Seed2027"
    return argparse.Namespace(
        repo=root,
        experiment=experiment,
        result=result,
        python=python,
        duoba_source=root / "GROVE-AD-V3" / "grove_ad.py",
        paano_root=root / "external" / "PaAno",
        paano_one=root / "GROVE-AD-V3" / "paano_official_one.py",
        gboc_root=root / "external" / "GBOC_official_55a2a31",
        gboc_one=root / "spectral_tsad_v5" / "baselines" / "gboc" / "run_one.py",
        gboc_config=root
        / "spectral_tsad_v5"
        / "baselines"
        / "gboc"
        / "configs"
        / "official_release.json",
        memto_root=root / "external" / "MEMTO_official",
        memto_one=experiment / "run_one_memto.py",
        dcdetector_root=root / "external" / "DCdetector",
        dcdetector_one=experiment / "run_one_dcdetector.py",
        seed=SEED,
    )


def source_hashes(args: argparse.Namespace) -> dict[str, str]:
    return {
        "common": sha256(args.experiment / "common.py"),
        "duoba": sha256(args.duoba_source),
        "controller": sha256(args.experiment / "controller.py"),
        "deep_runner": sha256(args.experiment / "run_one_deep.py"),
        "classical_runner": sha256(args.experiment / "run_one_classical.py"),
        "paano_one": sha256(args.paano_one),
        "gboc_one": sha256(args.gboc_one),
        "gboc_config": sha256(args.gboc_config),
        "memto_one": sha256(args.memto_one),
        "dcdetector_one": sha256(args.dcdetector_one),
    }


def validate_frozen_inputs(args: argparse.Namespace) -> dict[str, str]:
    actual = source_hashes(args)
    if actual != EXPECTED_HASHES:
        raise RuntimeError(f"frozen source hash mismatch: {actual}")
    input_manifest = args.repo / "server_b_input_manifest.json"
    if not input_manifest.is_file():
        raise RuntimeError(f"missing verified input manifest: {input_manifest}")
    payload = json.loads(input_manifest.read_text(encoding="utf-8"))
    if payload.get("status") != "verified" or payload.get("seed") != SEED:
        raise RuntimeError("Server B input manifest is not verified for seed 2027")
    return actual


ENVIRONMENT_SCRIPT = r'''
import hashlib, json, pathlib, platform, subprocess, sys, torch
freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
print(json.dumps({
 'python_executable':str(pathlib.Path(sys.executable).resolve()),
 'python':platform.python_version(),
 'torch':torch.__version__,
 'torch_cuda':torch.version.cuda,
 'cudnn':torch.backends.cudnn.version(),
 'devices':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
 'pip_freeze_sha256':hashlib.sha256(freeze.encode()).hexdigest(),
}))
'''


def runtime_environment(python: Path) -> dict[str, Any]:
    return json.loads(
        subprocess.check_output([str(python), "-c", ENVIRONMENT_SCRIPT], text=True)
    )


def environment_matches(reported: dict[str, Any], current: dict[str, Any]) -> bool:
    for key, value in reported.items():
        if key == "python_executable":
            if Path(str(value)).resolve() != Path(str(current.get(key, ""))).resolve():
                return False
        elif current.get(key) != value:
            return False
    return True


def valid_job_record(path: Path, job: Job) -> bool:
    method, track, file_name = job
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if (
        payload.get("method") != method
        or payload.get("track") != track
        or payload.get("file") != file_name
        or payload.get("dataset") != dataset_name(file_name)
        or payload.get("seed") != SEED
        or payload.get("error") is not None
    ):
        return False
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return False
    try:
        values = [float(metrics[name]) for name in METRICS]
    except (KeyError, TypeError, ValueError):
        return False
    return all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values)


def validate_job(root: Path, job: Job) -> None:
    method, track, file_name = job
    if method not in ALL_METHODS:
        raise ValueError(f"unsupported method in job list: {method}")
    if track not in {"U", "M"}:
        raise ValueError(f"invalid track in job list: {track}")
    if not file_name or Path(file_name).name != file_name:
        raise ValueError(f"job file must be a basename: {file_name!r}")
    data_path = root / "data" / f"TSB-AD-{track}" / file_name
    if not data_path.is_file():
        raise FileNotFoundError(f"job input is missing: {data_path}")
    dataset_name(file_name)


def load_exact_job_list(path: Path) -> tuple[list[Job], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("jobs") if isinstance(payload, dict) else payload
    if isinstance(payload, dict) and payload.get("seed", SEED) != SEED:
        raise ValueError("exact job list has an unexpected seed")
    if not isinstance(records, list) or not records:
        raise ValueError("exact job list must contain a nonempty `jobs` list")
    jobs: list[Job] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"job {index} is not an object")
        try:
            jobs.append(
                (str(record["method"]), str(record["track"]), str(record["file"]))
            )
        except KeyError as error:
            raise ValueError(f"job {index} lacks {error.args[0]!r}") from error
    return jobs, {
        "mode": "exact_job_list",
        "path": str(path.resolve()),
        "sha256": sha256(path),
    }


def load_split_jobs(
    split_plan_path: Path,
    root: Path,
    methods: tuple[str, ...],
) -> tuple[list[Job], dict[str, Any]]:
    integrity = verify_input_integrity(root, split_plan_path)
    payload = json.loads(split_plan_path.read_text(encoding="utf-8"))
    if payload.get("seed") != SEED:
        raise ValueError("split plan has an unexpected seed")
    records = payload.get("formal_files")
    if not isinstance(records, list) or not records:
        raise ValueError("split plan has no formal files")
    names = [Path(str(record["path"])).name for record in records]
    jobs = [(method, "M", name) for name in names for method in methods]
    return jobs, {
        "mode": "verified_split_plan",
        "path": str(split_plan_path.resolve()),
        "sha256": integrity["plan_sha256"],
        "input_digest": integrity["input_digest"],
    }


def exact_input_inventory(root: Path, jobs: Iterable[Job]) -> list[dict[str, Any]]:
    unique = sorted({(track, file_name) for _, track, file_name in jobs})
    inventory = []
    for track, file_name in unique:
        path = root / "data" / f"TSB-AD-{track}" / file_name
        inventory.append(
            {
                "track": track,
                "file": file_name,
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return inventory


def job_dict(job: Job) -> dict[str, str]:
    return {"method": job[0], "track": job[1], "file": job[2]}


def sort_jobs(
    args: argparse.Namespace,
    jobs: list[Job],
    mode: str,
    deferred_methods: set[str],
) -> list[Job]:
    if mode == "input":
        ordered = list(jobs)
    else:
        def cost(job: Job) -> tuple[float, str, str, str]:
            method, track, file_name = job
            size = (args.repo / "data" / f"TSB-AD-{track}" / file_name).stat().st_size
            return (
                METHOD_COST[method] * math.log2(max(size, 2)),
                method,
                track,
                file_name,
            )

        ordered = sorted(jobs, key=cost, reverse=True)

    # Preserve the selected within-group order while moving explicitly deferred
    # methods to the tail. This changes only launch order, never job membership.
    return [job for job in ordered if job[0] not in deferred_methods] + [
        job for job in ordered if job[0] in deferred_methods
    ]


def parse_methods(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ("GBOC",)
    methods = tuple(item.strip() for item in value.split(",") if item.strip())
    if (
        not methods
        or len(set(methods)) != len(methods)
        or not set(methods).issubset(ALL_METHODS)
    ):
        raise ValueError(f"invalid method selection: {methods}")
    return methods


def parse_gpu_workers(
    gpus_text: str,
    workers_per_gpu: int,
    worker_map_text: str | None,
) -> dict[str, int]:
    gpus = tuple(item.strip() for item in gpus_text.split(",") if item.strip())
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError(f"invalid GPU list: {gpus}")
    if workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be positive")
    result = {gpu: workers_per_gpu for gpu in gpus}
    if worker_map_text:
        parsed: dict[str, int] = {}
        for part in worker_map_text.split(","):
            key, separator, raw_count = part.strip().partition("=")
            if not separator or not key or not raw_count:
                raise ValueError(f"invalid GPU worker entry: {part!r}")
            count = int(raw_count)
            if count < 1 or key in parsed:
                raise ValueError(f"invalid GPU worker entry: {part!r}")
            parsed[key] = count
        if set(parsed) != set(gpus):
            raise ValueError("--gpu-worker-map must cover exactly --gpus")
        result = parsed
    return result


def parse_method_limits(values: list[str], methods: set[str], slots: int) -> dict[str, int]:
    limits = {method: slots for method in methods}
    seen: set[str] = set()
    for part in values:
        key, separator, raw_count = part.strip().partition("=")
        if not separator or key not in methods or key in seen:
            raise ValueError(f"invalid method limit: {part!r}")
        count = int(raw_count)
        if count < 1 or count > slots:
            raise ValueError(f"method limit must be in [1,{slots}]: {part!r}")
        limits[key] = count
        seen.add(key)
    return limits


def assert_no_persistent_artifacts(result: Path) -> None:
    if not result.exists():
        return
    forbidden = []
    for path in result.rglob("*"):
        relative_parts = {part.lower() for part in path.relative_to(result).parts}
        if relative_parts & FORBIDDEN_DIRECTORY_NAMES:
            forbidden.append(path)
        elif path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            forbidden.append(path)
        if len(forbidden) >= 10:
            break
    if forbidden:
        raise RuntimeError(
            "result root contains persistent score/checkpoint artifacts: "
            + ", ".join(str(path) for path in forbidden)
        )


def unexpected_unit_records(result: Path, jobs: list[Job]) -> list[Path]:
    expected = {standard_path(result, *job).resolve() for job in jobs}
    units_root = result / "units"
    if not units_root.exists():
        return []
    return [path for path in units_root.rglob("*.json") if path.resolve() not in expected]


def validate_method_execution(
    root: Path,
    result: Path,
    method_plan_path: Path,
    methods: set[str],
) -> tuple[dict[str, argparse.Namespace], dict[str, Any], dict[str, str]]:
    raw_plan = json.loads(method_plan_path.read_text(encoding="utf-8"))
    if not isinstance(raw_plan, dict) or not methods.issubset(raw_plan):
        raise RuntimeError("method plan does not cover the exact scheduled method set")

    args_by_method: dict[str, argparse.Namespace] = {}
    execution: dict[str, Any] = {}
    environments: dict[Path, dict[str, Any]] = {}
    hashes: dict[str, str] | None = None

    for method in sorted(methods):
        spec = raw_plan[method]
        if not isinstance(spec, dict):
            raise RuntimeError(f"invalid method plan entry for {method}")
        python = Path(os.path.abspath(str(spec.get("python", ""))))
        parity_name = spec.get("parity_report")
        if (
            not python.is_file()
            or not isinstance(parity_name, str)
            or Path(parity_name).name != parity_name
        ):
            raise RuntimeError(f"invalid execution environment for {method}")

        method_args = paths(root, result, python)
        method_hashes = validate_frozen_inputs(method_args)
        if hashes is None:
            hashes = method_hashes
        elif hashes != method_hashes:
            raise RuntimeError("method environments see different frozen sources")

        parity_path = root / "parity_pilot" / parity_name
        parity = json.loads(parity_path.read_text(encoding="utf-8"))
        if python not in environments:
            environments[python] = runtime_environment(python)
        current = environments[python]
        expected = spec.get("environment", current)
        candidate_environment = parity.get("candidate_environment")
        if parity.get("seed") != SEED or parity.get("source_sha256") != method_hashes:
            raise RuntimeError(f"parity report does not bind the frozen run for {method}")
        if parity.get("file") != PARITY_FILE or parity.get("policy_set_before_run") != PARITY_POLICY:
            raise RuntimeError(f"parity protocol mismatch for {method}")
        gate = parity.get("methods", {}).get(method, {})
        if not gate.get("passed"):
            raise RuntimeError(f"{method} did not pass the frozen parity gate")
        candidate_name = parity.get("candidate_name")
        if not isinstance(candidate_name, str) or Path(candidate_name).name != candidate_name:
            raise RuntimeError(f"invalid parity candidate for {method}")
        pilot_job = (method, "M", PARITY_FILE)
        candidate = standard_path(root / "parity_pilot" / candidate_name, *pilot_job)
        reference = root / "parity_pilot" / "reference" / f"{method}.json"
        if not valid_job_record(candidate, pilot_job) or not valid_job_record(reference, pilot_job):
            raise RuntimeError(f"invalid parity record for {method}")
        if gate.get("candidate_record_sha256") != sha256(candidate):
            raise RuntimeError(f"parity candidate changed for {method}")
        if gate.get("reference_record_sha256") != sha256(reference):
            raise RuntimeError(f"parity reference changed for {method}")
        if not isinstance(candidate_environment, dict) or not environment_matches(
            candidate_environment, current
        ):
            raise RuntimeError(f"formal and parity interpreters differ for {method}")
        if expected != current:
            raise RuntimeError(f"formal environment fingerprint changed for {method}")

        args_by_method[method] = method_args
        execution[method] = {
            "python": str(python),
            "parity_report": parity_name,
            "parity_report_sha256": sha256(parity_path),
            "environment": current,
        }

    if hashes is None:
        raise RuntimeError("no method was selected")
    execution["method_plan_sha256"] = sha256(method_plan_path)
    return args_by_method, execution, hashes


def canonical_commands(
    args_by_method: dict[str, argparse.Namespace],
    result: Path,
    jobs: list[Job],
    gpus: Iterable[str],
) -> dict[str, Any]:
    records = []
    scratch = Path("/__UNIT_SCRATCH__")
    for job in jobs:
        target = standard_path(result, *job)
        for gpu in gpus:
            command, environment, raw_path = command_for(
                args_by_method[job[0]], *job, target, gpu, scratch
            )
            records.append(
                {
                    "job": job_dict(job),
                    "physical_gpu": gpu,
                    "argv_with_scratch_placeholder": command,
                    "raw_path_with_scratch_placeholder": (
                        str(raw_path) if raw_path is not None else None
                    ),
                    "environment": {
                        key: environment.get(key)
                        for key in (
                            "CUDA_VISIBLE_DEVICES",
                            "OMP_NUM_THREADS",
                            "MKL_NUM_THREADS",
                            "OPENBLAS_NUM_THREADS",
                        )
                    },
                }
            )
    return {
        "scratch_placeholder": str(scratch),
        "commands": records,
    }


def verify_or_write(path: Path, payload: dict[str, Any], hash_key: str) -> None:
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != payload:
            raise RuntimeError(f"refusing to replace a different {path.name}")
        return
    atomic_json(path, payload)
    if payload.get(hash_key) is None:
        raise RuntimeError(f"audit payload lacks {hash_key}")


@dataclass(frozen=True)
class WorkerSpec:
    gpu: str
    slot: int

    @property
    def name(self) -> str:
        return f"gpu{self.gpu}-slot{self.slot}"


class WorkAllocator:
    def __init__(self, jobs: list[Job], method_limits: dict[str, int]) -> None:
        self.pending = list(jobs)
        self.method_limits = method_limits
        self.active_by_method: dict[str, int] = defaultdict(int)
        self.active_total = 0
        self.stop_new = False
        self.condition = threading.Condition()

    def acquire(self) -> Job | None:
        with self.condition:
            while True:
                if self.stop_new:
                    return None
                for index, job in enumerate(self.pending):
                    method = job[0]
                    if self.active_by_method[method] < self.method_limits[method]:
                        self.pending.pop(index)
                        self.active_by_method[method] += 1
                        self.active_total += 1
                        return job
                if not self.pending:
                    return None
                if self.active_total == 0:
                    raise RuntimeError("method limits made the pending queue unschedulable")
                self.condition.wait()

    def release(self, job: Job, failed: bool) -> None:
        with self.condition:
            method = job[0]
            self.active_by_method[method] -= 1
            self.active_total -= 1
            if failed:
                self.stop_new = True
            self.condition.notify_all()

    def remaining(self) -> int:
        with self.condition:
            return len(self.pending)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audited multi-process-per-GPU Server B accuracy scheduler"
    )
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/AAAI"))
    parser.add_argument("--result", type=Path, required=True)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--job-list", type=Path)
    source_group.add_argument("--split-plan", type=Path)
    parser.add_argument(
        "--methods",
        help="comma-separated methods for split mode; default: GBOC",
    )
    parser.add_argument("--method-plan", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument(
        "--gpu-worker-map",
        help="optional exact map such as 0=2,1=2; must cover --gpus",
    )
    parser.add_argument(
        "--method-limit",
        action="append",
        default=[],
        help="global concurrent-unit cap, repeat as METHOD=COUNT",
    )
    parser.add_argument("--schedule-order", choices=("lpt", "input"), default="lpt")
    parser.add_argument(
        "--defer-method",
        action="append",
        default=[],
        help="move this method to the tail without changing shard membership; repeatable",
    )
    parser.add_argument("--preflight-only", action="store_true")
    cli = parser.parse_args()

    if fcntl is None:
        raise RuntimeError("the high-throughput scheduler requires Linux flock")
    root = cli.root.resolve()
    result = cli.result.resolve()
    method_plan_path = cli.method_plan.resolve()
    result.mkdir(parents=True, exist_ok=True)
    lock_stream = (result / "shard.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another scheduler already owns this result root") from error

    methods_from_cli = parse_methods(cli.methods)
    if cli.job_list:
        if cli.methods is not None:
            raise ValueError("--methods is not used with an exact --job-list")
        source_path = cli.job_list.resolve()
        jobs, job_source = load_exact_job_list(source_path)
    else:
        source_path = (
            cli.split_plan.resolve()
            if cli.split_plan
            else root / "server_b_split_plan.json"
        )
        jobs, job_source = load_split_jobs(source_path, root, methods_from_cli)

    if len(set(jobs)) != len(jobs):
        raise ValueError("job source contains duplicate units")
    for job in jobs:
        validate_job(root, job)
    scheduled_methods = {job[0] for job in jobs}
    deferred_methods = tuple(dict.fromkeys(cli.defer_method))
    if any(method not in scheduled_methods for method in deferred_methods):
        raise ValueError(
            f"deferred methods must belong to the scheduled shard: {deferred_methods}"
        )
    args_by_method, method_execution, hashes = validate_method_execution(
        root, result, method_plan_path, scheduled_methods
    )
    base_args = args_by_method[next(iter(sorted(scheduled_methods)))]
    ordered_jobs = sort_jobs(
        base_args, jobs, cli.schedule_order, set(deferred_methods)
    )

    gpu_workers = parse_gpu_workers(
        cli.gpus, cli.workers_per_gpu, cli.gpu_worker_map
    )
    worker_specs = [
        WorkerSpec(gpu, slot)
        for gpu, count in gpu_workers.items()
        for slot in range(1, count + 1)
    ]
    method_limits = parse_method_limits(
        cli.method_limit, scheduled_methods, len(worker_specs)
    )

    assert_no_persistent_artifacts(result)
    unexpected = unexpected_unit_records(result, ordered_jobs)
    if unexpected:
        raise RuntimeError(
            f"result root contains out-of-plan unit records: {unexpected[:3]}"
        )
    input_inventory = exact_input_inventory(root, ordered_jobs)
    command_body = canonical_commands(
        args_by_method, result, ordered_jobs, gpu_workers.keys()
    )
    command_payload = {
        **command_body,
        "command_sha256": canonical_hash(command_body),
    }
    scheduler_sha256 = sha256(Path(__file__).resolve())
    immutable_body: dict[str, Any] = {
        "protocol": PROTOCOL_NAME,
        "seed": SEED,
        "scheduler_sha256": scheduler_sha256,
        "controller_command_builder_sha256": hashes["controller"],
        "command_sha256": command_payload["command_sha256"],
        "job_source": job_source,
        "requested_jobs": [job_dict(job) for job in jobs],
        "scheduled_jobs": [job_dict(job) for job in ordered_jobs],
        "input_inventory": input_inventory,
        "input_inventory_sha256": canonical_hash(input_inventory),
        "methods": sorted(scheduled_methods),
        "method_execution": method_execution,
        "source_sha256": hashes,
        "metrics": list(METRICS),
        "gpu_workers": gpu_workers,
        "method_limits": method_limits,
        "schedule_order": cli.schedule_order,
        "deferred_methods": list(deferred_methods),
        "expected_units": len(ordered_jobs),
        "accuracy_only": True,
        "paper_efficiency_valid": False,
        "efficiency_invalid_reason": (
            "multiple concurrent units may share each GPU and host resources"
        ),
        "persistent_scores": False,
        "persistent_checkpoints": False,
        "fail_stop_policy": "stop assigning new units after the first failed unit",
    }
    protocol_payload = {
        **immutable_body,
        "protocol_sha256": canonical_hash(immutable_body),
    }

    completed_at_start = {
        job for job in ordered_jobs if valid_job_record(standard_path(result, *job), job)
    }
    preflight = {
        "status": "preflight_passed",
        "result": str(result),
        "expected_units": len(ordered_jobs),
        "completed_units": len(completed_at_start),
        "pending_units": len(ordered_jobs) - len(completed_at_start),
        "workers": [spec.name for spec in worker_specs],
        "method_limits": method_limits,
        "scheduler_sha256": scheduler_sha256,
        "command_sha256": command_payload["command_sha256"],
        "protocol_sha256": protocol_payload["protocol_sha256"],
        "paper_efficiency_valid": False,
    }
    if cli.preflight_only:
        print(json.dumps(preflight, indent=2, ensure_ascii=False), flush=True)
        return 0

    command_path = result / "high_throughput_commands.json"
    protocol_path = result / "high_throughput_protocol.json"
    manifest_path = result / "high_throughput_manifest.json"
    verify_or_write(command_path, command_payload, "command_sha256")
    verify_or_write(protocol_path, protocol_payload, "protocol_sha256")

    previous: dict[str, Any] = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        immutable_keys = tuple(protocol_payload)
        mismatches = [
            key for key in immutable_keys if previous.get(key) != protocol_payload.get(key)
        ]
        if mismatches:
            raise RuntimeError(
                f"refusing to resume a different high-throughput identity: {mismatches}"
            )

    manifest: dict[str, Any] = {
        **protocol_payload,
        "status": "running",
        "started_at": previous.get("started_at", now()),
        "resumed_at": now() if previous else None,
        "resume_count": int(previous.get("resume_count", 0)) + (1 if previous else 0),
        "previous_error_units": int(previous.get("error_units", 0)) if previous else 0,
        "completed_units": len(completed_at_start),
        "error_units": 0,
        "pending_units": len(ordered_jobs) - len(completed_at_start),
    }
    atomic_json(manifest_path, manifest)
    print(
        f"[{now()}] high-throughput shard starts at "
        f"{len(completed_at_start)}/{len(ordered_jobs)} with {len(worker_specs)} workers",
        flush=True,
    )

    pending = [job for job in ordered_jobs if job not in completed_at_start]
    allocator = WorkAllocator(pending, method_limits)
    events: queue.Queue[dict[str, Any]] = queue.Queue()

    def worker(spec: WorkerSpec) -> None:
        while True:
            job = allocator.acquire()
            if job is None:
                break
            try:
                item = run_unit(args_by_method[job[0]], *job, spec.gpu)
                item = {**item, "worker": spec.name}
                target = standard_path(result, *job)
                if item.get("status") == "complete" and not valid_job_record(target, job):
                    item = {
                        **item,
                        "status": "error",
                        "error": "runner produced an invalid formal record",
                    }
                assert_no_persistent_artifacts(result)
            except Exception as error:
                item = {
                    "status": "error",
                    **job_dict(job),
                    "gpu": spec.gpu,
                    "worker": spec.name,
                    "error": f"{type(error).__name__}: {error}",
                }
            failed = item.get("status") == "error"
            allocator.release(job, failed=failed)
            events.put({"kind": "unit", "item": item, "job": job_dict(job)})
        events.put({"kind": "worker_done", "worker": spec.name})

    threads = [
        threading.Thread(target=worker, args=(spec,), name=spec.name, daemon=False)
        for spec in worker_specs
    ]
    for thread in threads:
        thread.start()

    completed = set(completed_at_start)
    errors: list[dict[str, Any]] = []
    finished_workers = 0
    while finished_workers < len(threads):
        event = events.get()
        if event["kind"] == "worker_done":
            finished_workers += 1
            continue
        item = event["item"]
        job_record = event["job"]
        job = (job_record["method"], job_record["track"], job_record["file"])
        if item.get("status") in {"complete", "skipped"} and valid_job_record(
            standard_path(result, *job), job
        ):
            completed.add(job)
        else:
            errors.append(item)
        manifest.update(
            {
                "completed_units": len(completed),
                "error_units": len(errors),
                "pending_units": allocator.remaining(),
                "active_workers": len(threads) - finished_workers,
                "last_unit": item,
                "updated_at": now(),
            }
        )
        if errors:
            manifest["first_error"] = errors[0]
        atomic_json(manifest_path, manifest)  # The main thread is the sole writer.
        print(
            f"[{len(completed)}/{len(ordered_jobs)}] {item}",
            flush=True,
        )

    for thread in threads:
        thread.join()
    while not events.empty():
        event = events.get_nowait()
        if event.get("kind") != "worker_done":
            raise RuntimeError("unprocessed unit event after all workers exited")

    assert_no_persistent_artifacts(result)
    final_completed = {
        job for job in ordered_jobs if valid_job_record(standard_path(result, *job), job)
    }
    manifest.update(
        {
            "completed_units": len(final_completed),
            "error_units": len(errors),
            "pending_units": len(ordered_jobs) - len(final_completed),
            "active_workers": 0,
            "status": (
                "complete"
                if len(final_completed) == len(ordered_jobs) and not errors
                else "failed"
                if errors
                else "incomplete"
            ),
            "completed_at": now(),
        }
    )
    atomic_json(manifest_path, manifest)
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
