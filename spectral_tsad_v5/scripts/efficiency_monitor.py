from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any, Mapping, Sequence, TextIO


EFFICIENCY_SCHEMA_VERSION = 1


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _linux_process_table() -> dict[int, tuple[int, float]]:
    """Return pid -> (ppid, RSS MiB) without adding a psutil dependency."""
    table: dict[int, tuple[int, float]] = {}
    proc = Path("/proc")
    if not proc.exists():
        return table
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        ppid = 0
        rss_kib = 0.0
        for line in status.splitlines():
            if line.startswith("PPid:"):
                ppid = int(line.split()[1])
            elif line.startswith("VmRSS:"):
                rss_kib = float(line.split()[1])
        table[int(entry.name)] = (ppid, rss_kib / 1024.0)
    return table


def _process_tree(root_pid: int, table: Mapping[int, tuple[int, float]]) -> set[int]:
    selected = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _) in table.items():
            if ppid in selected and pid not in selected:
                selected.add(pid)
                changed = True
    return selected


def _gpu_process_memory_mib(pids: set[int]) -> tuple[float, dict[str, float]]:
    if not pids:
        return 0.0, {}
    command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return 0.0, {}
    by_gpu: dict[str, float] = {}
    for line in output.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[1])
            used = float(parts[2])
        except ValueError:
            continue
        if pid in pids:
            by_gpu[parts[0]] = by_gpu.get(parts[0], 0.0) + used
    return sum(by_gpu.values()), by_gpu


def gpu_inventory() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in output.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) == 4:
            try:
                total = float(parts[3])
            except ValueError:
                total = None
            rows.append({"index": parts[0], "uuid": parts[1], "name": parts[2], "memory_total_mib": total})
    return rows


class EfficiencyMonitor:
    """Sample wall time, process-tree RSS, and process-attributed GPU memory."""

    def __init__(
        self,
        root_pid: int | None = None,
        *,
        sample_interval_s: float = 0.5,
        include_gpu: bool = True,
    ) -> None:
        self.root_pid = int(root_pid or os.getpid())
        self.sample_interval_s = float(sample_interval_s)
        self.include_gpu = bool(include_gpu)
        self._started_perf: float | None = None
        self._started_epoch: float | None = None
        self._stopped_perf: float | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples = 0
        self._peak_cpu_rss_mib = 0.0
        self._peak_gpu_process_mib = 0.0
        self._peak_gpu_by_uuid_mib: dict[str, float] = {}

    def _sample(self) -> None:
        table = _linux_process_table()
        pids = _process_tree(self.root_pid, table)
        rss = sum(table.get(pid, (0, 0.0))[1] for pid in pids)
        self._peak_cpu_rss_mib = max(self._peak_cpu_rss_mib, rss)
        if self.include_gpu:
            gpu_total, gpu_by_uuid = _gpu_process_memory_mib(pids)
            self._peak_gpu_process_mib = max(self._peak_gpu_process_mib, gpu_total)
            for uuid, used in gpu_by_uuid.items():
                self._peak_gpu_by_uuid_mib[uuid] = max(self._peak_gpu_by_uuid_mib.get(uuid, 0.0), used)
        self._samples += 1

    def _loop(self) -> None:
        while not self._stop_event.wait(self.sample_interval_s):
            self._sample()

    def start(self) -> "EfficiencyMonitor":
        if self._started_perf is not None:
            return self
        self._started_perf = time.perf_counter()
        self._started_epoch = time.time()
        self._sample()
        self._thread = threading.Thread(target=self._loop, name=f"efficiency-{self.root_pid}", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        if self._started_perf is None:
            self.start()
        self._stopped_perf = time.perf_counter()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.sample_interval_s * 3))
        self._sample()
        return {
            "efficiency_schema_version": EFFICIENCY_SCHEMA_VERSION,
            "wall_time_s": self._stopped_perf - float(self._started_perf),
            "peak_cpu_process_tree_rss_mib": self._peak_cpu_rss_mib,
            "peak_gpu_process_tree_memory_mib": self._peak_gpu_process_mib,
            "peak_gpu_by_uuid_mib": self._peak_gpu_by_uuid_mib,
            "root_pid": self.root_pid,
            "measurement_scope": "root process plus descendants",
            "gpu_attribution": "nvidia-smi per-process memory; unrelated jobs on the same GPU excluded",
            "sample_interval_s": self.sample_interval_s,
            "samples": self._samples,
            "started_at_epoch_s": self._started_epoch,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_inventory": gpu_inventory() if self.include_gpu else [],
        }

    def __enter__(self) -> "EfficiencyMonitor":
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()


def run_monitored_subprocess(
    command: Sequence[str],
    *,
    cwd: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    stdout: TextIO | int | None = None,
    stderr: TextIO | int | None = None,
    check: bool = False,
    include_gpu: bool = True,
    sample_interval_s: float = 0.5,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=None if env is None else dict(env),
        stdout=stdout,
        stderr=stderr,
        text=False,
    )
    monitor = EfficiencyMonitor(
        process.pid,
        sample_interval_s=sample_interval_s,
        include_gpu=include_gpu,
    ).start()
    returncode = process.wait()
    efficiency = monitor.stop()
    completed = subprocess.CompletedProcess(list(command), returncode)
    if check and returncode:
        raise subprocess.CalledProcessError(returncode, list(command))
    return completed, efficiency


def tagged_record(
    efficiency: Mapping[str, Any],
    *,
    method: str,
    track: str,
    seed: int,
    phase: str,
    scope: str,
    files_count: int,
    points_count: int | None = None,
    file: str | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    row = dict(efficiency)
    row.update(
        {
            "method": method,
            "track": track,
            "seed": int(seed),
            "phase": phase,
            "scope": scope,
            "files_count": int(files_count),
            "points_count": None if points_count is None else int(points_count),
            "file": file,
            "command": None if command is None else list(command),
        }
    )
    return row
