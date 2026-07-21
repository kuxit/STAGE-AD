from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time
from typing import Any


METRICS = (
    "VUS-PR",
    "VUS-ROC",
    "R-based-F1",
    "AUC-PR",
    "AUC-ROC",
    "Standard-F1",
)
TARGET_FILES = (
    "141_CATSv2_id_4_Sensor_tr_41727_1st_41827.csv",
    "138_CATSv2_id_1_Sensor_tr_16568_1st_16668.csv",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def unit_path(result: Path, file_name: str) -> Path:
    return result / "units" / "GBOC" / "M" / "CATSv2" / f"{Path(file_name).stem}.json"


def valid_record(path: Path, file_name: str) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = [float(payload["metrics"][name]) for name in METRICS]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False
    return (
        payload.get("method") == "GBOC"
        and payload.get("track") == "M"
        and payload.get("dataset") == "CATSv2"
        and payload.get("file") == file_name
        and payload.get("seed") == 2027
        and payload.get("error") is None
        and all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values)
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def group_members(pgid: int) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            if os.getpgid(pid) != pgid:
                continue
        except (ProcessLookupError, PermissionError):
            continue
        members.append({"pid": pid, "cmdline": cmdline(pid)})
    return sorted(members, key=lambda item: int(item["pid"]))


def verify_supervisor(pid: int, pgid: int, result: Path) -> None:
    if os.getpgrp() == pgid:
        raise RuntimeError("watcher must run in a different process group (use setsid)")
    try:
        actual_pgid = os.getpgid(pid)
        actual_sid = os.getsid(pid)
    except ProcessLookupError as error:
        raise RuntimeError(f"supervisor PID {pid} is not running") from error
    command = cmdline(pid)
    if actual_pgid != pgid or actual_sid != pgid:
        raise RuntimeError(
            f"supervisor identity mismatch: pid={pid} pgid={actual_pgid} sid={actual_sid}, expected {pgid}"
        )
    if "shard_server_b.py" not in command or str(result) not in command:
        raise RuntimeError(f"PID {pid} is not the expected Server B supervisor: {command}")


def atomic_marker(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wait for the two in-flight GBOC records, then stop the old Server B process group."
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("/root/autodl-tmp/AAAI/results/DUOBA_10SUBSET_BASELINES/seed2027_v2_server_b"),
    )
    parser.add_argument("--supervisor-pid", type=int, default=11141)
    parser.add_argument("--expected-pgid", type=int, default=11141)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--max-wait-seconds", type=float, default=21600.0)
    parser.add_argument("--term-timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--abandon-inflight",
        action="store_true",
        help=(
            "Stop immediately after auditing the currently active boundary units. "
            "Any target without a valid formal record is recorded for a later clean rerun."
        ),
    )
    parser.add_argument(
        "--marker",
        type=Path,
        default=Path(
            "/root/autodl-tmp/AAAI/results/DUOBA_10SUBSET_BASELINES/"
            "server_b_boundary_stop.json"
        ),
    )
    args = parser.parse_args()
    result = args.result.resolve()
    marker = args.marker.resolve()
    if args.poll_seconds <= 0 or args.max_wait_seconds <= 0 or args.term_timeout_seconds <= 0:
        raise ValueError("all timeouts must be positive")
    if marker.exists():
        raise RuntimeError(f"refusing to reuse existing audit marker: {marker}")

    verify_supervisor(args.supervisor_pid, args.expected_pgid, result)
    initial_members = group_members(args.expected_pgid)
    for file_name in TARGET_FILES:
        path = unit_path(result, file_name)
        if not valid_record(path, file_name) and not any(
            file_name in str(member["cmdline"]) for member in initial_members
        ):
            raise RuntimeError(f"target is neither complete nor active in the old process group: {file_name}")

    print(f"[{now()}] armed for {', '.join(TARGET_FILES)}", flush=True)
    if not args.abandon_inflight:
        deadline = time.monotonic() + args.max_wait_seconds
        previous: tuple[str, ...] | None = None
        while True:
            complete = tuple(
                file_name for file_name in TARGET_FILES if valid_record(unit_path(result, file_name), file_name)
            )
            if complete != previous:
                print(
                    f"[{now()}] valid boundary records: {len(complete)}/{len(TARGET_FILES)} {complete}",
                    flush=True,
                )
                previous = complete
            if len(complete) == len(TARGET_FILES):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("boundary records did not become valid before max-wait-seconds")
            verify_supervisor(args.supervisor_pid, args.expected_pgid, result)
            time.sleep(args.poll_seconds)

    records = {}
    rerun_required = []
    for file_name in TARGET_FILES:
        path = unit_path(result, file_name)
        if valid_record(path, file_name):
            records[file_name] = {"status": "valid", "path": str(path), "sha256": sha256(path)}
        else:
            records[file_name] = {"status": "inflight_abandoned", "path": str(path)}
            rerun_required.append(file_name)
    verify_supervisor(args.supervisor_pid, args.expected_pgid, result)
    members_before = group_members(args.expected_pgid)
    audit: dict[str, Any] = {
        "status": "ready_to_stop",
        "protocol": "server-b-throughput-boundary-v1",
        "created_at": now(),
        "result_root": str(result),
        "supervisor_pid": args.supervisor_pid,
        "expected_pgid": args.expected_pgid,
        "boundary_records": records,
        "abandon_inflight": bool(args.abandon_inflight),
        "rerun_required": rerun_required,
        "members_before_sigterm": members_before,
        "starts_new_scheduler": False,
    }
    atomic_marker(marker, audit)

    if rerun_required and not args.abandon_inflight:
        raise RuntimeError(f"boundary records unexpectedly became invalid: {rerun_required}")
    print(
        f"[{now()}] boundary audited; sending SIGTERM to PGID {args.expected_pgid}; "
        f"rerun_required={rerun_required}",
        flush=True,
    )
    os.killpg(args.expected_pgid, signal.SIGTERM)
    term_deadline = time.monotonic() + args.term_timeout_seconds
    while group_members(args.expected_pgid) and time.monotonic() < term_deadline:
        time.sleep(0.25)

    remaining = group_members(args.expected_pgid)
    escalated = False
    if remaining:
        escalated = True
        print(f"[{now()}] PGID still has members; sending SIGKILL: {remaining}", flush=True)
        os.killpg(args.expected_pgid, signal.SIGKILL)
        kill_deadline = time.monotonic() + 10.0
        while group_members(args.expected_pgid) and time.monotonic() < kill_deadline:
            time.sleep(0.25)

    audit.update(
        {
            "status": "stopped" if not group_members(args.expected_pgid) else "stop_incomplete",
            "completed_at": now(),
            "sigkill_escalated": escalated,
            "members_after_stop": group_members(args.expected_pgid),
        }
    )
    atomic_marker(marker, audit)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    return 0 if audit["status"] == "stopped" else 1


if __name__ == "__main__":
    raise SystemExit(main())
