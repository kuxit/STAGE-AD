from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


CONTROLLER_PID = 3222485
CONTROLLER = Path("/home/taoxie/AAAI/DuoBa-Baseline-Seed2027/controller.py")
CONTROLLER_SHA256 = "88a394978d2547608e2ee58e30e9797a58a6bc950ca950c8c802a8b567dd8d78"
TARGET_FILE = "083_LTDB_id_5_Medical_tr_15502_1st_15602.csv"
TARGET_RUNNER = "/spectral_tsad_v5/baselines/gboc/run_one.py"
RESULT = Path("/home/taoxie/AAAI/results/DUOBA_10SUBSET_BASELINES/seed2027_v2")
COMPLETE_MARKER = RESULT / "SERVER_B_SHARD_VERIFIED_COMPLETE.json"
PAUSED_MARKER = RESULT / "SERVER_A_PAUSED_FOR_SERVER_B.json"
LOG = RESULT / "server_a_boundary_guard.log"
EXPECTED_PLAN_SHA256 = "543c884bfc761f5fa767392b5bc29c9b7c3db6f56a07d2abd641cd3f14c90178"
EXPECTED_METHODS = {
    "PatchTST", "AnomalyTransformer", "TimesNet", "TranAD", "USAD",
    "OmniAnomaly", "PaAno", "GBOC", "MEMTO", "DCdetector",
}
EXPECTED_INPUT_DIGEST = {
    "tree_sha256": "0ed191fde5e5e2bb4d8b2a290238e5092b1799ef3146f391ccb5118293c2bd7b",
    "file_count": 9143,
    "total_bytes": 767676814,
    "selected_data_files": 44,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def log(message: str) -> None:
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(f"[{now()}] {message}\n")


def controller_command() -> str:
    return Path(f"/proc/{CONTROLLER_PID}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")


def controller_identity() -> dict[str, object]:
    proc = Path(f"/proc/{CONTROLLER_PID}")
    stat = (proc / "stat").read_text(encoding="utf-8")
    fields = stat.rpartition(")")[2].split()
    return {
        "start_time_ticks": fields[19],
        "uid": proc.stat().st_uid,
        "command": controller_command(),
    }


def process_table() -> list[tuple[int, int, str]]:
    rows: list[tuple[int, int, str]] = []
    output = subprocess.check_output(["ps", "-eo", "pid=,ppid=,args="], text=True)
    for line in output.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) == 3:
            rows.append((int(fields[0]), int(fields[1]), fields[2]))
    return rows


def descendants(rows: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    descendant_ids = {CONTROLLER_PID}
    changed = True
    while changed:
        changed = False
        for pid, ppid, _ in rows:
            if ppid in descendant_ids and pid not in descendant_ids:
                descendant_ids.add(pid)
                changed = True
    return [row for row in rows if row[0] in descendant_ids and row[0] != CONTROLLER_PID]


def controller_stopped() -> bool:
    status = Path(f"/proc/{CONTROLLER_PID}/status").read_text(encoding="utf-8")
    state = next(line for line in status.splitlines() if line.startswith("State:"))
    return "T (stopped)" in state or "t (tracing stop)" in state


def valid_completion_marker() -> bool:
    try:
        payload = json.loads(COMPLETE_MARKER.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        payload.get("status") == "verified_complete"
        and payload.get("protocol") == "duoba-10subset-seed2027-v2-server-b-split1"
        and payload.get("seed") == 2027
        and payload.get("split_plan_sha256") == EXPECTED_PLAN_SHA256
        and payload.get("input_digest") == EXPECTED_INPUT_DIGEST
        and set(payload.get("methods", ())) == EXPECTED_METHODS
        and payload.get("formal_files") == 43
        and payload.get("verified_units") == 430
    )


def atomic_marker(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    if digest(CONTROLLER) != CONTROLLER_SHA256:
        raise RuntimeError("controller hash mismatch")
    command = controller_command()
    if str(CONTROLLER) not in command or "--seed 2027" not in command:
        raise RuntimeError(f"unexpected controller command: {command}")
    identity = controller_identity()
    log("guard started")
    while True:
        if COMPLETE_MARKER.exists():
            if valid_completion_marker():
                log("validated server B completion marker already exists; guard exits without pausing")
                return 0
            log("invalid server B completion marker detected; guard exits with failure")
            return 3
        if not Path(f"/proc/{CONTROLLER_PID}").exists():
            log("controller disappeared; guard exits with failure")
            return 2
        if controller_identity() != identity or digest(CONTROLLER) != CONTROLLER_SHA256:
            log("controller identity or frozen hash changed; guard exits with failure")
            return 4
        rows = process_table()
        matching = [
            {"pid": pid, "ppid": ppid, "command": command}
            for pid, ppid, command in descendants(rows)
            if TARGET_FILE in command and TARGET_RUNNER in command
        ]
        if matching:
            if controller_identity() != identity:
                log("controller identity changed immediately before pause; guard exits with failure")
                return 5
            os.kill(CONTROLLER_PID, signal.SIGSTOP)
            for _ in range(20):
                if controller_stopped():
                    break
                time.sleep(0.1)
            else:
                log("SIGSTOP did not place the controller in stopped state")
                return 6
            payload = {
                "status": "controller_paused",
                "controller_pid": CONTROLLER_PID,
                "controller_identity": identity,
                "paused_at": now(),
                "trigger_file": TARGET_FILE,
                "trigger_processes": matching,
                "resume_policy": "only after SERVER_B_SHARD_VERIFIED_COMPLETE.json is validated",
            }
            atomic_marker(PAUSED_MARKER, payload)
            log(f"controller SIGSTOP sent; trigger={matching}")
            return 0
        time.sleep(0.25)


if __name__ == "__main__":
    raise SystemExit(main())
