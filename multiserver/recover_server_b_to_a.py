from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile


CONTROLLER_PID = 3222485
RESULT = Path("/home/taoxie/AAAI/results/DUOBA_10SUBSET_BASELINES/seed2027_v2")
STAGE = Path(
    "/home/taoxie/AAAI/DuoBa-Baseline-Seed2027/multiserver/"
    "server_b_recovery_20260720T1423Z"
)
PAUSED_MARKER = RESULT / "SERVER_A_PAUSED_FOR_SERVER_B.json"
RECOVERY_MARKER = RESULT / "SERVER_B_OFFLINE_RECOVERY_MERGED.json"
PLAN = STAGE / "server_b_split_plan.json"
ARCHIVE = STAGE / "server_b_valid_396_20260720T1441Z.tar.gz"

EXPECTED_PLAN_SHA256 = "543c884bfc761f5fa767392b5bc29c9b7c3db6f56a07d2abd641cd3f14c90178"
EXPECTED_ARCHIVE_SHA256 = "8d66ff5d396c6afd07188449ca14933a025927085c0c354336cc2d54b7f89d8b"
EXPECTED_INVENTORY_SHA256 = "1b99c8518452125783ec37ea6247621922599df4080f148cfb88495f061a0468"
EXPECTED_VALID_UNITS = 396
EXPECTED_TOTAL_UNITS = 430
EXPECTED_METHODS = {
    "PatchTST", "AnomalyTransformer", "TimesNet", "TranAD", "USAD",
    "OmniAnomaly", "PaAno", "GBOC", "MEMTO", "DCdetector",
}
METRICS = {
    "VUS-PR", "VUS-ROC", "R-based-F1", "AUC-PR", "AUC-ROC", "Standard-F1",
}
FROZEN = {
    Path("/home/taoxie/AAAI/DuoBa-Baseline-Seed2027/common.py"):
        "e1c8cb0d732dd18bde179ef552577eeea06151903e8f8821a1afe31e6e67ba59",
    Path("/home/taoxie/AAAI/GROVE-AD-V3/grove_ad.py"):
        "94f31c50641c1c097b4b60cad50994d99dd6148148f2ae35cddc51579d242dd3",
    Path("/home/taoxie/AAAI/DuoBa-Baseline-Seed2027/controller.py"):
        "88a394978d2547608e2ee58e30e9797a58a6bc950ca950c8c802a8b567dd8d78",
    Path("/home/taoxie/AAAI/DuoBa-Baseline-Seed2027/run_one_deep.py"):
        "a8bfb73293005ba5e9da9c3fc321e4f1b9cbae7db908434ac03568abd459840f",
    Path("/home/taoxie/AAAI/DuoBa-Baseline-Seed2027/run_one_classical.py"):
        "d97a1e4083bd1157c73afc3255b35eb871add08361c95c8e52cf8a95f90ca83c",
    Path("/home/taoxie/AAAI/GROVE-AD-V3/paano_official_one.py"):
        "00437318c9a6b6ded357fd88892b336ea6cebc5fe6c7755a08bfa909d67ddf75",
    Path("/home/taoxie/AAAI/spectral_tsad_v5/baselines/gboc/run_one.py"):
        "3861da207667fe4944a45712088a9072abec07fa7ce2167bbcb8751f29082e19",
    Path("/home/taoxie/AAAI/spectral_tsad_v5/baselines/gboc/configs/official_release.json"):
        "1839041de874a6a0586628274c51a14a4b6b66a049c2a7d46d38e2513d511fab",
    Path("/home/taoxie/AAAI/DuoBa-Baseline-Seed2027/run_one_memto.py"):
        "e923c6187b0954e0c01cc3fb09eade7a037b3e66148386bab19327f3d6901371",
    Path("/home/taoxie/AAAI/DuoBa-Baseline-Seed2027/run_one_dcdetector.py"):
        "2fb559dcfb3a7deb5b0a5c745ac5b524969f1cabf95946d2b05d1178dae1bb38",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=path.name + ".tmp.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def controller_stopped() -> bool:
    status = Path(f"/proc/{CONTROLLER_PID}/status").read_text(encoding="utf-8")
    state = next(line for line in status.splitlines() if line.startswith("State:"))
    return "T (stopped)" in state or "t (tracing stop)" in state


def controller_identity() -> dict[str, object]:
    proc = Path(f"/proc/{CONTROLLER_PID}")
    stat = (proc / "stat").read_text(encoding="utf-8")
    fields = stat.rpartition(")")[2].split()
    command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    return {"start_time_ticks": fields[19], "uid": proc.stat().st_uid, "command": command}


def active_descendants() -> list[dict[str, object]]:
    output = subprocess.check_output(
        ["ps", "-eo", "pid=,ppid=,stat=,args="], text=True
    )
    rows: list[tuple[int, int, str, str]] = []
    for line in output.splitlines():
        fields = line.strip().split(None, 3)
        if len(fields) == 4:
            rows.append((int(fields[0]), int(fields[1]), fields[2], fields[3]))
    descendants = {CONTROLLER_PID}
    changed = True
    while changed:
        changed = False
        for pid, ppid, _, _ in rows:
            if ppid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return [
        {"pid": pid, "ppid": ppid, "stat": stat, "command": command}
        for pid, ppid, stat, command in rows
        if pid != CONTROLLER_PID and pid in descendants and not stat.startswith("Z")
    ]


def complete_record(path: Path) -> bool:
    try:
        item = json.loads(path.read_text(encoding="utf-8"))
        metrics = item.get("metrics", {})
        return not item.get("error") and set(metrics) == METRICS and all(
            math.isfinite(float(metrics[name])) and 0.0 <= float(metrics[name]) <= 1.0
            for name in METRICS
        )
    except Exception:
        return False


def expected_members(plan: dict[str, object]) -> set[tuple[str, str, str, str]]:
    files: list[tuple[str, str, str]] = []
    for item in plan["formal_files"]:  # type: ignore[index]
        file_name = Path(item["path"]).name
        files.append(("M", file_name.split("_")[1], file_name))
    return {
        (method, track, dataset, file_name)
        for method in EXPECTED_METHODS
        for track, dataset, file_name in files
    }


def validate_stage() -> tuple[dict[tuple[str, str, str, str], Path], set[tuple[str, str, str, str]], str]:
    if sha256(PLAN) != EXPECTED_PLAN_SHA256:
        raise RuntimeError("split plan hash mismatch")
    if sha256(ARCHIVE) != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("recovery archive hash mismatch")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    expected = expected_members(plan)
    if len(expected) != EXPECTED_TOTAL_UNITS:
        raise RuntimeError(f"expected membership has {len(expected)} units")
    valid: dict[tuple[str, str, str, str], Path] = {}
    for path in sorted((STAGE / "units").rglob("*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        key = (item.get("method"), item.get("track"), item.get("dataset"), item.get("file"))
        if key not in expected:
            raise RuntimeError(f"unexpected staged member: {key}")
        if key in valid:
            raise RuntimeError(f"duplicate staged member: {key}")
        if item.get("seed") != 2027 or item.get("training_prefix_policy") != "filename_declared_label_blind":
            raise RuntimeError(f"invalid seed or prefix policy: {key}")
        if not complete_record(path):
            raise RuntimeError(f"invalid staged record: {path}")
        valid[key] = path
    if len(valid) != EXPECTED_VALID_UNITS:
        raise RuntimeError(f"staged unit count is {len(valid)}, expected {EXPECTED_VALID_UNITS}")
    inventory = hashlib.sha256()
    for key, path in sorted(valid.items()):
        inventory.update("/".join(key).encode("utf-8"))
        inventory.update(b"\0")
        inventory.update(bytes.fromhex(sha256(path)))
    inventory_sha256 = inventory.hexdigest()
    if inventory_sha256 != EXPECTED_INVENTORY_SHA256:
        raise RuntimeError("staged inventory hash mismatch")
    return valid, expected, inventory_sha256


def target_path(key: tuple[str, str, str, str]) -> Path:
    method, track, dataset, file_name = key
    return RESULT / "units" / method / track / dataset / f"{Path(file_name).stem}.json"


def main() -> int:
    if RECOVERY_MARKER.exists():
        print("recovery marker already exists; refusing a second merge", file=sys.stderr)
        return 20
    for path, expected_hash in FROZEN.items():
        if sha256(path) != expected_hash:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    paused = json.loads(PAUSED_MARKER.read_text(encoding="utf-8"))
    if paused.get("status") != "controller_paused" or paused.get("controller_pid") != CONTROLLER_PID:
        raise RuntimeError("invalid planned-pause marker")
    identity = controller_identity()
    if identity != paused.get("controller_identity"):
        raise RuntimeError("controller identity differs from planned-pause marker")
    if "--seed 2027" not in identity["command"] or "controller.py" not in identity["command"]:
        raise RuntimeError("unexpected controller command")
    if not controller_stopped():
        raise RuntimeError("controller is not stopped at the planned boundary")
    active = active_descendants()
    if active:
        print(json.dumps({"status": "not_ready", "active_descendants": active}, indent=2))
        return 10
    valid, expected, inventory_sha256 = validate_stage()
    copied: list[tuple[str, str, str, str]] = []
    retained: list[tuple[str, str, str, str]] = []
    for key, source in sorted(valid.items()):
        target = target_path(key)
        if target.exists():
            if not complete_record(target):
                raise RuntimeError(f"existing authoritative record is incomplete: {target}")
            retained.append(key)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + ".tmp.", delete=False) as stream:
            temporary = Path(stream.name)
            with source.open("rb") as source_stream:
                shutil.copyfileobj(source_stream, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if sha256(temporary) != sha256(source):
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"copy verification failed: {target}")
        os.replace(temporary, target)
        copied.append(key)
    present = {key for key in expected if complete_record(target_path(key))}
    missing = expected - present
    marker = {
        "status": "offline_recovery_merged",
        "merged_at": now(),
        "controller_pid": CONTROLLER_PID,
        "controller_identity": identity,
        "source": "complete set of 396 finished records rescued after server B was temporarily reopened",
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "inventory_sha256": inventory_sha256,
        "staged_valid_units": len(valid),
        "copied_units": len(copied),
        "retained_authoritative_units": len(retained),
        "present_split_units_after_merge": len(present),
        "remaining_units_for_original_controller": len(missing),
        "remaining_by_method": dict(sorted(Counter(key[0] for key in missing).items())),
        "runtime_policy": "all shared-resource runtimes are invalid for the paper efficiency table",
        "resume_policy": "resume the unchanged original controller to skip valid records and run only missing units",
    }
    atomic_json(RECOVERY_MARKER, marker)
    if controller_identity() != identity or not controller_stopped() or active_descendants():
        raise RuntimeError("controller state changed immediately before resume")
    os.kill(CONTROLLER_PID, signal.SIGCONT)
    print(json.dumps(marker, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
