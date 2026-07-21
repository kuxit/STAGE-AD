from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import signal
import shlex
import sys
import tempfile
import time
from typing import Any

import paramiko


METHODS = (
    "PatchTST",
    "AnomalyTransformer",
    "TimesNet",
    "TranAD",
    "USAD",
    "OmniAnomaly",
    "PaAno",
    "GBOC",
    "MEMTO",
    "DCdetector",
    "DuoBa",
)
METRICS = ("VUS-PR", "VUS-ROC", "R-based-F1", "AUC-PR", "AUC-ROC", "Standard-F1")
EXCLUDED = "124_TAO_id_9_Environment_tr_500_1st_1.csv"
EXPECTED_FILES = 43
EXPECTED_PLAN_SHA256 = "543c884bfc761f5fa767392b5bc29c9b7c3db6f56a07d2abd641cd3f14c90178"
CONTROLLER_SHA256 = "88a394978d2547608e2ee58e30e9797a58a6bc950ca950c8c802a8b567dd8d78"
EXPECTED_SOURCE_SHA256 = {
    "common": "e1c8cb0d732dd18bde179ef552577eeea06151903e8f8821a1afe31e6e67ba59",
    "duoba": "94f31c50641c1c097b4b60cad50994d99dd6148148f2ae35cddc51579d242dd3",
    "controller": CONTROLLER_SHA256,
    "deep_runner": "a8bfb73293005ba5e9da9c3fc321e4f1b9cbae7db908434ac03568abd459840f",
    "classical_runner": "d97a1e4083bd1157c73afc3255b35eb871add08361c95c8e52cf8a95f90ca83c",
    "paano_one": "00437318c9a6b6ded357fd88892b336ea6cebc5fe6c7755a08bfa909d67ddf75",
    "gboc_one": "3861da207667fe4944a45712088a9072abec07fa7ce2167bbcb8751f29082e19",
    "gboc_config": "1839041de874a6a0586628274c51a14a4b6b66a049c2a7d46d38e2513d511fab",
    "memto_one": "e923c6187b0954e0c01cc3fb09eade7a037b3e66148386bab19327f3d6901371",
    "dcdetector_one": "2fb559dcfb3a7deb5b0a5c745ac5b524969f1cabf95946d2b05d1178dae1bb38",
}
A_ROOT = PurePosixPath("/home/taoxie/AAAI")
B_ROOT = PurePosixPath("/root/autodl-tmp/AAAI")
A_RESULT = A_ROOT / "results/DUOBA_10SUBSET_BASELINES/seed2027_v2"
B_RESULT = B_ROOT / "results/DUOBA_10SUBSET_BASELINES/seed2027_v2_server_b"


def connect(host: str, port: int, user: str, **kwargs: Any) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username=user, timeout=20, banner_timeout=20, auth_timeout=20, **kwargs)
    return client


def read_json(sftp: paramiko.SFTPClient, path: PurePosixPath) -> dict[str, Any]:
    with sftp.open(str(path), "rb") as stream:
        return json.loads(stream.read().decode("utf-8"))


def read_bytes(sftp: paramiko.SFTPClient, path: PurePosixPath) -> bytes:
    with sftp.open(str(path), "rb") as stream:
        return stream.read()


def run_json(client: paramiko.SSHClient, command: str) -> dict[str, Any]:
    _, stdout, stderr = client.exec_command(command, timeout=900)
    payload = stdout.read()
    error = stderr.read().decode("utf-8", "replace")
    status = stdout.channel.recv_exit_status()
    if status != 0:
        raise RuntimeError(f"remote validation failed ({status}): {error}")
    return json.loads(payload.decode("utf-8"))


def canonical_plan_sha256(plan: dict[str, Any]) -> str:
    unsigned = dict(plan)
    unsigned.pop("plan_sha256", None)
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def valid_record(payload: dict[str, Any], method: str, file_name: str) -> bool:
    if payload.get("method") != method or payload.get("track") != "M":
        return False
    if payload.get("dataset") != dataset(file_name):
        return False
    if payload.get("file") != file_name or payload.get("seed") != 2027 or payload.get("error") is not None:
        return False
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return False
    for name in METRICS:
        try:
            value = float(metrics[name])
        except (KeyError, TypeError, ValueError):
            return False
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            return False
    return True


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def dataset(file_name: str) -> str:
    return file_name.split("_", 2)[1]


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


def remote_environment(client: paramiko.SSHClient, python: str) -> dict[str, Any]:
    if not python.startswith("/root/autodl-tmp/duoba-env") or not python.endswith("/bin/python"):
        raise RuntimeError(f"unexpected server B interpreter path: {python}")
    encoded = base64.b64encode(ENVIRONMENT_SCRIPT.encode()).decode()
    command = f"{shlex.quote(python)} -c {shlex.quote('import base64;exec(base64.b64decode(' + repr(encoded) + '))')}"
    return run_json(client, command)


def environment_matches(reported: dict[str, Any], current: dict[str, Any]) -> bool:
    for key, value in reported.items():
        if key == "python_executable":
            if PurePosixPath(str(value)) != PurePosixPath(str(current.get(key, ""))):
                # The older pilot recorded the bin/python symlink while the
                # formal fingerprint records its resolved python3.10 target.
                if PurePosixPath(str(value)).parent != PurePosixPath(str(current.get(key, ""))).parent:
                    return False
        elif current.get(key) != value:
            return False
    return True


def validate_a_pause(
    a: paramiko.SSHClient,
    a_sftp: paramiko.SFTPClient,
    formal_files: tuple[str, ...],
) -> dict[str, Any]:
    marker_path = A_RESULT / "SERVER_A_PAUSED_FOR_SERVER_B.json"
    marker = read_json(a_sftp, marker_path)
    if marker.get("status") != "controller_paused" or marker.get("controller_pid") != 3222485:
        raise RuntimeError("server A does not have the expected pause marker")
    proc = PurePosixPath("/proc/3222485")
    stat = read_bytes(a_sftp, proc / "stat").decode("utf-8")
    fields = stat.rpartition(")")[2].split()
    command = read_bytes(a_sftp, proc / "cmdline").replace(b"\0", b" ").decode("utf-8", "replace")
    identity = {
        "start_time_ticks": fields[19],
        "uid": a_sftp.stat(str(proc)).st_uid,
        "command": command,
    }
    if marker.get("controller_identity") != identity:
        raise RuntimeError("server A controller identity differs from the pause marker")
    status = read_bytes(a_sftp, proc / "status").decode("utf-8")
    state = next(line for line in status.splitlines() if line.startswith("State:"))
    if "T (stopped)" not in state and "t (tracing stop)" not in state:
        raise RuntimeError(f"server A controller is not stopped: {state}")
    controller = A_ROOT / "DuoBa-Baseline-Seed2027/controller.py"
    if sha256(read_bytes(a_sftp, controller)) != CONTROLLER_SHA256:
        raise RuntimeError("server A controller hash changed")
    if str(controller) not in command or "--seed 2027" not in command:
        raise RuntimeError("server A controller command changed")
    _, stdout, stderr = a.exec_command("ps -eo args=", timeout=30)
    process_text = stdout.read().decode("utf-8", "replace")
    if stdout.channel.recv_exit_status() != 0:
        raise RuntimeError(stderr.read().decode("utf-8", "replace"))
    active = [name for name in formal_files if name in process_text]
    if active:
        raise RuntimeError(f"server A still has active processes in the server B shard: {active[:3]}")
    return identity


def expected_jobs(
    b_sftp: paramiko.SFTPClient,
    methods: tuple[str, ...],
    plan: dict[str, Any],
) -> list[tuple[str, str]]:
    units = B_RESULT / "units"
    file_names = tuple(Path(record["path"]).name for record in plan["formal_files"])
    expected = {(method, file_name) for method in methods for file_name in file_names}
    actual: set[tuple[str, str]] = set()
    for method in methods:
        for subset in ("SVDB", "TAO", "CATSv2"):
            directory = units / method / "M" / subset
            for entry in b_sftp.listdir_attr(str(directory)):
                if not entry.filename.endswith(".json"):
                    continue
                file_name = entry.filename[:-5] + ".csv"
                if file_name == EXCLUDED:
                    raise RuntimeError(f"excluded pilot file found in formal shard: {method}")
                actual.add((method, file_name))
    if actual != expected:
        missing = sorted(expected - actual)[:3]
        extra = sorted(actual - expected)[:3]
        raise RuntimeError(f"unexpected B unit inventory: missing={missing} extra={extra}")
    return sorted(expected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-a", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    a_key = Path.home() / ".ssh" / "codex_duoba_server_a"
    b_key = Path.home() / ".ssh" / "codex_duoba_server_b"
    a = connect("10.21.23.183", 22, "taoxie", key_filename=str(a_key))
    b = connect("connect.westc.seetacloud.com", 21249, "root", key_filename=str(b_key))
    a_sftp = a.open_sftp()
    b_sftp = b.open_sftp()
    report: dict[str, Any] = {"status": "running", "copied": 0, "identical": 0, "conflicts": [], "invalid": []}
    try:
        manifest_bytes = read_bytes(b_sftp, B_RESULT / "shard_manifest.json")
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        methods = tuple(manifest.get("methods", ()))
        if not methods or len(set(methods)) != len(methods) or not set(methods).issubset(METHODS):
            raise RuntimeError(f"invalid B method shard: {methods}")
        expected_units = EXPECTED_FILES * len(methods)
        if manifest.get("expected_units") != expected_units:
            raise RuntimeError("B manifest expected-unit count is inconsistent with its method shard")
        if manifest.get("status") != "complete" or manifest.get("completed_units") != expected_units:
            raise RuntimeError(f"B shard is not complete: {manifest}")
        if int(manifest.get("error_units", -1)) != 0:
            raise RuntimeError("B shard contains errors")
        if manifest.get("source_sha256") != EXPECTED_SOURCE_SHA256:
            raise RuntimeError("B shard source hashes differ from the frozen run")
        if manifest.get("datasets") != ["SVDB", "TAO", "CATSv2"]:
            raise RuntimeError("B manifest contains an unexpected dataset shard")
        if manifest.get("excluded_files") != [EXCLUDED]:
            raise RuntimeError("B manifest contains an unexpected exclusion boundary")
        plan_bytes = read_bytes(b_sftp, B_ROOT / "server_b_split_plan.json")
        plan = json.loads(plan_bytes.decode("utf-8"))
        plan_sha = canonical_plan_sha256(plan)
        if plan.get("plan_sha256") != plan_sha or plan_sha != EXPECTED_PLAN_SHA256:
            raise RuntimeError("server B split plan hash mismatch")
        formal_files = tuple(Path(record["path"]).name for record in plan.get("formal_files", ()))
        if len(formal_files) != EXPECTED_FILES or len(set(formal_files)) != EXPECTED_FILES:
            raise RuntimeError("server B split plan has an invalid file inventory")
        if manifest.get("formal_files") != list(formal_files):
            raise RuntimeError("server B manifest and split plan have different file inventories")
        if manifest.get("split_plan_sha256") != plan_sha:
            raise RuntimeError("server B manifest uses an unexpected split plan")
        integrity_command = (
            "/root/autodl-tmp/duoba-env/bin/python "
            "/root/autodl-tmp/AAAI/DuoBa-Baseline-Seed2027/multiserver/input_integrity.py "
            "--root /root/autodl-tmp/AAAI --verify-plan /root/autodl-tmp/AAAI/server_b_split_plan.json"
        )
        integrity = run_json(b, integrity_command)
        if integrity.get("plan_sha256") != plan_sha or integrity.get("input_digest") != plan.get("input_digest"):
            raise RuntimeError("server B input tree changed after formal execution")
        if manifest.get("input_digest") != integrity.get("input_digest"):
            raise RuntimeError("server B manifest input digest mismatch")
        execution = manifest.get("method_execution")
        if not isinstance(execution, dict) or set(execution) != set(methods):
            raise RuntimeError("B manifest does not define every formal method environment")
        for method in methods:
            spec = execution.get(method)
            if not isinstance(spec, dict):
                raise RuntimeError(f"invalid B execution entry for {method}")
            parity_name = spec.get("parity_report")
            if not isinstance(parity_name, str) or "/" in parity_name or "\\" in parity_name:
                raise RuntimeError(f"invalid parity report name for {method}")
            parity_bytes = read_bytes(b_sftp, B_ROOT / "parity_pilot" / parity_name)
            if spec.get("parity_report_sha256") != sha256(parity_bytes):
                raise RuntimeError(f"parity report changed after formal execution for {method}")
            parity = json.loads(parity_bytes.decode("utf-8"))
            if parity.get("seed") != 2027:
                raise RuntimeError(f"B parity report uses an unexpected seed for {method}")
            if manifest.get("source_sha256") != parity.get("source_sha256"):
                raise RuntimeError(f"B formal shard and parity report use different sources for {method}")
            policy = parity.get("policy_set_before_run")
            if policy != {"max_metric_absolute_difference": 0.005, "mean_metric_absolute_difference": 0.002}:
                raise RuntimeError(f"B parity policy changed for {method}")
            if parity.get("file") != EXCLUDED:
                raise RuntimeError(f"B parity file changed for {method}")
            current_environment = remote_environment(b, spec.get("python", ""))
            if spec.get("environment") != current_environment:
                raise RuntimeError(f"B formal environment changed after execution for {method}")
            reported_environment = parity.get("candidate_environment")
            if not isinstance(reported_environment, dict) or not environment_matches(reported_environment, current_environment):
                raise RuntimeError(f"B formal and parity environments differ for {method}")
            if not parity.get("methods", {}).get(method, {}).get("passed"):
                raise RuntimeError(f"B parity gate did not pass for {method}")
            candidate_name = parity.get("candidate_name")
            if not isinstance(candidate_name, str) or "/" in candidate_name or "\\" in candidate_name:
                raise RuntimeError(f"B parity candidate is not identified for {method}")
            gate = parity["methods"][method]
            stem = Path(EXCLUDED).stem + ".json"
            candidate_path = B_ROOT / "parity_pilot" / candidate_name / "units" / method / "M" / "TAO" / stem
            reference_path = B_ROOT / "parity_pilot" / "reference" / f"{method}.json"
            candidate_bytes = read_bytes(b_sftp, candidate_path)
            reference_bytes = read_bytes(b_sftp, reference_path)
            if gate.get("candidate_record_sha256") != sha256(candidate_bytes):
                raise RuntimeError(f"B parity candidate changed for {method}")
            if gate.get("reference_record_sha256") != sha256(reference_bytes):
                raise RuntimeError(f"B parity reference changed for {method}")
            if not valid_record(json.loads(candidate_bytes), method, EXCLUDED):
                raise RuntimeError(f"B parity candidate is invalid for {method}")
            if not valid_record(json.loads(reference_bytes), method, EXCLUDED):
                raise RuntimeError(f"B parity reference is invalid for {method}")
        inputs = read_json(b_sftp, B_ROOT / "server_b_input_manifest.json")
        if inputs.get("status") != "verified" or inputs.get("seed") != 2027:
            raise RuntimeError("B input manifest is not verified")
        if inputs.get("input_digest") != integrity.get("input_digest"):
            raise RuntimeError("B verified input manifest no longer matches the input tree")

        jobs = expected_jobs(b_sftp, methods, plan)
        payloads: dict[tuple[str, str], bytes] = {}
        for method, file_name in jobs:
            subset = dataset(file_name)
            b_path = B_RESULT / "units" / method / "M" / subset / (Path(file_name).stem + ".json")
            payload = read_bytes(b_sftp, b_path)
            parsed = json.loads(payload.decode("utf-8"))
            if not valid_record(parsed, method, file_name):
                report["invalid"].append(str(b_path))
            else:
                payloads[(method, file_name)] = payload
        if report["invalid"] or len(payloads) != expected_units:
            raise RuntimeError("server B contains invalid formal records")

        controller_identity = validate_a_pause(a, a_sftp, formal_files)
        writes: list[tuple[PurePosixPath, bytes]] = []
        for method, file_name in jobs:
            subset = dataset(file_name)
            payload = payloads[(method, file_name)]
            a_path = A_RESULT / "units" / method / "M" / subset / (Path(file_name).stem + ".json")
            try:
                existing = read_bytes(a_sftp, a_path)
            except FileNotFoundError:
                existing = None
            if existing is not None:
                try:
                    existing_payload = json.loads(existing.decode("utf-8"))
                except Exception:
                    existing_payload = {}
                if valid_record(existing_payload, method, file_name):
                    if sha256(existing) == sha256(payload):
                        report["identical"] += 1
                    else:
                        report["conflicts"].append({"method": method, "file": file_name})
                    continue
                report["conflicts"].append({"method": method, "file": file_name, "reason": "invalid existing record"})
                continue
            writes.append((a_path, payload))

        if report["invalid"] or report["conflicts"]:
            raise RuntimeError("merge preflight found invalid records or conflicts")

        staged: list[tuple[PurePosixPath, PurePosixPath]] = []
        try:
            for a_path, payload in writes:
                parent = a_path.parent
                try:
                    a_sftp.stat(str(parent))
                except FileNotFoundError:
                    _, stdout, stderr = a.exec_command(f"mkdir -p {shlex.quote(str(parent))}")
                    if stdout.channel.recv_exit_status() != 0:
                        raise RuntimeError(stderr.read().decode("utf-8", "replace"))
                temporary = PurePosixPath(str(a_path) + f".incoming.server-b.{sha256(payload)}")
                with a_sftp.open(str(temporary), "wb") as stream:
                    stream.write(payload)
                staged.append((temporary, a_path))
            # All payloads and destinations have passed validation before the
            # first authoritative rename. A is stopped throughout this phase.
            for temporary, a_path in staged:
                a_sftp.posix_rename(str(temporary), str(a_path))
                report["copied"] += 1
        except Exception:
            for temporary, _ in staged:
                try:
                    a_sftp.remove(str(temporary))
                except OSError:
                    pass
            raise

        # Re-read all authoritative targets after the atomic writes.
        verified = 0
        for method, file_name in jobs:
            subset = dataset(file_name)
            a_path = A_RESULT / "units" / method / "M" / subset / (Path(file_name).stem + ".json")
            payload = read_json(a_sftp, a_path)
            if not valid_record(payload, method, file_name):
                raise RuntimeError(f"A target failed post-merge validation: {a_path}")
            verified += 1
        if verified != expected_units:
            raise RuntimeError(f"post-merge count mismatch: {verified}")

        report.update(
            {
                "status": "verified_complete",
                "protocol": "duoba-10subset-seed2027-v2-server-b-split1",
                "seed": 2027,
                "verified_units": verified,
                "methods": list(methods),
                "formal_files": len(formal_files),
                "split_plan_sha256": plan_sha,
                "controller_identity": controller_identity,
                "b_manifest_sha256": sha256(manifest_bytes),
                "input_digest": inputs["input_digest"],
            }
        )
        encoded = (json.dumps(report, indent=2) + "\n").encode()
        marker = A_RESULT / "SERVER_B_SHARD_VERIFIED_COMPLETE.json"
        temporary = PurePosixPath(str(marker) + f".tmp.{os.getpid()}")
        with a_sftp.open(str(temporary), "wb") as stream:
            stream.write(encoded)
        a_sftp.posix_rename(str(temporary), str(marker))

        if args.resume_a:
            validate_a_pause(a, a_sftp, formal_files)
            _, stdout, stderr = a.exec_command("kill -CONT 3222485")
            if stdout.channel.recv_exit_status() != 0:
                raise RuntimeError(stderr.read().decode("utf-8", "replace"))
            for _ in range(30):
                time.sleep(0.1)
                status = read_bytes(a_sftp, PurePosixPath("/proc/3222485/status")).decode("utf-8")
                state = next(line for line in status.splitlines() if line.startswith("State:"))
                if "T (stopped)" not in state and "t (tracing stop)" not in state:
                    break
            else:
                raise RuntimeError("server A controller did not resume after SIGCONT")
            report["controller_a_resumed"] = True
            encoded = (json.dumps(report, indent=2) + "\n").encode()
            temporary = PurePosixPath(str(marker) + f".tmp.resume.{os.getpid()}")
            with a_sftp.open(str(temporary), "wb") as stream:
                stream.write(encoded)
            a_sftp.posix_rename(str(temporary), str(marker))
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    except Exception as error:
        report.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        a_sftp.close()
        b_sftp.close()
        a.close()
        b.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
