from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time

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
PILOT_FILE = "124_TAO_id_9_Environment_tr_500_1st_1"


def connect(host: str, port: int, user: str, password: str) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        host,
        port=port,
        username=user,
        password=password,
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
    )
    return client


def run(client: paramiko.SSHClient, command: str, timeout: int = 120) -> str:
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    output = stdout.read().decode("utf-8", "replace")
    error = stderr.read().decode("utf-8", "replace")
    status = stdout.channel.recv_exit_status()
    if status != 0:
        raise RuntimeError(f"remote command failed ({status}): {command}\n{error}")
    return output


def stream_tree(source: paramiko.SSHClient, target: paramiko.SSHClient) -> None:
    source_command = (
        "tar -C /home/taoxie/AAAI -czf - "
        "data external GROVE-AD-V3 DuoBa-Baseline-Seed2027 "
        "spectral_tsad_v5/baselines/gboc spectral_tsad_v5/scripts/efficiency_monitor.py"
    )
    target_command = "mkdir -p /root/autodl-tmp/AAAI && tar -C /root/autodl-tmp/AAAI -xzf -"
    source_channel = source.get_transport().open_session()
    target_channel = target.get_transport().open_session()
    source_channel.exec_command(source_command)
    target_channel.exec_command(target_command)
    transferred = 0
    last_report = time.monotonic()
    while True:
        if source_channel.recv_ready():
            chunk = source_channel.recv(1024 * 1024)
            if chunk:
                target_channel.sendall(chunk)
                transferred += len(chunk)
                if time.monotonic() - last_report >= 15:
                    print(f"compressed transfer: {transferred / (1024**3):.2f} GiB", flush=True)
                    last_report = time.monotonic()
                continue
        if source_channel.exit_status_ready() and not source_channel.recv_ready():
            break
        time.sleep(0.01)
    source_error = bytearray()
    while source_channel.recv_stderr_ready():
        source_error.extend(source_channel.recv_stderr(65536))
    source_status = source_channel.recv_exit_status()
    target_channel.shutdown_write()
    while not target_channel.exit_status_ready():
        time.sleep(0.05)
    target_error = bytearray()
    while target_channel.recv_stderr_ready():
        target_error.extend(target_channel.recv_stderr(65536))
    target_status = target_channel.recv_exit_status()
    if source_status != 0 or target_status != 0:
        raise RuntimeError(
            f"tar bridge failed source={source_status} target={target_status} "
            f"source_error={source_error.decode('utf-8', 'replace')} "
            f"target_error={target_error.decode('utf-8', 'replace')}"
        )
    print(f"compressed transfer complete: {transferred / (1024**3):.2f} GiB", flush=True)


def upload_multiserver(target: paramiko.SSHClient, local_dir: Path) -> None:
    remote_dir = "/root/autodl-tmp/AAAI/DuoBa-Baseline-Seed2027/multiserver"
    run(target, f"mkdir -p {shlex.quote(remote_dir)}")
    sftp = target.open_sftp()
    try:
        for path in sorted(local_dir.iterdir()):
            if path.is_file():
                sftp.put(str(path), f"{remote_dir}/{path.name}")
    finally:
        sftp.close()
    run(target, f"chmod 700 {remote_dir}/setup_server_b.sh")


def copy_references(source: paramiko.SSHClient, target: paramiko.SSHClient) -> None:
    destination = "/root/autodl-tmp/AAAI/parity_pilot/reference"
    run(target, f"mkdir -p {destination}")
    source_sftp = source.open_sftp()
    target_sftp = target.open_sftp()
    try:
        for method in METHODS:
            remote = (
                "/home/taoxie/AAAI/results/DUOBA_10SUBSET_BASELINES/seed2027_v2/"
                f"units/{method}/M/TAO/{PILOT_FILE}.json"
            )
            with source_sftp.open(remote, "rb") as stream:
                payload = stream.read()
            parsed = json.loads(payload.decode("utf-8"))
            if parsed.get("method") != method or parsed.get("seed") != 2027 or parsed.get("error") is not None:
                raise RuntimeError(f"invalid parity reference: {method}")
            with target_sftp.open(f"{destination}/{method}.json", "wb") as stream:
                stream.write(payload)
    finally:
        source_sftp.close()
        target_sftp.close()


REMOTE_DIGEST_SCRIPT = r'''
import hashlib, json, sys
from pathlib import Path
root=Path(sys.argv[1])
sys.path.insert(0,str(root/'DuoBa-Baseline-Seed2027'))
from common import dataset_name
from controller import selected_files
pilot='124_TAO_id_9_Environment_tr_500_1st_1.csv'
names=[n for n in selected_files(root)['M'] if dataset_name(n) in {'SVDB','TAO','CATSv2'}]
if pilot not in names: names.append(pilot)
data_paths=[root/'data'/'TSB-AD-M'/n for n in sorted(set(names))]
trees=[
 root/'external', root/'GROVE-AD-V3',
 root/'spectral_tsad_v5'/'baselines'/'gboc',
]
paths=list(data_paths)
paths.append(root/'spectral_tsad_v5'/'scripts'/'efficiency_monitor.py')
for tree in trees:
    paths.extend(sorted(
        p for p in tree.rglob('*')
        if p.is_file()
        and '.git' not in p.parts
        and '__pycache__' not in p.parts
        and p.suffix not in {'.pyc', '.pyo'}
    ))
h=hashlib.sha256(); count=0; total=0
for path in sorted(set(paths), key=lambda p:str(p.relative_to(root))):
    rel=str(path.relative_to(root)).replace('\\','/')
    d=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): d.update(chunk)
    size=path.stat().st_size
    h.update(rel.encode()+b'\0'+str(size).encode()+b'\0'+d.hexdigest().encode()+b'\n')
    count+=1; total+=size
print(json.dumps({'tree_sha256':h.hexdigest(),'file_count':count,'total_bytes':total,'selected_data_files':len(set(data_paths))}))
'''


def remote_digest(client: paramiko.SSHClient, root: str) -> dict[str, object]:
    import base64

    encoded = base64.b64encode(REMOTE_DIGEST_SCRIPT.encode()).decode()
    python = (
        "/home/taoxie/AAAI/.venv/bin/python"
        if root == "/home/taoxie/AAAI"
        else "/root/autodl-tmp/duoba-env/bin/python"
    )
    command = (
        f"{python} -c {shlex.quote('import base64;exec(base64.b64decode(' + repr(encoded) + '))')} "
        f"{shlex.quote(root)}"
    )
    return json.loads(run(client, command, timeout=900))


def write_verified_manifest(target: paramiko.SSHClient, digest: dict[str, object]) -> None:
    payload = {
        "status": "verified",
        "seed": 2027,
        "source_server": "10.21.23.183",
        "target_server": "connect.westc.seetacloud.com:21249",
        "scope": "selected SVDB/TAO/CATSv2 data plus frozen external/code trees",
        "input_digest": digest,
    }
    encoded = (json.dumps(payload, indent=2) + "\n").encode()
    sftp = target.open_sftp()
    try:
        with sftp.open("/root/autodl-tmp/AAAI/server_b_input_manifest.json", "wb") as stream:
            stream.write(encoded)
    finally:
        sftp.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-multiserver", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    password_a = os.environ["DUOBA_SERVER_A_PASSWORD"]
    password_b = os.environ["DUOBA_SERVER_B_PASSWORD"]
    source = connect("10.21.23.183", 22, "taoxie", password_a)
    target = connect("connect.westc.seetacloud.com", 21249, "root", password_b)
    try:
        if not args.verify_only:
            stream_tree(source, target)
            upload_multiserver(target, args.local_multiserver.resolve())
            copy_references(source, target)
        else:
            upload_multiserver(target, args.local_multiserver.resolve())
        digest_a = remote_digest(source, "/home/taoxie/AAAI")
        digest_b = remote_digest(target, "/root/autodl-tmp/AAAI")
        if digest_a != digest_b:
            raise RuntimeError(f"input digest mismatch: A={digest_a} B={digest_b}")
        write_verified_manifest(target, digest_b)
        print(json.dumps({"status": "verified", "digest": digest_b}, indent=2), flush=True)
    finally:
        source.close()
        target.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
