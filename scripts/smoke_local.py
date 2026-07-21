from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import sys
from types import SimpleNamespace


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import controller  # noqa: E402
from common import METRICS, atomic_json, complete_record  # noqa: E402


ALL_METHODS = controller.METHOD_ORDER


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def choose_file(project: Path, track: str, explicit: str | None) -> str:
    candidates = controller.selected_files(project)[track]
    if explicit:
        if explicit not in candidates:
            raise ValueError(f"{explicit} is not in the locked {track} Eval selection")
        return explicit
    compatible = []
    for name in candidates:
        parts = Path(name).stem.split("_")
        boundary = int(parts[parts.index("tr") + 1])
        if boundary >= 500:
            path = project / "data" / f"TSB-AD-{track}" / name
            compatible.append((path.stat().st_size, name))
    if not compatible:
        raise RuntimeError(f"no smoke-compatible {track} file")
    return min(compatible)[1]


def validate_result(path: Path) -> dict[str, float]:
    if not complete_record(path):
        raise ValueError(f"invalid result: {path}")
    item = json.loads(path.read_text(encoding="utf-8"))
    metrics = {name: float(item["metrics"][name]) for name in METRICS}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError(f"non-finite metric: {path}")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Sequential one-GPU portability smoke test")
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--result", type=Path)
    parser.add_argument("--tracks", default="U,M", help="comma-separated subset of U,M")
    parser.add_argument("--methods", default=",".join(ALL_METHODS))
    parser.add_argument("--file-u")
    parser.add_argument("--file-m")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    project = args.project.resolve()
    tracks = tuple(item.strip() for item in args.tracks.split(",") if item.strip())
    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    unknown_tracks = set(tracks) - {"U", "M"}
    unknown_methods = set(methods) - set(ALL_METHODS)
    if unknown_tracks or unknown_methods:
        raise ValueError(f"unknown tracks={sorted(unknown_tracks)} methods={sorted(unknown_methods)}")
    if not tracks or not methods:
        raise ValueError("at least one track and one method are required")

    result = args.result
    if result is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        result = project / "results" / "smoke" / f"local_00gwk_{stamp}"
    result = result.resolve()
    result.mkdir(parents=True, exist_ok=True)

    files = {
        track: choose_file(project, track, args.file_u if track == "U" else args.file_m)
        for track in tracks
    }
    run_args = SimpleNamespace(
        repo=project,
        experiment=project,
        result=result,
        python=Path(os.path.abspath(args.python)),
        stage_source=project / "STAGE" / "stage.py",
        paano_root=project / "external" / "PaAno",
        paano_one=(
            project / "compat" / "paano_official_one_windows.py"
            if os.name == "nt"
            else project / "compat" / "paano_official_one_linux.py"
        ),
        gboc_root=project / "external" / "GBOC_official_55a2a31",
        gboc_one=project / "spectral_tsad_v5" / "baselines" / "gboc" / "run_one.py",
        gboc_config=project / "spectral_tsad_v5" / "baselines" / "gboc" / "configs" / "official_release.json",
        memto_root=project / "external" / "MEMTO_official",
        memto_one=project / "run_one_memto.py",
        dcdetector_root=project / "external" / "DCdetector",
        dcdetector_one=project / "run_one_dcdetector.py",
        seed=args.seed,
    )

    required_paths = [
        run_args.python,
        run_args.stage_source,
        run_args.paano_root / "model.py",
        run_args.gboc_root / "models" / "GBOC.py",
        run_args.memto_root / "model" / "Transformer.py",
        run_args.dcdetector_root / "model" / "DCdetector.py",
        project / "external" / "PAI" / "third_party" / "TSB-AD" / "TSB_AD" / "HP_list.py",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("run scripts/prepare_local_layout.ps1 first; missing: " + "; ".join(missing))

    manifest_path = result / "smoke_manifest.json"
    jobs = [(method, track, files[track]) for track in tracks for method in methods]
    manifest = {
        "schema_version": "stage-local-smoke-v1",
        "purpose": "portability only; never use runtime in the paper efficiency table",
        "runtime_eligible_for_paper": False,
        "seed": args.seed,
        "started_at": utc_now(),
        "python": platform.python_version(),
        "python_executable": str(run_args.python),
        "platform": platform.platform(),
        "gpu": args.gpu,
        "files": files,
        "methods": methods,
        "expected_units": len(jobs),
        "completed_units": 0,
        "error_units": 0,
        "units": [],
        "status": "running",
    }
    atomic_json(manifest_path, manifest)

    for index, (method, track, file_name) in enumerate(jobs, start=1):
        gpu = None if method in controller.NON_DEEP else args.gpu
        outcome = controller.run_unit(run_args, method, track, file_name, gpu)
        target = controller.standard_path(result, method, track, file_name)
        if outcome["status"] in {"complete", "skipped"}:
            outcome["metrics"] = validate_result(target)
            manifest["completed_units"] += 1
        else:
            manifest["error_units"] += 1
        manifest["units"].append(outcome)
        manifest["updated_at"] = utc_now()
        atomic_json(manifest_path, manifest)
        print(f"[{index}/{len(jobs)}] {outcome['status']} {method} {track} {file_name}", flush=True)

    manifest["status"] = "complete" if manifest["error_units"] == 0 else "incomplete"
    manifest["completed_at"] = utc_now()
    atomic_json(manifest_path, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "completed_units": manifest["completed_units"],
        "error_units": manifest["error_units"],
        "result": str(result),
    }, indent=2))
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
