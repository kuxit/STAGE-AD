from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import controller


def main() -> int:
    intervals: list[tuple[str, str | None, float, float]] = []
    lock = threading.Lock()

    def fake_run_unit(args, method, track, file_name, gpu):
        started = time.perf_counter()
        time.sleep(0.012 if method == "PaAno" else 0.02)
        finished = time.perf_counter()
        with lock:
            intervals.append((method, gpu, started, finished))
        return {
            "status": "complete",
            "method": method,
            "track": track,
            "file": file_name,
            "gpu": gpu,
        }

    selected = {
        "U": [f"{index:03d}_UCR_id_{index}_Sensor_tr_64_1st_80.csv" for index in range(12)],
        "M": [f"{index:03d}_TAO_id_{index}_Sensor_tr_64_1st_80.csv" for index in range(12)],
    }

    with tempfile.TemporaryDirectory(prefix="stage-scheduler-test-") as temporary:
        result = Path(temporary) / "result"
        argv = [
            "controller.py",
            "--repo", str(PROJECT),
            "--experiment", str(PROJECT),
            "--result", str(result),
            "--python", sys.executable,
            "--memto-python", sys.executable,
            "--stage-source", str(PROJECT / "STAGE" / "stage.py"),
            "--paano-root", str(PROJECT),
            "--paano-one", str(PROJECT / "compat" / "paano_official_one_linux.py"),
            "--gboc-root", str(PROJECT),
            "--gboc-one", str(PROJECT / "spectral_tsad_v5" / "baselines" / "gboc" / "run_one.py"),
            "--gboc-config", str(PROJECT / "spectral_tsad_v5" / "baselines" / "gboc" / "configs" / "official_release.json"),
            "--memto-root", str(PROJECT),
            "--memto-one", str(PROJECT / "run_one_memto.py"),
            "--dcdetector-root", str(PROJECT),
            "--dcdetector-one", str(PROJECT / "run_one_dcdetector.py"),
            "--phase", "baseline",
            "--scheduler", "throughput",
            "--gpus", "0,1",
            "--cpu-workers", "8",
            "--gboc-workers-per-gpu", "4",
        ]
        with patch.object(controller, "selected_files", return_value=selected), \
             patch.object(controller, "run_unit", side_effect=fake_run_unit), \
             patch.object(controller, "sha256", return_value="0" * 64), \
             patch.object(sys, "argv", argv):
            code = controller.main()

        if code != 0:
            raise AssertionError(f"controller returned {code}")
        manifest = json.loads((result / "run_manifest_baseline.json").read_text(encoding="utf-8"))
        if manifest["status"] != "complete" or manifest["error_units"] != 0:
            raise AssertionError("fake throughput run did not complete cleanly")

    paano = [item for item in intervals if item[0] == "PaAno"]
    other_gpu = [item for item in intervals if item[1] is not None and item[0] != "PaAno"]
    cpu = [item for item in intervals if item[1] is None]
    if not paano or not other_gpu or not cpu:
        raise AssertionError("scheduler did not exercise every resource class")
    paano_finished = max(item[3] for item in paano)
    if min(item[2] for item in other_gpu) < paano_finished:
        raise AssertionError("a GPU baseline overlapped the exclusive PaAno phase")
    if min(item[2] for item in cpu) < paano_finished:
        raise AssertionError("CPU jobs overlapped the exclusive PaAno phase")

    gboc = [item for item in intervals if item[0] == "GBOC"]
    regular = [
        item
        for item in other_gpu
        if item[0] not in {"GBOC", "STAGE"}
    ]
    overlap = any(max(g[2], r[2]) < min(g[3], r[3]) for g in gboc for r in regular)
    if not overlap:
        raise AssertionError("GBOC did not overlap a regular deep-baseline lane")

    events = sorted(
        [(start, 1) for _, _, start, _ in gboc]
        + [(finish, -1) for _, _, _, finish in gboc],
        key=lambda item: (item[0], -item[1]),
    )
    active = 0
    peak = 0
    for _, delta in events:
        active += delta
        peak = max(peak, active)
    if peak < 4:
        raise AssertionError(f"expected concurrent GBOC lanes, observed peak={peak}")

    print(
        json.dumps(
            {
                "status": "PASS",
                "exclusive_paano_units": len(paano),
                "gboc_peak_concurrency": peak,
                "gboc_regular_overlap": overlap,
                "total_fake_units": len(intervals),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
