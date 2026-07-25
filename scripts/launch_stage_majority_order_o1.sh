#!/usr/bin/env bash
set -euo pipefail

APP="${STAGE_O1_APP:-/root/autodl-tmp/STAGE-AD-majority-order-aware}"
PYTHON="${STAGE_PYTHON:-/root/autodl-tmp/envs/stage-py310/bin/python}"
METRICS_ROOT="${STAGE_METRICS_ROOT:-/root/autodl-tmp/STAGE-AD/external}"
RESULT_ROOT="${STAGE_O1_RESULT_ROOT:-/root/autodl-tmp/results/STAGE_majority_order_o1/seed2026}"
GPUS="${STAGE_O1_GPUS:-0}"
WORKERS_PER_GPU="${STAGE_O1_WORKERS_PER_GPU:-3}"
PROTOCOL="$APP/configs/stage_majority_order_o1.json"
EXPECTED="$APP/configs/stage_majority_order_o1_expected_identity.json"
RUNNER="$APP/scripts/stage_vuspr_search.py"

test -x "$PYTHON"
test -f "$PROTOCOL"
test -f "$EXPECTED"
test -f "$RUNNER"
test -d "$APP/data/TSB-AD-U"
test -d "$APP/data/TSB-AD-M"
test -d "$APP/data/File_List"
test -d "$METRICS_ROOT"

(
    cd "$APP"
    sha256sum --check checksums/stage-majority-order-o1.sha256
)

"$PYTHON" "$RUNNER" \
    --repo "$APP" \
    --protocol "$PROTOCOL" \
    --result-root "$RESULT_ROOT" \
    --metrics-root "$METRICS_ROOT" \
    plan

"$PYTHON" - "$RESULT_ROOT/stage_vuspr_search_plan.json" "$EXPECTED" <<'PY'
import json
import sys

plan = json.load(open(sys.argv[1], encoding="utf-8"))
expected = json.load(open(sys.argv[2], encoding="utf-8"))
checks = {
    "plan_fingerprint": plan["plan_fingerprint"]
    == expected["plan_fingerprint"],
    "protocol_sha256": plan["protocol_sha256"]
    == expected["protocol_sha256"],
    "protocol_fingerprint": plan["protocol_fingerprint"]
    == expected["protocol_fingerprint"],
    "source_sha256": plan["source_sha256"]
    == expected["source_sha256"],
    "manifest_sha256": plan["official_tuning_manifest_sha256"]
    == expected["official_tuning_manifest_sha256"],
    "series_count": plan["series_count"] == expected["series_count"],
    "physical_units": plan["expected_units"]
    == expected["expected_physical_units"],
    "logical_units": plan["expected_logical_units"]
    == expected["expected_logical_units"],
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"frozen O1 identity mismatch: {failed}")
print("frozen O1 identity audit passed")
PY

exec "$PYTHON" "$RUNNER" \
    --repo "$APP" \
    --protocol "$PROTOCOL" \
    --result-root "$RESULT_ROOT" \
    --metrics-root "$METRICS_ROOT" \
    run-scores \
    --python "$PYTHON" \
    --gpus "$GPUS" \
    --workers-per-gpu "$WORKERS_PER_GPU"
