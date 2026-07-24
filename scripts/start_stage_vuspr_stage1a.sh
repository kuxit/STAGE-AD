#!/usr/bin/env bash
set -euo pipefail

APP="${STAGE_VUSPR_APP:-/root/autodl-tmp/STAGE-AD-vuspr-tuning}"
LOG_DIR="${STAGE_LOG_DIR:-/root/autodl-tmp/logs}"
PID_FILE="$LOG_DIR/stage_vuspr_stage1a.pid"
LOG_FILE="$LOG_DIR/stage_vuspr_stage1a.log"
LAUNCHER="$APP/scripts/launch_stage_vuspr_stage1a.sh"

test -x "$LAUNCHER"
mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  prior_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$prior_pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$prior_pid" 2>/dev/null; then
    echo "STAGE VUS-PR controller is already alive: PID=$prior_pid" >&2
    exit 1
  fi
fi

nohup setsid bash "$LAUNCHER" >"$LOG_FILE" 2>&1 < /dev/null &
pid="$!"
temporary="$PID_FILE.tmp.$$"
printf '%s\n' "$pid" > "$temporary"
mv "$temporary" "$PID_FILE"

sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  echo "STAGE VUS-PR controller exited during startup; inspect $LOG_FILE" >&2
  exit 1
fi
echo "started STAGE VUS-PR controller PID=$pid log=$LOG_FILE"
