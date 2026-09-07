#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT_DIR/artifacts/gcp_credits_reconciliation/power_timeseries}"
DURATION_SECONDS="${DURATION_SECONDS:-604800}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-60}"

mkdir -p "$OUT_DIR"

ssh "michael@odyn-dgx1" "timeout $DURATION_SECONDS nvidia-smi --query-gpu=timestamp,index,name,power.draw,utilization.gpu --format=csv,noheader,nounits -l $INTERVAL_SECONDS" > "$OUT_DIR/odyn-dgx1_nvidia.csv" || true
ssh "michael@odyn-dgx3" "timeout $DURATION_SECONDS nvidia-smi --query-gpu=timestamp,index,name,power.draw,utilization.gpu --format=csv,noheader,nounits -l $INTERVAL_SECONDS" > "$OUT_DIR/odyn-dgx3_nvidia.csv" || true
ssh "odyn-radeon2" "end=\$((\$(date +%s) + $DURATION_SECONDS)); while [ \$(date +%s) -lt \$end ]; do ts=\$(date -u +%Y-%m-%dT%H:%M:%SZ); payload=\$(rocm-smi --showpower --showuse --json 2>/dev/null || printf '{}'); printf '{\"timestamp\":\"%s\",\"payload\":%s}\n' \"\$ts\" \"\$payload\"; sleep $INTERVAL_SECONDS; done" > "$OUT_DIR/odyn-radeon2_rocm.jsonl" || true

printf 'Saved telemetry captures to %s\n' "$OUT_DIR"
