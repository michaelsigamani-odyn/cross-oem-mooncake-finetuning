#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT_DIR/artifacts/gcp_credits_reconciliation/raw}"

if [[ -z "${LATITUDE_API_KEY:-}" ]]; then
  printf 'LATITUDE_API_KEY is required\n' >&2
  exit 1
fi

if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
  printf 'RUNPOD_API_KEY is required\n' >&2
  exit 1
fi

read -r LAST_MONTH_START LAST_MONTH_END <<<"$(python3 - <<'PY'
from datetime import datetime, timezone
now = datetime.now(timezone.utc)
start_this = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
year = start_this.year if start_this.month > 1 else start_this.year - 1
month = start_this.month - 1 if start_this.month > 1 else 12
start_last = start_this.replace(year=year, month=month)
print(start_last.strftime('%Y-%m-%dT%H:%M:%SZ'), start_this.strftime('%Y-%m-%dT%H:%M:%SZ'))
PY
)"

mkdir -p "$OUT_DIR"

curl -s -H "Authorization: Bearer $LATITUDE_API_KEY" "https://api.latitude.sh/projects" > "$OUT_DIR/latitude_projects.json"
jq -r '.data[].id' "$OUT_DIR/latitude_projects.json" | while read -r project_id; do
  curl -s -G -H "Authorization: Bearer $LATITUDE_API_KEY" --data-urlencode "filter[project]=$project_id" "https://api.latitude.sh/billing/usage" > "$OUT_DIR/latitude_billing_usage_${project_id}.json"
  curl -s -G -H "Authorization: Bearer $LATITUDE_API_KEY" --data-urlencode "filter[project]=$project_id" "https://api.latitude.sh/servers" > "$OUT_DIR/latitude_servers_${project_id}.json"
done

curl -s -G -H "Authorization: Bearer $RUNPOD_API_KEY" \
  --data-urlencode "bucketSize=month" \
  --data-urlencode "startTime=$LAST_MONTH_START" \
  --data-urlencode "endTime=$LAST_MONTH_END" \
  "https://rest.runpod.io/v1/billing/pods" > "$OUT_DIR/runpod_billing_pods_last_full_month.json"

curl -s -G -H "Authorization: Bearer $RUNPOD_API_KEY" \
  --data-urlencode "bucketSize=month" \
  --data-urlencode "startTime=$LAST_MONTH_START" \
  --data-urlencode "endTime=$LAST_MONTH_END" \
  "https://rest.runpod.io/v1/billing/endpoints" > "$OUT_DIR/runpod_billing_endpoints_last_full_month.json"

curl -s -G -H "Authorization: Bearer $RUNPOD_API_KEY" \
  --data-urlencode "bucketSize=month" \
  --data-urlencode "startTime=$LAST_MONTH_START" \
  --data-urlencode "endTime=$LAST_MONTH_END" \
  "https://rest.runpod.io/v1/billing/networkvolumes" > "$OUT_DIR/runpod_billing_networkvolumes_last_full_month.json"

jq -n \
  --arg generated_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg start "$LAST_MONTH_START" \
  --arg end "$LAST_MONTH_END" \
  '{generated_at: $generated_at, last_full_month_start: $start, last_full_month_end: $end}' > "$OUT_DIR/pull_metadata.json"

printf 'Saved billing snapshots to %s\n' "$OUT_DIR"
