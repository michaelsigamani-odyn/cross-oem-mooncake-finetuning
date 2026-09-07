# GCP credits spend reconciliation (2026-09-07)

## What was pulled

- Automated pull script: `scripts/pull_spend_sources.sh`
- Forward collection script for 7-day power traces: `scripts/capture_gpu_power_timeseries.sh`
- Raw outputs: `artifacts/gcp_credits_reconciliation/raw`
- Pull timestamp: `2026-09-07T13:21:46Z`
- Last full month window used for Runpod: `2026-08-01T00:00:00Z` to `2026-09-01T00:00:00Z`

## Reconciled values (with sources)

| Line item | Value | Status | Source |
|---|---:|---|---|
| Latitude H100 (Dallas) | 1 x `g3.h100.small` at `1230 USD/month` | Confirmed | `artifacts/gcp_credits_reconciliation/raw/latitude_billing_usage_proj_dYjv5rEmoNqp7.json` and `artifacts/gcp_credits_reconciliation/raw/latitude_servers_proj_dYjv5rEmoNqp7.json` |
| Runpod Pods (last full month) | `0 USD` | Confirmed | `artifacts/gcp_credits_reconciliation/raw/runpod_billing_pods_last_full_month.json` |
| Runpod Endpoints (last full month) | `0 USD` | Confirmed | `artifacts/gcp_credits_reconciliation/raw/runpod_billing_endpoints_last_full_month.json` |
| Runpod Network Volumes (last full month) | `0 USD` | Confirmed | `artifacts/gcp_credits_reconciliation/raw/runpod_billing_networkvolumes_last_full_month.json` |

## Items still not source-backed

- Gcore H100 rentals (2 x Luxembourg demo + 1 x Dallas prod) remain verbal assumptions (`1400 USD/GPU/month`) with no contract/invoice artifact in this workspace.
- Vast.ai and Lambda monthly totals are unresolved: no invoice export, billing screenshot archive, or usable API credential was discoverable in this workspace.
- Claude and Codex 6-month billing averages are unresolved: available keys do not provide usable billing-history access for the required org-level history pull.
- Owned-hardware electricity model remains unresolved: no persisted 7-day utilization/power time series was available in-repo or in obvious remote log paths.
- Site electricity tariffs (London office, Raghu house) were not found as bills/contracts in this workspace.

## Acceptance criteria status

- [x] Latitude: active spend confirmed via API-backed billing + server metadata.
- [ ] Gcore H100s: still assumption-only; invoice or service order not found.
- [~] vast.ai / Lambda / Runpod: Runpod is sourced; vast.ai and Lambda are unsourced.
- [ ] Claude + Codex 6-month average: insufficient evidence.
- [ ] Owned-hardware power draw 7-day series: insufficient evidence.
- [ ] Electricity cost roll-up: blocked by missing time-series and tariff documents.
- [x] Every number in `spend_reconciliation.csv` is explicitly labeled by source quality.
- [x] Reproducibility: API pull script and raw snapshots saved.

## Reproduction

```bash
./scripts/pull_spend_sources.sh
```

This regenerates the Latitude and Runpod snapshots under `artifacts/gcp_credits_reconciliation/raw`.

For owned-hardware telemetry capture:

```bash
DURATION_SECONDS=604800 INTERVAL_SECONDS=60 ./scripts/capture_gpu_power_timeseries.sh
```
