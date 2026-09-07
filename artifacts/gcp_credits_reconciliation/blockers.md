# Blockers and evidence gaps

## OpenAI / Codex billing history

- Attempted endpoint: `GET https://api.openai.com/v1/organization/costs?start_time=...&end_time=...&bucket_width=1m`
- Result: `invalid_api_key`
- Impact: 6-month Codex average and high/low outlier months cannot be computed from source data.

## Anthropic / Claude billing history

- Attempted endpoint: `GET https://api.anthropic.com/v1/organizations`
- Result: `authentication_error` requiring admin or organization-scoped key
- Impact: 6-month Claude average and outlier months cannot be computed from source data.

## Gcore invoice-backed rates

- No Gcore contract/invoice file was found in this workspace, `~/Documents`, or `~/Downloads` using targeted filename searches.
- Impact: H100 rental rates remain verbal assumptions.

## vast.ai and Lambda invoices

- No provider invoices, exports, or billing screenshots were found in the workspace using targeted filename searches.
- Impact: monthly spend is unresolved for these two providers.

## Owned-hardware telemetry and tariffs

- No persisted 7-day power/utilization series was found in this repo.
- Remote hosts reachable for spot checks (`odyn-radeon2`, `odyn-dgx1`, `odyn-dgx3`) did not expose obvious ready-made week-long GPU power logs in queried default paths.
- No electricity tariff documents were found for London office or Raghu's house.
- Impact: per-device and total monthly electricity cost roll-up cannot be computed honestly.
