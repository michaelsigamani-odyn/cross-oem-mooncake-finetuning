| Machine | DP | Backend | Training tokens | Runtime | tok/s | Avg W | tok/J | kWh/M tokens | Energy cost/M tokens | Power source | Power samples |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Radeon | 1 | ROCm | 1280 | 2.15 | 594.29 | 45.71 | 13.0011 | 0.017485 | 0.004721 GBP | amd_smi | 6 |
| DGX Spark (single) | 1 | CUDA | 1280 | 1.75 | 733.36 | 12.17 | 60.2349 | 0.003728 | 0.001007 GBP | nvml | 6 |

| Check | Result |
|---|---|
| DGX control detected | PASS |
| Radeon control detected | PASS |
| CUDA version captured | PASS |
| ROCm version captured | PASS |
| DGX power sampled | PASS |
| Radeon power sampled | PASS |
| DGX tokens/J calculated | PASS |
| Radeon tokens/J calculated | PASS |
| DGX energy cost/M calculated | PASS |
| Radeon energy cost/M calculated | PASS |
| Full Dagster run completed | PASS |