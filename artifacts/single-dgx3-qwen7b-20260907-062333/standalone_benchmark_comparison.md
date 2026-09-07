| Machine | DP | Backend | Training tokens | Runtime | tok/s | Avg W | tok/J | kWh/M tokens | Energy cost/M tokens | Power source | Power samples |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| DGX Spark (single) | 1 | CUDA | 1280 | 3.25 | 393.84 | 26.47 | 14.8762 | 0.017027 | 0.004597 GBP | nvidia-smi | 11 |

| Check | Result |
|---|---|
| DGX control detected | PASS |
| Radeon control detected | FAIL |
| CUDA version captured | PASS |
| ROCm version captured | FAIL |
| DGX power sampled | PASS |
| Radeon power sampled | FAIL |
| DGX tokens/J calculated | PASS |
| Radeon tokens/J calculated | FAIL |
| DGX energy cost/M calculated | PASS |
| Radeon energy cost/M calculated | FAIL |
| Full Dagster run completed | PASS |