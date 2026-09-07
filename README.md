# Cross-OEM Checkpoint Resume and Fine-Tuning

This repository demonstrates a controlled cross-OEM training workflow:

- train on one machine,
- transfer checkpoint artifacts,
- resume on a different machine/vendor,
- validate continuity with telemetry and summary checks.

Current scope is intentionally sequential (single active training phase at a time). This is designed to prove portability and recovery semantics first; data-parallel or multi-job orchestration can be layered on top.

## What this repo is for

- Validate checkpoint portability across heterogeneous GPU environments.
- Keep machine-specific details out of orchestration logic via structured config.
- Collect enough metrics to compare runtime, energy efficiency, and transfer overhead.
- Provide a Dagster-based control plane for local experimentation.

## What this repo is not

- It is not yet a fully parallel cross-machine trainer.
- It is not yet an inference disaggregation benchmark.
- It is not yet a production scheduler for large job fleets.

## Architecture at a glance

- Run config: `configs/run.json`
- Machine catalog: `configs/machines.json`
- Dagster definitions: `src/cross_oem_migration/orchestration/dagster/definitions.py`
- Legacy monolith (kept for migration/backward compatibility): `finetuning_sequential.py`

## Prerequisites

- Python 3.12
- SSH access to participating machines (key-based authentication preferred)
- `conda` or another environment manager

Optional but commonly needed:

- `sshpass` (only if you must use password-based SSH)
- vendor GPU telemetry libraries (`nvidia-ml-py` and/or AMD SMI bindings)

## Quickstart

```bash
python3 --version
conda create -y -n cross-oem-migration python=3.12
conda activate cross-oem-migration

pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"

cp .env.example .env
source .env

export DAGSTER_HOME="${HOME}/.dagster_home"
mkdir -p "${DAGSTER_HOME}"

dagster dev -w workspace.yaml
```

Expected output includes:

```text
Serving dagster-webserver on http://127.0.0.1:3000
```

Open `http://127.0.0.1:3000`, materialize the job/assets, and inspect run metadata.

## Configuration model

`configs/run.json` contains run-level concerns:

- source/target host selection
- model, steps, transfer backend
- checkpoint and timeout controls
- benchmark controls

`configs/machines.json` contains machine-level concerns:

- hostname, vendor, architecture expectations
- Python path and runtime mode
- optional transfer control-plane metadata
- optional distributed topology information

This split is deliberate: it keeps hardware details modular and avoids vendor-specific branching in orchestration paths.

## Secrets and credentials

Do not commit secrets into git-tracked config files.

Use environment variables for sensitive values:

```bash
export CROSS_OEM_SSH_PASSWORD="<only-if-needed>"
```

Recommended authentication order:

1. SSH keys and `~/.ssh/config` aliases.
2. SSH agent-backed keys.
3. Password auth only when key auth is not available.

Operational rules:

- Keep `.env` local only.
- Rotate secrets if any credential is accidentally committed.
- Prefer per-user shell exports or secret managers over static plaintext files.

## Telemetry and measurement

The goal is to normalize comparable metrics across NVIDIA and AMD hosts.

Core metrics include:

- training tokens processed
- runtime seconds
- tokens/second
- average and peak GPU power
- GPU energy joules
- tokens/joule
- checkpoint transfer throughput

Suggested packages:

- Host metrics: `psutil`
- Export format: `prometheus-client`
- NVIDIA: `nvidia-ml-py`
- AMD: ROCm AMD SMI Python bindings

For NVIDIA, sample NVML counters at a fixed interval and integrate power over time for energy. For AMD, use AMD SMI equivalents; API field names can differ between ROCm versions and should be verified on the target environment.

## Current roadmap

- Add a two-node A100 data-parallel stage for direct comparison with sequential resume.
- Add disaggregated prefill/inference profiling.
- Add structured profiling flows (including planned Vidur-based inference analysis).

## Troubleshooting

- If Dagster cannot load definitions, verify `workspace.yaml` pathing and editable install (`pip install -e .`).
- If SSH commands hang, check host aliases, key permissions, and timeout settings in `configs/run.json`.
- If telemetry is sparse, reduce sample interval and verify GPU library availability on each host.
