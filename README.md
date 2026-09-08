# Odyn Compute Profiler 

This repository tries to demonstrate a repeatable cross-OEM workflow for any member of Odyn who wants to replicate results:

- train on one machine,
- transfer checkpoint artifacts,
- resume on a different machine/vendor,
- validate continuity with telemetry and summary checks.
- So far mi300x, radeon, DGX-Spark connected via QSFP

Current scope is intentionally sequential (single active training phase at a time). This is designed to prove portability and recovery semantics first; data-parallel or multi-job orchestration can be layered on top.

## Prerequisites

- Python 3.12
- SSH access to participating machines (key-based authentication preferred)
- `conda` or another environment manager

Optional:

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

<img width="1348" height="930" alt="Screenshot 2026-09-07 at 20 26 05" src="https://github.com/user-attachments/assets/8426355f-8115-4d16-a444-ef70170f92f1" />


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


## Telemetry and measurement

This is the most important aspect of the repo currently. The goal is to normalize comparable metrics across NVIDIA and AMD hosts.

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

## Things I was thinking to do. Give me suggestions if you feel like. Thanks.

- Add a two-node A100 data-parallel stage for direct comparison with sequential resume.
- Add disaggregated prefill/inference profiling.
- Add structured profiling flows (including planned Vidur-based inference analysis).


