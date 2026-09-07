# Cross-OEM Migration (refactored)

Fine-tunes on one GPU vendor, migrates the checkpoint to another (NVIDIA <-> AMD),
resumes training, and produces an auditable cross-OEM report. Orchestrated with
Dagster **software-defined assets** (not the legacy `@op`/`@job` API the original
repo used).

Read **MIGRATION_NOTES.md** before relying on this in place of the original repo --
it documents exactly what was ported at full fidelity vs. simplified.

## Layout

```
configs/                      # run.json (parameters) + machines.json (hardware catalog)
src/cross_oem_migration/
  config/                     # dataclasses + loader -- no dagster, no subprocess
  execution/                  # Executor interface: SSHExecutor, LocalExecutor
  hardware/                   # GpuVendorAdapter interface: NvidiaAdapter, AmdAdapter
  transfer/                   # TransferBackend interface: scp, mooncake_tcp
  workloads/                  # Workload interface: FineTuningWorkload, InferenceWorkload (stub)
  data/                       # DatasetProvider / ArtifactStore / RunMetricsDatabase (sqlite)
  reporting/                  # pure energy/cost math + formula cross-checks
  orchestration/dagster/      # the ONLY place that imports `dagster`
    resources.py
    assets/                   # @asset graph: preflight -> train -> transfer -> resume -> report -> publish
    jobs.py, definitions.py   # entry point
scripts/                      # standalone scripts that actually run ON the remote hosts
  finetuning/train_lora_migration.py
  transfer/mooncake_tcp_agent.py
  validation/{checkpoint_io,validate_resume}.py
tests/
```

Every package above `orchestration/` is plain Python with no framework dependency --
you can `import cross_oem_migration.reporting.metrics` from a notebook, a pytest
file, or a different orchestrator entirely.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env   # only if a host needs password auth -- see below
export DAGSTER_HOME="${HOME}/.dagster_home"; mkdir -p "$DAGSTER_HOME"

dagster dev -w workspace.yaml
```

Open http://127.0.0.1:3000 -- you should see 9 assets in the graph (preflight x2,
training, transfer, resume x2, reporting x2, published) plus 2 asset checks
attached to `published_manifest`, not a single opaque job step.

## SSH access

**The original repo's `repo_config.json` and README committed a real SSH
password in plaintext.** That's fixed here: `configs/run.json` has no
`ssh_password` field, and if a host genuinely needs password auth, set
`CROSS_OEM_SSH_PASSWORD` as an environment variable (see `.env.example`), never
in a committed file. Prefer SSH keys + `~/.ssh/config` over passwords entirely.

## Configuration

- `configs/run.json` -- which two hosts, which model, which dataset, transfer backend.
- `configs/machines.json` -- the hardware catalog (ask #4): one entry per machine,
  vendor, architecture, container image, python interpreter. Add a machine by
  adding an entry here, not by editing pipeline code.

## Tests

```bash
pytest tests/
```
