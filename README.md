# Cross-Vendor Fine-Tuning Portability Test

This repository now starts with the simplest portability check: resume LoRA fine-tuning from AMD/ROCm on NVIDIA/CUDA without Mooncake.

## User Story 1 (No Mooncake)

Question answered: can the training workload itself resume across vendors?

### Hosts

- AMD source: `odyn-radeon2`
- NVIDIA destination: `odyn-dgx3`
- Optional second NVIDIA control run: `odyn-dgx1`

### What this test does

1. Fine-tunes a tiny Llama-family model on AMD.
2. Stops at step `10` and saves checkpoint `checkpoint-10`.
3. Terminates AMD process naturally after checkpoint save.
4. Copies checkpoint with ordinary `scp` (no Mooncake).
5. Resumes training on NVIDIA from the copied checkpoint.
6. Confirms global step advanced beyond saved step.

### Resumable state included

Hugging Face `Trainer` checkpoint contains:

- model and LoRA adapter weights
- optimizer state
- scheduler state
- RNG/trainer state including current global step

### Files

- `train_lora_migration.py`: tiny LoRA train/resume workload.
- `validate_resume.py`: verifies step continuity and loss jump tolerance.
- `dagster_basic_portability.py`: same flow packaged as a small Dagster job.
- `repo_config.json`: single root config for hosts, steps, thresholds, and runtime paths.

### Run

```bash
dagster dev -w ./workspace.yaml
```

### Run with Dagster Web UI

Start the Dagster web app with the workspace:

```bash
"/Users/michaelsigamani/cross-vendor-mooncake-test/.conda/bin/dagster" dev -w "/Users/michaelsigamani/cross-vendor-mooncake-test/workspace.yaml"
```

Then open `http://127.0.0.1:3000` and launch `portability_job`. Runtime values are read from `repo_config.json`.

### Assets, provenance, schema, and lineage

The job emits materializations for control, transfer, and published data stages. In the Dagster asset graph and event logs you will see:

- control assets: `portability/settings`, `portability/control/host_preparation`, `portability/control/environment_sync`
- source/transfer/target assets: `portability/source/amd/checkpoint`, `portability/transfer/checkpoint`, `portability/target/nvidia/resume_training`, `portability/target/nvidia/resume_validation`
- published data assets: `portability/published/<asset_path>` and `portability/published/assets_manifest`

Each materialization includes metadata for:

- provenance (`source_host`, `target_host`, `run_id`, operation details)
- schema/fields (`run_id`, `source_host`, `target_host`, `checkpoint_step`, `final_step`, `status`, `artifact_path`)
- lineage path (`lineage_path`) to make the control -> source -> transfer -> target -> published flow explicit

Validation now also writes a proof artifact at `~/cross-vendor-mooncake-test/nvidia_run_<run_id>/resume_validation_proof.json` on the NVIDIA host. The proof captures source/resumed steps, first resumed log step, resume pointer, and optimizer/scheduler/trainer-state file sizes for both checkpoints.

### Asset checks

Two checks are attached to `portability/published/assets_manifest`:

- `saved_assets_exist`: fails if expected saved artifacts are missing or zero-byte files
- `saved_assets_supported_type`: verifies saved artifacts are one of `directory`, `json`, `jsonl`, `csv`, `parquet`, `arrow`, `sqlite`, `db`, `txt`

Run checks in the UI from the `portability/published/assets_manifest` asset details page after a materialization run.

## User Story 2 (Mooncake, after Story 1 passes)

Mooncake transfer helper remains available:

- `mooncake_checkpoint_transfer.py`

Use these only after basic ROCm->CUDA resume is proven.

## User Story 3 (Real hardware, pre-registered, control + reproducibility)

This stage runs a bidirectional relay on real Spark and Radeon hardware with a fixed plan committed before execution:

- pre-registration: `experiments/story3_preregistration.json`
- relay harness: `failover_harness.py`
- checkpoint structure checks: `checkpoint_io.py`
- dataset used for training: `data/story3_dataset.jsonl`

### Execute

```bash
dagster dev -w ./workspace.yaml
```

Story 3 settings are fully config-driven from `repo_config.json`.

- `story3.run_environments` controls which environments execute (`dev`, `stage`, `prod`).
- `story3.environments.stage.spark_hosts` must contain exactly 2 DGX hosts.
- `story3.environments.prod.spark_hosts` must contain exactly 8 A100 hosts.
- each environment maps one Radeon host plus one-or-more Spark/A100 hosts.

Example topology section:

```json
"story3": {
  "run_environments": ["dev", "stage", "prod"],
  "environments": {
    "dev": {"radeon_host": "odyn-radeon2", "spark_hosts": ["dgx-spark"]},
    "stage": {"radeon_host": "odyn-radeon2", "spark_hosts": ["odyn-dgx1", "odyn-dgx3"]},
    "prod": {"radeon_host": "odyn-radeon2", "spark_hosts": ["a100-01", "a100-02", "a100-03", "a100-04", "a100-05", "a100-06", "a100-07", "a100-08"]}
  }
}
```

### What is recorded for independent verification

- per-seed trial folder: `artifacts/story3/seed-<seed>/`
- raw `run_summary.json` files for cross-vendor and control resumes
- raw `manifest.json` files for Spark and Radeon checkpoints
- raw `resume_validation_proof.json` files for each boundary
- full per-trial verdict in `trial_summary.json`
- aggregate verdict in `artifacts/story3/story3_report.json`

### Scope guardrails

Interpret results only as evidence for single-GPU, single-node, short-horizon failover for this model and setup. Do not generalize this result to multi-GPU, tensor-parallel, larger models, or long-horizon stability without separate testing.

## Known portability risks

- Version mismatch across PyTorch/Transformers/PEFT can break optimizer-state load.
- Vendor-specific fused optimizers can break checkpoint compatibility.
- BF16/FP16 differences may cause loss deltas; check for trend continuity, not exact equality.

## Dagster Cloud

Use `dagster_cloud.yaml` as the code-location manifest for cloud deployments. It defines two locations:

- `basic_portability` -> `dagster_basic_portability.py`
- `story3_failover` -> `dagster_story3_failover.py`

Cloud prerequisites and exact deployment commands depend on your Dagster Cloud tenancy and CLI setup. In this environment, there is insufficient evidence to validate a specific `dagster-cloud` CLI command because the CLI is not installed (`dagster-cloud: command not found`).
