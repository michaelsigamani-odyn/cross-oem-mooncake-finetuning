# User Story: Reliable AMD to NVIDIA Portability Run

## Story
As an ML platform engineer,
I want the portability Dagster job to train or reuse an AMD checkpoint, transfer it to NVIDIA, resume training, and validate continuity,
so that cross-vendor checkpoint portability is reproducible and operationally reliable.

## Acceptance Criteria
1. Running `python3 dagster_basic_portability.py` completes with `RUN_SUCCESS` for `portability_job` under normal network conditions.
2. If `~/cross-vendor-mooncake-test/amd_run_<run_id>/checkpoint-<checkpoint_step>/checkpoint_meta.json` already exists on AMD, `train_on_amd` reuses it and does not retrain.
3. `validate_resume` passes and writes `~/cross-vendor-mooncake-test/nvidia_run_<run_id>/resume_validation_proof.json` on NVIDIA.
4. Validation continuity uses resumed trace evidence (`loss_trace`) and confirms at least one logged step greater than source `saved_step`.
5. Validation enforces optimizer state consistency using `--max-optimizer-norm-delta` from `repo_config.json`.
6. Pipeline SSH calls include keepalive/connect-timeout settings and retry failed commands based on `command_retries`.
7. When `basic_portability.amd_use_gpu` is `false`, AMD training runs with `CUDA_VISIBLE_DEVICES=''` to avoid GPU-path crashes.
8. For the configured run (`run_id=amd-to-nvidia-basic`), proof reports `source_saved_step=10`, `resumed_saved_step=30`, and non-failing optimizer norm delta.

## Definition of Done
- The Dagster run succeeds end-to-end at least once with the current config.
- `resume_validation_proof.json` exists and matches criteria.
- Saved assets are materialized under local `artifacts/<run_id>/`.
