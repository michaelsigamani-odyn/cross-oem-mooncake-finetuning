import json
from typing import Any, Dict

from dagster import AssetExecutionContext, asset

from ....data.filesystem import LocalFilesystemDatasetProvider
from ....remote_paths import resolve_remote_root
from ....workloads.finetuning import FineTuningJobSpec, FineTuningWorkload
from ..resources import DataResource, ExecutorResource, SettingsResource
from .transfer import transferred_checkpoint


@asset(group_name="resume", deps=[transferred_checkpoint], description="Resumes training from the transferred checkpoint on the target host.")
def resumed_checkpoint(context: AssetExecutionContext, settings: SettingsResource, executor: ExecutorResource, data: DataResource) -> Dict[str, Any]:
    cfg = settings.get()
    exec_ = executor.get(cfg)
    dataset: LocalFilesystemDatasetProvider = data.dataset_provider(cfg, exec_)
    workload = FineTuningWorkload()
    remote_root = resolve_remote_root(exec_, cfg.target_host, cfg.remote_root, cfg.command_retries)

    run_root = f"{remote_root}/nvidia_run_{cfg.run_id}"
    remote_dataset_path = dataset.sync_to_host(cfg.dataset_path, cfg.target_host)
    spec = FineTuningJobSpec(
        model_id=cfg.model_id, dataset_path=remote_dataset_path, output_dir=run_root,
        stop_step=cfg.final_step, max_steps=cfg.final_step,
        telemetry_interval_seconds=cfg.training_telemetry_interval_seconds,
        use_gpu=cfg.target_use_gpu, resume_from_checkpoint=f"{remote_root}/checkpoint-{cfg.checkpoint_step}",
    )
    python_cmd = cfg.host_python_cmd_overrides.get(cfg.target_host, cfg.target_python_cmd)
    command = workload.build_command(python_cmd=python_cmd, remote_root=remote_root, spec=spec)
    exec_.run(command, host=cfg.target_host, retries=cfg.command_retries, timeout_seconds=cfg.torchrun_timeout_seconds, operation="resume_on_target")

    summary = json.loads(exec_.run(f"cat {run_root}/run_summary.json", host=cfg.target_host).stdout)
    context.add_output_metadata({"run_root": run_root, "final_loss": summary.get("final_loss")})
    return summary


@asset(group_name="resume", deps=[resumed_checkpoint], description="Independently verifies that resume genuinely continued training (extra steps, loss/optimizer continuity).")
def resume_validation(context: AssetExecutionContext, settings: SettingsResource, executor: ExecutorResource) -> Dict[str, Any]:
    cfg = settings.get()
    exec_ = executor.get(cfg)
    remote_root = resolve_remote_root(exec_, cfg.target_host, cfg.remote_root, cfg.command_retries)
    run_root = f"{remote_root}/nvidia_run_{cfg.run_id}"
    proof_out = f"{run_root}/resume_validation_proof.json"
    python_cmd = cfg.host_python_cmd_overrides.get(cfg.target_host, cfg.target_python_cmd)
    command = (
        f"{python_cmd} {remote_root}/scripts/validation/validate_resume.py "
        f"--source-checkpoint {remote_root}/checkpoint-{cfg.checkpoint_step} "
        f"--resumed-summary {run_root}/run_summary.json "
        f"--resumed-checkpoint {run_root}/checkpoint-{cfg.final_step} "
        f"--proof-out {proof_out} "
        f"--min-extra-steps {cfg.min_extra_steps} --max-loss-delta {cfg.max_loss_delta} "
        f"--max-optimizer-norm-delta {cfg.max_optimizer_norm_delta}"
    )
    exec_.run(command, host=cfg.target_host, retries=cfg.command_retries, operation="validate_resume")
    proof = json.loads(exec_.run(f"cat {proof_out}", host=cfg.target_host).stdout)
    context.add_output_metadata({"proof_path": proof_out, "resume_validation_ok": True,
                                  "min_extra_steps": cfg.min_extra_steps, "max_loss_delta": cfg.max_loss_delta})
    return proof
