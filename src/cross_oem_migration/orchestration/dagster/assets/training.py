"""Training asset. Ports `train_on_source` (original lines 1520-1556):
runs the fine-tuning workload on the source host and materializes the
resulting checkpoint as an asset. All command-building logic now lives in
workloads.finetuning.FineTuningWorkload, not inline here.
"""
from typing import Any, Dict

from dagster import AssetExecutionContext, MetadataValue, asset

from ....data.filesystem import LocalFilesystemDatasetProvider
from ....remote_paths import resolve_remote_root
from ....workloads.finetuning import FineTuningJobSpec, FineTuningWorkload
from ..resources import DataResource, ExecutorResource, SettingsResource
from .runtime_sync import source_runtime_scripts_synced


@asset(group_name="training", deps=[source_runtime_scripts_synced], description="Fine-tunes the base model on the source host up to `checkpoint_step`.")
def source_checkpoint(context: AssetExecutionContext, settings: SettingsResource, executor: ExecutorResource, data: DataResource) -> Dict[str, Any]:
    cfg = settings.get()
    exec_ = executor.get(cfg)
    dataset: LocalFilesystemDatasetProvider = data.dataset_provider(cfg, exec_)
    workload = FineTuningWorkload()
    remote_root = resolve_remote_root(exec_, cfg.source_host, cfg.remote_root, cfg.command_retries)

    run_root = f"{remote_root}/amd_run_{cfg.run_id}"
    checkpoint_meta = f"{run_root}/checkpoint-{cfg.checkpoint_step}/checkpoint_meta.json"
    summary_path = f"{run_root}/run_summary.json"

    if exec_.exists(cfg.source_host, checkpoint_meta) and exec_.exists(cfg.source_host, summary_path):
        # Mirrors the original reuse check (source_checkpoint_reusable,
        # lines 1576-1616) via FineTuningWorkload.is_checkpoint_reusable.
        import json
        existing = exec_.run(f"cat {summary_path}", host=cfg.source_host, allow_failure=True)
        if existing.ok:
            summary = json.loads(existing.stdout)
            if workload.is_checkpoint_reusable(summary, minimum_power_samples=cfg.minimum_power_samples,
                                                 expected_telemetry_interval_seconds=cfg.training_telemetry_interval_seconds):
                context.add_output_metadata({"status": "reused", "run_root": run_root})
                return summary

    exec_.run(f"rm -rf {run_root}", host=cfg.source_host, retries=cfg.command_retries, allow_failure=True)
    remote_dataset_path = dataset.sync_to_host(cfg.dataset_path, cfg.source_host)

    spec = FineTuningJobSpec(
        model_id=cfg.model_id, dataset_path=remote_dataset_path, output_dir=run_root,
        stop_step=cfg.checkpoint_step, max_steps=cfg.checkpoint_step,
        telemetry_interval_seconds=cfg.training_telemetry_interval_seconds,
        use_gpu=cfg.source_use_gpu, data_parallel_size=cfg.source_data_parallel_size,
    )
    python_cmd = cfg.host_python_cmd_overrides.get(cfg.source_host, cfg.source_python_cmd)
    command = workload.build_command(python_cmd=python_cmd, remote_root=remote_root, spec=spec)
    exec_.run(command, host=cfg.source_host, retries=cfg.command_retries, timeout_seconds=cfg.torchrun_timeout_seconds, operation="train_on_source")

    summary_result = exec_.run(f"cat {summary_path}", host=cfg.source_host)
    import json
    summary = json.loads(summary_result.stdout)
    context.add_output_metadata({
        "status": "created", "run_root": run_root,
        "final_loss": summary.get("final_loss"), "tokens_per_second": summary.get("tokens_per_second"),
    })
    return summary
