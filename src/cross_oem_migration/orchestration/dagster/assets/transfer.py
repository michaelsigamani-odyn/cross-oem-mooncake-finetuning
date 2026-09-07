from dataclasses import asdict
from typing import Any, Dict

from dagster import AssetExecutionContext, asset

from ....remote_paths import resolve_remote_root
from ..resources import ExecutorResource, SettingsResource, TransferResource
from .runtime_sync import target_runtime_scripts_synced
from .training import source_checkpoint


@asset(group_name="transfer", deps=[source_checkpoint, target_runtime_scripts_synced], description="Transfers the checkpoint from source_host to target_host and verifies it.")
def transferred_checkpoint(context: AssetExecutionContext, settings: SettingsResource, executor: ExecutorResource, transfer: TransferResource) -> Dict[str, Any]:
    cfg = settings.get()
    exec_ = executor.get(cfg)
    backend = transfer.get(cfg, exec_)
    source_root = resolve_remote_root(exec_, cfg.source_host, cfg.remote_root, cfg.command_retries)
    destination_root = resolve_remote_root(exec_, cfg.target_host, cfg.remote_root, cfg.command_retries)
    source_path = f"{source_root}/amd_run_{cfg.run_id}/checkpoint-{cfg.checkpoint_step}"
    destination_path = f"{destination_root}/checkpoint-{cfg.checkpoint_step}"

    result = backend.transfer(source_host=cfg.source_host, source_path=source_path,
                               destination_host=cfg.target_host, destination_path=destination_path)
    result = backend.verify(result)
    if not result.success or not result.checksum_ok:
        raise RuntimeError(f"Checkpoint transfer failed or checksum mismatch: {result.details}")

    context.add_output_metadata({
        "backend": result.backend, "bytes_transferred": result.bytes_transferred,
        "throughput_mb_s": round(result.throughput_bytes_per_second / (1024 * 1024), 2),
        "checksum_ok": result.checksum_ok,
    })
    return asdict(result)
