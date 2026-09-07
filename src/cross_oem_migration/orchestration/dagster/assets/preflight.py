"""Preflight assets. Converted from the original `@op`-based
`preflight_gpu_kernel_execution` (lines 2178-2219) to `@asset` so each
host's preflight shows up as its own node in the Dagster asset graph
(Web UI: Assets tab / global asset lineage graph) instead of being an
anonymous step inside an opaque job run.
"""
import json
from typing import Any, Dict

from dagster import AssetExecutionContext, MetadataValue, asset

from ....hardware import GpuVendorAdapter
from ..resources import ExecutorResource, HardwareResource, SettingsResource


def _run_preflight(settings, executor, hardware: HardwareResource, host: str) -> Dict[str, Any]:
    machine = settings.machine_for_host(host)
    vendor = machine.vendor if machine else "unknown"
    adapter: GpuVendorAdapter = hardware.adapter_for(vendor) if vendor != "unknown" else hardware.adapter_for("nvidia")
    python_cmd = settings.host_python_cmd_overrides.get(host) or (machine.python_cmd if machine else settings.source_python_cmd)
    script = adapter.preflight_script()
    command = f"{python_cmd} -c {json.dumps(script)}"
    result = executor.run(command, host=host, retries=settings.command_retries, allow_failure=True,
                           timeout_seconds=180.0, operation=f"probe_kernel_preflight_{host}")
    if not result.ok:
        raise RuntimeError(f"Preflight probe failed on host={host} (exit={result.returncode}). stderr={result.stderr[-2000:]}")
    lines = [l for l in result.stdout.splitlines() if l.strip()]
    payload = json.loads(lines[-1]) if lines else {}
    parsed = adapter.parse_preflight(host, payload)
    if not parsed.training_allowed:
        raise RuntimeError(f"GPU training not allowed on host={host}: {parsed.reason}")
    return payload


@asset(group_name="preflight", description="GPU vendor detection + real kernel execution check on the source host.")
def source_gpu_preflight(context: AssetExecutionContext, settings: SettingsResource, executor: ExecutorResource, hardware: HardwareResource) -> Dict[str, Any]:
    cfg = settings.get()
    payload = _run_preflight(cfg, executor.get(cfg), hardware, cfg.source_host)
    context.add_output_metadata({"vendor": payload.get("gpu_vendor"), "device_name": payload.get("gpu_model"), "architecture": payload.get("gpu_architecture")})
    return payload


@asset(group_name="preflight", description="GPU vendor detection + real kernel execution check on the target host.")
def target_gpu_preflight(context: AssetExecutionContext, settings: SettingsResource, executor: ExecutorResource, hardware: HardwareResource) -> Dict[str, Any]:
    cfg = settings.get()
    payload = _run_preflight(cfg, executor.get(cfg), hardware, cfg.target_host)
    context.add_output_metadata({"vendor": payload.get("gpu_vendor"), "device_name": payload.get("gpu_model"), "architecture": payload.get("gpu_architecture")})
    return payload
