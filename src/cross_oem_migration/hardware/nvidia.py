"""NVIDIA-specific hardware behavior. Telemetry query ported from
`query_nvidia_smi_sample` / `parse_nvidia_smi_row` (original
train_lora_migration.py lines 438-442, 514-520)."""
import re
import subprocess
import time
from typing import Any, Dict, Optional

from ._preflight_script import SHARED_PREFLIGHT_SCRIPT
from .base import GpuPreflightResult, GpuVendorAdapter


class NvidiaAdapter(GpuVendorAdapter):
    vendor = "nvidia"

    def preflight_script(self) -> str:
        return SHARED_PREFLIGHT_SCRIPT

    def parse_preflight(self, host: str, payload: Dict[str, Any]) -> GpuPreflightResult:
        return GpuPreflightResult(
            host=host,
            vendor=payload.get("gpu_vendor", "nvidia"),
            device_name=payload.get("gpu_model"),
            architecture=payload.get("gpu_architecture"),
            gpu_detected=bool(payload.get("gpu_detected")),
            training_allowed=bool(payload.get("training_allowed")),
            reason=payload.get("gpu_kernel_reason"),
            raw=payload,
        )

    def telemetry_sample_script(self) -> str:
        # Executed in-process by the training workload (not remotely via
        # Executor) -- see workloads/finetuning.py GpuTelemetryMonitor.
        return "nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits"

    def query_telemetry_sample(self) -> Dict[str, Optional[float]]:
        now = time.perf_counter()
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=False,
        )
        row = next((line for line in (out.stdout or "").splitlines() if line.strip()), "")
        parts = [item.strip() for item in row.split(",")]
        util = _parse_float(parts[0] if len(parts) > 0 else None)
        mem_mib = _parse_float(parts[1] if len(parts) > 1 else None)
        power = _parse_float(parts[2] if len(parts) > 2 else None)
        mem_bytes = None if mem_mib is None else int(mem_mib * 1024 * 1024)
        return {"timestamp": now, "gpu_utilization_percent": util, "memory_used_bytes": mem_bytes, "power_watts": power}

    def default_torch_install_command(self, python_cmd: str) -> str:
        return f'{python_cmd} -c "import torch" || {python_cmd} -m pip install torch'


def _parse_float(value):
    if value is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None
