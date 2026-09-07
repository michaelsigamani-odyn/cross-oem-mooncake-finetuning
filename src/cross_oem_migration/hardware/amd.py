"""AMD-specific hardware behavior (ROCm / amd-smi / rocm-smi). Telemetry
query ported from `query_amd_smi_sample` / `amd_smi_command` /
`parse_amd_smi_row` and JSON-flattening helpers (original
train_lora_migration.py lines 445-512)."""
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._preflight_script import SHARED_PREFLIGHT_SCRIPT
from .base import GpuPreflightResult, GpuVendorAdapter


class AmdAdapter(GpuVendorAdapter):
    vendor = "amd"

    def preflight_script(self) -> str:
        return SHARED_PREFLIGHT_SCRIPT

    def parse_preflight(self, host: str, payload: Dict[str, Any]) -> GpuPreflightResult:
        return GpuPreflightResult(
            host=host,
            vendor=payload.get("gpu_vendor", "amd"),
            device_name=payload.get("gpu_model"),
            architecture=payload.get("gpu_architecture"),
            gpu_detected=bool(payload.get("gpu_detected")),
            training_allowed=bool(payload.get("training_allowed")),
            reason=payload.get("gpu_kernel_reason"),
            raw=payload,
        )

    def telemetry_sample_script(self) -> str:
        return "amd-smi metric --utilization --vram-usage --power --json (falls back to rocm-smi)"

    def query_telemetry_sample(self) -> Dict[str, Optional[float]]:
        now = time.perf_counter()
        stdout = self._amd_smi_stdout()
        payload = _parse_json(stdout)
        util = _parse_float(_extract_value(payload, ["gpu", "gpu_use", "gpu_use_percent", "utilization", "gpu_util_percent", "GFX Activity"]))
        mem = self._parse_memory_bytes(payload)
        power = _parse_float(_extract_value(payload, ["power", "socket_power", "current_socket_power", "average_socket_power", "Average Graphics Package Power (W)"]))
        return {"timestamp": now, "gpu_utilization_percent": util, "memory_used_bytes": mem, "power_watts": power}

    def default_torch_install_command(self, python_cmd: str) -> str:
        return (
            f'{python_cmd} -c "import torch" || {python_cmd} -m pip install '
            f"--index-url https://download.pytorch.org/whl/rocm6.3 torch==2.9.1+rocm6.3"
        )

    # -- vendor-private helpers -------------------------------------------------
    def _amd_smi_command(self) -> Optional[List[str]]:
        binary = next((p for p in [shutil.which("amd-smi"), "/opt/rocm/bin/amd-smi", "/opt/rocm-5.7.0/bin/amd-smi"] if p and Path(p).exists()), None)
        return None if binary is None else [binary, "metric", "--utilization", "--vram-usage", "--power", "--json"]

    def _amd_smi_stdout(self) -> str:
        command = self._amd_smi_command()
        if command is not None:
            out = subprocess.run(command, text=True, capture_output=True, check=False)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout
        out = subprocess.run(["rocm-smi", "--showuse", "--showmemuse", "--showpower", "--json"], text=True, capture_output=True, check=False)
        return out.stdout if out.returncode == 0 else ""

    def _parse_memory_bytes(self, payload: Dict[str, Any]) -> Optional[int]:
        direct = _parse_float(_extract_value(payload, ["vram_used", "used_vram", "used_memory", "gpu_memory_usage", "mem_use"]))
        if direct is not None and direct > 0:
            return int(direct)
        mib = _parse_float(_extract_value(payload, ["VRAM Total Used Memory (MiB)", "used vram (mb)", "vram usage"]))
        return None if mib is None else int(mib * 1024 * 1024)


def _parse_json(stdout: str) -> Dict[str, Any]:
    try:
        loaded = json.loads(stdout)
    except Exception:
        return {}
    if isinstance(loaded, dict):
        return loaded
    return loaded[0] if isinstance(loaded, list) and loaded and isinstance(loaded[0], dict) else {}


def _flatten_items(payload: Dict[str, Any], prefix: str = "") -> List[tuple]:
    if not isinstance(payload, dict):
        return []
    rows = []
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        rows.extend(_flatten_items(value, name) if isinstance(value, dict) else [(name, value)])
    return rows


def _flatten_scalar(value: Any) -> Optional[str]:
    return None if isinstance(value, (dict, list)) else str(value)


def _extract_value(payload: Dict[str, Any], keys: List[str]) -> Optional[str]:
    values = [_flatten_scalar(v) for name, v in _flatten_items(payload) if any(k.lower() in name.lower() for k in keys)]
    return next((v for v in values if v is not None), None)


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None
