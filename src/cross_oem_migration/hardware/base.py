"""Hardware abstraction (ask #4): one interface, one implementation per
vendor. Nothing outside this package should ever branch on
`if vendor == "amd"` / `if vendor == "nvidia"` again -- that logic was
previously smeared across `kernel_preflight_script`, `run_iperf3_*`,
`training_kwargs`, `gpu_metrics`, and `query_gpu_sample` in the original
files. Adding a third vendor (e.g. Intel Gaudi) means adding one adapter
class here and one registry entry -- Open/Closed Principle.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class GpuPreflightResult:
    host: str
    vendor: str
    device_name: Optional[str]
    architecture: Optional[str]
    gpu_detected: bool
    training_allowed: bool
    reason: Optional[str]
    raw: Dict[str, Any]


class GpuVendorAdapter(ABC):
    """Everything that differs between NVIDIA and AMD lives behind this
    interface: how to detect the GPU, how to read power telemetry, and how
    to install the right torch wheel for that vendor's runtime."""

    vendor: str

    @abstractmethod
    def preflight_script(self) -> str:
        """A self-contained Python script (run remotely via the Executor)
        that detects the GPU, runs a real kernel on it, and reports power
        telemetry. Returns JSON on stdout."""

    @abstractmethod
    def parse_preflight(self, host: str, payload: Dict[str, Any]) -> GpuPreflightResult:
        ...

    @abstractmethod
    def telemetry_sample_script(self) -> str:
        """A short script that returns one instantaneous power/utilization
        sample -- used by the training workload's telemetry monitor."""

    @abstractmethod
    def default_torch_install_command(self, python_cmd: str) -> str:
        ...
