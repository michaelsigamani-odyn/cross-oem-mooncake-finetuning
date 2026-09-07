"""Pure data model for run configuration.

This module intentionally contains ZERO behavior tied to Dagster, SSH,
GPU vendors, or subprocess execution. It is safe to import from anywhere
(assets, tests, CLI scripts, notebooks) without dragging in the rest of
the stack. This is the fix for the original `finetuning_sequential.py`
lines 16-227, where the config dataclasses lived in the same file as the
Dagster ops that consumed them.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class TransferNodeConfig:
    server_name: str
    control_host: str
    control_port: int
    python_cmd: str


@dataclass(frozen=True)
class MooncakeTransferConfig:
    metadata_server: str
    protocol: str
    python_cmd: str
    chunk_mb: int
    max_inflight: int
    source: TransferNodeConfig
    destination: TransferNodeConfig


@dataclass(frozen=True)
class TransferSettings:
    backend: str  # "scp" | "mooncake_tcp" -- see transfer/registry.py for the OCP-friendly lookup
    mooncake: Optional[MooncakeTransferConfig]


@dataclass(frozen=True)
class MachineTransferInfo:
    control_host: Optional[str]
    control_port: Optional[int]


@dataclass(frozen=True)
class MachineSpec:
    """One row of the hardware catalog. This is the *only* place hardware
    facts (vendor, arch, container image, ROCm path...) should live -- see
    configs/machines.json rather than scattering vendor `if` statements
    through orchestration code."""

    key: str
    hostname: str
    vendor: str  # "nvidia" | "amd" -- resolved to an adapter via hardware/registry.py
    device_name: str
    expected_gpu_architecture: str
    container_image: Optional[str]
    execution_mode: str  # "host" | "docker"
    torch_install_command: Optional[str]
    hf_cache_dir: Optional[str]
    rocm_path: Optional[str]
    electricity_price_per_kwh: float
    currency: str
    python_cmd: str
    data_parallel_size: int
    distributed_hosts: List[str]
    ddp_master_addr: Optional[str]
    nccl_socket_ifname: Optional[str]
    transfer: MachineTransferInfo


@dataclass(frozen=True)
class StandaloneBenchmarkConfig:
    enabled: bool
    machine_keys: List[str]
    run_id_suffix: str
    steps: int
    max_steps: int
    max_length: int
    per_device_batch_size: int
    learning_rate: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    telemetry_interval_seconds: float
    model_id: str


@dataclass(frozen=True)
class PortabilitySettings:
    """Run-level configuration. Secrets (ssh_password) are deliberately
    NOT read from the committed config file -- see config/loader.py."""

    local_root: str
    remote_root: str
    source_host: str
    target_host: str
    checkpoint_step: int
    final_step: int
    run_id: str
    model_id: str
    dataset_path: str
    ssh_password: Optional[str]
    asset_paths: List[str]
    command_retries: int
    min_extra_steps: int
    max_loss_delta: float
    max_optimizer_norm_delta: float
    source_use_gpu: bool
    target_use_gpu: bool
    source_data_parallel_size: int
    source_python_cmd: str
    target_python_cmd: str
    host_python_cmd_overrides: Dict[str, str]
    electricity_price_gbp_per_kwh: float
    revenue_per_million_tokens_gbp: Optional[float]
    minimum_telemetry_coverage_percent: float
    maximum_energy_consistency_error_percent: float
    minimum_power_samples: int
    recommended_minimum_energy_benchmark_seconds: float
    source_system_overhead_watts: Optional[float]
    target_system_overhead_watts: Optional[float]
    training_telemetry_interval_seconds: float
    transfer: TransferSettings
    machine_catalog: Dict[str, MachineSpec]
    standalone_benchmark: StandaloneBenchmarkConfig
    ssh_connect_timeout_seconds: int
    command_timeout_seconds: float
    torchrun_timeout_seconds: float
    rendezvous_timeout_seconds: int
    process_group_timeout_seconds: int
    transfer_timeout_seconds: float

    def machine_for_host(self, host: str) -> Optional[MachineSpec]:
        return next((m for m in self.machine_catalog.values() if m.hostname == host), None)
