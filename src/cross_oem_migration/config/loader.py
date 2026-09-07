"""Loads PortabilitySettings + the hardware catalog from configs/*.json.

Direct port of the parsing logic in the original `finetuning_sequential.py`
(load_settings / machine_catalog / machine_spec / standalone_benchmark_config
/ build_transfer_settings, originally lines 230-421), split out so it can be
unit tested without importing Dagster or subprocess.

SECURITY FIX vs. the original repo: `repo_config.json` used to contain
`"ssh_password": "michael"` in plaintext, committed to git, and repeated in
the README. That is not acceptable for anything beyond a throwaway local
demo. This loader now reads the password (if any) from the
CROSS_OEM_SSH_PASSWORD environment variable, or from an SSH key (preferred:
leave ssh_password unset and rely on `~/.ssh/config` + agent auth). See
.env.example.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .models import (
    MachineSpec,
    MachineTransferInfo,
    MooncakeTransferConfig,
    PortabilitySettings,
    StandaloneBenchmarkConfig,
    TransferNodeConfig,
    TransferSettings,
)

DEFAULT_SSH_CONNECT_TIMEOUT_SECONDS = 15
DEFAULT_COMMAND_TIMEOUT_SECONDS = 600.0
DEFAULT_TORCHRUN_TIMEOUT_SECONDS = 1800.0
DEFAULT_RDZV_TIMEOUT_SECONDS = 180
DEFAULT_PROCESS_GROUP_TIMEOUT_SECONDS = 300
DEFAULT_TRANSFER_TIMEOUT_SECONDS = 1200.0

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


def nullable(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in {"", "none", "null"} else text


def as_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_json(name: str) -> dict:
    path = CONFIG_DIR / name
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_settings(config_dir: Optional[Path] = None) -> PortabilitySettings:
    """Build settings from configs/run.json + configs/machines.json,
    overlaid with environment variables for anything secret."""
    global CONFIG_DIR
    if config_dir is not None:
        CONFIG_DIR = config_dir

    run = _read_json("run.json")
    common = run["common"]
    basic = run["finetuning_sequential"]

    dataset_path = str(basic.get("dataset_path", "data/quant_mentor_500_alpaca.jsonl"))
    source_host = str(basic["source_host"])
    target_host = str(basic["target_host"])
    host_python_cmd_overrides = {str(k): str(v) for k, v in dict(basic.get("host_python_cmd_overrides", {})).items()}
    revenue = as_float(basic.get("revenue_per_million_tokens_gbp"))
    transfer = _build_transfer_settings(basic)
    catalog = load_machine_catalog(basic)
    standalone = _standalone_benchmark_config(run, basic, catalog)

    ssh_password = os.environ.get("CROSS_OEM_SSH_PASSWORD") or nullable(basic.get("ssh_password"))

    return PortabilitySettings(
        local_root=str(common["local_root_dir"]),
        remote_root=str(common["remote_root_dir"]),
        source_host=source_host,
        target_host=target_host,
        checkpoint_step=int(basic["checkpoint_step"]),
        final_step=int(basic["final_step"]),
        run_id=str(basic["run_id"]),
        model_id=str(basic.get("model_id", "Qwen/Qwen2.5-1.5B")),
        dataset_path=dataset_path,
        ssh_password=ssh_password,
        asset_paths=list(basic["asset_paths"]),
        command_retries=int(basic["command_retries"]),
        min_extra_steps=int(basic["min_extra_steps"]),
        max_loss_delta=float(basic["max_loss_delta"]),
        max_optimizer_norm_delta=float(basic.get("max_optimizer_norm_delta", 1e-4)),
        source_use_gpu=bool(basic.get("source_use_gpu", True)),
        target_use_gpu=bool(basic.get("target_use_gpu", True)),
        source_data_parallel_size=int(basic.get("source_data_parallel_size", 1)),
        source_python_cmd=str(basic.get("source_python_cmd", "/usr/bin/python3.12")),
        target_python_cmd=str(basic.get("target_python_cmd", "/usr/bin/python3.12")),
        host_python_cmd_overrides=host_python_cmd_overrides,
        electricity_price_gbp_per_kwh=float(basic.get("electricity_price_gbp_per_kwh", 0.27)),
        revenue_per_million_tokens_gbp=None if revenue is None else float(revenue),
        minimum_telemetry_coverage_percent=float(basic.get("minimum_telemetry_coverage_percent", 90.0)),
        maximum_energy_consistency_error_percent=float(basic.get("maximum_energy_consistency_error_percent", 10.0)),
        minimum_power_samples=int(basic.get("minimum_power_samples", 10)),
        recommended_minimum_energy_benchmark_seconds=float(basic.get("recommended_minimum_energy_benchmark_seconds", 30.0)),
        source_system_overhead_watts=as_float(basic.get("source_system_overhead_watts")),
        target_system_overhead_watts=as_float(basic.get("target_system_overhead_watts")),
        training_telemetry_interval_seconds=float(basic.get("training_telemetry_interval_seconds", 0.25)),
        transfer=transfer,
        machine_catalog=catalog,
        standalone_benchmark=standalone,
        ssh_connect_timeout_seconds=int(basic.get("ssh_connect_timeout_seconds", DEFAULT_SSH_CONNECT_TIMEOUT_SECONDS)),
        command_timeout_seconds=float(basic.get("command_timeout_seconds", DEFAULT_COMMAND_TIMEOUT_SECONDS)),
        torchrun_timeout_seconds=float(basic.get("torchrun_timeout_seconds", DEFAULT_TORCHRUN_TIMEOUT_SECONDS)),
        rendezvous_timeout_seconds=int(basic.get("rendezvous_timeout_seconds", DEFAULT_RDZV_TIMEOUT_SECONDS)),
        process_group_timeout_seconds=int(basic.get("process_group_timeout_seconds", DEFAULT_PROCESS_GROUP_TIMEOUT_SECONDS)),
        transfer_timeout_seconds=float(basic.get("transfer_timeout_seconds", DEFAULT_TRANSFER_TIMEOUT_SECONDS)),
    )


def load_machine_catalog(basic: Optional[Dict[str, Any]] = None) -> Dict[str, MachineSpec]:
    """Hardware catalog now lives in its own file (configs/machines.json)
    instead of being interleaved with run parameters -- this is the fix
    for ask #4 (abstract hardware-specific config away from the run config
    and away from vendor `if` statements in the pipeline code)."""
    section = _read_json("machines.json")
    return {key: _machine_spec(key, value) for key, value in section.items()}


def _machine_spec(key: str, payload: Dict[str, Any]) -> MachineSpec:
    transfer = payload.get("transfer", {}) if isinstance(payload, dict) else {}
    hostname = str(payload.get("hostname", "")).strip()
    raw_hosts = payload.get("distributed_hosts", []) if isinstance(payload.get("distributed_hosts", []), list) else []
    hosts = [str(item).strip() for item in raw_hosts if str(item).strip()]
    distributed_hosts = hosts if hosts else ([hostname] if hostname else [])
    return MachineSpec(
        key=key,
        hostname=hostname,
        vendor=str(payload.get("vendor", "unknown")).strip().lower(),
        device_name=str(payload.get("device_name", "unknown")).strip(),
        expected_gpu_architecture=str(payload.get("expected_gpu_architecture", "unknown")).strip(),
        container_image=nullable(payload.get("container_image")),
        execution_mode=str(payload.get("execution_mode", "docker" if payload.get("container_image") else "host")).strip().lower(),
        torch_install_command=nullable(payload.get("torch_install_command")),
        hf_cache_dir=nullable(payload.get("hf_cache_dir")),
        rocm_path=nullable(payload.get("rocm_path")),
        electricity_price_per_kwh=float(payload.get("electricity_price_per_kwh", 0.27)),
        currency=str(payload.get("currency", "GBP")).strip().upper(),
        python_cmd=str(payload.get("python_cmd", "/usr/bin/python3.12")).strip(),
        data_parallel_size=int(payload.get("data_parallel_size", 1)),
        distributed_hosts=distributed_hosts,
        ddp_master_addr=nullable(payload.get("ddp_master_addr")),
        nccl_socket_ifname=nullable(payload.get("nccl_socket_ifname")),
        transfer=MachineTransferInfo(nullable(transfer.get("control_host")), as_int(transfer.get("control_port"))),
    )


def _standalone_benchmark_config(run: Dict[str, Any], basic: Dict[str, Any], catalog: Dict[str, MachineSpec]) -> StandaloneBenchmarkConfig:
    section = dict(run.get("standalone_benchmark", {}))
    machine_keys = [str(item) for item in section.get("machine_keys", []) if str(item) in catalog]
    default_steps = int(section.get("steps", int(basic.get("checkpoint_step", 10))))
    return StandaloneBenchmarkConfig(
        enabled=bool(section.get("enabled", True)),
        machine_keys=machine_keys,
        run_id_suffix=str(section.get("run_id_suffix", "baseline")),
        steps=default_steps,
        max_steps=int(section.get("max_steps", default_steps)),
        max_length=int(section.get("max_length", 128)),
        per_device_batch_size=int(section.get("per_device_batch_size", 1)),
        learning_rate=float(section.get("learning_rate", 2e-4)),
        lora_r=int(section.get("lora_r", 8)),
        lora_alpha=int(section.get("lora_alpha", 16)),
        lora_dropout=float(section.get("lora_dropout", 0.05)),
        telemetry_interval_seconds=float(section.get("telemetry_interval_seconds", basic.get("training_telemetry_interval_seconds", 0.25))),
        model_id=str(section.get("model_id", "Qwen/Qwen2.5-7B")),
    )


def _build_transfer_settings(basic: dict) -> TransferSettings:
    backend = str(basic.get("transfer_backend", "scp"))
    mooncake = _mooncake_transfer(basic.get("transfer"))
    return TransferSettings(backend, mooncake)


def _mooncake_transfer(config: Optional[dict]) -> Optional[MooncakeTransferConfig]:
    if not config:
        return None
    return MooncakeTransferConfig(
        metadata_server=str(config.get("metadata_server", "P2PHANDSHAKE")),
        protocol=str(config.get("protocol", "tcp")),
        python_cmd=str(config.get("python_cmd", "/usr/bin/python3.12")),
        chunk_mb=int(config.get("chunk_mb", 64)),
        max_inflight=int(config.get("max_inflight", 1)),
        source=_node_transfer(config, "source"),
        destination=_node_transfer(config, "destination"),
    )


def _node_transfer(config: dict, key: str) -> TransferNodeConfig:
    section = config.get(key)
    if not section:
        raise KeyError(f"transfer.{key} missing in configs/run.json")
    return TransferNodeConfig(
        server_name=str(section["server_name"]),
        control_host=str(section["control_host"]),
        control_port=int(section["control_port"]),
        python_cmd=str(section.get("python_cmd", "")),
    )
