from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import os
import re
import shlex
import subprocess
import time

from dagster import AssetCheckResult, AssetKey, AssetMaterialization, In, MetadataValue, Nothing, OpExecutionContext, Out, TableColumn, TableSchema, asset_check, job, op


@dataclass
class PortabilitySettings:
    local_root: str
    remote_root: str
    source_host: str
    target_host: str
    checkpoint_step: int
    final_step: int
    run_id: str
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
    transfer: "TransferSettings"
    machine_catalog: Dict[str, "MachineSpec"]
    standalone_benchmark: "StandaloneBenchmarkConfig"


@dataclass
class TransferNodeConfig:
    server_name: str
    control_host: str
    control_port: int
    python_cmd: str


@dataclass
class MooncakeTransferConfig:
    metadata_server: str
    protocol: str
    python_cmd: str
    chunk_mb: int
    max_inflight: int
    source: TransferNodeConfig
    destination: TransferNodeConfig


@dataclass
class TransferSettings:
    backend: str
    mooncake: Optional[MooncakeTransferConfig]


@dataclass
class TransferResult:
    success: bool
    backend: str
    bytes_transferred: int
    elapsed_seconds: float
    throughput_bytes_per_second: float
    checksum_ok: bool
    resume_validation_ok: bool
    details: Dict[str, Any]


@dataclass
class MachineTransferInfo:
    control_host: Optional[str]
    control_port: Optional[int]


@dataclass
class MachineSpec:
    key: str
    hostname: str
    vendor: str
    device_name: str
    expected_gpu_architecture: str
    container_image: Optional[str]
    execution_mode: str
    torch_install_command: Optional[str]
    hf_cache_dir: Optional[str]
    rocm_path: Optional[str]
    electricity_price_per_kwh: float
    currency: str
    python_cmd: str
    transfer: MachineTransferInfo


@dataclass
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


PORTABILITY_SCHEMA = TableSchema(
    columns=[
        TableColumn(name="run_id", type="string"),
        TableColumn(name="source_host", type="string"),
        TableColumn(name="target_host", type="string"),
        TableColumn(name="checkpoint_step", type="int"),
        TableColumn(name="final_step", type="int"),
        TableColumn(name="status", type="string"),
        TableColumn(name="artifact_path", type="string"),
    ]
)

PUBLISHED_MANIFEST_KEY = AssetKey(["portability", "published", "assets_manifest"])
MIN_TORCH_VERSION = "2.6"
TORCH_SECURITY_CVE = "CVE-2025-32434"
REQUIRED_PHASE_FIELDS = [
    "hostname",
    "vendor",
    "device_name",
    "gpu_architecture",
    "torch_version",
    "gpu_requested",
    "gpu_detected",
    "gpu_kernel_test_passed",
    "gpu_execution_verified",
    "first_step",
    "last_step",
    "useful_training_tokens",
    "training_runtime_seconds",
    "tokens_per_second",
    "mean_gpu_utilization_percent",
    "peak_gpu_utilization_percent",
    "peak_accelerator_memory_bytes",
    "mean_gpu_power_watts",
    "peak_gpu_power_watts",
    "gpu_energy_joules",
    "gpu_energy_kwh",
    "tokens_per_joule",
    "energy_cost_gbp",
    "energy_cost_per_million_tokens_gbp",
    "energy_break_even_gbp_per_million_tokens",
    "final_loss",
]
SOURCE_PHASE_REQUIRED_FIELDS = [*REQUIRED_PHASE_FIELDS, "checkpoint_size_bytes", "checkpoint_save_seconds"]
DESTINATION_PHASE_REQUIRED_FIELDS = [*REQUIRED_PHASE_FIELDS, "checkpoint_load_seconds", "resume_to_first_step_seconds"]


def load_settings() -> PortabilitySettings:
    config = repo_config()
    common = config["common"]
    basic = config["basic_portability"]
    dataset_path = str(basic.get("dataset_path", "data/story3_dataset.jsonl"))
    source_use_gpu = bool(basic.get("source_use_gpu", basic.get("amd_use_gpu", True)))
    target_use_gpu = bool(basic.get("target_use_gpu", True))
    source_host = str(basic["source_host"]) if "source_host" in basic else str(basic["amd_host"])
    target_host = str(basic["target_host"]) if "target_host" in basic else str(basic["nvidia_host"])
    source_python_cmd = str(basic.get("source_python_cmd", "/usr/bin/python3.12"))
    target_python_cmd = str(basic.get("target_python_cmd", "/usr/bin/python3.12"))
    host_python_cmd_overrides = {str(key): str(value) for key, value in dict(basic.get("host_python_cmd_overrides", {})).items()}
    revenue = nullable(basic.get("revenue_per_million_tokens_gbp"))
    transfer = build_transfer_settings(basic)
    catalog = machine_catalog(config, basic)
    standalone = standalone_benchmark_config(config, basic, catalog)
    return PortabilitySettings(
        str(common["local_root_dir"]),
        str(common["remote_root_dir"]),
        source_host,
        target_host,
        int(basic["checkpoint_step"]),
        int(basic["final_step"]),
        str(basic["run_id"]),
        dataset_path,
        nullable(basic["ssh_password"]),
        list(basic["asset_paths"]),
        int(basic["command_retries"]),
        int(basic["min_extra_steps"]),
        float(basic["max_loss_delta"]),
        float(basic.get("max_optimizer_norm_delta", 1e-4)),
        source_use_gpu,
        target_use_gpu,
        int(basic.get("source_data_parallel_size", 1)),
        source_python_cmd,
        target_python_cmd,
        host_python_cmd_overrides,
        float(basic.get("electricity_price_gbp_per_kwh", 0.27)),
        None if revenue is None else float(revenue),
        float(basic.get("minimum_telemetry_coverage_percent", 90.0)),
        float(basic.get("maximum_energy_consistency_error_percent", 10.0)),
        int(basic.get("minimum_power_samples", 10)),
        float(basic.get("recommended_minimum_energy_benchmark_seconds", 30.0)),
        as_float(basic.get("source_system_overhead_watts")),
        as_float(basic.get("target_system_overhead_watts")),
        float(basic.get("training_telemetry_interval_seconds", 0.25)),
        transfer,
        catalog,
        standalone,
    )


def machine_catalog(config: Dict[str, Any], basic: Dict[str, Any]) -> Dict[str, MachineSpec]:
    section = dict(config.get("machine_catalog", {}))
    defaults = default_machine_catalog(basic)
    payload = {**defaults, **section}
    return {key: machine_spec(key, value) for key, value in payload.items()}


def default_machine_catalog(basic: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        "amd_existing": {
            "hostname": str(basic.get("source_host", basic.get("amd_host", ""))),
            "vendor": "amd",
            "device_name": "unknown",
            "expected_gpu_architecture": "unknown",
            "container_image": None,
            "electricity_price_per_kwh": float(basic.get("electricity_price_gbp_per_kwh", 0.27)),
            "currency": "GBP",
            "python_cmd": str(basic.get("source_python_cmd", "/usr/bin/python3.12")),
            "transfer": {
                "control_host": str(basic.get("transfer", {}).get("source", {}).get("control_host", "")) or None,
                "control_port": as_int(basic.get("transfer", {}).get("source", {}).get("control_port")),
            },
        },
        "nvidia_control": {
            "hostname": str(basic.get("target_host", basic.get("nvidia_host", ""))),
            "vendor": "nvidia",
            "device_name": "unknown",
            "expected_gpu_architecture": "unknown",
            "container_image": None,
            "electricity_price_per_kwh": float(basic.get("electricity_price_gbp_per_kwh", 0.27)),
            "currency": "GBP",
            "python_cmd": str(basic.get("target_python_cmd", "/usr/bin/python3.12")),
            "transfer": {
                "control_host": str(basic.get("transfer", {}).get("destination", {}).get("control_host", "")) or None,
                "control_port": as_int(basic.get("transfer", {}).get("destination", {}).get("control_port")),
            },
        },
    }


def machine_spec(key: str, payload: Dict[str, Any]) -> MachineSpec:
    transfer = payload.get("transfer", {}) if isinstance(payload, dict) else {}
    return MachineSpec(
        key,
        str(payload.get("hostname", "")).strip(),
        str(payload.get("vendor", "unknown")).strip().lower(),
        str(payload.get("device_name", "unknown")).strip(),
        str(payload.get("expected_gpu_architecture", "unknown")).strip(),
        nullable(payload.get("container_image")),
        str(payload.get("execution_mode", "docker" if payload.get("container_image") else "host")).strip().lower(),
        nullable(payload.get("torch_install_command")),
        nullable(payload.get("hf_cache_dir")),
        nullable(payload.get("rocm_path")),
        float(payload.get("electricity_price_per_kwh", payload.get("electricity_price_gbp_per_kwh", 0.27))),
        str(payload.get("currency", "GBP")).strip().upper(),
        str(payload.get("python_cmd", "/usr/bin/python3.12")).strip(),
        MachineTransferInfo(nullable(transfer.get("control_host")), as_int(transfer.get("control_port"))),
    )


def standalone_benchmark_config(config: Dict[str, Any], basic: Dict[str, Any], catalog: Dict[str, MachineSpec]) -> StandaloneBenchmarkConfig:
    section = dict(config.get("standalone_benchmark", {}))
    machine_keys = [str(item) for item in section.get("machine_keys", ["nvidia_control", "amd_existing"]) if str(item) in catalog]
    default_steps = int(section.get("steps", int(basic.get("checkpoint_step", 10))))
    return StandaloneBenchmarkConfig(
        bool(section.get("enabled", True)),
        machine_keys,
        str(section.get("run_id_suffix", "baseline")),
        default_steps,
        int(section.get("max_steps", default_steps)),
        int(section.get("max_length", 128)),
        int(section.get("per_device_batch_size", 1)),
        float(section.get("learning_rate", 2e-4)),
        int(section.get("lora_r", 8)),
        int(section.get("lora_alpha", 16)),
        float(section.get("lora_dropout", 0.05)),
        float(section.get("telemetry_interval_seconds", basic.get("training_telemetry_interval_seconds", 0.25))),
        str(section.get("model_id", "Qwen/Qwen2.5-7B")),
    )


def build_transfer_settings(basic: dict) -> TransferSettings:
    backend = str(basic.get("transfer_backend", "scp"))
    mooncake = mooncake_transfer(basic.get("transfer"))
    return TransferSettings(backend, mooncake)


def mooncake_transfer(config: Optional[dict]) -> Optional[MooncakeTransferConfig]:
    if not config:
        return None
    source = node_transfer(config, "source")
    destination = node_transfer(config, "destination")
    return MooncakeTransferConfig(
        str(config.get("metadata_server", "P2PHANDSHAKE")),
        str(config.get("protocol", "tcp")),
        str(config.get("python_cmd", "/usr/bin/python3.12")),
        int(config.get("chunk_mb", 64)),
        int(config.get("max_inflight", 1)),
        source,
        destination,
    )


def node_transfer(config: dict, key: str) -> TransferNodeConfig:
    section = config.get(key)
    if not section:
        raise KeyError(f"transfer.{key} missing in repo_config.json")
    return TransferNodeConfig(
        str(section["server_name"]),
        str(section["control_host"]),
        int(section["control_port"]),
        str(section.get("python_cmd", "")),
    )


def repo_config() -> dict:
    path = Path(__file__).with_name("repo_config.json")
    return json.loads(path.read_text())


def nullable(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in {"", "none", "null"} else text


def sshpass_env(settings: PortabilitySettings) -> Optional[dict]:
    return {**os.environ, "SSHPASS": settings.ssh_password} if settings.ssh_password else None


def ssh_options(settings: PortabilitySettings) -> List[str]:
    batch = "no" if settings.ssh_password else "yes"
    base = ["-o", "IdentitiesOnly=yes", "-o", f"BatchMode={batch}", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=120"]
    auth = ["-o", "PreferredAuthentications=publickey,password,keyboard-interactive", "-o", "NumberOfPasswordPrompts=1"]
    return base + auth if settings.ssh_password else base


def run_shell(command: List[str], retries: int, env: Optional[dict] = None, allow_failure: bool = False) -> subprocess.CompletedProcess:
    for attempt in range(retries + 1):
        result = subprocess.run(command, check=False, text=True, capture_output=True, env=env)
        if result.returncode == 0 or allow_failure:
            return result
        if attempt == retries:
            joined = " ".join(shlex.quote(part) for part in command)
            raise RuntimeError(f"command failed: {joined}\nexit={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}")
        if result.returncode != 0:
            time.sleep(2)
    return subprocess.CompletedProcess(command, 1)


def ssh_cmd(settings: PortabilitySettings, host: str, command: str) -> List[str]:
    return (["sshpass", "-e"] if settings.ssh_password else []) + ["ssh", *ssh_options(settings), host, command]


def scp_cmd(settings: PortabilitySettings, source: str, target: str, recursive: bool = False) -> List[str]:
    return (["sshpass", "-e"] if settings.ssh_password else []) + ["scp", *ssh_options(settings), *( ["-r"] if recursive else []), source, target]


def remote_ckpt(settings: PortabilitySettings) -> str:
    return f"checkpoint-{settings.checkpoint_step}"


def pip_bootstrap(settings: PortabilitySettings, python_cmd: str) -> str:
    check = f"{shlex.quote(python_cmd)} -m pip --version >/dev/null 2>&1"
    install = f"(printf %s {shlex.quote(settings.ssh_password)} | sudo -S -k apt-get update && printf %s {shlex.quote(settings.ssh_password)} | sudo -S -k apt-get install -y python3-pip)"
    return f"{check} || true" if not settings.ssh_password else f"{check} || {install}"


def local_asset_path(settings: PortabilitySettings, asset_path: str) -> Path:
    return Path(settings.local_root) / asset_path


def remote_asset_path(settings: PortabilitySettings, asset_path: str) -> str:
    return f"{settings.remote_root}/{asset_path}"


def normalized_asset_path(path: str) -> str:
    return path.lstrip("./") if not path.startswith(("/", "~/")) else path


def remote_dataset_path(settings: PortabilitySettings) -> str:
    path = normalized_asset_path(settings.dataset_path)
    return path if path.startswith(("/", "~/")) else f"{settings.remote_root}/{path}"


def expanded_remote_root(settings: PortabilitySettings, host: str) -> str:
    if not settings.remote_root.startswith("~/"):
        return settings.remote_root
    username = host.split("@", 1)[0] if "@" in host else ""
    if username:
        return f"/home/{username}/{settings.remote_root[2:]}"
    return f"/root/{settings.remote_root[2:]}"


def sync_paths(settings: PortabilitySettings) -> List[str]:
    merged = [*settings.asset_paths, settings.dataset_path]
    return list(dict.fromkeys(normalized_asset_path(path) for path in merged))


def sync_asset_to_host(settings: PortabilitySettings, host: str, asset_path: str) -> None:
    local_path = local_asset_path(settings, asset_path)
    if not local_path.exists():
        return
    mkdir_cmd = f"mkdir -p {settings.remote_root}/{asset_path}" if local_path.is_dir() else f"mkdir -p {settings.remote_root}/{Path(asset_path).parent}"
    run_shell(ssh_cmd(settings, host, mkdir_cmd), settings.command_retries, env=sshpass_env(settings))
    run_shell(scp_cmd(settings, str(local_path), f"{host}:{settings.remote_root}/{Path(asset_path).parent}", recursive=True), settings.command_retries, env=sshpass_env(settings))


def remote_exists(settings: PortabilitySettings, host: str, remote_path: str) -> bool:
    result = run_shell(ssh_cmd(settings, host, f"test -e {remote_path}"), settings.command_retries, env=sshpass_env(settings), allow_failure=True)
    return result.returncode == 0


def save_asset_from_host(settings: PortabilitySettings, host: str, asset_path: str) -> None:
    local_target = saved_local_asset_path(settings, asset_path)
    local_target.parent.mkdir(parents=True, exist_ok=True)
    remote_path = remote_asset_path(settings, asset_path)
    if not remote_exists(settings, host, remote_path):
        return
    recursive = remote_is_dir(settings, host, remote_path)
    target = str(local_target.parent if recursive else local_target)
    run_shell(scp_cmd(settings, f"{host}:{remote_path}", target, recursive=recursive), settings.command_retries, env=sshpass_env(settings))


def remote_is_dir(settings: PortabilitySettings, host: str, remote_path: str) -> bool:
    result = run_shell(ssh_cmd(settings, host, f"test -d {remote_path}"), settings.command_retries, env=sshpass_env(settings), allow_failure=True)
    return result.returncode == 0


def saved_local_asset_path(settings: PortabilitySettings, asset_path: str) -> Path:
    return Path(settings.local_root) / "artifacts" / settings.run_id / normalized_asset_path(asset_path)


def output_asset_paths(settings: PortabilitySettings) -> List[str]:
    run = f"nvidia_run_{settings.run_id}"
    source = f"amd_run_{settings.run_id}"
    transfer = f"{run}/transfer_tests"
    defaults = [
        f"{source}/run_summary.json",
        f"{source}/gpu_telemetry_raw.json",
        f"{run}/gpu_telemetry_raw.json",
        f"{run}/run_summary.json",
        f"{run}/resume_validation_proof.json",
        f"{run}/checkpoint-{settings.final_step}/trainer_state.json",
        f"{source}/checkpoint-{settings.checkpoint_step}/checkpoint_meta.json",
        f"{transfer}/iperf3_max_tcp_capability.json",
        f"{transfer}/scp_rsync_file_transfer_baseline.json",
        f"{transfer}/mooncake_tcp_application_path.json",
        f"{transfer}/transfer_test_matrix.json",
    ]
    return list(dict.fromkeys([*settings.asset_paths, *defaults]))


def transfer_tests_dir(settings: PortabilitySettings) -> str:
    return f"{expanded_remote_root(settings, settings.target_host)}/nvidia_run_{settings.run_id}/transfer_tests"


def transfer_test_payload(name: str, path_label: str, artifact_path: str, details: Dict[str, Any]) -> Dict[str, Any]:
    return {"test": name, "path": path_label, "artifact_path": artifact_path, "details": details}


def write_remote_json(settings: PortabilitySettings, host: str, remote_path: str, payload: Dict[str, Any]) -> None:
    python_cmd = shlex.quote(execution_python_cmd(settings, host))
    payload_text = json.dumps(payload, indent=2)
    script = "import pathlib; p=pathlib.Path(%r).expanduser(); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(%r)" % (remote_path, payload_text)
    run_shell(ssh_cmd(settings, host, f"{python_cmd} -c {shlex.quote(script)}"), settings.command_retries, env=sshpass_env(settings))


def transfer_test_artifacts(settings: PortabilitySettings, transfer_result: TransferResult, iperf3_details: Optional[Dict[str, Any]] = None, rsync_details: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    root = transfer_tests_dir(settings)
    iperf3_payload = iperf3_details if iperf3_details is not None else run_iperf3_baseline(settings)
    rsync_payload = rsync_details if rsync_details is not None else run_rsync_baseline(settings)
    mooncake = transfer_test_payload(
        "Mooncake TCP",
        "Mooncake application path",
        f"{root}/mooncake_tcp_application_path.json",
        {"backend": transfer_result.backend, "success": transfer_result.success, "bytes_transferred": transfer_result.bytes_transferred, "elapsed_seconds": transfer_result.elapsed_seconds, "throughput_bytes_per_second": transfer_result.throughput_bytes_per_second, "manifest_path": str(transfer_result.details.get("manifest_path", ""))},
    )
    iperf3 = transfer_test_payload("iperf3", "Maximum TCP capability", f"{root}/iperf3_max_tcp_capability.json", iperf3_payload)
    baseline = transfer_test_payload("SCP/rsync", "File-transfer baseline", f"{root}/scp_rsync_file_transfer_baseline.json", rsync_payload)
    return [iperf3, baseline, mooncake]


def compact_error(stdout: str, stderr: str) -> str:
    text = stderr.strip() or stdout.strip() or "command produced no output"
    return text if len(text) <= 400 else f"{text[:400]}..."


def extract_json_blob(text: str) -> Optional[Dict[str, Any]]:
    matches = re.findall(r"\{.*\}", text, flags=re.DOTALL)
    return json.loads(matches[-1]) if matches else None


def iperf_target_host(settings: PortabilitySettings) -> str:
    mooncake = settings.transfer.mooncake
    if mooncake and mooncake.destination.control_host.strip() not in {"", "0.0.0.0", "127.0.0.1", "localhost"}:
        return mooncake.destination.control_host.strip()
    return settings.target_host.split("@", 1)[-1]


def iperf_seconds(payload: Dict[str, Any]) -> float:
    end = payload.get("end", {})
    segment = end.get("sum_received") or end.get("sum_sent") or {}
    return float(segment.get("seconds", 0.0))


def iperf_bits_per_second(payload: Dict[str, Any]) -> float:
    end = payload.get("end", {})
    segment = end.get("sum_received") or end.get("sum_sent") or {}
    return float(segment.get("bits_per_second", 0.0))


def run_iperf3_baseline(settings: PortabilitySettings) -> Dict[str, Any]:
    session = f"{settings.run_id}_{settings.checkpoint_step}_{int(time.time() * 1000)}"
    pid_path = f"/tmp/iperf3_server_{session}.pid"
    start_cmd = f"nohup iperf3 -s -1 -J >/tmp/iperf3_server_{session}.json 2>&1 & echo $! > {shlex.quote(pid_path)}"
    run_shell(ssh_cmd(settings, settings.target_host, start_cmd), settings.command_retries, env=sshpass_env(settings))
    return run_iperf3_client(settings, pid_path)


def run_iperf3_client(settings: PortabilitySettings, pid_path: str) -> Dict[str, Any]:
    try:
        time.sleep(1)
        host = iperf_target_host(settings)
        command = f"iperf3 -c {shlex.quote(host)} -J -t 10"
        output = run_shell(ssh_cmd(settings, settings.source_host, command), 0, env=sshpass_env(settings), allow_failure=True)
        return parse_iperf3_result(output)
    except Exception as exc:
        return {"success": False, "error": str(exc)}
    finally:
        cleanup = f"test -f {shlex.quote(pid_path)} && kill $(cat {shlex.quote(pid_path)}) >/dev/null 2>&1 || true"
        run_shell(ssh_cmd(settings, settings.target_host, cleanup), settings.command_retries, env=sshpass_env(settings), allow_failure=True)


def parse_iperf3_result(output: subprocess.CompletedProcess) -> Dict[str, Any]:
    if output.returncode != 0:
        return {"success": False, "error": compact_error(output.stdout, output.stderr)}
    payload = extract_json_blob(output.stdout)
    if payload is None:
        return {"success": False, "error": "iperf3 output did not contain JSON"}
    return {"success": True, "elapsed_seconds": iperf_seconds(payload), "throughput_bits_per_second": iperf_bits_per_second(payload)}


def source_checkpoint_dir(settings: PortabilitySettings) -> str:
    return f"{expanded_remote_root(settings, settings.source_host)}/amd_run_{settings.run_id}/{remote_ckpt(settings)}"


def rsync_remote_shell(settings: PortabilitySettings) -> str:
    return " ".join(shlex.quote(part) for part in ["ssh", *ssh_options(settings)])


def ssh_host_name(host: str) -> str:
    return host.split("@", 1)[-1]


def rsync_known_hosts_command(target_host: str) -> str:
    known_hosts = '"$HOME/.ssh/known_hosts"'
    target = shlex.quote(ssh_host_name(target_host))
    return f'mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh" && touch {known_hosts} && chmod 600 {known_hosts} && (ssh-keygen -F {target} -f {known_hosts} >/dev/null || ssh-keyscan -H {target} >> {known_hosts})'


def ensure_rsync_ssh_ready(settings: PortabilitySettings) -> None:
    command = rsync_known_hosts_command(settings.target_host)
    run_shell(ssh_cmd(settings, settings.source_host, command), settings.command_retries, env=sshpass_env(settings))


def rsync_source_command(settings: PortabilitySettings, source_ckpt: str, destination_tmp: str) -> str:
    source_path = shlex.quote(f"{source_ckpt}/")
    destination_path = shlex.quote(f"{settings.target_host}:{destination_tmp}/")
    remote_shell = shlex.quote(rsync_remote_shell(settings))
    return f"rsync -a --stats -e {remote_shell} {source_path} {destination_path}"


def parse_rsync_transferred_bytes(stdout: str) -> Optional[int]:
    match = re.search(r"Total transferred file size:\s*([0-9,]+)\s*bytes", stdout)
    return int(match.group(1).replace(",", "")) if match else None


def run_rsync_baseline(settings: PortabilitySettings) -> Dict[str, Any]:
    source_ckpt = source_checkpoint_dir(settings)
    stamp = f"{settings.run_id}_{settings.checkpoint_step}_{int(time.time() * 1000)}"
    destination_tmp = f"{transfer_tests_dir(settings)}/rsync_benchmark_{stamp}"
    return execute_rsync_baseline(settings, source_ckpt, destination_tmp)


def execute_rsync_baseline(settings: PortabilitySettings, source_ckpt: str, destination_tmp: str) -> Dict[str, Any]:
    try:
        run_shell(ssh_cmd(settings, settings.target_host, f"mkdir -p {shlex.quote(destination_tmp)}"), settings.command_retries, env=sshpass_env(settings))
        ensure_rsync_ssh_ready(settings)
        start = time.perf_counter()
        rsync_cmd = rsync_source_command(settings, source_ckpt, destination_tmp)
        output = run_shell(ssh_cmd(settings, settings.source_host, rsync_cmd), 0, env=sshpass_env(settings), allow_failure=True)
        elapsed = time.perf_counter() - start
        return parse_rsync_result(output, elapsed)
    except Exception as exc:
        return {"success": False, "error": str(exc)}
    finally:
        run_shell(ssh_cmd(settings, settings.target_host, f"rm -rf {shlex.quote(destination_tmp)}"), settings.command_retries, env=sshpass_env(settings), allow_failure=True)


def parse_rsync_result(output: subprocess.CompletedProcess, elapsed: float) -> Dict[str, Any]:
    if output.returncode != 0:
        return {"success": False, "error": compact_error(output.stdout, output.stderr)}
    bytes_transferred = parse_rsync_transferred_bytes(output.stdout)
    if bytes_transferred is None:
        return {"success": False, "error": "rsync stats missing transferred bytes"}
    throughput = 0.0 if elapsed <= 0 else float(bytes_transferred) / elapsed
    return {"success": True, "bytes_transferred": int(bytes_transferred), "elapsed_seconds": elapsed, "throughput_bytes_per_second": throughput}


def throughput_percent(numerator: float, denominator: float) -> Optional[float]:
    return None if denominator <= 0 else round((numerator / denominator) * 100.0, 2)


def transfer_test_comparison(records: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    mooncake = next((item for item in records if item["test"] == "Mooncake TCP"), None)
    iperf3 = next((item for item in records if item["test"] == "iperf3"), None)
    rsync = next((item for item in records if item["test"] == "SCP/rsync"), None)
    mooncake_tp = float((mooncake or {}).get("details", {}).get("throughput_bytes_per_second", 0.0))
    iperf_tp = float((iperf3 or {}).get("details", {}).get("throughput_bits_per_second", 0.0)) / 8.0
    rsync_tp = float((rsync or {}).get("details", {}).get("throughput_bytes_per_second", 0.0))
    return {"mooncake_vs_iperf_percent": throughput_percent(mooncake_tp, iperf_tp), "mooncake_vs_rsync_percent": throughput_percent(mooncake_tp, rsync_tp)}


def as_float(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> Optional[int]:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def energy_cost_gbp(energy_kwh: Optional[float], price: float) -> Optional[float]:
    return None if energy_kwh is None else energy_kwh * price


def cost_per_million_tokens(cost: Optional[float], useful_tokens: Optional[int]) -> Optional[float]:
    if cost is None or useful_tokens is None or useful_tokens <= 0:
        return None
    return cost / useful_tokens * 1_000_000.0


def extract_transfer_record(payload: Dict[str, Any], name: str) -> Dict[str, Any]:
    tests = payload.get("tests", []) if isinstance(payload, dict) else []
    return next((item for item in tests if item.get("test") == name), {})


def migration_transfer_seconds(settings: PortabilitySettings, transfer_payload: Dict[str, Any], transfer_result: TransferResult) -> Optional[float]:
    mooncake_seconds = as_float(extract_transfer_record(transfer_payload, "Mooncake TCP").get("details", {}).get("elapsed_seconds"))
    rsync_seconds = as_float(extract_transfer_record(transfer_payload, "SCP/rsync").get("details", {}).get("elapsed_seconds"))
    if settings.transfer.backend == "mooncake_tcp":
        return mooncake_seconds
    if settings.transfer.backend == "scp":
        return as_float(transfer_result.elapsed_seconds)
    return rsync_seconds


def normalize_machine_summary(settings: PortabilitySettings, summary: Dict[str, Any], preflight: Dict[str, Any], gpu_requested: bool, role: str) -> Dict[str, Any]:
    useful_tokens = as_int(summary.get("useful_training_tokens"))
    runtime_seconds = as_float(summary.get("runtime_seconds"))
    gpu_energy_joules = as_float(summary.get("gpu_energy_joules"))
    gpu_energy_kwh = as_float(summary.get("gpu_energy_kwh"))
    tokens_per_second = as_float(summary.get("tokens_per_second")) or safe_ratio(as_float(useful_tokens), runtime_seconds)
    tokens_per_joule = as_float(summary.get("tokens_per_joule")) or safe_ratio(as_float(useful_tokens), gpu_energy_joules)
    energy_cost = energy_cost_gbp(gpu_energy_kwh, settings.electricity_price_gbp_per_kwh)
    cost_per_m = cost_per_million_tokens(energy_cost, useful_tokens)
    kernel_test = as_bool(preflight.get("kernel_execution_test"))
    gpu_detected = as_bool(preflight.get("gpu_detected"))
    runtime_verified = as_bool(summary.get("gpu_runtime_verified"))
    gpu_execution_verified = bool(gpu_requested and kernel_test and gpu_detected and runtime_verified is not False) if gpu_requested else bool(runtime_verified)
    vendor = str(summary.get("vendor") or detected_vendor_from_versions(summary) or preflight_vendor(preflight) or "").strip().lower() or None
    first_step = phase_first_step(summary)
    payload = {
        "hostname": summary.get("hostname") or preflight.get("host") or role,
        "vendor": vendor,
        "device_name": summary.get("device") or preflight.get("device_name"),
        "gpu_architecture": summary.get("gpu_architecture") or preflight.get("gpu_architecture"),
        "torch_version": summary.get("torch_version") or summary.get("torch"),
        "gpu_requested": gpu_requested,
        "gpu_detected": gpu_detected,
        "gpu_kernel_test_passed": kernel_test if gpu_requested else None,
        "gpu_execution_verified": gpu_execution_verified,
        "gpu_runtime_verified": runtime_verified,
        "first_step": first_step,
        "last_step": as_int(summary.get("saved_step")),
        "useful_training_tokens": useful_tokens,
        "training_runtime_seconds": runtime_seconds,
        "tokens_per_second": tokens_per_second,
        "mean_gpu_utilization_percent": as_float(summary.get("mean_gpu_utilization_percent")),
        "peak_gpu_utilization_percent": as_float(summary.get("peak_gpu_utilization_percent")),
        "peak_accelerator_memory_bytes": as_int(summary.get("peak_accelerator_memory_bytes")),
        "mean_gpu_power_watts": as_float(summary.get("mean_power_watts")),
        "peak_gpu_power_watts": as_float(summary.get("peak_power_watts")),
        "gpu_energy_joules": gpu_energy_joules,
        "gpu_energy_kwh": gpu_energy_kwh,
        "tokens_per_joule": tokens_per_joule,
        "energy_cost_gbp": energy_cost,
        "energy_cost_per_million_tokens_gbp": cost_per_m,
        "energy_break_even_gbp_per_million_tokens": cost_per_m,
        "gpu_energy_cost_gbp": energy_cost,
        "gpu_energy_cost_per_million_tokens_gbp": cost_per_m,
        "gpu_energy_break_even_per_million_tokens_gbp": cost_per_m,
        "system_energy_kwh": None,
        "system_energy_source": "not_measured",
        "final_loss": as_float(summary.get("final_loss")),
        "learning_rate": as_float(summary.get("learning_rate")),
        "optimizer_learning_rate_before_save": as_float(summary.get("optimizer_learning_rate_before_save")),
        "optimizer_learning_rate_after_resume_load": as_float(summary.get("optimizer_learning_rate_after_resume_load")),
        "optimizer_exp_avg_sq_norm_before_save": as_float(summary.get("optimizer_exp_avg_sq_norm_before_save")),
        "optimizer_exp_avg_sq_norm_after_resume_load": as_float(summary.get("optimizer_exp_avg_sq_norm_after_resume_load")),
        "loss_trace": summary.get("loss_trace", []),
        "telemetry_source": summary.get("telemetry_source"),
        "telemetry_errors": summary.get("telemetry_errors", []),
        "telemetry_start_timestamp": summary.get("telemetry_start_timestamp"),
        "telemetry_end_timestamp": summary.get("telemetry_end_timestamp"),
        "telemetry_duration_seconds": as_float(summary.get("telemetry_duration_seconds")),
        "telemetry_sample_count": as_int(summary.get("telemetry_sample_count") or summary.get("raw_telemetry_sample_count")),
        "sampling_interval_seconds": as_float(summary.get("sampling_interval_seconds")),
    }
    if role == "source":
        payload["checkpoint_size_bytes"] = as_int(summary.get("checkpoint_size_bytes"))
        payload["checkpoint_save_seconds"] = as_float(summary.get("checkpoint_save_seconds"))
    if role == "target":
        payload["checkpoint_load_seconds"] = as_float(summary.get("checkpoint_load_seconds"))
        payload["resume_to_first_step_seconds"] = as_float(summary.get("resume_to_first_step_seconds"))
    if vendor == "amd":
        payload["hip_version"] = summary.get("hip_version") or preflight.get("hip_version")
    if vendor == "nvidia":
        payload["cuda_version"] = summary.get("cuda_version") or preflight.get("cuda_version")
    return payload


def phase_first_step(summary: Dict[str, Any]) -> Optional[int]:
    resumed = as_int(summary.get("first_step_after_resume"))
    return resumed if resumed is not None else as_int(summary.get("first_logged_step"))


def detected_vendor_from_versions(summary: Dict[str, Any]) -> Optional[str]:
    if summary.get("hip_version"):
        return "amd"
    if summary.get("cuda_version"):
        return "nvidia"
    return None


def preflight_vendor(preflight: Dict[str, Any]) -> Optional[str]:
    if not preflight:
        return None
    if preflight.get("hip_version"):
        return "amd"
    if preflight.get("cuda_version"):
        return "nvidia"
    return None


def missing_required_fields(payload: Dict[str, Any], required_fields: List[str]) -> List[str]:
    return [field for field in required_fields if payload.get(field) is None]


def run_direction_expected_vendors(run_id: str) -> Optional[tuple[str, str]]:
    text = run_id.strip().lower().replace("_", "-")
    aliases = {"amd": "amd", "radeon": "amd", "nvidia": "nvidia", "dgx": "nvidia"}
    if "-to-" not in text:
        return None
    source, destination = text.split("-to-", 1)
    return None if source not in aliases or destination not in aliases else (aliases[source], aliases[destination])


def required_metric_classification(source: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, str]:
    required = {f"source_training.{key}": "REQUIRED" for key in SOURCE_PHASE_REQUIRED_FIELDS}
    required.update({f"destination_training.{key}": "REQUIRED" for key in DESTINATION_PHASE_REQUIRED_FIELDS})
    optional = {
        "source_training.system_energy_kwh": "OPTIONAL_UNAVAILABLE",
        "destination_training.system_energy_kwh": "OPTIONAL_UNAVAILABLE",
    }
    not_applicable: Dict[str, str] = {}
    if source.get("vendor") == "amd":
        not_applicable["source_training.cuda_version"] = "NOT_APPLICABLE"
    if source.get("vendor") == "nvidia":
        not_applicable["source_training.hip_version"] = "NOT_APPLICABLE"
    if target.get("vendor") == "amd":
        not_applicable["destination_training.cuda_version"] = "NOT_APPLICABLE"
    if target.get("vendor") == "nvidia":
        not_applicable["destination_training.hip_version"] = "NOT_APPLICABLE"
    return {**required, **optional, **not_applicable}


def metrics_missing_paths(prefix: str, payload: Dict[str, Any], required_fields: List[str]) -> List[str]:
    return [f"{prefix}.{field}" for field in missing_required_fields(payload, required_fields)]


def truthy(value: Any) -> bool:
    return bool(as_bool(value))


def percent_ratio(numerator: Any, denominator: Any) -> Optional[float]:
    ratio = safe_ratio(as_float(numerator), as_float(denominator))
    if ratio is None:
        return None
    return max(0.0, min(ratio * 100.0, 100.0))


def energy_consistency_error(expected_joules: Any, integrated_joules: Any) -> Optional[float]:
    expected = as_float(expected_joules)
    integrated = as_float(integrated_joules)
    if expected is None or integrated is None or expected <= 0:
        return None
    return abs(integrated - expected) / expected * 100.0


def energy_validation_payload(settings: PortabilitySettings, phase: Dict[str, Any]) -> Dict[str, Any]:
    runtime = as_float(phase.get("training_runtime_seconds"))
    telemetry_duration = as_float(phase.get("telemetry_duration_seconds"))
    samples = as_int(phase.get("telemetry_sample_count"))
    mean_power = as_float(phase.get("mean_gpu_power_watts"))
    integrated = as_float(phase.get("gpu_energy_joules"))
    expected = None if mean_power is None or runtime is None else mean_power * runtime
    coverage = percent_ratio(telemetry_duration, runtime)
    consistency = energy_consistency_error(expected, integrated)
    coverage_passed = coverage is not None and coverage >= settings.minimum_telemetry_coverage_percent
    sample_passed = samples is not None and samples >= settings.minimum_power_samples
    consistency_passed = consistency is not None and consistency <= settings.maximum_energy_consistency_error_percent
    passed = coverage_passed and sample_passed and consistency_passed
    return {
        "telemetry_start_timestamp": phase.get("telemetry_start_timestamp"),
        "telemetry_end_timestamp": phase.get("telemetry_end_timestamp"),
        "telemetry_duration_seconds": telemetry_duration,
        "training_runtime_seconds": runtime,
        "telemetry_sample_count": samples,
        "sampling_interval_seconds": as_float(phase.get("sampling_interval_seconds")),
        "telemetry_coverage_percent": coverage,
        "mean_gpu_power_watts": mean_power,
        "expected_energy_from_mean_power_joules": expected,
        "integrated_gpu_energy_joules": integrated,
        "energy_consistency_error_percent": consistency,
        "minimum_telemetry_coverage_percent": settings.minimum_telemetry_coverage_percent,
        "maximum_energy_consistency_error_percent": settings.maximum_energy_consistency_error_percent,
        "minimum_power_samples": settings.minimum_power_samples,
        "coverage_check_passed": coverage_passed,
        "sample_count_check_passed": sample_passed,
        "energy_consistency_check_passed": consistency_passed,
        "passed": passed,
    }


def phase_energy_inputs_valid(phase: Dict[str, Any]) -> bool:
    checks = [
        as_int(phase.get("useful_training_tokens")),
        as_float(phase.get("training_runtime_seconds")),
        as_float(phase.get("tokens_per_joule")),
        as_float(phase.get("mean_gpu_power_watts")),
        as_float(phase.get("gpu_energy_joules")),
    ]
    return all(value is not None and value > 0 for value in checks)


def system_energy_entry(runtime_seconds: Any, gpu_energy_joules: Any, host_overhead_watts: Optional[float]) -> Dict[str, Any]:
    runtime = as_float(runtime_seconds)
    gpu_energy = as_float(gpu_energy_joules)
    if host_overhead_watts is None or runtime is None or gpu_energy is None:
        return {
            "energy_joules": None,
            "energy_kwh": None,
            "type": "not_measured",
            "method": None,
            "assumptions": None,
            "confidence": None,
        }
    overhead_joules = host_overhead_watts * runtime
    total_joules = gpu_energy + overhead_joules
    return {
        "energy_joules": total_joules,
        "energy_kwh": total_joules / 3_600_000.0,
        "type": "estimated",
        "method": "measured_gpu_energy_plus_configured_host_overhead",
        "assumptions": {"host_overhead_watts": host_overhead_watts},
        "confidence": "low",
    }


def metric_winner(left: Dict[str, Any], right: Dict[str, Any], key: str, lower_is_better: bool = False) -> Optional[str]:
    lv = as_float(left.get(key))
    rv = as_float(right.get(key))
    if lv is None or rv is None or lv == rv:
        return None
    if lower_is_better:
        return "source_training" if lv < rv else "destination_training"
    return "source_training" if lv > rv else "destination_training"


def format_value(value: Any, suffix: str = "", digits: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}{suffix}" if isinstance(value, float) else f"{value}{suffix}"
    return f"{value}{suffix}"


def render_side_by_side(report: Dict[str, Any]) -> str:
    source = report["source_training"]
    target = report["destination_training"]
    migration = report["migration"]
    continuity = report["continuity"]
    run = report["run"]
    validation = report.get("validation", {})
    source_steps = step_span(source.get("first_step"), source.get("last_step"))
    target_steps = step_span(target.get("first_step"), target.get("last_step"))
    source_energy_kj = kj_value(source.get("gpu_energy_joules"))
    target_energy_kj = kj_value(target.get("gpu_energy_joules"))
    iperf_mbps = mbps_from_bits(migration.get("iperf3", {}).get("throughput_bits_per_second"))
    rsync_mbps = mbps_from_bytes(migration.get("rsync", {}).get("throughput_bytes_per_second"))
    mooncake_mbps = mbps_from_bytes(migration.get("mooncake", {}).get("throughput_bytes_per_second"))
    lines = [
        "Cross-OEM LoRA Benchmark",
        "========================",
        "",
        f"Model: {run.get('model_id')}",
        f"Path: {source.get('hostname')} -> Mooncake -> {target.get('hostname')}",
        "",
        f"{'':24}{'Source':>16}{'Destination':>16}",
        "------------------------------------------------",
        f"{'GPU execution':24}{'PASS' if source.get('gpu_execution_verified') else 'FAIL':>16}{'PASS' if target.get('gpu_execution_verified') else 'FAIL':>16}",
        f"{'Training steps':24}{source_steps:>16}{target_steps:>16}",
        f"{'Useful tokens':24}{format_value(source.get('useful_training_tokens')):>16}{format_value(target.get('useful_training_tokens')):>16}",
        f"{'Runtime':24}{format_value(source.get('training_runtime_seconds'), ' s'):>16}{format_value(target.get('training_runtime_seconds'), ' s'):>16}",
        f"{'Tokens/sec':24}{format_value(source.get('tokens_per_second')):>16}{format_value(target.get('tokens_per_second')):>16}",
        f"{'Mean GPU util':24}{format_value(source.get('mean_gpu_utilization_percent'), '%'):>16}{format_value(target.get('mean_gpu_utilization_percent'), '%'):>16}",
        f"{'Peak accel memory':24}{format_value(gb_value(source.get('peak_accelerator_memory_bytes')), ' GB'):>16}{format_value(gb_value(target.get('peak_accelerator_memory_bytes')), ' GB'):>16}",
        f"{'Mean GPU power':24}{format_value(source.get('mean_gpu_power_watts'), ' W'):>16}{format_value(target.get('mean_gpu_power_watts'), ' W'):>16}",
        f"{'GPU energy':24}{format_value(source_energy_kj, ' kJ'):>16}{format_value(target_energy_kj, ' kJ'):>16}",
        f"{'Tokens/joule':24}{format_value(source.get('tokens_per_joule')):>16}{format_value(target.get('tokens_per_joule')):>16}",
        f"{'Energy GBP / 1M':24}{format_value(source.get('energy_cost_per_million_tokens_gbp'), ' GBP'):>16}{format_value(target.get('energy_cost_per_million_tokens_gbp'), ' GBP'):>16}",
        f"{'Energy break-even /1M':24}{format_value(source.get('energy_break_even_gbp_per_million_tokens'), ' GBP'):>16}{format_value(target.get('energy_break_even_gbp_per_million_tokens'), ' GBP'):>16}",
        "",
        "Migration",
        "------------------------------------------------",
        f"Checkpoint size           {format_value(gb_value(migration.get('checkpoint_logical_size_bytes')), ' GB')}",
        f"iperf3 ceiling            {format_value(iperf_mbps, ' Mbit/s')}",
        f"rsync                     {format_value(rsync_mbps, ' Mbit/s')}",
        f"Mooncake                  {format_value(mooncake_mbps, ' Mbit/s')}",
        f"Mooncake / TCP ceiling    {format_value(migration.get('mooncake_vs_iperf_percent'), '%')}",
        f"Mooncake / rsync          {format_value(migration.get('mooncake_vs_rsync_percent'), '%')}",
        "",
        f"Save checkpoint           {format_value(migration.get('checkpoint_save_seconds'), ' s')}",
        f"Transfer                  {format_value(migration.get('transfer_seconds'), ' s')}",
        f"Load checkpoint           {format_value(migration.get('checkpoint_load_seconds'), ' s')}",
        f"Resume -> first step      {format_value(migration.get('resume_to_first_step_seconds'), ' s')}",
        f"Total migration           {format_value(migration.get('migration_to_first_step_seconds'), ' s')}",
        "",
        f"Checksum                  {'PASS' if migration.get('checksum_verified') else 'FAIL'}",
        f"Optimizer continuity      {'PASS' if continuity.get('optimizer_state_verified') else 'FAIL'}",
        f"Training continuity       {'PASS' if continuity.get('resume_continuity_verified') else 'FAIL'}",
        "",
        f"Functional benchmark      {'PASS' if validation.get('functional_benchmark_complete') else 'FAIL'}",
        f"Transfer benchmark        {'PASS' if validation.get('transfer_benchmark_valid') else 'FAIL'}",
        f"GPU energy benchmark      {'PASS' if validation.get('energy_benchmark_valid') else 'FAIL'}",
        f"Economic comparison       {'PASS' if validation.get('economic_comparison_valid') else 'FAIL'}",
    ]
    standalone = report.get("standalone_benchmark", {})
    if standalone.get("enabled"):
        lines.extend(["", "Standalone Benchmark", "------------------------------------------------", standalone.get("table", "")])
    return "\n".join(lines)


def step_span(first_step: Any, last_step: Any) -> str:
    first = as_int(first_step)
    last = as_int(last_step)
    return "N/A" if first is None or last is None else f"{first}-{last}"


def kj_value(value: Any) -> Optional[float]:
    joules = as_float(value)
    return None if joules is None else joules / 1000.0


def mbps_from_bits(value: Any) -> Optional[float]:
    bits = as_float(value)
    return None if bits is None else bits / 1_000_000.0


def mbps_from_bytes(value: Any) -> Optional[float]:
    amount = as_float(value)
    return None if amount is None else amount * 8.0 / 1_000_000.0


def gb_value(value: Any) -> Optional[float]:
    amount = as_float(value)
    return None if amount is None else amount / (1024 ** 3)


@op(ins={"settings": In(PortabilitySettings), "transfer_result": In(TransferResult), "_ready": In(Nothing)}, out=Out(Nothing))
def record_transfer_tests(context: OpExecutionContext, settings: PortabilitySettings, transfer_result: TransferResult) -> None:
    host = settings.target_host
    records = transfer_test_artifacts(settings, transfer_result)
    comparison = transfer_test_comparison(records)
    for record in records:
        write_remote_json(settings, host, str(record["artifact_path"]), dict(record))
        emit_materialization(
            context,
            ["portability", "target", "nvidia", "transfer_tests", *asset_key_for_path(str(record["artifact_path"]).split(f"{settings.remote_root}/", 1)[-1])[2:]],
            "Transfer-path test artifact written on target host.",
            {
                **common_metadata(settings, "recorded", str(record["artifact_path"])),
                "transfer_test": record["test"],
                "transfer_path": record["path"],
                "details": MetadataValue.json(dict(record["details"])),
                "lineage_path": MetadataValue.json(["transfer", "verification", "target", "nvidia", "transfer_tests", record["test"]]),
            },
        )
    matrix_path = f"{transfer_tests_dir(settings)}/transfer_test_matrix.json"
    write_remote_json(settings, host, matrix_path, {"tests": records, "comparison": comparison})
    emit_materialization(
        context,
        ["portability", "target", "nvidia", "transfer_tests", "transfer_test_matrix"],
        "Transfer-path test matrix recorded.",
        {
            **common_metadata(settings, "recorded", matrix_path),
            "tests": MetadataValue.json(records),
            "comparison": MetadataValue.json(comparison),
            "lineage_path": MetadataValue.json(["transfer", "verification", "target", "nvidia", "transfer_tests", "transfer_test_matrix"]),
        },
    )


def infer_data_type(saved_path: Path) -> str:
    if saved_path.is_dir():
        return "directory"
    suffix = saved_path.suffix.lower()
    return suffix.lstrip(".") if suffix else "unknown"


def is_supported_data_type(data_type: str) -> bool:
    return data_type in {"directory", "json", "jsonl", "csv", "parquet", "arrow", "sqlite", "db", "txt"}


def asset_key_for_path(asset_path: str) -> List[str]:
    cleaned = asset_path.replace("\\", "/").strip("/")
    return ["portability", "data", *[part for part in cleaned.split("/") if part]]


def common_metadata(settings: PortabilitySettings, status: str, artifact_path: str) -> dict:
    return {
        "run_id": settings.run_id,
        "source_host": settings.source_host,
        "target_host": settings.target_host,
        "checkpoint_step": settings.checkpoint_step,
        "final_step": settings.final_step,
        "status": status,
        "artifact_path": MetadataValue.path(artifact_path),
        "schema": PORTABILITY_SCHEMA,
    }


def emit_materialization(context: OpExecutionContext, key: List[str], description: str, metadata: dict) -> None:
    context.log_event(AssetMaterialization(asset_key=key, description=description, metadata=metadata))


@op(out=Out(PortabilitySettings))
def settings_op(context: OpExecutionContext) -> PortabilitySettings:
    settings = load_settings()
    paths = output_asset_paths(settings)
    emit_materialization(
        context,
        ["portability", "settings"],
        "Portability run settings captured.",
        {
            **common_metadata(settings, "configured", str(Path(settings.local_root))),
            "lineage_path": MetadataValue.json(["control", "settings"]),
            "local_root": settings.local_root,
            "remote_root": settings.remote_root,
            "asset_paths": MetadataValue.json(paths),
            "command_retries": settings.command_retries,
            "min_extra_steps": settings.min_extra_steps,
            "max_loss_delta": settings.max_loss_delta,
            "source_data_parallel_size": settings.source_data_parallel_size,
            "host_python_cmd_overrides": MetadataValue.json(settings.host_python_cmd_overrides),
            "electricity_price_gbp_per_kwh": settings.electricity_price_gbp_per_kwh,
            "revenue_per_million_tokens_gbp": settings.revenue_per_million_tokens_gbp,
            "minimum_telemetry_coverage_percent": settings.minimum_telemetry_coverage_percent,
            "maximum_energy_consistency_error_percent": settings.maximum_energy_consistency_error_percent,
            "minimum_power_samples": settings.minimum_power_samples,
            "recommended_minimum_energy_benchmark_seconds": settings.recommended_minimum_energy_benchmark_seconds,
            "source_system_overhead_watts": settings.source_system_overhead_watts,
            "target_system_overhead_watts": settings.target_system_overhead_watts,
            "training_telemetry_interval_seconds": settings.training_telemetry_interval_seconds,
            "machine_catalog": MetadataValue.json(
                {
                    key: {
                        "hostname": machine.hostname,
                        "vendor": machine.vendor,
                        "device_name": machine.device_name,
                        "expected_gpu_architecture": machine.expected_gpu_architecture,
                        "container_image": machine.container_image,
                        "execution_mode": machine.execution_mode,
                        "torch_install_command": machine.torch_install_command,
                        "hf_cache_dir": machine.hf_cache_dir,
                        "rocm_path": machine.rocm_path,
                        "electricity_price_per_kwh": machine.electricity_price_per_kwh,
                        "currency": machine.currency,
                        "transfer": {
                            "control_host": machine.transfer.control_host,
                            "control_port": machine.transfer.control_port,
                        },
                    }
                    for key, machine in settings.machine_catalog.items()
                }
            ),
            "standalone_benchmark": MetadataValue.json(
                {
                    "enabled": settings.standalone_benchmark.enabled,
                    "machine_keys": settings.standalone_benchmark.machine_keys,
                    "run_id_suffix": settings.standalone_benchmark.run_id_suffix,
                    "steps": settings.standalone_benchmark.steps,
                    "max_steps": settings.standalone_benchmark.max_steps,
                    "max_length": settings.standalone_benchmark.max_length,
                    "per_device_batch_size": settings.standalone_benchmark.per_device_batch_size,
                    "model_id": settings.standalone_benchmark.model_id,
                }
            ),
        },
    )
    return settings


@op(ins={"settings": In(PortabilitySettings)}, out=Out(Nothing))
def prepare_hosts(context: OpExecutionContext, settings: PortabilitySettings) -> None:
    hosts = all_known_hosts(settings)
    for host in hosts:
        run_shell(ssh_cmd(settings, host, f"mkdir -p {settings.remote_root}"), settings.command_retries, env=sshpass_env(settings))
    emit_materialization(
        context,
        ["portability", "control", "host_preparation"],
        "Remote directories prepared on both hosts.",
        {
            **common_metadata(settings, "ready", settings.remote_root),
            "lineage_path": MetadataValue.json(["control", "settings", "control", "host_preparation"]),
            "provenance": MetadataValue.json({"operation": "mkdir", "hosts": hosts}),
        },
    )


@op(ins={"settings": In(PortabilitySettings), "_ready": In(Nothing)}, out=Out(Nothing))
def sync_and_install(context: OpExecutionContext, settings: PortabilitySettings) -> None:
    hosts = all_known_hosts(settings)
    files = ["requirements.txt", "train_lora_migration.py", "validate_resume.py", "checkpoint_io.py", "mooncake_tcp_agent.py"]
    for name in files:
        for host in hosts:
            run_shell(scp_cmd(settings, str(Path(settings.local_root) / name), f"{host}:{settings.remote_root}/"), settings.command_retries, env=sshpass_env(settings))
    paths = sync_paths(settings)
    for asset_path in paths:
        for host in hosts:
            sync_asset_to_host(settings, host, asset_path)
    install_hosts = [host for host in hosts if not machine_uses_container(settings, host)]
    for host in install_hosts:
        run_shell(ssh_cmd(settings, host, requirements_install_command(settings, host)), settings.command_retries, env=sshpass_env(settings))
    ensure_mooncake_dependency(settings, settings.source_host)
    ensure_mooncake_dependency(settings, settings.target_host)
    emit_materialization(
        context,
        ["portability", "control", "environment_sync"],
        "Code, dependencies, and input assets synced to source and target hosts.",
        {
            **common_metadata(settings, "synced", settings.remote_root),
            "lineage_path": MetadataValue.json(["control", "host_preparation", "control", "environment_sync"]),
            "files": MetadataValue.json(files),
            "asset_paths": MetadataValue.json(paths),
            "synced_hosts": MetadataValue.json(hosts),
            "host_installations": MetadataValue.json(install_hosts),
        },
    )


def machine_uses_container(settings: PortabilitySettings, host: str) -> bool:
    machine = machine_for_host(settings, host)
    return bool(machine and machine.container_image and machine.execution_mode == "docker")


@op(name="train_on_source", ins={"settings": In(PortabilitySettings), "gpu_preflight": In(dict), "standalone_result": In(dict)}, out=Out(Nothing))
def train_on_source(context: OpExecutionContext, settings: PortabilitySettings, gpu_preflight: Dict[str, Any], standalone_result: Dict[str, Any]) -> None:
    _ = standalone_result
    source_run_root = f"{settings.remote_root}/amd_run_{settings.run_id}"
    source_run_root_expanded = f"{expanded_remote_root(settings, settings.source_host)}/amd_run_{settings.run_id}"
    checkpoint_meta = f"{settings.remote_root}/amd_run_{settings.run_id}/{remote_ckpt(settings)}/checkpoint_meta.json"
    if remote_exists(settings, settings.source_host, checkpoint_meta) and source_checkpoint_reusable(settings):
        emit_materialization(
            context,
            ["portability", "source", "amd", "checkpoint"],
            "Existing source checkpoint reused.",
            {
                **common_metadata(settings, "checkpoint_reused", f"{source_run_root}/{remote_ckpt(settings)}"),
                "lineage_path": MetadataValue.json(["control", "environment_sync", "source", "amd", "checkpoint"]),
                "provenance": MetadataValue.json({"producer": settings.source_host, "mode": "reuse"}),
            },
        )
        return
    run_shell(ssh_cmd(settings, settings.source_host, f"rm -rf {shlex.quote(source_run_root_expanded)}"), settings.command_retries, env=sshpass_env(settings), allow_failure=True)
    if settings.source_use_gpu and not bool(gpu_preflight.get("source", {}).get("training_allowed", False)):
        raise RuntimeError("source training blocked because GPU kernel preflight did not pass")
    run = source_train_command(settings)
    run_shell(ssh_cmd(settings, settings.source_host, run), settings.command_retries, env=sshpass_env(settings))
    emit_materialization(
        context,
        ["portability", "source", "amd", "checkpoint"],
        "Checkpoint created on source host.",
        {
            **common_metadata(
                settings,
                "checkpoint_created",
                f"{settings.remote_root}/amd_run_{settings.run_id}/{remote_ckpt(settings)}",
            ),
            "lineage_path": MetadataValue.json(["control", "environment_sync", "source", "amd", "checkpoint"]),
            "provenance": MetadataValue.json({"producer": settings.source_host, "consumer": settings.target_host}),
        },
    )


def source_train_command(settings: PortabilitySettings) -> str:
    python_cmd = execution_python_cmd(settings, settings.source_host)
    script = f"{settings.remote_root}/train_lora_migration.py"
    args = f"--dataset-path {remote_dataset_path(settings)} --output-dir {settings.remote_root}/amd_run_{settings.run_id} --stop-step {settings.checkpoint_step} --max-steps 100 --telemetry-interval-seconds {settings.training_telemetry_interval_seconds}"
    if not settings.source_use_gpu:
        return f"CUDA_VISIBLE_DEVICES='' {shlex.quote(python_cmd)} {script} {args}"
    if settings.source_data_parallel_size <= 1:
        return f"{shlex.quote(python_cmd)} {script} {args}"
    return f"{shlex.quote(python_cmd)} -m torch.distributed.run --standalone --nproc_per_node={settings.source_data_parallel_size} {script} {args}"


def source_checkpoint_reusable(settings: PortabilitySettings) -> bool:
    path = f"{settings.remote_root}/amd_run_{settings.run_id}/run_summary.json"
    if not remote_exists(settings, settings.source_host, path):
        return False
    try:
        summary = pull_json(settings, settings.source_host, path)
    except Exception:
        return False
    required = [
        "hostname",
        "vendor",
        "gpu_architecture",
        "torch_version",
        "saved_step",
        "useful_training_tokens",
        "runtime_seconds",
        "mean_gpu_utilization_percent",
        "peak_gpu_utilization_percent",
        "peak_accelerator_memory_bytes",
        "mean_power_watts",
        "peak_power_watts",
        "gpu_energy_joules",
        "gpu_energy_kwh",
        "tokens_per_joule",
        "tokens_per_second",
        "checkpoint_size_bytes",
        "checkpoint_save_seconds",
        "final_loss",
        "telemetry_duration_seconds",
        "telemetry_sample_count",
        "sampling_interval_seconds",
    ]
    if missing_required_fields(summary, required):
        return False
    samples = as_int(summary.get("telemetry_sample_count") or summary.get("raw_telemetry_sample_count"))
    interval = as_float(summary.get("sampling_interval_seconds"))
    if samples is None or samples < settings.minimum_power_samples:
        return False
    if interval is None or abs(interval - settings.training_telemetry_interval_seconds) > 1e-9:
        return False
    return True


def transfer_checkpoint(settings: PortabilitySettings, source: str, destination: str, backend: str) -> TransferResult:
    if backend == "mooncake_tcp":
        return transfer_checkpoint_mooncake(settings, source, destination)
    if backend == "scp":
        return transfer_checkpoint_scp(settings, source, destination)
    raise RuntimeError(f"unsupported transfer backend: {backend}")


def transfer_checkpoint_scp(settings: PortabilitySettings, source: str, destination: str) -> TransferResult:
    local_stage = Path("/tmp") / settings.run_id
    source_ckpt = f"{settings.remote_root}/amd_run_{settings.run_id}/{remote_ckpt(settings)}"
    destination_ckpt = f"{settings.remote_root}/{remote_ckpt(settings)}"
    start = time.perf_counter()
    local_stage.mkdir(parents=True, exist_ok=True)
    run_shell(scp_cmd(settings, f"{source}:{source_ckpt}", str(local_stage), recursive=True), settings.command_retries, env=sshpass_env(settings))
    run_shell(scp_cmd(settings, str(local_stage / remote_ckpt(settings)), f"{destination}:{settings.remote_root}/", recursive=True), settings.command_retries, env=sshpass_env(settings))
    elapsed = time.perf_counter() - start
    verified = verify_transfer_manifests(settings, source, source_ckpt, destination, destination_ckpt)
    bytes_transferred = int(verified["total_bytes"])
    return TransferResult(
        bool(verified["checksum_ok"]),
        "scp",
        bytes_transferred,
        elapsed,
        0.0 if elapsed <= 0 else bytes_transferred / elapsed,
        bool(verified["checksum_ok"]),
        False,
        verified,
    )


def transfer_checkpoint_mooncake(settings: PortabilitySettings, source: str, destination: str) -> TransferResult:
    config = settings.transfer.mooncake
    if not config:
        raise RuntimeError("transfer backend mooncake_tcp selected but transfer configuration is missing")
    sync_mooncake_agent(settings, source)
    sync_mooncake_agent(settings, destination)
    source_ckpt = f"{settings.remote_root}/amd_run_{settings.run_id}/{remote_ckpt(settings)}"
    destination_ckpt = f"{settings.remote_root}/{remote_ckpt(settings)}"
    destination_tmp = f"{destination_ckpt}.tmp"
    control_host = receiver_control_host(config)
    payload: Dict[str, Any] = {}
    last_error: Optional[Exception] = None
    for attempt in range(3):
        session = f"{int(time.time() * 1000)}_{os.getpid()}_{attempt}"
        preferred_port = config.destination.control_port + (attempt * 50)
        control_port = reserve_receiver_control_port(settings, destination, preferred_port)
        pid_path = f"/tmp/mooncake_receiver_{settings.run_id}_{settings.checkpoint_step}_{session}_{control_port}.pid"
        log_path = f"/tmp/mooncake_receiver_{settings.run_id}_{settings.checkpoint_step}_{session}_{control_port}.log"
        receiver_cmd = mooncake_receiver_command(settings, config, destination, destination_ckpt, destination_tmp, pid_path, log_path, session, control_port)
        run_shell(ssh_cmd(settings, destination, receiver_cmd), settings.command_retries, env=sshpass_env(settings))
        try:
            wait_for_receiver(settings, source, destination, control_host, control_port, pid_path, log_path)
            sender_cmd = mooncake_sender_command(settings, config, source, source_ckpt, destination_ckpt, destination_tmp, control_host, session, control_port)
            sender_result = run_shell(ssh_cmd(settings, source, sender_cmd), 0, env=sshpass_env(settings))
            payload = parse_sender_result(sender_result.stdout)
            break
        except RuntimeError as exc:
            last_error = exc
            excerpt = receiver_log_excerpt(settings, destination, log_path)
            retryable = mooncake_retryable_failure(str(exc), excerpt)
            if not retryable or attempt == 2:
                raise RuntimeError(f"mooncake sender failed: {exc}; receiver_log={excerpt}") from exc
            time.sleep(2)
        finally:
            run_shell(ssh_cmd(settings, destination, f"test -f {pid_path} && kill $(cat {pid_path}) >/dev/null 2>&1 || true"), settings.command_retries, env=sshpass_env(settings), allow_failure=True)
    if not payload and last_error:
        raise RuntimeError(f"mooncake sender failed after retries: {last_error}")
    return TransferResult(
        bool(payload.get("success", False)),
        str(payload.get("backend", "mooncake_tcp")),
        int(payload.get("bytes_transferred", 0)),
        float(payload.get("elapsed_seconds", 0.0)),
        float(payload.get("throughput_bytes_per_second", 0.0)),
        bool(payload.get("checksum_ok", False)),
        False,
        payload,
    )


def sync_mooncake_agent(settings: PortabilitySettings, host: str) -> None:
    local_agent = str(Path(__file__).with_name("mooncake_tcp_agent.py"))
    remote_parent = str(Path(settings.remote_root))
    run_shell(ssh_cmd(settings, host, f"mkdir -p {shlex.quote(remote_parent)}"), settings.command_retries, env=sshpass_env(settings))
    run_shell(scp_cmd(settings, local_agent, f"{host}:{settings.remote_root}/", recursive=False), settings.command_retries, env=sshpass_env(settings))


def reserve_receiver_control_port(settings: PortabilitySettings, destination: str, preferred_port: int) -> int:
    for offset in range(100):
        port = preferred_port + offset
        if receiver_control_port_available(settings, destination, port):
            return port
    raise RuntimeError(f"unable to reserve receiver control port starting at {preferred_port}")


def mooncake_retryable_failure(error_text: str, receiver_excerpt: str) -> bool:
    text = f"{error_text} {receiver_excerpt}".lower()
    patterns = ["address already in use", "connection refused", "not reachable", "exited before becoming reachable"]
    return any(pattern in text for pattern in patterns)


def receiver_control_port_available(settings: PortabilitySettings, destination: str, control_port: int) -> bool:
    python_cmd = mooncake_python_cmd(settings, destination)
    script = "import socket,sys; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); rc=0\ntry: s.bind(('0.0.0.0', int(sys.argv[1])))\nexcept OSError: rc=1\ns.close(); raise SystemExit(rc)"
    command = f"{shlex.quote(python_cmd)} -c {shlex.quote(script)} {control_port}"
    result = run_shell(ssh_cmd(settings, destination, command), settings.command_retries, env=sshpass_env(settings), allow_failure=True)
    return result.returncode == 0


def mooncake_receiver_command(settings: PortabilitySettings, config: MooncakeTransferConfig, destination: str, destination_ckpt: str, destination_tmp: str, pid_path: str, log_path: str, session: str, control_port: int) -> str:
    chunk_bytes = config.chunk_mb * 1024 * 1024
    python_cmd = mooncake_python_cmd(settings, destination)
    parts = [
        f"{shlex.quote(python_cmd)} {settings.remote_root}/mooncake_tcp_agent.py",
        "--mode receiver",
        f"--metadata-server {shlex.quote(config.metadata_server)}",
        f"--protocol {shlex.quote(config.protocol)}",
        f"--local-server-name {shlex.quote(config.destination.server_name)}",
        f"--peer-server-name {shlex.quote(config.source.server_name)}",
        f"--chunk-bytes {chunk_bytes}",
        f"--max-inflight {config.max_inflight}",
        f"--destination-dir {shlex.quote(destination_ckpt)}",
        f"--destination-tmp-dir {shlex.quote(destination_tmp)}",
        f"--session-id {shlex.quote(session)}",
        "--control-host 0.0.0.0",
        f"--control-port {control_port}",
    ]
    prefix = mooncake_runtime_prefix(config)
    return f"nohup setsid {prefix} {' '.join(parts)} </dev/null > {shlex.quote(log_path)} 2>&1 & echo $! > {shlex.quote(pid_path)}"


def mooncake_sender_command(settings: PortabilitySettings, config: MooncakeTransferConfig, source: str, source_ckpt: str, destination_ckpt: str, destination_tmp: str, control_host: str, session: str, control_port: int) -> str:
    chunk_bytes = config.chunk_mb * 1024 * 1024
    python_cmd = mooncake_python_cmd(settings, source)
    parts = [
        f"{shlex.quote(python_cmd)} {settings.remote_root}/mooncake_tcp_agent.py",
        "--mode sender",
        f"--metadata-server {shlex.quote(config.metadata_server)}",
        f"--protocol {shlex.quote(config.protocol)}",
        f"--local-server-name {shlex.quote(config.source.server_name)}",
        f"--peer-server-name {shlex.quote(config.destination.server_name)}",
        f"--chunk-bytes {chunk_bytes}",
        f"--max-inflight {config.max_inflight}",
        f"--source-dir {shlex.quote(source_ckpt)}",
        f"--destination-dir {shlex.quote(destination_ckpt)}",
        f"--destination-tmp-dir {shlex.quote(destination_tmp)}",
        f"--session-id {shlex.quote(session)}",
        f"--control-host {shlex.quote(control_host)}",
        f"--control-port {control_port}",
    ]
    return f"{mooncake_runtime_prefix(config)} {' '.join(parts)}"


def mooncake_runtime_prefix(config: MooncakeTransferConfig) -> str:
    if config.protocol.strip().lower() != "tcp":
        return "env"
    return "env MC_FORCE_TCP=1 MC_TE_FILTERS=__no_such_rdma_device__"


def receiver_control_host(config: MooncakeTransferConfig) -> str:
    host = config.destination.control_host.strip()
    if host and host not in {"0.0.0.0", "127.0.0.1", "localhost"}:
        return host
    return config.destination.server_name.split(":", 1)[0]


def wait_for_receiver(settings: PortabilitySettings, source: str, destination: str, control_host: str, control_port: int, pid_path: str, log_path: str) -> None:
    python_cmd = mooncake_python_cmd(settings, source)
    probe = (
        f"{shlex.quote(python_cmd)} -c \"import json,socket; "
        f"s=socket.create_connection(({control_host!r}, {control_port}), timeout=2); "
        "s.sendall(b'{\\\"command\\\": \\\"hello\\\"}\\n'); "
        "r=s.makefile('rb').readline().decode('utf-8'); "
        "s.close(); "
        "raise SystemExit(0 if r.strip() else 2)\""
    )
    for _ in range(20):
        result = run_shell(ssh_cmd(settings, source, probe), settings.command_retries, env=sshpass_env(settings), allow_failure=True)
        if result.returncode == 0:
            return
        alive = run_shell(ssh_cmd(settings, destination, f"test -f {pid_path} && kill -0 $(cat {pid_path})"), settings.command_retries, env=sshpass_env(settings), allow_failure=True)
        if alive.returncode != 0:
            excerpt = receiver_log_excerpt(settings, destination, log_path)
            raise RuntimeError(f"mooncake receiver process exited before becoming reachable at {control_host}:{control_port}; log={excerpt}")
        time.sleep(1)
    excerpt = receiver_log_excerpt(settings, destination, log_path)
    raise RuntimeError(f"mooncake receiver not reachable at {control_host}:{control_port}; log={excerpt}")


def receiver_log_excerpt(settings: PortabilitySettings, destination: str, log_path: str) -> str:
    python_cmd = mooncake_python_cmd(settings, destination)
    command = "%s -c \"from pathlib import Path; p=Path(%r); text=p.read_text(errors='replace') if p.exists() else 'log file missing'; lines=text.splitlines(); print(' | '.join(lines[-20:]))\"" % (shlex.quote(python_cmd), log_path)
    result = run_shell(ssh_cmd(settings, destination, command), settings.command_retries, env=sshpass_env(settings), allow_failure=True)
    return result.stdout.strip() if result.stdout.strip() else "no log output"


def mooncake_python_cmd(settings: PortabilitySettings, host: str) -> str:
    mooncake = settings.transfer.mooncake
    if not mooncake:
        return "/usr/bin/python3.12"
    command = mooncake.python_cmd
    if host == settings.source_host and mooncake.source.python_cmd:
        command = mooncake.source.python_cmd
    if host == settings.target_host and mooncake.destination.python_cmd:
        command = mooncake.destination.python_cmd
    if command.startswith("~/") and "@" in host:
        username = host.split("@", 1)[0]
        return f"/home/{username}/{command[2:]}"
    return command


def execution_python_cmd(settings: PortabilitySettings, host: str) -> str:
    if host in settings.host_python_cmd_overrides:
        command = settings.host_python_cmd_overrides[host]
    else:
        source_cmd = getattr(settings, "source_python_cmd", "/usr/bin/python3.12")
        target_cmd = getattr(settings, "target_python_cmd", "/usr/bin/python3.12")
        command = source_cmd if host == settings.source_host else target_cmd
    command = machine_python_cmd(settings, host) or command
    if command.startswith("~/") and "@" in host:
        username = host.split("@", 1)[0]
        return f"/home/{username}/{command[2:]}"
    return command


def machine_python_cmd(settings: PortabilitySettings, host: str) -> Optional[str]:
    machine = machine_for_host(settings, host)
    return None if machine is None else machine.python_cmd


def machine_for_host(settings: PortabilitySettings, host: str) -> Optional[MachineSpec]:
    return next((item for item in settings.machine_catalog.values() if item.hostname == host), None)


def benchmark_machines(settings: PortabilitySettings) -> List[MachineSpec]:
    return [settings.machine_catalog[key] for key in settings.standalone_benchmark.machine_keys if key in settings.machine_catalog]


def all_known_hosts(settings: PortabilitySettings) -> List[str]:
    hosts = [settings.source_host, settings.target_host, *[machine.hostname for machine in benchmark_machines(settings)]]
    return list(dict.fromkeys([host for host in hosts if host]))


def requirements_install_command(settings: PortabilitySettings, host: str) -> str:
    python_cmd = shlex.quote(execution_python_cmd(settings, host))
    requirements = f"{settings.remote_root}/requirements.txt"
    detect = f"IS_VENV=$({python_cmd} -c \"import sys; print(1 if sys.prefix != sys.base_prefix else 0)\")"
    venv = f"{python_cmd} -m pip install -U pip && {python_cmd} -m pip install -r {requirements}"
    system = f"{pip_bootstrap(settings, execution_python_cmd(settings, host))} && {python_cmd} -m pip install --user --break-system-packages -U pip && {python_cmd} -m pip install --user --break-system-packages -r {requirements}"
    base = f"{detect}; if [ \"$IS_VENV\" = \"1\" ]; then {venv}; else {system}; fi"
    torch_override = machine_torch_install(settings, host)
    return base if not torch_override else f"{base} && {torch_override}"


def machine_torch_install(settings: PortabilitySettings, host: str) -> Optional[str]:
    machine = machine_for_host(settings, host)
    return None if machine is None else machine.torch_install_command


def min_torch_tuple() -> tuple[int, int]:
    major, minor = MIN_TORCH_VERSION.split(".", 1)
    return int(major), int(minor)


def parse_torch_tuple(version: str) -> Optional[tuple[int, int]]:
    match = re.search(r"(\d+)\.(\d+)", version)
    return (int(match.group(1)), int(match.group(2))) if match else None


def version_probe_script() -> str:
    return "import json,torch,transformers; print(json.dumps({'torch_version': torch.__version__, 'transformers_version': transformers.__version__}))"


def architecture_probe_script() -> str:
    return (
        "import json,re,subprocess,torch;"
        "run=lambda cmd: subprocess.run(cmd, check=False, text=True, capture_output=True).stdout;"
        "roc=run(['rocminfo']);"
        "gfx=re.search(r'(gfx[0-9a-zA-Z]+)', roc);"
        "nvs=run(['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader']);"
        "cc=re.search(r'(\\d+)\\.(\\d+)', nvs);"
        "runtime=(gfx.group(1) if gfx else ('sm_%s%s' % (cc.group(1), cc.group(2)) if cc else ''));"
        "runtime=runtime.split(':', 1)[0];"
        "arches=[];"
        "\ntry:\n arches=[str(x) for x in torch.cuda.get_arch_list()]\nexcept Exception:\n arches=[]\n"
        "\nif (not runtime):\n"
        "\n try:\n"
        "\n  if torch.cuda.is_available():\n"
        "\n   prop=torch.cuda.get_device_properties(0);"
        "\n   gcn=str(getattr(prop, 'gcnArchName', '')).split(':', 1)[0];"
        "\n   cap=getattr(torch.cuda, 'get_device_capability', lambda *_: (0, 0))(0);"
        "\n   runtime=gcn or ('sm_%s%s' % (cap[0], cap[1]) if cap != (0, 0) else '');"
        "\n except Exception:\n"
        "\n  runtime=''\n"
        "print(json.dumps({'runtime_gpu_arch': runtime, 'torch_arch_list': arches}))"
    )


def probe_command(settings: PortabilitySettings, host: str, script: str) -> str:
    python_cmd = shlex.quote(execution_python_cmd(settings, host))
    return f"{python_cmd} -c {shlex.quote(script)}"


def parse_probe_payload(host: str, stdout: str, verify_cmd: str) -> Dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"Preflight probe returned no output on host={host}. Verification command: {verify_cmd}")
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Preflight probe returned non-JSON output on host={host}. Verification command: {verify_cmd}. Output: {stdout}") from exc


def run_probe(settings: PortabilitySettings, host: str, script: str, verify_cmd: str, retries: int) -> Dict[str, Any]:
    result = run_shell(ssh_cmd(settings, host, probe_command(settings, host, script)), retries, env=sshpass_env(settings), allow_failure=True)
    if result.returncode != 0:
        raise RuntimeError(f"Preflight probe failed on host={host} (exit={result.returncode}). Verification command: {verify_cmd}. stderr: {result.stderr.strip()}")
    return parse_probe_payload(host, result.stdout, verify_cmd)


def run_probe_optional(settings: PortabilitySettings, host: str, script: str, verify_cmd: str) -> Dict[str, Any]:
    result = run_shell(ssh_cmd(settings, host, probe_command(settings, host, script)), 0, env=sshpass_env(settings), allow_failure=True)
    return {} if result.returncode != 0 else parse_probe_payload(host, result.stdout, verify_cmd)


def machine_probe_command(settings: PortabilitySettings, machine: MachineSpec, script: str) -> str:
    if not machine.container_image or machine.execution_mode != "docker":
        return f"{machine_runtime_exports(machine)}{probe_command(settings, machine.hostname, script)}"
    inner = f"python3 -c {shlex.quote(script)}"
    return docker_command(settings, machine, inner)


def docker_command(settings: PortabilitySettings, machine: MachineSpec, inner: str) -> str:
    mount = expanded_remote_root(settings, machine.hostname)
    run = [
        "docker run --rm",
        "--network host",
        "--ipc=host",
        "--shm-size 16g",
        "--device /dev/kfd",
        "--device /dev/dri",
        "--group-add video",
        f"-v {shlex.quote(mount)}:/workspace",
        "-w /workspace",
        shlex.quote(str(machine.container_image)),
        "bash -lc",
        shlex.quote(inner),
    ]
    return " ".join(run)


def run_machine_probe(settings: PortabilitySettings, machine: MachineSpec, script: str) -> Dict[str, Any]:
    command = machine_probe_command(settings, machine, script)
    result = run_shell(ssh_cmd(settings, machine.hostname, command), settings.command_retries, env=sshpass_env(settings), allow_failure=True)
    if result.returncode != 0:
        raise RuntimeError(f"Preflight probe failed on host={machine.hostname} (exit={result.returncode}). command={command}. stderr={result.stderr.strip()}")
    payload = parse_probe_payload(machine.hostname, result.stdout, command)
    return {"verify_command": command, **payload}


def assert_torch_compatible(host: str, torch_version: str, transformers_version: str, verify_cmd: str) -> None:
    parsed = parse_torch_tuple(torch_version)
    if parsed is None:
        raise RuntimeError(f"Preflight could not parse torch version on host={host}: torch={torch_version}, transformers={transformers_version}. Required torch>={MIN_TORCH_VERSION} ({TORCH_SECURITY_CVE}). Verification command: {verify_cmd}")
    if parsed < min_torch_tuple():
        raise RuntimeError(f"Preflight torch compatibility failed on host={host}: torch={torch_version}, transformers={transformers_version}, required=torch>={MIN_TORCH_VERSION} due to {TORCH_SECURITY_CVE}. Upgrade torch in the runtime environment used by training; bypassing via weights_only=False is not supported. Verification command: {verify_cmd}")


def architecture_warning(host: str, payload: Dict[str, Any]) -> Optional[str]:
    runtime_arch = str(payload.get("runtime_gpu_arch", "")).strip()
    torch_arches = [str(item).strip() for item in payload.get("torch_arch_list", []) if str(item).strip()]
    if not runtime_arch or not torch_arches:
        return None
    matches = [item for item in torch_arches if runtime_arch in item or item in runtime_arch]
    return None if matches else f"Potential torch architecture mismatch on host={host}: runtime_gpu_arch={runtime_arch}, torch_arch_list={torch_arches}. Runtime may fail with invalid device function."


def kernel_preflight_script() -> str:
    return (
        "import json,platform,re,socket,subprocess,torch;"
        "run=lambda cmd: subprocess.run(cmd, check=False, text=True, capture_output=True);"
        "gpu_detected=bool(torch.cuda.is_available());"
        "hip=getattr(torch.version,'hip',None);"
        "cuda=getattr(torch.version,'cuda',None);"
        "vendor=('amd' if hip else ('nvidia' if cuda else 'unknown'));"
        "hostname=socket.gethostname();"
        "device_name=torch.cuda.get_device_name(0) if gpu_detected else None;"
        "runtime='';"
        "roc_text='';"
        "\ntry:\n roc=run(['rocminfo']); roc_text=roc.stdout or ''\n"
        "except Exception:\n roc_text=''\n"
        "m=re.search(r'(gfx[0-9a-zA-Z]+)', roc_text);"
        "runtime=(m.group(1) if m else runtime);"
        "\nif (not runtime) and gpu_detected:\n"
        " try:\n"
        "  prop=torch.cuda.get_device_properties(0); runtime=str(getattr(prop,'gcnArchName','')).split(':',1)[0]\n"
        " except Exception:\n"
        "  runtime=runtime\n"
        "\nif (not runtime) and gpu_detected and cuda:\n"
        " try:\n"
        "  cap=torch.cuda.get_device_capability(0); runtime='sm_%s%s' % (cap[0], cap[1])\n"
        " except Exception:\n"
        "  runtime=runtime\n"
        "total_memory=None;"
        "\nif gpu_detected:\n"
        " try:\n"
        "  total_memory=int(getattr(torch.cuda.get_device_properties(0),'total_memory',0)) or None\n"
        " except Exception:\n"
        "  total_memory=None\n"
        "power_ok=False; power_reason='';"
        "\nif vendor=='amd':\n"
        " p=run(['rocm-smi','--showpower','--json']); txt=(p.stdout or '').strip();\n"
        " try:\n"
        "  obj=json.loads(txt) if txt else {}; rows=[v for k,v in obj.items() if str(k).startswith('card') and isinstance(v,dict)]; card=(rows[0] if rows else {}); vals=[str(v) for k,v in card.items() if 'Power' in str(k)]; num=next((re.search(r'-?\\d+(?:\\.\\d+)?', v).group(0) for v in vals if re.search(r'-?\\d+(?:\\.\\d+)?', v)), None); power_ok=(p.returncode==0 and num is not None); power_reason=('' if power_ok else ('power reading is unavailable from rocm-smi' if p.returncode==0 else (p.stderr.strip() or txt or 'rocm-smi unavailable')))\n"
        " except Exception:\n"
        "  power_ok=False; power_reason=('rocm-smi power payload parse failed' if p.returncode==0 else (p.stderr.strip() or txt or 'rocm-smi unavailable'))\n"
        "\nelif vendor=='nvidia':\n"
        " p=run(['nvidia-smi','--query-gpu=power.draw','--format=csv,noheader,nounits']); power_ok=(p.returncode==0); power_reason=('' if power_ok else (p.stderr.strip() or p.stdout.strip() or 'nvidia-smi unavailable'))\n"
        "\nelse:\n"
        " power_ok=False; power_reason='vendor not detected'\n"
        "payload={'hostname':hostname,'gpu_vendor':vendor,'gpu_model':device_name,'gpu_architecture':runtime or None,'total_gpu_memory_bytes':total_memory,'rocm_version':hip,'torch_version':torch.__version__,'python_version':platform.python_version(),'gpu_detected':gpu_detected,'power_telemetry_available':power_ok,'power_telemetry_reason':(power_reason or None),'hip_version':hip,'cuda_version':cuda,'torch_arch_list':[str(x) for x in (torch.cuda.get_arch_list() if gpu_detected else [])]};"
        "\nif not gpu_detected:\n"
        " payload.update({'kernel_execution_test':False,'training_allowed':False,'gpu_kernel_reason':'torch.cuda.is_available() is false'}); print(json.dumps(payload)); raise SystemExit(0)\n"
        "\ntry:\n"
        " x=torch.ones(1024, device='cuda'); y=(x*2).sum().item(); torch.cuda.synchronize(); payload.update({'kernel_execution_test':True,'training_allowed':True,'kernel_result':y,'gpu_kernel_reason':None})\n"
        "except Exception as exc:\n"
        " payload.update({'kernel_execution_test':False,'training_allowed':False,'gpu_kernel_reason':str(exc).splitlines()[0]})\n"
        "print(json.dumps(payload))"
    )


def gpu_preflight_payload(settings: PortabilitySettings, host: str) -> Dict[str, Any]:
    verify_cmd = probe_command(settings, host, kernel_preflight_script())
    payload = run_probe(settings, host, kernel_preflight_script(), verify_cmd, settings.command_retries)
    return {
        "host": host,
        "verify_command": verify_cmd,
        **payload,
        "device_name": payload.get("device_name") or payload.get("gpu_model"),
        "vendor": payload.get("gpu_vendor"),
        "error": payload.get("error") or payload.get("gpu_kernel_reason"),
    }


def ensure_gpu_training_allowed(payload: Dict[str, Any]) -> None:
    if payload.get("training_allowed"):
        return
    arch = payload.get("gpu_architecture")
    torch_version = payload.get("torch_version")
    hip_version = payload.get("hip_version")
    error = payload.get("error")
    detail = {
        "gpu_requested": True,
        "gpu_detected": payload.get("gpu_detected"),
        "gpu_architecture": arch,
        "kernel_execution_test": payload.get("kernel_execution_test"),
        "training_allowed": False,
        "torch_version": torch_version,
        "hip_version": hip_version,
        "error": error,
    }
    raise RuntimeError(json.dumps(detail))


@op(ins={"settings": In(PortabilitySettings), "_ready": In(Nothing)}, out=Out(dict))
def preflight_gpu_kernel_execution(context: OpExecutionContext, settings: PortabilitySettings) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    checks_by_role: Dict[str, Dict[str, Any]] = {}
    if settings.source_use_gpu:
        source_check = {"role": "source", "gpu_requested": True, **gpu_preflight_payload(settings, settings.source_host)}
        checks.append(source_check)
        checks_by_role["source"] = source_check
    else:
        checks_by_role["source"] = {"role": "source", "gpu_requested": False, "training_allowed": True, "gpu_detected": None, "kernel_execution_test": None}
    if settings.target_use_gpu:
        target_check = {"role": "target", "gpu_requested": True, **gpu_preflight_payload(settings, settings.target_host)}
        checks.append(target_check)
        checks_by_role["target"] = target_check
    else:
        checks_by_role["target"] = {"role": "target", "gpu_requested": False, "training_allowed": True, "gpu_detected": None, "kernel_execution_test": None}
    failed = [payload for payload in checks if not payload.get("training_allowed")]
    if failed:
        emit_materialization(
            context,
            ["portability", "control", "preflight_gpu_kernel_execution"],
            "GPU kernel execution preflight failed.",
            {
                **common_metadata(settings, "failed", settings.remote_root),
                "lineage_path": MetadataValue.json(["control", "preflight_runtime_compatibility", "control", "preflight_gpu_kernel_execution"]),
                "results": MetadataValue.json(checks),
            },
        )
        ensure_gpu_training_allowed(failed[0])
    emit_materialization(
        context,
        ["portability", "control", "preflight_gpu_kernel_execution"],
        "GPU kernel execution preflight completed.",
        {
            **common_metadata(settings, "compatible", settings.remote_root),
            "lineage_path": MetadataValue.json(["control", "preflight_runtime_compatibility", "control", "preflight_gpu_kernel_execution"]),
            "results": MetadataValue.json(checks),
        },
    )
    return checks_by_role


@op(ins={"settings": In(PortabilitySettings), "_ready": In(Nothing)}, out=Out(Nothing))
def preflight_torch_transformers(context: OpExecutionContext, settings: PortabilitySettings) -> None:
    source_verify = probe_command(settings, settings.source_host, version_probe_script())
    target_verify = probe_command(settings, settings.target_host, version_probe_script())
    target_arch_verify = probe_command(settings, settings.target_host, architecture_probe_script())
    source_payload = run_probe(settings, settings.source_host, version_probe_script(), source_verify, settings.command_retries)
    target_payload = run_probe(settings, settings.target_host, version_probe_script(), target_verify, settings.command_retries)
    source_torch = str(source_payload.get("torch_version", ""))
    target_torch = str(target_payload.get("torch_version", ""))
    source_transformers = str(source_payload.get("transformers_version", ""))
    target_transformers = str(target_payload.get("transformers_version", ""))
    assert_torch_compatible(settings.source_host, source_torch, source_transformers, source_verify)
    assert_torch_compatible(settings.target_host, target_torch, target_transformers, target_verify)
    target_arch = run_probe_optional(settings, settings.target_host, architecture_probe_script(), target_arch_verify)
    warning = architecture_warning(settings.target_host, target_arch)
    if warning:
        context.log.warning(warning)
    emit_materialization(
        context,
        ["portability", "control", "preflight_runtime_compatibility"],
        "Preflight torch/transformers compatibility verified on source and target hosts.",
        {
            **common_metadata(settings, "compatible", settings.remote_root),
            "lineage_path": MetadataValue.json(["control", "environment_sync", "control", "preflight_runtime_compatibility"]),
            "required_torch_version": MIN_TORCH_VERSION,
            "cve": TORCH_SECURITY_CVE,
            "source_versions": MetadataValue.json({"host": settings.source_host, "torch": source_torch, "transformers": source_transformers, "verify_command": source_verify}),
            "target_versions": MetadataValue.json({"host": settings.target_host, "torch": target_torch, "transformers": target_transformers, "verify_command": target_verify}),
            "target_arch_probe": MetadataValue.json({"verify_command": target_arch_verify, **target_arch}),
            "target_arch_warning": warning or "",
        },
    )


def standalone_run_id(settings: PortabilitySettings, machine: MachineSpec) -> str:
    return f"standalone_{machine.key}_{settings.run_id}_{settings.standalone_benchmark.run_id_suffix}"


def standalone_roots(settings: PortabilitySettings, machine: MachineSpec) -> tuple[str, str]:
    host_root = expanded_remote_root(settings, machine.hostname)
    run_name = standalone_run_id(settings, machine)
    return f"{settings.remote_root}/{run_name}", f"{host_root}/{run_name}"


def standalone_dataset_path(settings: PortabilitySettings, machine: MachineSpec) -> str:
    relative = normalized_asset_path(settings.dataset_path)
    if machine.container_image and machine.execution_mode == "docker":
        return f"/workspace/{relative}" if not relative.startswith("/") else relative
    if relative.startswith(("/", "~/")):
        return relative if not relative.startswith("~/") else f"{expanded_remote_root(settings, machine.hostname)}/{relative[2:]}"
    return f"{expanded_remote_root(settings, machine.hostname)}/{relative}"


def standalone_training_command(settings: PortabilitySettings, machine: MachineSpec, host_run_root: str) -> str:
    sb = settings.standalone_benchmark
    dataset = standalone_dataset_path(settings, machine)
    run_name = Path(host_run_root).name
    container_output = f"/workspace/{run_name}"
    args = (
        f"--model-id {shlex.quote(sb.model_id)} "
        f"--dataset-path {shlex.quote(dataset)} "
        f"--output-dir {shlex.quote(host_run_root if not (machine.container_image and machine.execution_mode == 'docker') else container_output)} "
        f"--stop-step {sb.steps} "
        f"--max-steps {sb.max_steps} "
        f"--per-device-batch-size {sb.per_device_batch_size} "
        f"--max-length {sb.max_length} "
        f"--learning-rate {sb.learning_rate} "
        f"--lora-r {sb.lora_r} "
        f"--lora-alpha {sb.lora_alpha} "
        f"--lora-dropout {sb.lora_dropout} "
        f"--telemetry-interval-seconds {sb.telemetry_interval_seconds}"
    )
    if machine.container_image and machine.execution_mode == "docker":
        install = "python3 -m pip install -U pip >/tmp/pip.log 2>&1 && python3 -m pip install -r /workspace/requirements.txt >>/tmp/pip.log 2>&1"
        train = f"python3 /workspace/train_lora_migration.py {args}"
        return docker_command(settings, machine, f"{install} && {train}")
    python_cmd = shlex.quote(execution_python_cmd(settings, machine.hostname))
    script = f"{expanded_remote_root(settings, machine.hostname)}/train_lora_migration.py"
    env_prefix = machine_runtime_exports(machine)
    return f"{env_prefix}{python_cmd} {shlex.quote(script)} {args}"


def machine_runtime_exports(machine: MachineSpec) -> str:
    commands: List[str] = []
    if machine.hf_cache_dir:
        root = shlex.quote(machine.hf_cache_dir)
        hub = shlex.quote(f"{machine.hf_cache_dir.rstrip('/')}/hub")
        transformers = shlex.quote(f"{machine.hf_cache_dir.rstrip('/')}/transformers")
        xdg = shlex.quote(str(Path(machine.hf_cache_dir).parent))
        commands.append(f"mkdir -p {root} {hub} {transformers}")
        commands.append(f"export HF_HOME={root}")
        commands.append(f"export HUGGINGFACE_HUB_CACHE={hub}")
        commands.append(f"export HF_HUB_CACHE={hub}")
        commands.append(f"export TRANSFORMERS_CACHE={transformers}")
        commands.append(f"export XDG_CACHE_HOME={xdg}")
    if machine.rocm_path:
        commands.append(f"export ROCM_PATH={shlex.quote(machine.rocm_path)}")
    return "" if not commands else "; ".join(commands) + "; "


def ensure_required_preflight_fields(machine: MachineSpec, payload: Dict[str, Any]) -> Dict[str, str]:
    required = {
        "hostname": payload.get("hostname"),
        "gpu_vendor": payload.get("gpu_vendor"),
        "gpu_model": payload.get("gpu_model"),
        "gpu_architecture": payload.get("gpu_architecture"),
        "total_gpu_memory_bytes": payload.get("total_gpu_memory_bytes"),
        "torch_version": payload.get("torch_version"),
        "python_version": payload.get("python_version"),
        "gpu_detected": payload.get("gpu_detected"),
        "kernel_execution_test": payload.get("kernel_execution_test"),
        "power_telemetry_available": payload.get("power_telemetry_available"),
    }
    missing = {field: f"{field} not detected on host={machine.hostname}" for field, value in required.items() if value is None or value == ""}
    if str(payload.get("gpu_vendor", "")).lower() == "amd" and not payload.get("rocm_version"):
        missing["rocm_version"] = f"rocm_version not detected on host={machine.hostname}; torch.version.hip is empty"
    return missing


def architecture_matches(machine: MachineSpec, payload: Dict[str, Any]) -> bool:
    expected = machine.expected_gpu_architecture.strip().lower()
    actual = str(payload.get("gpu_architecture", "")).strip().lower()
    return bool(expected in {"", "unknown"} or (actual and expected in actual))


def machine_preflight_payload(settings: PortabilitySettings, machine: MachineSpec) -> Dict[str, Any]:
    payload = run_machine_probe(settings, machine, kernel_preflight_script())
    missing = ensure_required_preflight_fields(machine, payload)
    arch_ok = architecture_matches(machine, payload)
    arch_reason = None if arch_ok else f"expected {machine.expected_gpu_architecture} but detected {payload.get('gpu_architecture')}"
    return {
        "machine_key": machine.key,
        "hostname": machine.hostname,
        "container_image": machine.container_image,
        "expected_gpu_architecture": machine.expected_gpu_architecture,
        **payload,
        "missing_reasons": missing,
        "architecture_match": arch_ok,
        "architecture_reason": arch_reason,
    }


def benchmark_row(machine: MachineSpec, summary: Dict[str, Any], preflight: Dict[str, Any]) -> Dict[str, Any]:
    tokens = as_int(summary.get("useful_training_tokens"))
    runtime = as_float(summary.get("runtime_seconds"))
    avg_watts = as_float(summary.get("mean_power_watts"))
    tps = as_float(summary.get("tokens_per_second")) or safe_ratio(as_float(tokens), runtime)
    energy_joules = as_float(summary.get("gpu_energy_joules")) or (None if avg_watts is None or runtime is None else avg_watts * runtime)
    tpj = safe_ratio(as_float(tokens), None if avg_watts is None or runtime is None else avg_watts * runtime)
    cost_per_million = None if tpj is None else machine.electricity_price_per_kwh / (3.6 * tpj)
    reason = preflight.get("power_telemetry_reason") or "power telemetry unavailable"
    unavailable = {
        "average_power_watts": reason if avg_watts is None else None,
        "total_energy_joules": reason if energy_joules is None else None,
        "tokens_per_joule": reason if tpj is None else None,
        "energy_cost_per_million_tokens": reason if cost_per_million is None else None,
    }
    return {
        "machine_key": machine.key,
        "machine": machine_label(machine),
        "hostname": machine.hostname,
        "backend": "ROCm" if machine.vendor == "amd" else "CUDA",
        "vendor": machine.vendor,
        "device_name": machine.device_name,
        "expected_gpu_architecture": machine.expected_gpu_architecture,
        "detected_gpu_architecture": preflight.get("gpu_architecture"),
        "container_image": machine.container_image,
        "execution_mode": machine.execution_mode,
        "electricity_price_per_kwh": machine.electricity_price_per_kwh,
        "currency": machine.currency,
        "training_tokens_processed": tokens,
        "runtime_seconds": runtime,
        "tokens_per_second": tps,
        "average_power_watts": avg_watts,
        "total_energy_joules": energy_joules,
        "tokens_per_joule": tpj,
        "energy_cost_per_million_tokens": cost_per_million,
        "gpu_detected": preflight.get("gpu_detected"),
        "gpu_kernel_test_passed": preflight.get("kernel_execution_test"),
        "gpu_runtime_verified": summary.get("gpu_runtime_verified"),
        "power_telemetry_available": preflight.get("power_telemetry_available"),
        "power_telemetry_reason": preflight.get("power_telemetry_reason"),
        "metric_unavailable_reasons": {k: v for k, v in unavailable.items() if v},
        "required_field_reasons": preflight.get("missing_reasons", {}),
    }


def machine_label(machine: MachineSpec) -> str:
    aliases = {"nvidia_control": "NVIDIA control", "amd_existing": "Existing AMD", "mi300x": "MI300X"}
    return aliases.get(machine.key, machine.key)


def benchmark_row_errors(row: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not truthy(row.get("gpu_detected")):
        errors.append(f"{row.get('machine_key')}: torch did not detect GPU")
    if not truthy(row.get("gpu_kernel_test_passed")):
        errors.append(f"{row.get('machine_key')}: GPU kernel test failed")
    if not truthy(row.get("gpu_runtime_verified")):
        errors.append(f"{row.get('machine_key')}: training runtime did not verify GPU execution")
    if not truthy(row.get("power_telemetry_available")):
        errors.append(f"{row.get('machine_key')}: power telemetry unavailable ({row.get('power_telemetry_reason')})")
    for key in ["training_tokens_processed", "runtime_seconds", "tokens_per_second", "average_power_watts", "total_energy_joules", "tokens_per_joule", "energy_cost_per_million_tokens"]:
        value = as_float(row.get(key)) if key != "training_tokens_processed" else as_int(row.get(key))
        if value is None:
            errors.append(f"{row.get('machine_key')}: missing {key}")
    if row.get("required_field_reasons"):
        errors.append(f"{row.get('machine_key')}: missing preflight fields: {row.get('required_field_reasons')}")
    return errors


def render_standalone_table(rows: List[Dict[str, Any]]) -> str:
    header = "| Machine | Backend | Training tokens | Runtime | tok/s | Avg W | tok/J | Energy cost/M tokens |"
    divider = "|---|---|---:|---:|---:|---:|---:|---:|"
    lines = [header, divider]
    for row in rows:
        runtime = format_value(as_float(row.get("runtime_seconds")), "", 2)
        tps = format_value(as_float(row.get("tokens_per_second")), "", 2)
        watts = format_value(as_float(row.get("average_power_watts")), "", 2)
        tpj = format_value(as_float(row.get("tokens_per_joule")), "", 4)
        cost = format_value(as_float(row.get("energy_cost_per_million_tokens")), "", 6)
        lines.append(
            "| {machine} | {backend} | {tokens} | {runtime} | {tps} | {watts} | {tpj} | {cost} {currency} |".format(
                machine=row.get("machine"),
                backend=row.get("backend"),
                tokens=row.get("training_tokens_processed") or 0,
                runtime=runtime,
                tps=tps,
                watts=watts,
                tpj=tpj,
                cost=cost,
                currency=row.get("currency") or "GBP",
            )
        )
    return "\n".join(lines)


@op(ins={"settings": In(PortabilitySettings), "_ready": In(dict)}, out=Out(dict))
def run_standalone_benchmarks(context: OpExecutionContext, settings: PortabilitySettings, _ready: Dict[str, Any]) -> Dict[str, Any]:
    if not settings.standalone_benchmark.enabled:
        return {"enabled": False, "rows": [], "validation_errors": []}
    rows: List[Dict[str, Any]] = []
    preflight_rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    for machine in benchmark_machines(settings):
        _, host_run_root = standalone_roots(settings, machine)
        run_shell(ssh_cmd(settings, machine.hostname, f"rm -rf {shlex.quote(host_run_root)}"), settings.command_retries, env=sshpass_env(settings), allow_failure=True)
        preflight = machine_preflight_payload(settings, machine)
        preflight_rows.append(preflight)
        if preflight.get("missing_reasons"):
            errors.append(f"{machine.key}: missing preflight metadata {preflight.get('missing_reasons')}")
            continue
        if not preflight.get("architecture_match"):
            errors.append(f"{machine.key}: architecture mismatch {preflight.get('architecture_reason')}")
            continue
        if not preflight.get("training_allowed"):
            errors.append(f"{machine.key}: preflight training not allowed ({preflight.get('gpu_kernel_reason')})")
            continue
        run_shell(ssh_cmd(settings, machine.hostname, standalone_training_command(settings, machine, host_run_root)), settings.command_retries, env=sshpass_env(settings))
        summary = pull_json(settings, machine.hostname, f"{host_run_root}/run_summary.json")
        row = benchmark_row(machine, summary, preflight)
        rows.append(row)
        errors.extend(benchmark_row_errors(row))
    table = render_standalone_table(rows)
    payload = {
        "enabled": True,
        "rows": rows,
        "preflight": preflight_rows,
        "validation_errors": sorted(dict.fromkeys(errors)),
        "table": table,
    }
    local_root = Path(settings.local_root) / "artifacts" / settings.run_id
    local_root.mkdir(parents=True, exist_ok=True)
    json_path = local_root / "standalone_benchmark_comparison.json"
    text_path = local_root / "standalone_benchmark_comparison.md"
    json_path.write_text(json.dumps(payload, indent=2))
    text_path.write_text(table)
    emit_materialization(
        context,
        ["portability", "published", "standalone_benchmark_comparison"],
        "Standalone GPU benchmark matrix completed.",
        {
            **common_metadata(settings, "persisted", str(json_path)),
            "machines": MetadataValue.json([machine.key for machine in benchmark_machines(settings)]),
            "row_count": len(rows),
            "validation_errors": MetadataValue.json(payload["validation_errors"]),
            "table": MetadataValue.md(table),
        },
    )
    if errors:
        raise RuntimeError(f"standalone benchmark validation failed: {errors}")
    return payload


def ensure_mooncake_dependency(settings: PortabilitySettings, host: str) -> None:
    if settings.transfer.backend != "mooncake_tcp":
        return
    python_cmd = mooncake_python_cmd(settings, host)
    install = mooncake_install_command(python_cmd)
    verify = f"{shlex.quote(python_cmd)} -c \"from mooncake.engine import TransferEngine; print('mooncake ok')\""
    run_shell(ssh_cmd(settings, host, f"{install} && {verify}"), settings.command_retries, env=sshpass_env(settings))


def mooncake_install_command(python_cmd: str) -> str:
    quoted = shlex.quote(python_cmd)
    detect = f"{quoted} -c \"import sys; print(1 if sys.prefix != sys.base_prefix else 0)\""
    venv = f"{quoted} -m pip install --index-url https://pypi.org/simple mooncake-transfer-engine-non-cuda"
    system = f"{quoted} -m pip install --user --break-system-packages --index-url https://pypi.org/simple mooncake-transfer-engine-non-cuda"
    return f"IS_VENV=$({detect}) && [ \"$IS_VENV\" = \"1\" ] && {venv} || {system}"


def parse_sender_result(stdout: str) -> Dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    try:
        payload = json.loads(lines[-1]) if lines else {}
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"mooncake sender returned non-JSON output: {stdout}") from exc
    if not payload:
        raise RuntimeError("mooncake sender produced no JSON result")
    return payload


def verify_transfer_manifests(settings: PortabilitySettings, source: str, source_ckpt: str, destination: str, destination_ckpt: str) -> Dict[str, Any]:
    stamp = f"{settings.run_id}_{settings.checkpoint_step}_{int(time.time())}"
    source_manifest_remote = f"/tmp/source_manifest_{stamp}.json"
    destination_manifest_remote = f"/tmp/destination_manifest_{stamp}.json"
    source_manifest_local = Path("/tmp") / f"source_manifest_{stamp}.json"
    destination_manifest_local = Path("/tmp") / f"destination_manifest_{stamp}.json"
    src_py = shlex.quote(execution_python_cmd(settings, source))
    dst_py = shlex.quote(execution_python_cmd(settings, destination))
    src_cmd = f"{src_py} {settings.remote_root}/checkpoint_io.py --mode manifest --checkpoint-a {source_ckpt} --output {source_manifest_remote}"
    dst_cmd = f"{dst_py} {settings.remote_root}/checkpoint_io.py --mode manifest --checkpoint-a {destination_ckpt} --output {destination_manifest_remote}"
    run_shell(ssh_cmd(settings, source, src_cmd), settings.command_retries, env=sshpass_env(settings))
    run_shell(ssh_cmd(settings, destination, dst_cmd), settings.command_retries, env=sshpass_env(settings))
    run_shell(scp_cmd(settings, f"{source}:{source_manifest_remote}", str(source_manifest_local)), settings.command_retries, env=sshpass_env(settings))
    run_shell(scp_cmd(settings, f"{destination}:{destination_manifest_remote}", str(destination_manifest_local)), settings.command_retries, env=sshpass_env(settings))
    source_manifest = json.loads(source_manifest_local.read_text())
    destination_manifest = json.loads(destination_manifest_local.read_text())
    mismatches = strict_manifest_mismatches(source_manifest, destination_manifest)
    return {
        "checksum_ok": not mismatches,
        "mismatches": mismatches,
        "file_count": int(source_manifest.get("file_count", 0)),
        "total_bytes": int(source_manifest.get("total_bytes", 0)),
    }


def strict_manifest_mismatches(source_manifest: Dict[str, Any], destination_manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    destination_files = {item["relative_path"]: item for item in destination_manifest.get("files", [])}
    mismatches: List[Dict[str, Any]] = []
    for source_item in source_manifest.get("files", []):
        relative_path = source_item["relative_path"]
        destination_item = destination_files.get(relative_path)
        if destination_item is None:
            mismatches.append({"relative_path": relative_path, "error": "missing"})
            continue
        if int(source_item["size_bytes"]) != int(destination_item["size_bytes"]) or str(source_item["sha256"]) != str(destination_item["sha256"]):
            mismatches.append(
                {
                    "relative_path": relative_path,
                    "source_size": int(source_item["size_bytes"]),
                    "destination_size": int(destination_item["size_bytes"]),
                    "source_sha256": str(source_item["sha256"]),
                    "destination_sha256": str(destination_item["sha256"]),
                }
            )
    source_paths = {item["relative_path"] for item in source_manifest.get("files", [])}
    for relative_path in sorted(path for path in destination_files.keys() if path not in source_paths):
        mismatches.append({"relative_path": relative_path, "error": "unexpected"})
    return mismatches


@op(ins={"settings": In(PortabilitySettings), "_ready": In(Nothing)}, out=Out(TransferResult))
def copy_checkpoint(context: OpExecutionContext, settings: PortabilitySettings) -> TransferResult:
    result = transfer_checkpoint(settings, settings.source_host, settings.target_host, settings.transfer.backend)
    if not result.success:
        raise RuntimeError(f"checkpoint transfer failed via backend={result.backend}: {result.details}")
    emit_materialization(
        context,
        ["portability", "transfer", "checkpoint"],
        f"Checkpoint transferred from AMD to NVIDIA via backend={result.backend}.",
        {
            **common_metadata(settings, "transferred", f"{settings.remote_root}/{remote_ckpt(settings)}"),
            "lineage_path": MetadataValue.json(["source", "amd", "checkpoint", "transfer", "checkpoint", "target", "nvidia", "checkpoint"]),
            "provenance": MetadataValue.json(
                {
                    "source_checkpoint": f"{settings.remote_root}/amd_run_{settings.run_id}/{remote_ckpt(settings)}",
                    "target_checkpoint": f"{settings.remote_root}/{remote_ckpt(settings)}",
                    "backend": result.backend,
                    "details": result.details,
                }
            ),
            "transfer_backend": result.backend,
            "bytes_transferred": result.bytes_transferred,
            "elapsed_seconds": result.elapsed_seconds,
            "throughput_bytes_per_second": result.throughput_bytes_per_second,
            "checksum_ok": result.checksum_ok,
        },
    )
    return result


@op(ins={"settings": In(PortabilitySettings), "transfer_result": In(TransferResult)}, out=Out(TransferResult))
def verify_transfer(context: OpExecutionContext, settings: PortabilitySettings, transfer_result: TransferResult) -> TransferResult:
    if not transfer_result.checksum_ok:
        raise RuntimeError(f"checkpoint verification failed for backend={transfer_result.backend}: {transfer_result.details}")
    emit_materialization(
        context,
        ["portability", "transfer", "verification"],
        "Transfer verification passed; resume gate opened.",
        {
            **common_metadata(settings, "verified", f"{settings.remote_root}/{remote_ckpt(settings)}"),
            "lineage_path": MetadataValue.json(["transfer", "checkpoint", "transfer", "verification"]),
            "transfer_backend": transfer_result.backend,
            "bytes_transferred": transfer_result.bytes_transferred,
            "elapsed_seconds": transfer_result.elapsed_seconds,
            "throughput_bytes_per_second": transfer_result.throughput_bytes_per_second,
            "checksum_ok": transfer_result.checksum_ok,
        },
    )
    return transfer_result


@op(name="resume_on_target", ins={"settings": In(PortabilitySettings), "verified_transfer": In(TransferResult)}, out=Out(Nothing))
def resume_on_target(context: OpExecutionContext, settings: PortabilitySettings, verified_transfer: TransferResult) -> None:
    if not verified_transfer.checksum_ok:
        raise RuntimeError("resume_on_nvidia blocked because checkpoint transfer verification did not pass")
    python_cmd = shlex.quote(execution_python_cmd(settings, settings.target_host))
    device_prefix = "" if settings.target_use_gpu else "CUDA_VISIBLE_DEVICES='' "
    run = f"{device_prefix}{python_cmd} {settings.remote_root}/train_lora_migration.py --dataset-path {remote_dataset_path(settings)} --output-dir {settings.remote_root}/nvidia_run_{settings.run_id} --resume-from {settings.remote_root}/{remote_ckpt(settings)} --stop-step {settings.final_step} --max-steps {settings.final_step} --telemetry-interval-seconds {settings.training_telemetry_interval_seconds}"
    run_shell(ssh_cmd(settings, settings.target_host, run), settings.command_retries, env=sshpass_env(settings))
    emit_materialization(
        context,
        ["portability", "target", "nvidia", "resume_training"],
        "Training resumed from transferred checkpoint on target host.",
        {
            **common_metadata(settings, "resumed", f"{settings.remote_root}/nvidia_run_{settings.run_id}/run_summary.json"),
            "lineage_path": MetadataValue.json(["transfer", "checkpoint", "target", "nvidia", "resume_training"]),
            "provenance": MetadataValue.json({"resume_from": f"{settings.remote_root}/{remote_ckpt(settings)}", "runner": settings.target_host, "transfer_backend": verified_transfer.backend}),
        },
    )


@op(ins={"settings": In(PortabilitySettings), "_ready": In(Nothing), "verified_transfer": In(TransferResult)})
def validate_resume(context: OpExecutionContext, settings: PortabilitySettings, verified_transfer: TransferResult) -> None:
    resumed_checkpoint = f"{settings.remote_root}/nvidia_run_{settings.run_id}/checkpoint-{settings.final_step}"
    proof_out = f"{settings.remote_root}/nvidia_run_{settings.run_id}/resume_validation_proof.json"
    python_cmd = shlex.quote(execution_python_cmd(settings, settings.target_host))
    cmd = f"{python_cmd} {settings.remote_root}/validate_resume.py --source-checkpoint {settings.remote_root}/{remote_ckpt(settings)} --resumed-summary {settings.remote_root}/nvidia_run_{settings.run_id}/run_summary.json --resumed-checkpoint {resumed_checkpoint} --proof-out {proof_out} --min-extra-steps {settings.min_extra_steps} --max-loss-delta {settings.max_loss_delta} --max-optimizer-norm-delta {settings.max_optimizer_norm_delta}"
    run_shell(ssh_cmd(settings, settings.target_host, cmd), settings.command_retries, env=sshpass_env(settings))
    emit_materialization(
        context,
        ["portability", "target", "nvidia", "resume_validation"],
        "Resume validation completed.",
        {
            **common_metadata(settings, "validated", f"{settings.remote_root}/nvidia_run_{settings.run_id}/run_summary.json"),
            "lineage_path": MetadataValue.json(["target", "nvidia", "resume_training", "target", "nvidia", "resume_validation"]),
            "validation_policy": MetadataValue.json({"min_extra_steps": settings.min_extra_steps, "max_loss_delta": settings.max_loss_delta}),
            "proof_path": MetadataValue.path(proof_out),
            "transfer_backend": verified_transfer.backend,
            "resume_validation_ok": True,
        },
    )


def pull_json(settings: PortabilitySettings, host: str, remote_path: str) -> Dict[str, Any]:
    temp = Path("/tmp") / f"portability_pull_{settings.run_id}_{int(time.time() * 1000)}.json"
    run_shell(scp_cmd(settings, f"{host}:{remote_path}", str(temp)), settings.command_retries, env=sshpass_env(settings))
    payload = json.loads(temp.read_text())
    temp.unlink(missing_ok=True)
    return payload


def read_transfer_payload(settings: PortabilitySettings) -> Dict[str, Any]:
    path = f"{transfer_tests_dir(settings)}/transfer_test_matrix.json"
    return pull_json(settings, settings.target_host, path)


def read_resume_proof(settings: PortabilitySettings) -> Dict[str, Any]:
    path = f"{settings.remote_root}/nvidia_run_{settings.run_id}/resume_validation_proof.json"
    return pull_json(settings, settings.target_host, path)


def source_summary(settings: PortabilitySettings) -> Dict[str, Any]:
    path = f"{settings.remote_root}/amd_run_{settings.run_id}/run_summary.json"
    return pull_json(settings, settings.source_host, path)


def target_summary(settings: PortabilitySettings) -> Dict[str, Any]:
    path = f"{settings.remote_root}/nvidia_run_{settings.run_id}/run_summary.json"
    return pull_json(settings, settings.target_host, path)


def migration_to_first_step_seconds(report: Dict[str, Any]) -> Optional[float]:
    save_seconds = as_float(report.get("checkpoint_save_seconds"))
    transfer_seconds = as_float(report.get("transfer_seconds"))
    load_seconds = as_float(report.get("checkpoint_load_seconds"))
    resume_seconds = as_float(report.get("resume_to_first_step_seconds"))
    parts = [value for value in [save_seconds, transfer_seconds, load_seconds, resume_seconds] if value is not None]
    return None if not parts else float(sum(parts))


def ratio(destination_value: Any, source_value: Any) -> Optional[float]:
    return safe_ratio(as_float(destination_value), as_float(source_value))


def loss_after_resume(summary: Dict[str, Any], source_step: Optional[int]) -> Optional[float]:
    if source_step is None:
        return None
    trace = summary.get("loss_trace", []) if isinstance(summary, dict) else []
    rows = [item for item in trace if isinstance(item, dict) and as_int(item.get("step")) is not None]
    next_row = next((item for item in rows if int(item.get("step")) > source_step), None)
    return as_float((next_row or {}).get("loss"))


def transfer_to_first_step_seconds(migration: Dict[str, Any]) -> Optional[float]:
    transfer_seconds = as_float(migration.get("transfer_seconds"))
    load_seconds = as_float(migration.get("checkpoint_load_seconds"))
    resume_seconds = as_float(migration.get("resume_to_first_step_seconds"))
    parts = [value for value in [transfer_seconds, load_seconds, resume_seconds] if value is not None]
    return None if not parts else float(sum(parts))


def training_phase_errors(prefix: str, payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    useful = as_int(payload.get("useful_training_tokens"))
    runtime = as_float(payload.get("training_runtime_seconds"))
    tps = as_float(payload.get("tokens_per_second"))
    util_mean = as_float(payload.get("mean_gpu_utilization_percent"))
    util_peak = as_float(payload.get("peak_gpu_utilization_percent"))
    mem_peak = as_int(payload.get("peak_accelerator_memory_bytes"))
    power_mean = as_float(payload.get("mean_gpu_power_watts"))
    power_peak = as_float(payload.get("peak_gpu_power_watts"))
    energy_j = as_float(payload.get("gpu_energy_joules"))
    energy_kwh = as_float(payload.get("gpu_energy_kwh"))
    tpj = as_float(payload.get("tokens_per_joule"))
    add = errors.append
    if useful is not None and useful <= 0:
        add(f"{prefix}.useful_training_tokens must be > 0")
    if runtime is not None and runtime <= 0:
        add(f"{prefix}.training_runtime_seconds must be > 0")
    if tps is not None and tps <= 0:
        add(f"{prefix}.tokens_per_second must be > 0")
    if util_mean is not None and (util_mean < 0 or util_mean > 100):
        add(f"{prefix}.mean_gpu_utilization_percent outside [0,100]")
    if util_peak is not None and (util_peak < 0 or util_peak > 100):
        add(f"{prefix}.peak_gpu_utilization_percent outside [0,100]")
    if util_mean is not None and util_peak is not None and util_peak < util_mean:
        add(f"{prefix}.peak_gpu_utilization_percent must be >= mean")
    if mem_peak is not None and mem_peak <= 0:
        add(f"{prefix}.peak_accelerator_memory_bytes must be > 0")
    if power_mean is not None and power_mean <= 0:
        add(f"{prefix}.mean_gpu_power_watts must be > 0")
    if power_peak is not None and power_peak <= 0:
        add(f"{prefix}.peak_gpu_power_watts must be > 0")
    if power_mean is not None and power_peak is not None and power_peak < power_mean:
        add(f"{prefix}.peak_gpu_power_watts must be >= mean")
    if energy_j is not None and energy_j <= 0:
        add(f"{prefix}.gpu_energy_joules must be > 0")
    if energy_kwh is not None and energy_kwh <= 0:
        add(f"{prefix}.gpu_energy_kwh must be > 0")
    if tpj is not None and tpj <= 0:
        add(f"{prefix}.tokens_per_joule must be > 0")
    if energy_j is not None and energy_kwh is not None and abs((energy_j / 3_600_000.0) - energy_kwh) > 1e-6:
        add(f"{prefix}.gpu_energy_kwh inconsistent with gpu_energy_joules")
    return errors


def build_validation(
    settings: PortabilitySettings,
    source: Dict[str, Any],
    target: Dict[str, Any],
    migration: Dict[str, Any],
    continuity: Dict[str, Any],
    source_energy_validation: Dict[str, Any],
    target_energy_validation: Dict[str, Any],
    system_energy: Dict[str, Any],
) -> Dict[str, Any]:
    required_missing = [
        *metrics_missing_paths("source_training", source, SOURCE_PHASE_REQUIRED_FIELDS),
        *metrics_missing_paths("destination_training", target, DESTINATION_PHASE_REQUIRED_FIELDS),
    ]
    errors: List[str] = []
    warnings: List[str] = []
    expected = run_direction_expected_vendors(settings.run_id)
    if expected:
        if source.get("vendor") != expected[0] or target.get("vendor") != expected[1]:
            errors.append(f"run_id direction mismatch: expected source={expected[0]}, destination={expected[1]} but got source={source.get('vendor')}, destination={target.get('vendor')}")
    if settings.source_use_gpu and not truthy(source.get("gpu_execution_verified")):
        errors.append("source_training.gpu_execution_verified is false while gpu_requested=true")
    if settings.target_use_gpu and not truthy(target.get("gpu_execution_verified")):
        errors.append("destination_training.gpu_execution_verified is false while gpu_requested=true")
    source_first, source_last = as_int(source.get("first_step")), as_int(source.get("last_step"))
    target_first, target_last = as_int(target.get("first_step")), as_int(target.get("last_step"))
    if source_first is not None and source_last is not None and source_first > source_last:
        errors.append("source phase step bounds invalid")
    if target_first is not None and target_last is not None and target_first > target_last:
        errors.append("destination phase step bounds invalid")
    if source_last is not None and target_first is not None and target_first != source_last + 1:
        errors.append(f"phase continuity invalid: expected destination first step {source_last + 1}, got {target_first}")
    errors.extend(training_phase_errors("source_training", source))
    errors.extend(training_phase_errors("destination_training", target))
    if not truthy(migration.get("iperf3", {}).get("success")):
        errors.append("migration.iperf3.success is false")
    if not truthy(migration.get("rsync", {}).get("success")):
        errors.append("migration.rsync.success is false")
    if not truthy(migration.get("mooncake", {}).get("success")):
        errors.append("migration.mooncake.success is false")
    if not truthy(migration.get("checksum_verified")):
        errors.append("migration.checksum_verified is false")
    if as_float(migration.get("checkpoint_save_seconds")) is None:
        required_missing.append("migration.checkpoint_save_seconds")
    if as_float(migration.get("checkpoint_load_seconds")) is None:
        required_missing.append("migration.checkpoint_load_seconds")
    if as_float(migration.get("resume_to_first_step_seconds")) is None:
        required_missing.append("migration.resume_to_first_step_seconds")
    if as_float(migration.get("migration_to_first_step_seconds")) is None:
        required_missing.append("migration.migration_to_first_step_seconds")
    if as_int(migration.get("checkpoint_logical_size_bytes")) is None:
        required_missing.append("migration.checkpoint_logical_size_bytes")
    if as_int(migration.get("transfer_payload_bytes")) is None:
        required_missing.append("migration.transfer_payload_bytes")
    if not truthy(continuity.get("step_continuity_verified")):
        errors.append("continuity.step_continuity_verified is false")
    if not truthy(continuity.get("optimizer_state_verified")):
        errors.append("continuity.optimizer_state_verified is false")
    if not truthy(continuity.get("learning_rate_continuity_verified")):
        errors.append("continuity.learning_rate_continuity_verified is false")
    if not truthy(continuity.get("resume_continuity_verified")):
        errors.append("continuity.resume_continuity_verified is false")
    if not truthy(source_energy_validation.get("coverage_check_passed")):
        errors.append("Source telemetry coverage below required threshold")
    if not truthy(target_energy_validation.get("coverage_check_passed")):
        errors.append("Destination telemetry coverage below required threshold")
    if not truthy(source_energy_validation.get("sample_count_check_passed")):
        errors.append("Insufficient power telemetry samples for source training")
    if not truthy(target_energy_validation.get("sample_count_check_passed")):
        errors.append("Insufficient power telemetry samples for destination training")
    if not truthy(source_energy_validation.get("energy_consistency_check_passed")):
        errors.append("Source GPU energy consistency error above tolerance")
    if not truthy(target_energy_validation.get("energy_consistency_check_passed")):
        errors.append("Destination GPU energy consistency error above tolerance")
    source_runtime = as_float(source.get("training_runtime_seconds"))
    target_runtime = as_float(target.get("training_runtime_seconds"))
    if source_runtime is not None and source_runtime < settings.recommended_minimum_energy_benchmark_seconds:
        warnings.append(f"Source training phase is only {source_runtime:.2f} seconds; power-efficiency result may be noisy")
    if target_runtime is not None and target_runtime < settings.recommended_minimum_energy_benchmark_seconds:
        warnings.append(f"Destination training phase is only {target_runtime:.2f} seconds; power-efficiency result may be noisy")
    if system_energy.get("source", {}).get("type") == "not_measured" or system_energy.get("target", {}).get("type") == "not_measured":
        warnings.append("Whole-system energy is not measured")
    direction_ok = expected is None or (source.get("vendor") == expected[0] and target.get("vendor") == expected[1])
    source_steps_ok = source_first is not None and source_last is not None and source_first <= source_last
    target_steps_ok = target_first is not None and target_last is not None and target_first <= target_last
    sequential_steps_ok = source_last is not None and target_first is not None and target_first == source_last + 1
    functional_benchmark_complete = direction_ok and source_steps_ok and target_steps_ok and sequential_steps_ok and truthy(source.get("gpu_execution_verified")) and truthy(target.get("gpu_execution_verified")) and not required_missing
    cross_oem_resume_valid = truthy(continuity.get("step_continuity_verified")) and truthy(continuity.get("optimizer_state_verified")) and truthy(continuity.get("learning_rate_continuity_verified")) and truthy(continuity.get("resume_continuity_verified"))
    transfer_benchmark_valid = truthy(migration.get("iperf3", {}).get("success")) and truthy(migration.get("rsync", {}).get("success")) and truthy(migration.get("mooncake", {}).get("success")) and truthy(migration.get("checksum_verified"))
    energy_benchmark_valid = phase_energy_inputs_valid(source) and phase_energy_inputs_valid(target) and truthy(source_energy_validation.get("passed")) and truthy(target_energy_validation.get("passed"))
    economic_comparison_valid = energy_benchmark_valid and transfer_benchmark_valid and cross_oem_resume_valid and not required_missing
    benchmark_complete = functional_benchmark_complete and cross_oem_resume_valid and transfer_benchmark_valid and energy_benchmark_valid and economic_comparison_valid and not errors
    return {
        "benchmark_complete": benchmark_complete,
        "cross_oem_gpu_benchmark_valid": benchmark_complete,
        "functional_benchmark_complete": functional_benchmark_complete,
        "cross_oem_resume_valid": cross_oem_resume_valid,
        "transfer_benchmark_valid": transfer_benchmark_valid,
        "energy_benchmark_valid": energy_benchmark_valid,
        "economic_comparison_valid": economic_comparison_valid,
        "required_metrics_missing": sorted(dict.fromkeys(required_missing)),
        "validation_errors": sorted(dict.fromkeys(errors)),
        "validation_warnings": sorted(dict.fromkeys(warnings)),
        "field_classification": required_metric_classification(source, target),
        "optional_unavailable_reasons": {
            "source_training.system_energy_kwh": "not_measured",
            "destination_training.system_energy_kwh": "not_measured",
            "system_energy.source": "not_measured" if system_energy.get("source", {}).get("type") == "not_measured" else "estimated",
            "system_energy.target": "not_measured" if system_energy.get("target", {}).get("type") == "not_measured" else "estimated",
        },
    }


@op(ins={"settings": In(PortabilitySettings), "_ready": In(Nothing), "verified_transfer": In(TransferResult), "gpu_preflight": In(dict), "standalone_benchmark": In(dict)}, out=Out(Nothing))
def build_cross_oem_report(context: OpExecutionContext, settings: PortabilitySettings, verified_transfer: TransferResult, gpu_preflight: Dict[str, Any], standalone_benchmark: Dict[str, Any]) -> None:
    source_raw = source_summary(settings)
    target_raw = target_summary(settings)
    source = normalize_machine_summary(settings, source_raw, gpu_preflight.get("source", {}), settings.source_use_gpu, "source")
    target = normalize_machine_summary(settings, target_raw, gpu_preflight.get("target", {}), settings.target_use_gpu, "target")
    transfer_payload = read_transfer_payload(settings)
    proof = read_resume_proof(settings)
    iperf3 = extract_transfer_record(transfer_payload, "iperf3").get("details", {})
    rsync = extract_transfer_record(transfer_payload, "SCP/rsync").get("details", {})
    mooncake = extract_transfer_record(transfer_payload, "Mooncake TCP").get("details", {})
    transfer_comparison = transfer_payload.get("comparison", {}) if isinstance(transfer_payload, dict) else {}
    migration = {
        "checkpoint_logical_size_bytes": source.get("checkpoint_size_bytes"),
        "transfer_payload_bytes": as_int(mooncake.get("bytes_transferred")),
        "checkpoint_save_seconds": source.get("checkpoint_save_seconds"),
        "iperf3": {
            "success": iperf3.get("success"),
            "throughput_bits_per_second": as_float(iperf3.get("throughput_bits_per_second")),
        },
        "rsync": {
            "success": rsync.get("success"),
            "bytes_transferred": as_int(rsync.get("bytes_transferred")),
            "elapsed_seconds": as_float(rsync.get("elapsed_seconds")),
            "throughput_bytes_per_second": as_float(rsync.get("throughput_bytes_per_second")),
        },
        "mooncake": {
            "success": mooncake.get("success"),
            "bytes_transferred": as_int(mooncake.get("bytes_transferred")),
            "elapsed_seconds": as_float(mooncake.get("elapsed_seconds")),
            "throughput_bytes_per_second": as_float(mooncake.get("throughput_bytes_per_second")),
        },
        "mooncake_vs_iperf_percent": as_float(transfer_comparison.get("mooncake_vs_iperf_percent")),
        "mooncake_vs_rsync_percent": as_float(transfer_comparison.get("mooncake_vs_rsync_percent")),
        "checkpoint_load_seconds": target.get("checkpoint_load_seconds"),
        "resume_to_first_step_seconds": target.get("resume_to_first_step_seconds"),
        "transfer_seconds": migration_transfer_seconds(settings, transfer_payload, verified_transfer),
        "checksum_verified": bool(verified_transfer.checksum_ok),
    }
    migration["transfer_to_first_step_seconds"] = transfer_to_first_step_seconds(migration)
    migration["migration_to_first_step_seconds"] = migration_to_first_step_seconds(
        {
            "checkpoint_save_seconds": migration.get("checkpoint_save_seconds"),
            "checkpoint_load_seconds": migration.get("checkpoint_load_seconds"),
            "resume_to_first_step_seconds": migration.get("resume_to_first_step_seconds"),
            "transfer_seconds": migration.get("transfer_seconds"),
        }
    )
    last_source_step = as_int(proof.get("source_saved_step")) or source.get("last_step")
    continuity = {
        "last_source_step": last_source_step,
        "first_destination_step": as_int(proof.get("first_step_after_handoff")) or target.get("first_step"),
        "loss_before_checkpoint": as_float(source_raw.get("loss_before_migration")) or source.get("final_loss"),
        "loss_after_resume": loss_after_resume(target_raw, last_source_step),
        "optimizer_state_before_save": as_float(proof.get("source_optimizer_exp_avg_sq_norm_before_save")),
        "optimizer_state_after_load": as_float(proof.get("resumed_optimizer_exp_avg_sq_norm_after_resume_load")),
        "learning_rate_before": as_float(proof.get("source_learning_rate_before_save")) or source.get("optimizer_learning_rate_before_save"),
        "learning_rate_after": as_float(proof.get("resumed_learning_rate_after_resume_load")) or target.get("optimizer_learning_rate_after_resume_load"),
        "continuity_verification_result": proof.get("checkpoint_validation_result"),
        "continuity_policy": proof.get("continuity_policy"),
        "step_continuity_verified": proof.get("step_continuity_verified"),
        "optimizer_state_verified": proof.get("optimizer_state_verified"),
        "learning_rate_continuity_verified": proof.get("learning_rate_continuity_verified"),
        "resume_continuity_verified": bool(proof.get("resume_continuity_verified", False)),
    }
    source_energy_validation = energy_validation_payload(settings, source)
    target_energy_validation = energy_validation_payload(settings, target)
    source["energy_validation"] = source_energy_validation
    target["energy_validation"] = target_energy_validation
    system_energy = {
        "source": system_energy_entry(source.get("training_runtime_seconds"), source.get("gpu_energy_joules"), settings.source_system_overhead_watts),
        "target": system_energy_entry(target.get("training_runtime_seconds"), target.get("gpu_energy_joules"), settings.target_system_overhead_watts),
    }
    source_system_cost = energy_cost_gbp(as_float(system_energy["source"].get("energy_kwh")), settings.electricity_price_gbp_per_kwh)
    target_system_cost = energy_cost_gbp(as_float(system_energy["target"].get("energy_kwh")), settings.electricity_price_gbp_per_kwh)
    source["system_energy_cost_gbp"] = source_system_cost
    source["system_energy_cost_per_million_tokens_gbp"] = cost_per_million_tokens(source_system_cost, as_int(source.get("useful_training_tokens")))
    source["system_energy_break_even_per_million_tokens_gbp"] = source.get("system_energy_cost_per_million_tokens_gbp")
    source["system_energy_cost_estimated"] = system_energy["source"].get("type") == "estimated"
    target["system_energy_cost_gbp"] = target_system_cost
    target["system_energy_cost_per_million_tokens_gbp"] = cost_per_million_tokens(target_system_cost, as_int(target.get("useful_training_tokens")))
    target["system_energy_break_even_per_million_tokens_gbp"] = target.get("system_energy_cost_per_million_tokens_gbp")
    target["system_energy_cost_estimated"] = system_energy["target"].get("type") == "estimated"
    comparison = {
        "higher_tokens_per_second": None,
        "higher_tokens_per_joule": None,
        "lower_energy_cost_per_million_tokens": None,
        "tokens_per_second_ratio_destination_vs_source": None,
        "tokens_per_joule_ratio_destination_vs_source": None,
        "energy_cost_per_million_tokens_ratio_destination_vs_source": None,
    }
    validation = build_validation(settings, source, target, migration, continuity, source_energy_validation, target_energy_validation, system_energy)
    if validation.get("economic_comparison_valid"):
        comparison = {
        "higher_tokens_per_second": metric_winner(source, target, "tokens_per_second"),
        "higher_tokens_per_joule": metric_winner(source, target, "tokens_per_joule"),
        "lower_energy_cost_per_million_tokens": metric_winner(source, target, "energy_cost_per_million_tokens_gbp", lower_is_better=True),
        "tokens_per_second_ratio_destination_vs_source": ratio(target.get("tokens_per_second"), source.get("tokens_per_second")),
        "tokens_per_joule_ratio_destination_vs_source": ratio(target.get("tokens_per_joule"), source.get("tokens_per_joule")),
        "energy_cost_per_million_tokens_ratio_destination_vs_source": ratio(target.get("energy_cost_per_million_tokens_gbp"), source.get("energy_cost_per_million_tokens_gbp")),
    }
    report = {
        "run": {
            "run_id": settings.run_id,
            "model_id": source_raw.get("model_id") or target_raw.get("model_id"),
            "training_type": source_raw.get("training_type") or target_raw.get("training_type") or "lora",
            "dataset_path": settings.dataset_path,
            "electricity_price_gbp_per_kwh": settings.electricity_price_gbp_per_kwh,
            "economic_scope": "energy-only break-even; excludes hardware, CPU/system energy, networking, storage, cooling, and operations",
        },
        "source_training": source,
        "destination_training": target,
        "migration": migration,
        "continuity": continuity,
        "system_energy": system_energy,
        "comparison": comparison,
        "validation": validation,
        "standalone_benchmark": standalone_benchmark,
    }
    report["run"]["cross_oem_gpu_benchmark_valid"] = bool(validation.get("cross_oem_gpu_benchmark_valid"))
    local_root = Path(settings.local_root) / "artifacts" / settings.run_id
    local_root.mkdir(parents=True, exist_ok=True)
    json_path = local_root / "cross_oem_metrics_report.json"
    text_path = local_root / "cross_oem_metrics_summary.txt"
    json_path.write_text(json.dumps(report, indent=2))
    text_path.write_text(render_side_by_side(report))
    if not validation.get("cross_oem_gpu_benchmark_valid"):
        context.log.warning(f"cross_oem_gpu_benchmark_valid=false: missing={validation.get('required_metrics_missing')} errors={validation.get('validation_errors')}")
    emit_materialization(
        context,
        ["portability", "published", "cross_oem_metrics_report"],
        "Cross-OEM metrics report generated from normalized machine metrics.",
        {
            **common_metadata(settings, "persisted", str(json_path)),
            "text_report": MetadataValue.path(str(text_path)),
            "cross_oem_gpu_benchmark_valid": bool(validation.get("cross_oem_gpu_benchmark_valid")),
            "benchmark_complete": bool(validation.get("benchmark_complete")),
            "required_metrics_missing": MetadataValue.json(validation.get("required_metrics_missing", [])),
            "validation_errors": MetadataValue.json(validation.get("validation_errors", [])),
            "standalone_validation_errors": MetadataValue.json(standalone_benchmark.get("validation_errors", [])),
            "higher_tokens_per_second": comparison.get("higher_tokens_per_second") or "unavailable",
            "higher_tokens_per_joule": comparison.get("higher_tokens_per_joule") or "unavailable",
            "lower_energy_cost_per_million_tokens": comparison.get("lower_energy_cost_per_million_tokens") or "unavailable",
            "resume_continuity_verified": bool(continuity.get("resume_continuity_verified")),
        },
    )


@op(ins={"settings": In(PortabilitySettings), "_ready": In(Nothing)})
def save_assets(context: OpExecutionContext, settings: PortabilitySettings) -> None:
    saved_assets = []
    for asset_path in output_asset_paths(settings):
        save_asset_from_host(settings, settings.target_host, asset_path)
        if not saved_local_asset_path(settings, asset_path).exists():
            save_asset_from_host(settings, settings.source_host, asset_path)
        local_saved_path = saved_local_asset_path(settings, asset_path)
        data_type = infer_data_type(local_saved_path) if local_saved_path.exists() else "missing"
        saved_assets.append({"asset_path": asset_path, "saved_path": str(local_saved_path), "data_type": data_type})
        emit_materialization(
            context,
            ["portability", "published", *asset_key_for_path(asset_path)[2:]],
            "Data asset copied from NVIDIA host to local artifacts.",
            {
                **common_metadata(settings, "persisted", str(local_saved_path)),
                "remote_source": MetadataValue.path(f"{settings.remote_root}/{asset_path}"),
                "local_destination": MetadataValue.path(str(local_saved_path)),
                "data_type": data_type,
                "lineage_path": MetadataValue.json(["target", "nvidia", "resume_validation", "published", asset_path]),
                "provenance": MetadataValue.json({"copied_from": settings.target_host, "pipeline_step": "save_assets"}),
            },
        )
    emit_materialization(
        context,
        PUBLISHED_MANIFEST_KEY.path,
        "Manifest of all assets persisted from NVIDIA to local artifacts.",
        {
            **common_metadata(settings, "persisted", str(Path(settings.local_root) / "artifacts" / settings.run_id)),
            "saved_assets": MetadataValue.json(saved_assets),
            "lineage_path": MetadataValue.json(["target", "nvidia", "resume_validation", "published", "assets_manifest"]),
        },
    )


def saved_assets_for_checks(settings: PortabilitySettings) -> List[Path]:
    return [saved_local_asset_path(settings, asset_path) for asset_path in output_asset_paths(settings)]


@asset_check(asset=PUBLISHED_MANIFEST_KEY, name="saved_assets_exist")
def saved_assets_exist_check() -> AssetCheckResult:
    settings = load_settings()
    saved_paths = saved_assets_for_checks(settings)
    missing = [str(path) for path in saved_paths if not path.exists()]
    empty = [str(path) for path in saved_paths if path.exists() and path.is_file() and path.stat().st_size == 0]
    return AssetCheckResult(passed=not missing and not empty, metadata={"missing_paths": MetadataValue.json(missing), "empty_files": MetadataValue.json(empty), "checked_paths": MetadataValue.json([str(path) for path in saved_paths])})


@asset_check(asset=PUBLISHED_MANIFEST_KEY, name="saved_assets_supported_type")
def saved_assets_supported_type_check() -> AssetCheckResult:
    settings = load_settings()
    saved_paths = [path for path in saved_assets_for_checks(settings) if path.exists()]
    typed_assets = [{"path": str(path), "data_type": infer_data_type(path)} for path in saved_paths]
    unsupported = [item for item in typed_assets if not is_supported_data_type(item["data_type"])]
    return AssetCheckResult(passed=not unsupported, metadata={"typed_assets": MetadataValue.json(typed_assets), "unsupported_assets": MetadataValue.json(unsupported), "supported_types": MetadataValue.json(sorted(["directory", "json", "jsonl", "csv", "parquet", "arrow", "sqlite", "db", "txt"]))})


@job
def portability_job() -> None:
    settings = settings_op()
    prepared = prepare_hosts(settings)
    synced = sync_and_install(settings, prepared)
    preflight = preflight_torch_transformers(settings, synced)
    gpu_ready = preflight_gpu_kernel_execution(settings, preflight)
    standalone = run_standalone_benchmarks(settings, gpu_ready)
    trained = train_on_source(settings=settings, gpu_preflight=gpu_ready, standalone_result=standalone)
    copied = copy_checkpoint(settings, trained)
    verified = verify_transfer(settings, copied)
    resumed = resume_on_target(settings, verified)
    validated = validate_resume(settings=settings, _ready=resumed, verified_transfer=verified)
    recorded = record_transfer_tests(settings=settings, transfer_result=verified, _ready=validated)
    report = build_cross_oem_report(settings=settings, _ready=recorded, verified_transfer=verified, gpu_preflight=gpu_ready, standalone_benchmark=standalone)
    save_assets(settings, report)


if __name__ == "__main__":
    result = portability_job.execute_in_process()
    raise SystemExit(0 if result.success else 1)
