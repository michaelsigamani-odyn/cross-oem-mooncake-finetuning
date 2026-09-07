from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import argparse
import inspect
import json
import math
import os
import platform
import random
import re
import shutil
import socket
import subprocess
import threading
import time

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling, Trainer
from transformers import TrainerCallback, TrainingArguments


@dataclass
class TrainConfig:
    model_id: str
    dataset_path: str
    output_dir: str
    resume_from: Optional[str]
    stop_step: int
    max_steps: int
    per_device_batch_size: int
    learning_rate: float
    seed: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    ddp_timeout_seconds: int
    max_length: int
    telemetry_interval_seconds: float


@dataclass
class TelemetrySample:
    timestamp: float
    gpu_utilization_percent: Optional[float]
    memory_used_bytes: Optional[int]
    power_watts: Optional[float]


@dataclass
class RuntimeMetrics:
    checkpoint_size_bytes: Optional[int]
    checkpoint_save_seconds: Optional[float]
    checkpoint_load_seconds: Optional[float]
    resume_to_first_step_seconds: Optional[float]
    first_step_after_resume: Optional[int]
    telemetry_start_timestamp: Optional[str]
    telemetry_end_timestamp: Optional[str]


@dataclass
class EnvironmentDiagnostics:
    hostname: str
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_version: Optional[str]
    hip_version: Optional[str]
    gpu_name: str
    model_device: str
    trainer_device: str


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    return to_config(args)


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--dataset-path", default="./data/quant_mentor_500_alpaca.jsonl")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--stop-step", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=140)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--ddp-timeout-seconds", type=int, default=300)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--telemetry-interval-seconds", type=float, default=0.25)


def to_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(args.model_id, args.dataset_path, args.output_dir, args.resume_from, args.stop_step, args.max_steps, args.per_device_batch_size, args.learning_rate, args.seed, args.lora_r, args.lora_alpha, args.lora_dropout, args.ddp_timeout_seconds, args.max_length, args.telemetry_interval_seconds)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataset(path: str) -> Dataset:
    records = read_jsonl(path)
    return Dataset.from_list([{"text": render_record(record)} for record in records])


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    assert rows, f"dataset is empty: {path}"
    return rows


def render_record(record: Dict[str, Any]) -> str:
    instruction = str(record.get("instruction", "")).strip()
    input_text = str(record.get("input", "")).strip()
    output_text = str(record.get("output", "")).strip()
    return "\n".join([f"Instruction: {instruction}", f"Input: {input_text or 'N/A'}", f"Response: {output_text}"])


def tokenize_rows(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    fn = lambda batch: tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)
    tokenized = dataset.map(fn, batched=True, remove_columns=["text"])
    return tokenized.with_format(type="torch")


def build_tokenizer(model_id: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def supports_bf16() -> bool:
    return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())


def build_model(cfg: TrainConfig) -> Any:
    dtype = torch.bfloat16 if supports_bf16() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=dtype)
    return apply_lora(model, cfg)


def apply_lora(model: Any, cfg: TrainConfig) -> Any:
    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, target_modules=["q_proj", "v_proj"], bias="none")
    return get_peft_model(model, lora)


def build_args(cfg: TrainConfig) -> TrainingArguments:
    bf16 = supports_bf16()
    kwargs = training_kwargs(cfg, bf16)
    return TrainingArguments(**supported_training_kwargs(kwargs))


def training_kwargs(cfg: TrainConfig, bf16: bool) -> Dict[str, Any]:
    return {"output_dir": cfg.output_dir, "per_device_train_batch_size": cfg.per_device_batch_size, "learning_rate": cfg.learning_rate, "logging_steps": 1, "max_steps": cfg.max_steps, "lr_scheduler_type": "linear", "warmup_ratio": 0.03, "save_strategy": "steps", "save_steps": cfg.stop_step, "save_total_limit": 2, "bf16": bf16, "fp16": not bf16, "optim": "adamw_torch", "report_to": [], "ddp_timeout": cfg.ddp_timeout_seconds}


def distributed_env_payload() -> Dict[str, Any]:
    local_ip = socket.gethostbyname(socket.gethostname())
    fields = ["MASTER_ADDR", "MASTER_PORT", "NNODES", "NODE_RANK", "WORLD_SIZE", "RANK", "LOCAL_RANK", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME"]
    return {**{name: str(os.environ.get(name, "")) for name in fields}, "hostname": socket.gethostname(), "host_ip": local_ip}


def log_distributed_env() -> None:
    print("DISTRIBUTED_ENV", json.dumps(distributed_env_payload(), sort_keys=True))


def supported_training_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    names = inspect.signature(TrainingArguments.__init__).parameters
    return {key: value for key, value in kwargs.items() if key in names}


def checkpoint_path(cfg: TrainConfig) -> Path:
    return Path(cfg.output_dir) / f"checkpoint-{cfg.stop_step}"


def resume_step(path: Optional[str]) -> Optional[int]:
    if not path:
        return None
    name = Path(path).name
    return int(name.split("checkpoint-")[-1]) if name.startswith("checkpoint-") else None


def last_loss(logs: List[Dict[str, Any]]) -> Optional[float]:
    losses = [entry.get("loss") for entry in logs if "loss" in entry]
    return float(losses[-1]) if losses else None


def first_logged_step(logs: List[Dict[str, Any]]) -> Optional[int]:
    steps = [entry.get("step") for entry in logs if "step" in entry]
    return int(steps[0]) if steps else None


def last_logged_step(logs: List[Dict[str, Any]]) -> Optional[int]:
    steps = [entry.get("step") for entry in logs if "step" in entry]
    return int(steps[-1]) if steps else None


def loss_trace(logs: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    rows = [{"step": int(entry["step"]), "loss": float(entry["loss"])} for entry in logs if "step" in entry and "loss" in entry]
    return sorted(rows, key=lambda item: item["step"])


def optimizer_exp_avg_sq_norm(optimizer: Any) -> Optional[float]:
    state = optimizer.state_dict().get("state", {}) if optimizer else {}
    values = [tensor.float().pow(2).sum().item() for item in state.values() for key, tensor in item.items() if key == "exp_avg_sq"]
    return math.sqrt(sum(values)) if values else None


def trainer_optimizer_norm(trainer: Trainer) -> Optional[float]:
    return optimizer_exp_avg_sq_norm(getattr(trainer, "optimizer", None))


def optimizer_learning_rate(optimizer: Any) -> Optional[float]:
    groups = list(getattr(optimizer, "param_groups", [])) if optimizer else []
    return None if not groups else float(groups[0].get("lr"))


def trainer_learning_rate(trainer: Trainer) -> Optional[float]:
    return optimizer_learning_rate(getattr(trainer, "optimizer", None))


def resume_probe_norm(trainer: Trainer) -> Optional[float]:
    probe = getattr(trainer, "_resume_probe", None)
    return None if probe is None else probe.norm_after_resume_load


def resume_probe_learning_rate(trainer: Trainer) -> Optional[float]:
    probe = getattr(trainer, "_resume_probe", None)
    return None if probe is None else probe.learning_rate_after_resume_load


def write_checkpoint_meta(trainer: Trainer, cfg: TrainConfig, monitor: "GpuTelemetryMonitor", runtime_seconds: float, runtime: RuntimeMetrics) -> None:
    path = checkpoint_path(cfg) / "checkpoint_meta.json"
    payload = meta_payload(trainer, cfg, monitor, runtime_seconds, runtime)
    path.write_text(json.dumps(payload, indent=2))


def write_run_summary(trainer: Trainer, cfg: TrainConfig, monitor: "GpuTelemetryMonitor", runtime_seconds: float, runtime: RuntimeMetrics) -> None:
    path = Path(cfg.output_dir) / "run_summary.json"
    payload = meta_payload(trainer, cfg, monitor, runtime_seconds, runtime)
    path.write_text(json.dumps(payload, indent=2))


def meta_payload(trainer: Trainer, cfg: TrainConfig, monitor: "GpuTelemetryMonitor", runtime_seconds: float, runtime: RuntimeMetrics) -> Dict[str, Any]:
    state = trainer.state
    optimizer_norm = trainer_optimizer_norm(trainer)
    saved_step = int(state.global_step)
    useful_tokens = useful_training_tokens(cfg, saved_step)
    gpu = gpu_metrics(monitor, runtime_seconds)
    gpu_energy_kwh = ratio(gpu.get("gpu_energy_joules"), 3_600_000.0)
    energy_kwh_per_million_tokens = ratio((1_000_000.0 * gpu_energy_kwh) if gpu_energy_kwh is not None else None, useful_tokens)
    peak_mem = peak_memory_bytes(gpu.get("peak_sampled_memory_bytes"))
    gpu_runtime_verified = bool(cuda_available_flag() and trainer_device_name(trainer).startswith("cuda") and model_device_name(trainer).startswith("cuda"))
    return {
        "saved_step": saved_step,
        "resume_loaded_from_step": resume_step(cfg.resume_from),
        "loss_before_migration": last_loss(state.log_history),
        "loss_trace": loss_trace(state.log_history),
        "first_logged_step": first_logged_step(state.log_history),
        "last_logged_step": last_logged_step(state.log_history),
        "optimizer_exp_avg_sq_norm": optimizer_norm,
        "optimizer_exp_avg_sq_norm_before_save": optimizer_norm,
        "optimizer_exp_avg_sq_norm_after_resume_load": resume_probe_norm(trainer),
        "learning_rate": trainer_learning_rate(trainer),
        "optimizer_learning_rate_before_save": trainer_learning_rate(trainer),
        "optimizer_learning_rate_after_resume_load": resume_probe_learning_rate(trainer),
        "seed": cfg.seed,
        "model_id": cfg.model_id,
        "dataset_path": cfg.dataset_path,
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "device": device_vendor(),
        "vendor": gpu_vendor(),
        "gpu_architecture": gpu_architecture(),
        "torch": torch.__version__,
        "torch_version": torch.__version__,
        "resume_from": cfg.resume_from,
        "runtime_seconds": runtime_seconds,
        "useful_training_tokens": useful_tokens,
        "tokens_per_second": ratio(useful_tokens, runtime_seconds),
        "final_loss": last_loss(state.log_history),
        "mean_gpu_utilization_percent": gpu.get("mean_gpu_utilization_percent"),
        "peak_gpu_utilization_percent": gpu.get("peak_gpu_utilization_percent"),
        "peak_accelerator_memory_bytes": peak_mem,
        "mean_power_watts": gpu.get("mean_power_watts"),
        "peak_power_watts": gpu.get("peak_power_watts"),
        "min_power_watts": gpu.get("min_power_watts"),
        "gpu_energy_joules": gpu.get("gpu_energy_joules"),
        "gpu_energy_kwh": gpu_energy_kwh,
        "tokens_per_joule": ratio(useful_tokens, gpu.get("gpu_energy_joules")),
        "energy_kwh_per_million_tokens": energy_kwh_per_million_tokens,
        "power_sample_count": gpu.get("power_sample_count"),
        "power_telemetry_available": gpu.get("power_telemetry_available"),
        "power_telemetry_source": gpu.get("power_telemetry_source"),
        "power_telemetry_reason": gpu.get("power_telemetry_reason"),
        "telemetry_source": gpu.get("telemetry_source"),
        "sampling_interval_seconds": cfg.telemetry_interval_seconds,
        "telemetry_duration_seconds": gpu.get("telemetry_duration_seconds"),
        "telemetry_sample_count": gpu.get("telemetry_sample_count"),
        "raw_telemetry_sample_count": gpu.get("telemetry_sample_count"),
        "telemetry_start_timestamp": runtime.telemetry_start_timestamp,
        "telemetry_end_timestamp": runtime.telemetry_end_timestamp,
        "gpu_runtime_verified": gpu_runtime_verified,
        "checkpoint_size_bytes": runtime.checkpoint_size_bytes,
        "checkpoint_save_seconds": runtime.checkpoint_save_seconds,
        "checkpoint_load_seconds": runtime.checkpoint_load_seconds,
        "resume_to_first_step_seconds": runtime.resume_to_first_step_seconds,
        "first_step_after_resume": runtime.first_step_after_resume,
        "max_sequence_length": cfg.max_length,
    }


def ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def useful_training_tokens(cfg: TrainConfig, saved_step: int) -> int:
    start = resume_step(cfg.resume_from) or 0
    return max(saved_step - start, 0) * cfg.per_device_batch_size * cfg.max_length


def gpu_vendor() -> str:
    if hip_version_string():
        return "amd"
    if cuda_version_string():
        return "nvidia"
    return "unknown"


def gpu_architecture() -> Optional[str]:
    if not torch.cuda.is_available():
        return None
    if hip_version_string():
        props = torch.cuda.get_device_properties(0)
        return str(getattr(props, "gcnArchName", "")).split(":", 1)[0] or None
    major, minor = torch.cuda.get_device_capability(0)
    return f"sm_{major}{minor}"


def peak_memory_bytes(sampled_peak: Optional[int]) -> Optional[int]:
    peak_alloc = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    values = [value for value in [sampled_peak, peak_alloc] if value is not None]
    return None if not values else max(values)


def gpu_metrics(monitor: "GpuTelemetryMonitor", runtime_seconds: float) -> Dict[str, Any]:
    util = [x.gpu_utilization_percent for x in monitor.samples if x.gpu_utilization_percent is not None]
    power = [x.power_watts for x in monitor.samples if x.power_watts is not None]
    peak_mem = [x.memory_used_bytes for x in monitor.samples if x.memory_used_bytes is not None]
    energy = integrate_energy_joules(monitor.samples) or ratio(sum(power) * runtime_seconds, len(power) or None)
    return {
        "mean_gpu_utilization_percent": ratio(sum(util), len(util) or None),
        "peak_gpu_utilization_percent": max(util) if util else None,
        "peak_sampled_memory_bytes": max(peak_mem) if peak_mem else None,
        "mean_power_watts": ratio(sum(power), len(power) or None),
        "peak_power_watts": max(power) if power else None,
        "min_power_watts": min(power) if power else None,
        "gpu_energy_joules": energy,
        "power_sample_count": len(power),
        "power_telemetry_available": bool(len(power) >= 2),
        "power_telemetry_source": telemetry_label() if len(power) >= 2 else None,
        "power_telemetry_reason": None if len(power) >= 2 else monitor.failure_reason,
        "telemetry_source": telemetry_label(),
        "telemetry_duration_seconds": runtime_seconds,
        "telemetry_sample_count": len(monitor.samples),
    }


def integrate_energy_joules(samples: List[TelemetrySample]) -> Optional[float]:
    rows = [x for x in samples if x.power_watts is not None]
    if len(rows) < 2:
        return None
    return sum((rows[i].power_watts + rows[i - 1].power_watts) * (rows[i].timestamp - rows[i - 1].timestamp) * 0.5 for i in range(1, len(rows)))


class GpuTelemetryMonitor:
    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = max(interval_seconds, 0.1)
        self.samples: List[TelemetrySample] = []
        self.failure_reason: str = "no telemetry samples"
        self._alive = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._alive = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._alive = False
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while self._alive:
            self._sample_once()
            time.sleep(self.interval_seconds)

    def _sample_once(self) -> None:
        if not torch.cuda.is_available():
            self.failure_reason = "torch.cuda.is_available() is false"
            return
        sample = query_gpu_sample()
        self.samples.append(sample)
        if sample.power_watts is not None:
            self.failure_reason = ""
        elif not self.failure_reason:
            self.failure_reason = f"{sample_source()} returned non-numeric power"


def query_gpu_sample() -> TelemetrySample:
    return query_amd_smi_sample() if hip_version_string() else query_nvidia_smi_sample()


def sample_source() -> str:
    return "amd-smi/rocm-smi" if hip_version_string() else "nvidia-smi"


def telemetry_label() -> str:
    return "amd_smi" if hip_version_string() else "nvml"


def query_nvidia_smi_sample() -> TelemetrySample:
    now = time.perf_counter()
    cmd = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"]
    out = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return parse_nvidia_smi_row(now, out.stdout if out.returncode == 0 else "")


def query_amd_smi_sample() -> TelemetrySample:
    now = time.perf_counter()
    return parse_amd_smi_row(now, amd_smi_stdout())


def amd_smi_stdout() -> str:
    command = amd_smi_command()
    if command is not None:
        out = subprocess.run(command, text=True, capture_output=True, check=False)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout
    return rocm_smi_stdout()


def amd_smi_command() -> Optional[List[str]]:
    binary = next((x for x in [shutil.which("amd-smi"), "/opt/rocm/bin/amd-smi", "/opt/rocm-5.7.0/bin/amd-smi"] if x and Path(x).exists()), None)
    return None if binary is None else [binary, "metric", "--utilization", "--vram-usage", "--power", "--json"]


def rocm_smi_stdout() -> str:
    out = subprocess.run(["rocm-smi", "--showuse", "--showmemuse", "--showpower", "--json"], text=True, capture_output=True, check=False)
    return out.stdout if out.returncode == 0 else ""


def parse_amd_smi_row(now: float, stdout: str) -> TelemetrySample:
    payload = parse_json(stdout)
    util = parse_float(extract_value(payload, ["gpu", "gpu_use", "gpu_use_percent", "utilization", "gpu_util_percent", "GFX Activity"] ))
    mem = parse_memory_bytes(payload)
    power = parse_float(extract_value(payload, ["power", "socket_power", "current_socket_power", "average_socket_power", "Average Graphics Package Power (W)"]))
    return TelemetrySample(now, util, mem, power)


def parse_json(stdout: str) -> Dict[str, Any]:
    try:
        loaded = json.loads(stdout)
    except Exception:
        return {}
    if isinstance(loaded, dict):
        return loaded
    return loaded[0] if isinstance(loaded, list) and loaded and isinstance(loaded[0], dict) else {}


def extract_value(payload: Dict[str, Any], keys: List[str]) -> Optional[str]:
    values = [flatten_scalar(v) for _, v in flatten_items(payload) if any(k.lower() in _.lower() for k in keys)]
    return next((v for v in values if v is not None), None)


def flatten_items(payload: Dict[str, Any], prefix: str = "") -> List[tuple[str, Any]]:
    if not isinstance(payload, dict):
        return []
    rows = []
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        rows.extend(flatten_items(value, name) if isinstance(value, dict) else [(name, value)])
    return rows


def flatten_scalar(value: Any) -> Optional[str]:
    return None if isinstance(value, (dict, list)) else str(value)


def parse_memory_bytes(payload: Dict[str, Any]) -> Optional[int]:
    direct = parse_float(extract_value(payload, ["vram_used", "used_vram", "used_memory", "gpu_memory_usage", "mem_use"]))
    if direct is not None and direct > 0:
        return int(direct)
    mib = parse_float(extract_value(payload, ["VRAM Total Used Memory (MiB)", "used vram (mb)", "vram usage"]))
    return mib_to_bytes(mib)


def parse_nvidia_smi_row(now: float, stdout: str) -> TelemetrySample:
    row = next((line for line in stdout.splitlines() if line.strip()), "")
    parts = [item.strip() for item in row.split(",")]
    util = parse_float(parts[0] if len(parts) > 0 else None)
    mem_mib = parse_float(parts[1] if len(parts) > 1 else None)
    power = parse_float(parts[2] if len(parts) > 2 else None)
    return TelemetrySample(now, util, mib_to_bytes(mem_mib), power)


def parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def mib_to_bytes(value: Optional[float]) -> Optional[int]:
    return None if value is None else int(value * 1024 * 1024)


def device_vendor() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(0)


def python_runtime() -> str:
    return platform.python_version()


def torch_release() -> str:
    return torch.__version__


def cuda_available_flag() -> bool:
    return bool(torch.cuda.is_available())


def cuda_version_string() -> Optional[str]:
    return getattr(torch.version, "cuda", None)


def hip_version_string() -> Optional[str]:
    return getattr(torch.version, "hip", None)


def trainer_device_name(trainer: Trainer) -> str:
    return str(trainer.args.device)


def model_device_name(trainer: Trainer) -> str:
    params = list(trainer.model.parameters())
    if not params:
        return "uninitialized"
    return str(params[0].device)


def ensure_model_device(trainer: Trainer) -> str:
    trainer.model.to(trainer.args.device)
    return model_device_name(trainer)


def environment_payload(trainer: Trainer, device: str) -> EnvironmentDiagnostics:
    return EnvironmentDiagnostics(socket.gethostname(), python_runtime(), torch_release(), cuda_available_flag(), cuda_version_string(), hip_version_string(), device_vendor(), device, trainer_device_name(trainer))


def log_environment(trainer: Trainer) -> None:
    actual_device = ensure_model_device(trainer)
    payload = environment_payload(trainer, actual_device)
    print("ENV_DIAGNOSTICS", json.dumps(asdict(payload), sort_keys=True))


class StopAndSaveCallback(TrainerCallback):
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.save_started_at: Optional[float] = None
        self.save_duration_seconds: Optional[float] = None
        self.checkpoint_size_bytes: Optional[int] = None

    def on_step_end(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> Any:
        if state.global_step != self.cfg.stop_step:
            return control
        self.save_started_at = time.perf_counter()
        control.should_save = True
        control.should_training_stop = True
        return control

    def on_save(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> Any:
        self.save_duration_seconds = None if self.save_started_at is None else time.perf_counter() - self.save_started_at
        self.checkpoint_size_bytes = dir_size_bytes(checkpoint_path(self.cfg))
        return control


class ResumeStateProbeCallback(TrainerCallback):
    def __init__(self) -> None:
        self.norm_after_resume_load: Optional[float] = None
        self.learning_rate_after_resume_load: Optional[float] = None

    def on_train_begin(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> Any:
        optimizer = kwargs.get("optimizer")
        self.norm_after_resume_load = optimizer_exp_avg_sq_norm(optimizer)
        self.learning_rate_after_resume_load = optimizer_learning_rate(optimizer)
        return control


class ResumeTimingCallback(TrainerCallback):
    def __init__(self, resume_step: int, started_at: float) -> None:
        self.resume_step = resume_step
        self.started_at = started_at
        self.checkpoint_load_seconds: Optional[float] = None
        self.resume_to_first_step_seconds: Optional[float] = None
        self.first_step_after_resume: Optional[int] = None

    def on_train_begin(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> Any:
        self.checkpoint_load_seconds = time.perf_counter() - self.started_at
        return control

    def on_log(self, args: TrainingArguments, state: Any, control: Any, logs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        if self.first_step_after_resume is not None or logs is None:
            return control
        current = int(logs.get("step", state.global_step or 0))
        if current <= self.resume_step:
            return control
        self.first_step_after_resume = current
        self.resume_to_first_step_seconds = time.perf_counter() - self.started_at
        return control


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    tokenizer = build_tokenizer(cfg.model_id)
    dataset = tokenize_rows(build_dataset(cfg.dataset_path), tokenizer, cfg.max_length)
    trainer = make_trainer(cfg, tokenizer, dataset)
    log_distributed_env()
    log_environment(trainer)
    monitor = GpuTelemetryMonitor(cfg.telemetry_interval_seconds)
    run_training(trainer, cfg, monitor)


def make_trainer(cfg: TrainConfig, tokenizer: AutoTokenizer, dataset: Dataset) -> Trainer:
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    model = build_model(cfg)
    args = build_args(cfg)
    probe = ResumeStateProbeCallback() if cfg.resume_from else None
    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=collator, callbacks=callbacks(cfg, probe))
    trainer._resume_probe = probe
    return trainer


def callbacks(cfg: TrainConfig, probe: Optional[ResumeStateProbeCallback]) -> List[TrainerCallback]:
    base = [] if cfg.resume_from else [StopAndSaveCallback(cfg)]
    return base + ([probe] if probe else [])


def run_training(trainer: Trainer, cfg: TrainConfig, monitor: GpuTelemetryMonitor) -> None:
    start = time.perf_counter()
    started_iso = now_iso()
    monitor.start()
    if cfg.resume_from:
        timing = ResumeTimingCallback(resume_step(cfg.resume_from) or 0, start)
        trainer.add_callback(timing)
        trainer.train(resume_from_checkpoint=cfg.resume_from)
        monitor.stop()
        runtime_seconds = time.perf_counter() - start
        runtime = RuntimeMetrics(None, None, timing.checkpoint_load_seconds, timing.resume_to_first_step_seconds, timing.first_step_after_resume, started_iso, now_iso())
        write_run_summary(trainer, cfg, monitor, runtime_seconds, runtime)
        return
    trainer.train()
    monitor.stop()
    runtime_seconds = time.perf_counter() - start
    saved = next((x for x in trainer.callback_handler.callbacks if isinstance(x, StopAndSaveCallback)), None)
    runtime = RuntimeMetrics(saved.checkpoint_size_bytes if saved else dir_size_bytes(checkpoint_path(cfg)), saved.save_duration_seconds if saved else None, None, None, None, started_iso, now_iso())
    write_checkpoint_meta(trainer, cfg, monitor, runtime_seconds, runtime)
    write_run_summary(trainer, cfg, monitor, runtime_seconds, runtime)


def dir_size_bytes(path: Path) -> int:
    files = [item for item in path.rglob("*") if item.is_file()]
    return int(sum(item.stat().st_size for item in files))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
