from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import argparse
import inspect
import json
import math
import platform
import random
import socket

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
    parser.add_argument("--dataset-path", default="./data/story3_dataset.jsonl")
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


def to_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(args.model_id, args.dataset_path, args.output_dir, args.resume_from, args.stop_step, args.max_steps, args.per_device_batch_size, args.learning_rate, args.seed, args.lora_r, args.lora_alpha, args.lora_dropout)


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


def tokenize_rows(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    fn = lambda batch: tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
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
    return {"output_dir": cfg.output_dir, "per_device_train_batch_size": cfg.per_device_batch_size, "learning_rate": cfg.learning_rate, "logging_steps": 1, "max_steps": cfg.max_steps, "lr_scheduler_type": "linear", "warmup_ratio": 0.03, "save_strategy": "steps", "save_steps": cfg.stop_step, "save_total_limit": 2, "bf16": bf16, "fp16": not bf16, "optim": "adamw_torch", "report_to": []}


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


def resume_probe_norm(trainer: Trainer) -> Optional[float]:
    probe = getattr(trainer, "_resume_probe", None)
    return None if probe is None else probe.norm_after_resume_load


def write_checkpoint_meta(trainer: Trainer, cfg: TrainConfig) -> None:
    path = checkpoint_path(cfg) / "checkpoint_meta.json"
    payload = meta_payload(trainer, cfg)
    path.write_text(json.dumps(payload, indent=2))


def write_run_summary(trainer: Trainer, cfg: TrainConfig) -> None:
    path = Path(cfg.output_dir) / "run_summary.json"
    payload = meta_payload(trainer, cfg)
    path.write_text(json.dumps(payload, indent=2))


def meta_payload(trainer: Trainer, cfg: TrainConfig) -> Dict[str, Any]:
    state = trainer.state
    optimizer_norm = trainer_optimizer_norm(trainer)
    return {
        "saved_step": int(state.global_step),
        "resume_loaded_from_step": resume_step(cfg.resume_from),
        "loss_before_migration": last_loss(state.log_history),
        "loss_trace": loss_trace(state.log_history),
        "first_logged_step": first_logged_step(state.log_history),
        "last_logged_step": last_logged_step(state.log_history),
        "optimizer_exp_avg_sq_norm": optimizer_norm,
        "optimizer_exp_avg_sq_norm_before_save": optimizer_norm,
        "optimizer_exp_avg_sq_norm_after_resume_load": resume_probe_norm(trainer),
        "seed": cfg.seed,
        "model_id": cfg.model_id,
        "dataset_path": cfg.dataset_path,
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "device": device_vendor(),
        "torch": torch.__version__,
        "resume_from": cfg.resume_from,
    }


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

    def on_step_end(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> Any:
        if state.global_step != self.cfg.stop_step:
            return control
        control.should_save = True
        control.should_training_stop = True
        return control


class ResumeStateProbeCallback(TrainerCallback):
    def __init__(self) -> None:
        self.norm_after_resume_load: Optional[float] = None

    def on_train_begin(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> Any:
        self.norm_after_resume_load = optimizer_exp_avg_sq_norm(kwargs.get("optimizer"))
        return control


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    tokenizer = build_tokenizer(cfg.model_id)
    dataset = tokenize_rows(build_dataset(cfg.dataset_path), tokenizer)
    trainer = make_trainer(cfg, tokenizer, dataset)
    log_environment(trainer)
    run_training(trainer, cfg)


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


def run_training(trainer: Trainer, cfg: TrainConfig) -> None:
    if cfg.resume_from:
        trainer.train(resume_from_checkpoint=cfg.resume_from)
        write_run_summary(trainer, cfg)
        return
    trainer.train()
    write_checkpoint_meta(trainer, cfg)
    write_run_summary(trainer, cfg)


def main() -> None:
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
