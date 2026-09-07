"""Fine-tuning workload. The actual training loop lives entirely in
`scripts/finetuning/train_lora_migration.py` -- a standalone script with
no knowledge of Dagster, SSH, or this package. This class's only job is
building the correct command line to invoke it, given a target host,
model, dataset and optional data-parallel width. This is the fix for ask
#2: the workload is now a swappable, independently testable, independently
runnable unit instead of being three inlined f-strings inside a Dagster op.
"""
import shlex
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base import Workload


@dataclass
class FineTuningJobSpec:
    model_id: str
    dataset_path: str
    output_dir: str
    stop_step: int
    max_steps: int
    telemetry_interval_seconds: float
    use_gpu: bool = True
    data_parallel_size: int = 1
    resume_from_checkpoint: Optional[str] = None
    max_length: int = 512
    per_device_batch_size: int = 4
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Distributed (multi-host DDP) fields -- see MIGRATION_NOTES.md: ported
    # at reduced fidelity vs. the original `standalone_training_command` /
    # `distributed_train_invocation` (original lines 2275-2347); single-host
    # and single-process-per-host paths are fully ported, multi-node
    # rendezvous wiring should be reviewed against the original before
    # relying on it in production.
    nnodes: int = 1
    node_rank: int = 0
    rdzv_endpoint: Optional[str] = None
    rendezvous_timeout_seconds: int = 180


class FineTuningWorkload(Workload):
    name = "finetuning"
    script_relpath = "scripts/finetuning/train_lora_migration.py"

    def build_command(self, *, python_cmd: str, remote_root: str, spec: FineTuningJobSpec, **kwargs: Any) -> str:
        script = f"{remote_root}/{self.script_relpath}"
        args = (
            f"--model-id {shlex.quote(spec.model_id)} "
            f"--dataset-path {shlex.quote(spec.dataset_path)} "
            f"--output-dir {shlex.quote(spec.output_dir)} "
            f"--stop-step {spec.stop_step} "
            f"--max-steps {spec.max_steps} "
            f"--max-length {spec.max_length} "
            f"--per-device-batch-size {spec.per_device_batch_size} "
            f"--learning-rate {spec.learning_rate} "
            f"--lora-r {spec.lora_r} --lora-alpha {spec.lora_alpha} --lora-dropout {spec.lora_dropout} "
            f"--telemetry-interval-seconds {spec.telemetry_interval_seconds}"
        )
        if spec.resume_from_checkpoint:
            args += f" --resume-from {shlex.quote(spec.resume_from_checkpoint)}"

        if not spec.use_gpu:
            return f"CUDA_VISIBLE_DEVICES='' {shlex.quote(python_cmd)} {script} {args}"
        if spec.data_parallel_size <= 1 and spec.nnodes <= 1:
            return f"{shlex.quote(python_cmd)} {script} {args}"

        distributed = (
            f"{shlex.quote(python_cmd)} -m torch.distributed.run "
            f"--nnodes={spec.nnodes} --node-rank={spec.node_rank} "
            f"--nproc_per_node={max(spec.data_parallel_size, 1)} "
        )
        if spec.nnodes > 1 and spec.rdzv_endpoint:
            distributed += (
                f"--rdzv_backend=c10d --rdzv_endpoint={spec.rdzv_endpoint} "
                f"--rdzv_conf=timeout={spec.rendezvous_timeout_seconds} "
            )
        else:
            distributed += "--standalone "
        return f"{distributed}{script} {args}"

    def parse_run_summary(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # run_summary.json is already in the shape the reporting layer
        # wants (see scripts/finetuning/train_lora_migration.py
        # write_run_summary) -- ported behavior is pass-through.
        return dict(payload)

    def is_checkpoint_reusable(self, summary: Dict[str, Any], *, minimum_power_samples: int, expected_telemetry_interval_seconds: float) -> bool:
        """Direct port of `source_checkpoint_reusable` (original lines
        1576-1616): decides whether an existing checkpoint's run_summary.json
        is complete/fresh enough to skip re-training."""
        required = [
            "hostname", "vendor", "gpu_architecture", "torch_version", "saved_step",
            "useful_training_tokens", "runtime_seconds", "mean_gpu_utilization_percent",
            "peak_gpu_utilization_percent", "peak_accelerator_memory_bytes", "mean_power_watts",
            "peak_power_watts", "gpu_energy_joules", "gpu_energy_kwh", "tokens_per_joule",
            "tokens_per_second", "checkpoint_size_bytes", "checkpoint_save_seconds", "final_loss",
            "telemetry_duration_seconds", "telemetry_sample_count", "sampling_interval_seconds",
        ]
        if any(field not in summary or summary[field] is None for field in required):
            return False
        samples = summary.get("telemetry_sample_count") or summary.get("raw_telemetry_sample_count")
        interval = summary.get("sampling_interval_seconds")
        try:
            if samples is None or int(samples) < minimum_power_samples:
                return False
            if interval is None or abs(float(interval) - expected_telemetry_interval_seconds) > 1e-9:
                return False
        except (TypeError, ValueError):
            return False
        return True
