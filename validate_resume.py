from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import argparse
import json


@dataclass
class ValidateConfig:
    source_checkpoint: str
    resumed_summary: str
    resumed_checkpoint: str
    proof_out: Optional[str]
    min_extra_steps: int
    max_loss_delta: float
    max_optimizer_norm_delta: float


def parse_args() -> ValidateConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--resumed-summary", required=True)
    parser.add_argument("--resumed-checkpoint", required=True)
    parser.add_argument("--proof-out", default=None)
    parser.add_argument("--min-extra-steps", type=int, default=1)
    parser.add_argument("--max-loss-delta", type=float, default=1.0)
    parser.add_argument("--max-optimizer-norm-delta", type=float, default=1e-4)
    args = parser.parse_args()
    return ValidateConfig(args.source_checkpoint, args.resumed_summary, args.resumed_checkpoint, args.proof_out, args.min_extra_steps, args.max_loss_delta, args.max_optimizer_norm_delta)


def read_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def latest_loss(summary: Dict[str, Any]) -> Optional[float]:
    value = summary.get("loss_before_migration")
    return None if value is None else float(value)


def check_step(source_meta: Dict[str, Any], resumed_summary: Dict[str, Any], min_extra_steps: int) -> None:
    source_step = int(source_meta["saved_step"])
    resumed_step = int(resumed_summary["saved_step"])
    assert resumed_step >= source_step + min_extra_steps, f"resume failed: expected >= {source_step + min_extra_steps}, got {resumed_step}"


def check_resume_pointer(source_checkpoint: str, resumed_summary: Dict[str, Any]) -> None:
    actual = str(resumed_summary.get("resume_from") or "")
    assert Path(actual).as_posix() == Path(source_checkpoint).as_posix(), f"resume pointer mismatch: expected {source_checkpoint}, got {actual}"


def check_log_continuity(source_meta: Dict[str, Any], resumed_summary: Dict[str, Any]) -> None:
    source_step = int(source_meta["saved_step"])
    trace = resumed_summary.get("loss_trace") or []
    resumed_steps = [int(item["step"]) for item in trace if "step" in item]
    first_after_handoff = next((step for step in resumed_steps if step > source_step), None)
    assert first_after_handoff is not None, f"resume continuity failed: no logged step after {source_step}"


def check_handoff_step(source_meta: Dict[str, Any], resumed_summary: Dict[str, Any]) -> None:
    source_step = int(source_meta["saved_step"])
    loaded_step = resumed_summary.get("resume_loaded_from_step")
    assert loaded_step is not None, "missing resume_loaded_from_step in resumed summary"
    assert int(loaded_step) == source_step, f"handoff step mismatch: expected {source_step}, got {loaded_step}"


def assert_non_empty(path: Path) -> int:
    size = path.stat().st_size
    assert size > 0, f"empty file: {path}"
    return int(size)


def check_checkpoint_state_files(checkpoint_dir: str) -> Dict[str, int]:
    names = ["optimizer.pt", "scheduler.pt", "trainer_state.json"]
    files = [Path(checkpoint_dir) / name for name in names]
    assert all(path.exists() for path in files), f"checkpoint missing state files in {checkpoint_dir}"
    return {path.name: assert_non_empty(path) for path in files}


def build_proof(source_meta: Dict[str, Any], resumed_summary: Dict[str, Any], source_files: Dict[str, int], resumed_files: Dict[str, int]) -> Dict[str, Any]:
    source_norm = source_meta.get("optimizer_exp_avg_sq_norm_before_save")
    resumed_norm = resumed_summary.get("optimizer_exp_avg_sq_norm_after_resume_load")
    return {
        "source_saved_step": int(source_meta["saved_step"]),
        "resumed_saved_step": int(resumed_summary["saved_step"]),
        "resumed_first_logged_step": resumed_summary.get("first_logged_step"),
        "resume_from": resumed_summary.get("resume_from"),
        "resume_loaded_from_step": resumed_summary.get("resume_loaded_from_step"),
        "source_optimizer_exp_avg_sq_norm_before_save": source_norm,
        "resumed_optimizer_exp_avg_sq_norm_after_resume_load": resumed_norm,
        "optimizer_norm_delta": None if source_norm is None or resumed_norm is None else abs(float(source_norm) - float(resumed_norm)),
        "source_state_file_sizes": source_files,
        "resumed_state_file_sizes": resumed_files,
    }


def write_proof(path: Optional[str], payload: Dict[str, Any]) -> None:
    if path is None:
        return
    Path(path).write_text(json.dumps(payload, indent=2))


def check_loss(source_meta: Dict[str, Any], resumed_summary: Dict[str, Any], max_loss_delta: float) -> None:
    source_loss = source_meta.get("loss_before_migration")
    resumed_loss = latest_loss(resumed_summary)
    if source_loss is None or resumed_loss is None:
        return
    delta = abs(float(source_loss) - float(resumed_loss))
    assert delta <= max_loss_delta, f"loss jump too high: delta={delta}"


def check_optimizer_norm(source_meta: Dict[str, Any], resumed_summary: Dict[str, Any], max_optimizer_norm_delta: float) -> None:
    source_norm = source_meta.get("optimizer_exp_avg_sq_norm_before_save")
    resumed_norm = resumed_summary.get("optimizer_exp_avg_sq_norm_after_resume_load")
    assert source_norm is not None, "missing optimizer_exp_avg_sq_norm_before_save in source checkpoint metadata"
    assert resumed_norm is not None, "missing optimizer_exp_avg_sq_norm_after_resume_load in resumed summary"
    delta = abs(float(source_norm) - float(resumed_norm))
    assert delta <= max_optimizer_norm_delta, f"optimizer exp_avg_sq norm mismatch: delta={delta}"


def main() -> None:
    cfg = parse_args()
    source_meta = read_json(f"{cfg.source_checkpoint}/checkpoint_meta.json")
    resumed = read_json(cfg.resumed_summary)
    source_files = check_checkpoint_state_files(cfg.source_checkpoint)
    resumed_files = check_checkpoint_state_files(cfg.resumed_checkpoint)
    check_step(source_meta, resumed, cfg.min_extra_steps)
    check_handoff_step(source_meta, resumed)
    check_resume_pointer(cfg.source_checkpoint, resumed)
    check_log_continuity(source_meta, resumed)
    check_loss(source_meta, resumed, cfg.max_loss_delta)
    check_optimizer_norm(source_meta, resumed, cfg.max_optimizer_norm_delta)
    write_proof(cfg.proof_out, build_proof(source_meta, resumed, source_files, resumed_files))
    print("resume validation passed")


if __name__ == "__main__":
    main()
