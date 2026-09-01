from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import argparse
import json
import math
import shlex
import subprocess

from checkpoint_io import assert_structurally_identical, read_manifest


@dataclass
class HarnessConfig:
    environment: str
    plan_path: str
    local_root: str
    remote_root: str
    spark_host: str
    radeon_host: str
    venv_path: str
    artifact_root: str
    ssh_password: str


@dataclass
class Plan:
    model_id: str
    dataset_path: str
    seeds: List[int]
    leg1_step: int
    leg2_step: int
    leg3_step: int
    min_extra_steps: int
    max_loss_delta: float
    max_optimizer_norm_delta: float
    max_trajectory_rmse: float


def parse_args() -> HarnessConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", default="dev")
    parser.add_argument("--plan-path", default="./experiments/story3_preregistration.json")
    parser.add_argument("--local-root", default=".")
    parser.add_argument("--remote-root", default="~/cross-vendor-mooncake-test")
    parser.add_argument("--spark-host", default="dgx-spark")
    parser.add_argument("--radeon-host", default="odyn-radeon2")
    parser.add_argument("--venv-path", default="~/cv-story3-venv")
    parser.add_argument("--artifact-root", default="./artifacts/story3")
    parser.add_argument("--ssh-password", default="")
    args = parser.parse_args()
    return HarnessConfig(args.environment, args.plan_path, args.local_root, args.remote_root, args.spark_host, args.radeon_host, args.venv_path, args.artifact_root, args.ssh_password)


def load_plan(path: str) -> Plan:
    payload = json.loads(Path(path).read_text())
    steps = payload["steps"]
    limits = payload["thresholds"]
    return Plan(payload["model_id"], payload["dataset_path"], payload["seeds"], steps["leg1_step"], steps["leg2_step"], steps["leg3_step"], limits["min_extra_steps"], limits["max_loss_delta"], limits["max_optimizer_norm_delta"], limits["max_trajectory_rmse"])


def run(command: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def sshpass_prefix(cfg: HarnessConfig) -> List[str]:
    return ["sshpass", "-p", cfg.ssh_password] if cfg.ssh_password else []


def ssh_options(cfg: HarnessConfig) -> List[str]:
    if not cfg.ssh_password:
        return ["-o", "IdentitiesOnly=yes"]
    return ["-o", "IdentitiesOnly=yes", "-o", "PreferredAuthentications=password", "-o", "PubkeyAuthentication=no", "-o", "NumberOfPasswordPrompts=1"]


def ssh(cfg: HarnessConfig, host: str, command: str) -> subprocess.CompletedProcess:
    return run([*sshpass_prefix(cfg), "ssh", *ssh_options(cfg), host, command])


def scp(cfg: HarnessConfig, source: str, target: str) -> subprocess.CompletedProcess:
    return run([*sshpass_prefix(cfg), "scp", *ssh_options(cfg), "-r", source, target])


def must_succeed(result: subprocess.CompletedProcess, label: str) -> None:
    assert result.returncode == 0, f"{label} failed\nstdout={result.stdout}\nstderr={result.stderr}"


def remote_python(cfg: HarnessConfig, host: str, code: str) -> str:
    activate = f"source {cfg.venv_path}/bin/activate"
    command = f"{activate} && python3 -c {shlex.quote(code)}"
    result = ssh(cfg, host, command)
    must_succeed(result, f"remote python on {host}")
    return result.stdout.strip()


def host_env(cfg: HarnessConfig, host: str) -> Dict[str, Any]:
    code = "import json,platform,socket,torch;print(json.dumps({'host':socket.gethostname(),'python':platform.python_version(),'torch':torch.__version__,'cuda':getattr(torch.version,'cuda',None),'hip':getattr(torch.version,'hip',None)}))"
    return json.loads(remote_python(cfg, host, code))


def setup_host(cfg: HarnessConfig, host: str) -> None:
    mkdir = ssh(cfg, host, f"mkdir -p {cfg.remote_root} {cfg.remote_root}/data")
    must_succeed(mkdir, f"mkdir on {host}")
    for name in ["requirements.txt", "train_lora_migration.py", "validate_resume.py", "checkpoint_io.py"]:
        copied = scp(cfg, f"{cfg.local_root}/{name}", f"{host}:{cfg.remote_root}/")
        must_succeed(copied, f"scp {name} to {host}")
    dataset = scp(cfg, f"{cfg.local_root}/data/story3_dataset.jsonl", f"{host}:{cfg.remote_root}/data/")
    must_succeed(dataset, f"scp dataset to {host}")
    venv = ssh(cfg, host, f"python3 -m venv {cfg.venv_path} && source {cfg.venv_path}/bin/activate && pip install -U pip && pip install -r {cfg.remote_root}/requirements.txt")
    must_succeed(venv, f"venv setup on {host}")


def train_cmd(cfg: HarnessConfig, plan: Plan, output_dir: str, seed: int, stop_step: int, max_steps: int, resume_from: Optional[str]) -> str:
    activate = f"source {cfg.venv_path}/bin/activate"
    resume = f" --resume-from {resume_from}" if resume_from else ""
    return f"{activate} && python3 {cfg.remote_root}/train_lora_migration.py --model-id {plan.model_id} --dataset-path {cfg.remote_root}/{plan.dataset_path} --output-dir {output_dir} --seed {seed} --stop-step {stop_step} --max-steps {max_steps}{resume}"


def validate_cmd(cfg: HarnessConfig, plan: Plan, source_ckpt: str, resumed_summary: str, resumed_ckpt: str, proof_out: str) -> str:
    activate = f"source {cfg.venv_path}/bin/activate"
    return f"{activate} && python3 {cfg.remote_root}/validate_resume.py --source-checkpoint {source_ckpt} --resumed-summary {resumed_summary} --resumed-checkpoint {resumed_ckpt} --proof-out {proof_out} --min-extra-steps {plan.min_extra_steps} --max-loss-delta {plan.max_loss_delta} --max-optimizer-norm-delta {plan.max_optimizer_norm_delta}"


def manifest_cmd(cfg: HarnessConfig, checkpoint_dir: str, out_path: str) -> str:
    activate = f"source {cfg.venv_path}/bin/activate"
    return f"{activate} && python3 {cfg.remote_root}/checkpoint_io.py --mode manifest --checkpoint-a {checkpoint_dir} --output {out_path}"


def pull_file(cfg: HarnessConfig, host: str, remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    copied = scp(cfg, f"{host}:{remote_path}", str(local_path))
    must_succeed(copied, f"pull {remote_path} from {host}")


def loss_series(path: Path) -> Dict[int, float]:
    payload = json.loads(path.read_text())
    return {int(item["step"]): float(item["loss"]) for item in payload.get("loss_trace", [])}


def trajectory_rmse(cross_summary: Path, control_summary: Path) -> float:
    cross = loss_series(cross_summary)
    control = loss_series(control_summary)
    common = sorted(set(cross.keys()) & set(control.keys()))
    assert common, "no overlapping loss steps between cross-vendor and control runs"
    squared = [(cross[step] - control[step]) ** 2 for step in common]
    return math.sqrt(sum(squared) / len(squared))


def run_trial(cfg: HarnessConfig, plan: Plan, seed: int) -> Dict[str, Any]:
    trial_id = f"seed-{seed}"
    trial_dir = Path(cfg.artifact_root) / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    logs: List[Dict[str, Any]] = []
    try:
        logs.extend([{"host": cfg.spark_host, "env": host_env(cfg, cfg.spark_host)}, {"host": cfg.radeon_host, "env": host_env(cfg, cfg.radeon_host)}])
        spark_leg1_dir = f"{cfg.remote_root}/relay_{trial_id}_spark_leg1"
        radeon_leg2_dir = f"{cfg.remote_root}/relay_{trial_id}_radeon_leg2"
        spark_leg3_dir = f"{cfg.remote_root}/relay_{trial_id}_spark_leg3"
        control_leg1_dir = f"{cfg.remote_root}/control_{trial_id}_spark_leg1"
        control_leg2_dir = f"{cfg.remote_root}/control_{trial_id}_spark_leg2"
        spark_ckpt = f"{spark_leg1_dir}/checkpoint-{plan.leg1_step}"
        radeon_ckpt = f"{radeon_leg2_dir}/checkpoint-{plan.leg2_step}"
        spark_final_ckpt = f"{spark_leg3_dir}/checkpoint-{plan.leg3_step}"
        control_ckpt = f"{control_leg1_dir}/checkpoint-{plan.leg1_step}"
        must_succeed(ssh(cfg, cfg.spark_host, train_cmd(cfg, plan, spark_leg1_dir, seed, plan.leg1_step, plan.leg1_step, None)), "spark leg1")
        must_succeed(scp(cfg, f"{cfg.spark_host}:{spark_ckpt}", f"{cfg.radeon_host}:{cfg.remote_root}/"), "spark->radeon copy")
        must_succeed(ssh(cfg, cfg.radeon_host, train_cmd(cfg, plan, radeon_leg2_dir, seed, plan.leg2_step, plan.leg2_step, f"{cfg.remote_root}/checkpoint-{plan.leg1_step}")), "radeon leg2")
        proof1 = f"{radeon_leg2_dir}/resume_validation_proof.json"
        must_succeed(ssh(cfg, cfg.radeon_host, validate_cmd(cfg, plan, f"{cfg.remote_root}/checkpoint-{plan.leg1_step}", f"{radeon_leg2_dir}/run_summary.json", radeon_ckpt, proof1)), "validate spark->radeon")
        spark_manifest_remote = f"{spark_leg1_dir}/manifest.json"
        radeon_manifest_remote = f"{radeon_leg2_dir}/manifest.json"
        must_succeed(ssh(cfg, cfg.spark_host, manifest_cmd(cfg, spark_ckpt, spark_manifest_remote)), "spark manifest")
        must_succeed(ssh(cfg, cfg.radeon_host, manifest_cmd(cfg, radeon_ckpt, radeon_manifest_remote)), "radeon manifest")
        spark_manifest_local = trial_dir / "spark_manifest.json"
        radeon_manifest_local = trial_dir / "radeon_manifest.json"
        pull_file(cfg, cfg.spark_host, spark_manifest_remote, spark_manifest_local)
        pull_file(cfg, cfg.radeon_host, radeon_manifest_remote, radeon_manifest_local)
        assert_structurally_identical(read_manifest(str(spark_manifest_local)), read_manifest(str(radeon_manifest_local)))
        must_succeed(scp(cfg, f"{cfg.radeon_host}:{radeon_ckpt}", f"{cfg.spark_host}:{cfg.remote_root}/"), "radeon->spark copy")
        must_succeed(ssh(cfg, cfg.spark_host, train_cmd(cfg, plan, spark_leg3_dir, seed, plan.leg3_step, plan.leg3_step, f"{cfg.remote_root}/checkpoint-{plan.leg2_step}")), "spark leg3")
        proof2 = f"{spark_leg3_dir}/resume_validation_proof.json"
        must_succeed(ssh(cfg, cfg.spark_host, validate_cmd(cfg, plan, f"{cfg.remote_root}/checkpoint-{plan.leg2_step}", f"{spark_leg3_dir}/run_summary.json", spark_final_ckpt, proof2)), "validate radeon->spark")
        must_succeed(ssh(cfg, cfg.spark_host, train_cmd(cfg, plan, control_leg1_dir, seed, plan.leg1_step, plan.leg1_step, None)), "control spark leg1")
        must_succeed(ssh(cfg, cfg.spark_host, train_cmd(cfg, plan, control_leg2_dir, seed, plan.leg2_step, plan.leg2_step, control_ckpt)), "control spark leg2")
        proofc = f"{control_leg2_dir}/resume_validation_proof.json"
        must_succeed(ssh(cfg, cfg.spark_host, validate_cmd(cfg, plan, control_ckpt, f"{control_leg2_dir}/run_summary.json", f"{control_leg2_dir}/checkpoint-{plan.leg2_step}", proofc)), "validate spark->spark control")
        cross_summary = trial_dir / "cross_run_summary.json"
        control_summary = trial_dir / "control_run_summary.json"
        pull_file(cfg, cfg.radeon_host, f"{radeon_leg2_dir}/run_summary.json", cross_summary)
        pull_file(cfg, cfg.spark_host, f"{control_leg2_dir}/run_summary.json", control_summary)
        rmse = trajectory_rmse(cross_summary, control_summary)
        assert rmse <= plan.max_trajectory_rmse, f"cross/control loss trajectory differs: rmse={rmse}"
        for remote_path, host in [(proof1, cfg.radeon_host), (proof2, cfg.spark_host), (proofc, cfg.spark_host), (f"{spark_leg3_dir}/run_summary.json", cfg.spark_host), (f"{radeon_leg2_dir}/run_summary.json", cfg.radeon_host), (f"{control_leg2_dir}/run_summary.json", cfg.spark_host)]:
            pull_file(cfg, host, remote_path, trial_dir / Path(remote_path).name)
        summary = {"trial": trial_id, "seed": seed, "status": "passed", "trajectory_rmse": rmse, "logs": logs}
        (trial_dir / "trial_summary.json").write_text(json.dumps(summary, indent=2))
        return summary
    except Exception as error:
        summary = {"trial": trial_id, "seed": seed, "status": "failed", "error": str(error), "logs": logs}
        (trial_dir / "trial_summary.json").write_text(json.dumps(summary, indent=2))
        return summary


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    passed = [item for item in results if item["status"] == "passed"]
    return {
        "trials": len(results),
        "passed": len(passed),
        "pass_fraction": f"{len(passed)}/{len(results)}",
        "results": results,
        "scope": "single-GPU, single-node, short-horizon Qwen 1.5B LoRA checkpoint failover between one Spark and one Radeon host",
    }


def main() -> None:
    cfg = parse_args()
    plan = load_plan(cfg.plan_path)
    setup_host(cfg, cfg.spark_host)
    setup_host(cfg, cfg.radeon_host)
    results = [run_trial(cfg, plan, seed) for seed in plan.seeds]
    report = {"environment": cfg.environment, **aggregate(results)}
    out = Path(cfg.artifact_root) / "story3_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
