from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import json
import os
import shlex
import subprocess
import time

from dagster import AssetCheckResult, AssetKey, AssetMaterialization, In, MetadataValue, Nothing, OpExecutionContext, Out, TableColumn, TableSchema, asset_check, job, op


@dataclass
class PortabilitySettings:
    local_root: str
    remote_root: str
    amd_host: str
    nvidia_host: str
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
    amd_use_gpu: bool


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


def load_settings() -> PortabilitySettings:
    config = repo_config()
    common = config["common"]
    basic = config["basic_portability"]
    dataset_path = str(basic.get("dataset_path", "data/story3_dataset.jsonl"))
    amd_use_gpu = bool(basic.get("amd_use_gpu", True))
    return PortabilitySettings(str(common["local_root_dir"]), str(common["remote_root_dir"]), str(basic["amd_host"]), str(basic["nvidia_host"]), int(basic["checkpoint_step"]), int(basic["final_step"]), str(basic["run_id"]), dataset_path, nullable(basic["ssh_password"]), list(basic["asset_paths"]), int(basic["command_retries"]), int(basic["min_extra_steps"]), float(basic["max_loss_delta"]), float(basic.get("max_optimizer_norm_delta", 1e-4)), amd_use_gpu)


def repo_config() -> dict:
    path = Path(__file__).with_name("repo_config.json")
    return json.loads(path.read_text())


def nullable(value: object) -> Optional[str]:
    text = str(value)
    return text if text else None


def sshpass_env(settings: PortabilitySettings) -> Optional[dict]:
    return {**os.environ, "SSHPASS": settings.ssh_password} if settings.ssh_password else None


def ssh_options(settings: PortabilitySettings) -> List[str]:
    base = ["-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=120"]
    return base + ["-o", "PreferredAuthentications=password", "-o", "PubkeyAuthentication=no", "-o", "NumberOfPasswordPrompts=1"] if settings.ssh_password else base


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


def pip_bootstrap(settings: PortabilitySettings) -> str:
    return "python3 -m pip --version >/dev/null 2>&1 || true" if not settings.ssh_password else f"python3 -m pip --version >/dev/null 2>&1 || (printf %s {shlex.quote(settings.ssh_password)} | sudo -S -k apt-get update && printf %s {shlex.quote(settings.ssh_password)} | sudo -S -k apt-get install -y python3-pip)"


def local_asset_path(settings: PortabilitySettings, asset_path: str) -> Path:
    return Path(settings.local_root) / asset_path


def remote_asset_path(settings: PortabilitySettings, asset_path: str) -> str:
    return f"{settings.remote_root}/{asset_path}"


def normalized_asset_path(path: str) -> str:
    return path.lstrip("./") if not path.startswith(("/", "~/")) else path


def remote_dataset_path(settings: PortabilitySettings) -> str:
    path = normalized_asset_path(settings.dataset_path)
    return path if path.startswith(("/", "~/")) else f"{settings.remote_root}/{path}"


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
    defaults = [
        f"{run}/run_summary.json",
        f"{run}/resume_validation_proof.json",
        f"{run}/checkpoint-{settings.final_step}/trainer_state.json",
        f"{source}/checkpoint-{settings.checkpoint_step}/checkpoint_meta.json",
    ]
    return list(dict.fromkeys([*settings.asset_paths, *defaults]))


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
        "source_host": settings.amd_host,
        "target_host": settings.nvidia_host,
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
        },
    )
    return settings


@op(ins={"settings": In(PortabilitySettings)}, out=Out(Nothing))
def prepare_hosts(context: OpExecutionContext, settings: PortabilitySettings) -> None:
    run_shell(ssh_cmd(settings, settings.amd_host, f"mkdir -p {settings.remote_root}"), settings.command_retries, env=sshpass_env(settings))
    run_shell(ssh_cmd(settings, settings.nvidia_host, f"mkdir -p {settings.remote_root}"), settings.command_retries, env=sshpass_env(settings))
    emit_materialization(
        context,
        ["portability", "control", "host_preparation"],
        "Remote directories prepared on both hosts.",
        {
            **common_metadata(settings, "ready", settings.remote_root),
            "lineage_path": MetadataValue.json(["control", "settings", "control", "host_preparation"]),
            "provenance": MetadataValue.json({"operation": "mkdir", "hosts": [settings.amd_host, settings.nvidia_host]}),
        },
    )


@op(ins={"settings": In(PortabilitySettings), "_ready": In(Nothing)}, out=Out(Nothing))
def sync_and_install(context: OpExecutionContext, settings: PortabilitySettings) -> None:
    files = ["requirements.txt", "train_lora_migration.py", "validate_resume.py"]
    for name in files:
        run_shell(scp_cmd(settings, str(Path(settings.local_root) / name), f"{settings.amd_host}:{settings.remote_root}/"), settings.command_retries, env=sshpass_env(settings))
        run_shell(scp_cmd(settings, str(Path(settings.local_root) / name), f"{settings.nvidia_host}:{settings.remote_root}/"), settings.command_retries, env=sshpass_env(settings))
    paths = sync_paths(settings)
    for asset_path in paths:
        sync_asset_to_host(settings, settings.amd_host, asset_path)
        sync_asset_to_host(settings, settings.nvidia_host, asset_path)
    run_shell(ssh_cmd(settings, settings.amd_host, f"{pip_bootstrap(settings)} && python3 -m pip install --user --break-system-packages -U pip && python3 -m pip install --user --break-system-packages -r {settings.remote_root}/requirements.txt"), settings.command_retries, env=sshpass_env(settings))
    run_shell(ssh_cmd(settings, settings.nvidia_host, f"{pip_bootstrap(settings)} && python3 -m pip install --user --break-system-packages -U pip && python3 -m pip install --user --break-system-packages -r {settings.remote_root}/requirements.txt"), settings.command_retries, env=sshpass_env(settings))
    emit_materialization(
        context,
        ["portability", "control", "environment_sync"],
        "Code, dependencies, and input assets synced to source and target hosts.",
        {
            **common_metadata(settings, "synced", settings.remote_root),
            "lineage_path": MetadataValue.json(["control", "host_preparation", "control", "environment_sync"]),
            "files": MetadataValue.json(files),
            "asset_paths": MetadataValue.json(paths),
        },
    )


@op(ins={"settings": In(PortabilitySettings), "_ready": In(Nothing)}, out=Out(Nothing))
def train_on_amd(context: OpExecutionContext, settings: PortabilitySettings) -> None:
    checkpoint_meta = f"{settings.remote_root}/amd_run_{settings.run_id}/{remote_ckpt(settings)}/checkpoint_meta.json"
    if remote_exists(settings, settings.amd_host, checkpoint_meta):
        emit_materialization(
            context,
            ["portability", "source", "amd", "checkpoint"],
            "Existing AMD checkpoint reused.",
            {
                **common_metadata(settings, "checkpoint_reused", f"{settings.remote_root}/amd_run_{settings.run_id}/{remote_ckpt(settings)}"),
                "lineage_path": MetadataValue.json(["control", "environment_sync", "source", "amd", "checkpoint"]),
                "provenance": MetadataValue.json({"producer": settings.amd_host, "mode": "reuse"}),
            },
        )
        return
    if settings.amd_use_gpu:
        probe = "python3 -c \"import torch; ok=bool(torch.cuda.is_available()); print(f'torch={torch.__version__} cuda_available={ok}'); raise SystemExit(0 if ok else 2)\""
        run_shell(ssh_cmd(settings, settings.amd_host, probe), settings.command_retries, env=sshpass_env(settings))
    prefix = "" if settings.amd_use_gpu else "CUDA_VISIBLE_DEVICES='' "
    run = f"{prefix}python3 {settings.remote_root}/train_lora_migration.py --dataset-path {remote_dataset_path(settings)} --output-dir {settings.remote_root}/amd_run_{settings.run_id} --stop-step {settings.checkpoint_step} --max-steps 100"
    run_shell(ssh_cmd(settings, settings.amd_host, run), settings.command_retries, env=sshpass_env(settings))
    emit_materialization(
        context,
        ["portability", "source", "amd", "checkpoint"],
        "Checkpoint created on AMD host.",
        {
            **common_metadata(
                settings,
                "checkpoint_created",
                f"{settings.remote_root}/amd_run_{settings.run_id}/{remote_ckpt(settings)}",
            ),
            "lineage_path": MetadataValue.json(["control", "environment_sync", "source", "amd", "checkpoint"]),
            "provenance": MetadataValue.json({"producer": settings.amd_host, "consumer": settings.nvidia_host}),
        },
    )


@op(ins={"settings": In(PortabilitySettings), "_ready": In(Nothing)}, out=Out(Nothing))
def copy_checkpoint(context: OpExecutionContext, settings: PortabilitySettings) -> None:
    local_stage = Path("/tmp") / settings.run_id
    local_stage.mkdir(parents=True, exist_ok=True)
    run_shell(scp_cmd(settings, f"{settings.amd_host}:{settings.remote_root}/amd_run_{settings.run_id}/{remote_ckpt(settings)}", str(local_stage), recursive=True), settings.command_retries, env=sshpass_env(settings))
    run_shell(scp_cmd(settings, str(local_stage / remote_ckpt(settings)), f"{settings.nvidia_host}:{settings.remote_root}/", recursive=True), settings.command_retries, env=sshpass_env(settings))
    emit_materialization(
        context,
        ["portability", "transfer", "checkpoint"],
        "Checkpoint transferred from AMD to NVIDIA via local staging.",
        {
            **common_metadata(settings, "transferred", f"{settings.remote_root}/{remote_ckpt(settings)}"),
            "lineage_path": MetadataValue.json(["source", "amd", "checkpoint", "transfer", "checkpoint", "target", "nvidia", "checkpoint"]),
            "provenance": MetadataValue.json(
                {
                    "stage_path": str(local_stage / remote_ckpt(settings)),
                    "source_checkpoint": f"{settings.remote_root}/amd_run_{settings.run_id}/{remote_ckpt(settings)}",
                    "target_checkpoint": f"{settings.remote_root}/{remote_ckpt(settings)}",
                }
            ),
        },
    )


@op(ins={"settings": In(PortabilitySettings), "_ready": In(Nothing)}, out=Out(Nothing))
def resume_on_nvidia(context: OpExecutionContext, settings: PortabilitySettings) -> None:
    run = f"python3 {settings.remote_root}/train_lora_migration.py --dataset-path {remote_dataset_path(settings)} --output-dir {settings.remote_root}/nvidia_run_{settings.run_id} --resume-from {settings.remote_root}/{remote_ckpt(settings)} --stop-step {settings.final_step} --max-steps {settings.final_step}"
    run_shell(ssh_cmd(settings, settings.nvidia_host, run), settings.command_retries, env=sshpass_env(settings))
    emit_materialization(
        context,
        ["portability", "target", "nvidia", "resume_training"],
        "Training resumed from transferred checkpoint on NVIDIA host.",
        {
            **common_metadata(settings, "resumed", f"{settings.remote_root}/nvidia_run_{settings.run_id}/run_summary.json"),
            "lineage_path": MetadataValue.json(["transfer", "checkpoint", "target", "nvidia", "resume_training"]),
            "provenance": MetadataValue.json({"resume_from": f"{settings.remote_root}/{remote_ckpt(settings)}", "runner": settings.nvidia_host}),
        },
    )


@op(ins={"settings": In(PortabilitySettings), "_ready": In(Nothing)})
def validate_resume(context: OpExecutionContext, settings: PortabilitySettings) -> None:
    resumed_checkpoint = f"{settings.remote_root}/nvidia_run_{settings.run_id}/checkpoint-{settings.final_step}"
    proof_out = f"{settings.remote_root}/nvidia_run_{settings.run_id}/resume_validation_proof.json"
    cmd = f"python3 {settings.remote_root}/validate_resume.py --source-checkpoint {settings.remote_root}/{remote_ckpt(settings)} --resumed-summary {settings.remote_root}/nvidia_run_{settings.run_id}/run_summary.json --resumed-checkpoint {resumed_checkpoint} --proof-out {proof_out} --min-extra-steps {settings.min_extra_steps} --max-loss-delta {settings.max_loss_delta} --max-optimizer-norm-delta {settings.max_optimizer_norm_delta}"
    run_shell(ssh_cmd(settings, settings.nvidia_host, cmd), settings.command_retries, env=sshpass_env(settings))
    emit_materialization(
        context,
        ["portability", "target", "nvidia", "resume_validation"],
        "Resume validation completed.",
        {
            **common_metadata(settings, "validated", f"{settings.remote_root}/nvidia_run_{settings.run_id}/run_summary.json"),
            "lineage_path": MetadataValue.json(["target", "nvidia", "resume_training", "target", "nvidia", "resume_validation"]),
            "validation_policy": MetadataValue.json({"min_extra_steps": settings.min_extra_steps, "max_loss_delta": settings.max_loss_delta}),
            "proof_path": MetadataValue.path(proof_out),
        },
    )


@op(ins={"settings": In(PortabilitySettings), "_ready": In(Nothing)})
def save_assets(context: OpExecutionContext, settings: PortabilitySettings) -> None:
    saved_assets = []
    for asset_path in output_asset_paths(settings):
        save_asset_from_host(settings, settings.nvidia_host, asset_path)
        if not saved_local_asset_path(settings, asset_path).exists():
            save_asset_from_host(settings, settings.amd_host, asset_path)
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
                "provenance": MetadataValue.json({"copied_from": settings.nvidia_host, "pipeline_step": "save_assets"}),
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
    return [saved_local_asset_path(settings, asset_path) for asset_path in settings.asset_paths]


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
    trained = train_on_amd(settings, synced)
    copied = copy_checkpoint(settings, trained)
    resumed = resume_on_nvidia(settings, copied)
    validated = validate_resume(settings, resumed)
    save_assets(settings, validated)


if __name__ == "__main__":
    result = portability_job.execute_in_process()
    raise SystemExit(0 if result.success else 1)
