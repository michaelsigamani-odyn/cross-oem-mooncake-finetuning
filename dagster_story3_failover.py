from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import json
import shlex
import subprocess

from dagster import In, Nothing, OpExecutionContext, Out, job, op


@dataclass
class Story3Settings:
    local_root: str
    remote_root: str
    venv_path: str
    artifact_root: str
    plan_path: str
    ssh_password: str
    run_environments: List[str]
    environment_map: Dict[str, Dict[str, List[str] | str]]


def read_repo_config() -> Dict[str, object]:
    path = Path(__file__).with_name("repo_config.json")
    return json.loads(path.read_text())


def require_prod_hosts(hosts: List[str]) -> List[str]:
    assert len(hosts) == 8, f"prod requires exactly 8 A100 hosts, got {len(hosts)}"
    return hosts


def validate_stage_hosts(hosts: List[str]) -> List[str]:
    assert len(hosts) == 2, f"stage requires exactly 2 DGX hosts, got {len(hosts)}"
    return hosts


def host_map(config: Dict[str, object]) -> Dict[str, Dict[str, List[str] | str]]:
    story3 = config["story3"]
    envs = story3["environments"]
    validate_stage_hosts(envs["stage"]["spark_hosts"])
    require_prod_hosts(envs["prod"]["spark_hosts"])
    return envs


def common_value(config: Dict[str, object], key: str) -> str:
    return str(config["common"][key])


def story3_value(config: Dict[str, object], key: str) -> str:
    return str(config["story3"][key])


def load_settings() -> Story3Settings:
    config = read_repo_config()
    runs = list(config["story3"]["run_environments"])
    assert set(runs).issubset({"dev", "stage", "prod"}), f"invalid run_environments: {runs}"
    return Story3Settings(common_value(config, "local_root_dir"), common_value(config, "remote_root_dir"), story3_value(config, "venv_path"), story3_value(config, "artifact_root"), story3_value(config, "plan_path"), story3_value(config, "ssh_password"), runs, host_map(config))


def run_local(command: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def assert_success(result: subprocess.CompletedProcess, label: str) -> None:
    assert result.returncode == 0, f"{label} failed\nstdout={result.stdout}\nstderr={result.stderr}"


def harness_cmd(settings: Story3Settings, env_name: str, spark_host: str, radeon_host: str, artifact_root: str) -> List[str]:
    password = ["--ssh-password", settings.ssh_password] if settings.ssh_password else []
    return ["python3", f"{settings.local_root}/failover_harness.py", "--plan-path", f"{settings.local_root}/{settings.plan_path.lstrip('./')}", "--local-root", settings.local_root, "--remote-root", settings.remote_root, "--spark-host", spark_host, "--radeon-host", radeon_host, "--venv-path", settings.venv_path, "--artifact-root", artifact_root, "--environment", env_name, *password]


def log_command(context: OpExecutionContext, command: List[str]) -> None:
    context.log.info("RUN %s", " ".join(shlex.quote(item) for item in command))


def run_matrix(context: OpExecutionContext, settings: Story3Settings) -> None:
    for env_name in settings.run_environments:
        env_settings = settings.environment_map[env_name]
        radeon_host = str(env_settings["radeon_host"])
        spark_hosts = list(env_settings["spark_hosts"])
        for spark_host in spark_hosts:
            root = f"{settings.artifact_root}/{env_name}/{spark_host}"
            command = harness_cmd(settings, env_name, spark_host, radeon_host, root)
            log_command(context, command)
            assert_success(run_local(command), f"story3 {env_name} {spark_host}")


@op(out=Out(Story3Settings))
def settings_op(context: OpExecutionContext) -> Story3Settings:
    settings = load_settings()
    context.log.info("Story3 run_environments=%s", settings.run_environments)
    return settings


@op(ins={"settings": In(Story3Settings)}, out=Out(Nothing))
def run_story3_op(context: OpExecutionContext, settings: Story3Settings) -> None:
    run_matrix(context, settings)


@job
def story3_failover_job() -> None:
    settings = settings_op()
    run_story3_op(settings)


if __name__ == "__main__":
    result = story3_failover_job.execute_in_process()
    raise SystemExit(0 if result.success else 1)
