"""Docker command wrapping for containerized machines (e.g. the MI300X
runpod host in configs/machines.json). Ported from `docker_command`
(original lines 1985-2001). Kept separate from hardware/ because this is
about *execution mode* (host vs. container), independent of GPU vendor."""
import shlex

from ..config.models import MachineSpec


def wrap_in_container(machine: MachineSpec, inner_command: str, mount_path: str) -> str:
    if not machine.container_image or machine.execution_mode != "docker":
        return inner_command
    run = [
        "docker run --rm",
        "--network host",
        "--ipc=host",
        "--shm-size 16g",
        "--device /dev/kfd",
        "--device /dev/dri",
        "--group-add video",
        f"-v {shlex.quote(mount_path)}:/workspace",
        "-w /workspace",
        shlex.quote(machine.container_image),
        "bash -lc",
        shlex.quote(inner_command),
    ]
    return " ".join(run)
