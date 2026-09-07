"""Backend lookup -- adding a new transfer strategy means adding one class
and one entry here (Open/Closed Principle), replacing the original
`if backend == "mooncake_tcp" / elif backend == "scp"` dispatch in
`transfer_checkpoint` (original lines 1619-1624)."""
from typing import Dict

from ..config.models import PortabilitySettings
from ..execution.base import Executor
from .base import TransferBackend
from .mooncake_backend import MooncakeTransferBackend
from .scp_backend import ScpTransferBackend


def get_backend(name: str, executor: Executor, settings: PortabilitySettings, remote_root: str) -> TransferBackend:
    if name == "scp":
        return ScpTransferBackend(executor, settings, f"{remote_root}/scripts/validation/checkpoint_io.py")
    if name == "mooncake_tcp":
        return MooncakeTransferBackend(executor, settings, f"{remote_root}/scripts/transfer/mooncake_tcp_agent.py")
    raise ValueError(f"unsupported transfer backend: {name!r}")
