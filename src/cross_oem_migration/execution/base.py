"""Executor interface (DIP): everything upstream (hardware probes, transfer
backends, workloads) depends on this abstraction, never on `subprocess` or
`ssh` directly. This replaces the free functions `run_shell` / `ssh_cmd` /
`scp_cmd` that were scattered through the original 3,547-line file and
called directly from ~40 different places.

Swapping SSH for, say, a Kubernetes exec, or a local-only dev mode, means
writing one new Executor implementation -- nothing else in the codebase
changes (Open/Closed Principle).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ExecResult:
    command: str
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class Executor(ABC):
    """Runs a command against a target (a host, or 'local'). Retries,
    timeouts, and start/end logging are the Executor's job so that callers
    (hardware probes, workloads, transfer backends) stay free of
    subprocess/ssh mechanics entirely."""

    @abstractmethod
    def run(
        self,
        command: str,
        *,
        host: Optional[str] = None,
        retries: int = 0,
        timeout_seconds: float = 600.0,
        allow_failure: bool = False,
        operation: str = "command",
    ) -> ExecResult:
        ...

    @abstractmethod
    def copy(
        self,
        source: str,
        destination: str,
        *,
        recursive: bool = False,
        retries: int = 0,
    ) -> ExecResult:
        """Copy a local path to/from a remote target. `destination`/`source`
        use the `host:path` convention, matching scp semantics."""
        ...

    @abstractmethod
    def exists(self, host: str, path: str) -> bool:
        ...
