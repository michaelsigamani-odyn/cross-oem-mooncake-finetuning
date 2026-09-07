"""LocalExecutor - runs commands on the machine running Dagster itself.
Useful for tests, dry-runs, and single-box demos where source_host/target_host
both resolve to localhost. Implements the same Executor contract as
SSHExecutor so assets/workloads never need to know which one they got."""
import shlex
import subprocess
import time
from typing import Optional

from .base import ExecResult, Executor
from .ssh import _tail, _utc_now


class LocalExecutor(Executor):
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
        for attempt in range(retries + 1):
            start = time.perf_counter()
            print(f"{_utc_now()} START {operation} host=local command={command}")
            try:
                proc = subprocess.run(shlex.split(command), check=False, text=True, capture_output=True, timeout=timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"HANG/TIMEOUT DETECTED\nstage: {operation}\ncommand: {command}\nlast stdout: {_tail(exc.stdout)}") from exc
            elapsed = time.perf_counter() - start
            result = ExecResult(command, proc.returncode, proc.stdout, proc.stderr, elapsed)
            if result.ok or allow_failure or attempt == retries:
                return result
            time.sleep(2)
        raise AssertionError("unreachable")

    def copy(self, source: str, destination: str, *, recursive: bool = False, retries: int = 0) -> ExecResult:
        flags = "-r " if recursive else ""
        return self.run(f"cp {flags}{source} {destination}", retries=retries)

    def exists(self, host: str, path: str) -> bool:
        return self.run(f"test -e {shlex.quote(path)}", allow_failure=True).ok
