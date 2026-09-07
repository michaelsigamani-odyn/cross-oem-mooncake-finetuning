"""SSH-backed Executor. Direct port of `run_shell` / `ssh_cmd` / `scp_cmd` /
`ssh_options` / `sshpass_env` from the original file (lines ~430-511),
wrapped behind the Executor interface instead of being called ad hoc."""
import os
import shlex
import subprocess
import time
from datetime import datetime, timezone
from typing import List, Optional

from .base import ExecResult, Executor


class SSHExecutor(Executor):
    def __init__(self, ssh_password: Optional[str] = None, connect_timeout_seconds: int = 15):
        self._ssh_password = ssh_password
        self._connect_timeout_seconds = connect_timeout_seconds

    def _ssh_options(self) -> List[str]:
        batch = "no" if self._ssh_password else "yes"
        base = [
            "-o", "IdentitiesOnly=yes",
            "-o", f"BatchMode={batch}",
            "-o", f"ConnectTimeout={self._connect_timeout_seconds}",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=8",
        ]
        if not self._ssh_password:
            return base
        return base + ["-o", "PreferredAuthentications=publickey,password,keyboard-interactive", "-o", "NumberOfPasswordPrompts=1"]

    def _env(self) -> Optional[dict]:
        return {**os.environ, "SSHPASS": self._ssh_password} if self._ssh_password else None

    def _ssh_prefix(self) -> List[str]:
        return ["sshpass", "-e"] if self._ssh_password else []

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
        full_command = self._ssh_prefix() + ["ssh", *self._ssh_options(), host, command] if host else shlex.split(command)
        return self._run_with_retries(full_command, host, retries, timeout_seconds, allow_failure, operation)

    def copy(self, source: str, destination: str, *, recursive: bool = False, retries: int = 0) -> ExecResult:
        command = self._ssh_prefix() + ["scp", *self._ssh_options(), *(["-r"] if recursive else []), source, destination]
        host = destination.split(":", 1)[0] if ":" in destination else (source.split(":", 1)[0] if ":" in source else None)
        return self._run_with_retries(command, host, retries, 600.0, False, "scp")

    def exists(self, host: str, path: str) -> bool:
        return self.run(f"test -e {shlex.quote(path)}", host=host, allow_failure=True).ok

    def _run_with_retries(
        self,
        command: List[str],
        host: Optional[str],
        retries: int,
        timeout_seconds: float,
        allow_failure: bool,
        operation: str,
    ) -> ExecResult:
        shell_text = " ".join(shlex.quote(part) for part in command)
        last: Optional[ExecResult] = None
        for attempt in range(retries + 1):
            start = time.perf_counter()
            print(f"{_utc_now()} START {operation} host={host or 'local'} command={shell_text} timeout_seconds={timeout_seconds}")
            try:
                proc = subprocess.run(command, check=False, text=True, capture_output=True, env=self._env(), timeout=timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                elapsed = time.perf_counter() - start
                raise RuntimeError(
                    f"HANG/TIMEOUT DETECTED\nstage: {operation}\nhost: {host or 'local'}\ncommand: {shell_text}\nelapsed: {elapsed:.3f}\n"
                    f"last stdout: {_tail(exc.stdout)}\nlast stderr: {_tail(exc.stderr)}"
                ) from exc
            elapsed = time.perf_counter() - start
            print(f"{_utc_now()} END {operation} duration={elapsed:.3f} returncode={proc.returncode}")
            result = ExecResult(shell_text, proc.returncode, proc.stdout, proc.stderr, elapsed)
            if result.ok:
                return result
            last = result
            if attempt == retries and allow_failure:
                return result
            if attempt == retries:
                raise RuntimeError(f"command failed: {shell_text}\nexit={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}")
            time.sleep(2)
        return last  # pragma: no cover - unreachable, retries always >= 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _tail(value: object, lines: int = 20) -> str:
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value or "")
    rows = [line for line in text.splitlines() if line.strip()]
    return "\n".join(rows[-lines:]) if rows else "[empty]"
