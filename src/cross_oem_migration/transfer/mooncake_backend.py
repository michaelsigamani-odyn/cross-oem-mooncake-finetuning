"""Mooncake TCP transfer backend -- ports checkpoints using the
`scripts/transfer/mooncake_tcp_agent.py` workload script running on both
hosts. Direct port of `transfer_checkpoint_mooncake` and its helpers
(original lines 1655-1833). The receiver/sender wire protocol and retry
logic are unchanged; only the SSH mechanics were swapped for the injected
Executor.
"""
import json
import os
import shlex
import time
from typing import Any, Dict, Optional

from ..config.models import MooncakeTransferConfig, PortabilitySettings
from ..execution.base import Executor
from .base import TransferBackend, TransferResult


class MooncakeTransferBackend(TransferBackend):
    name = "mooncake_tcp"

    def __init__(self, executor: Executor, settings: PortabilitySettings, remote_agent_path: str):
        if not settings.transfer.mooncake:
            raise RuntimeError("transfer backend mooncake_tcp selected but settings.transfer.mooncake is missing")
        self._executor = executor
        self._settings = settings
        self._config = settings.transfer.mooncake
        self._agent_path = remote_agent_path

    def transfer(self, *, source_host: str, source_path: str, destination_host: str, destination_path: str) -> TransferResult:
        settings, config = self._settings, self._config
        destination_tmp = f"{destination_path}.tmp"
        control_host = _receiver_control_host(config)
        payload: Dict[str, Any] = {}
        last_error: Optional[Exception] = None

        for attempt in range(3):
            session = f"{int(time.time() * 1000)}_{os.getpid()}_{attempt}"
            preferred_port = config.destination.control_port + (attempt * 50)
            control_port = self._reserve_control_port(destination_host, preferred_port)
            pid_path = f"/tmp/mooncake_receiver_{settings.run_id}_{settings.checkpoint_step}_{session}_{control_port}.pid"
            log_path = f"/tmp/mooncake_receiver_{settings.run_id}_{settings.checkpoint_step}_{session}_{control_port}.log"
            receiver_cmd = self._receiver_command(destination_path, destination_tmp, pid_path, log_path, session, control_port)
            self._executor.run(receiver_cmd, host=destination_host, retries=settings.command_retries,
                                timeout_seconds=settings.command_timeout_seconds, operation="mooncake_receiver_launch")
            try:
                self._wait_for_receiver(source_host, destination_host, control_host, control_port, pid_path, log_path)
                sender_cmd = self._sender_command(source_path, destination_path, destination_tmp, control_host, session, control_port)
                sender_result = self._executor.run(sender_cmd, host=source_host, timeout_seconds=settings.transfer_timeout_seconds, operation="mooncake_sender_transfer")
                payload = _parse_sender_result(sender_result.stdout)
                break
            except RuntimeError as exc:
                last_error = exc
                excerpt = self._receiver_log_excerpt(destination_host, log_path)
                if not _mooncake_retryable_failure(str(exc), excerpt) or attempt == 2:
                    raise RuntimeError(f"mooncake sender failed: {exc}; receiver_log={excerpt}") from exc
                time.sleep(2)
            finally:
                self._executor.run(f"test -f {pid_path} && kill $(cat {pid_path}) >/dev/null 2>&1 || true",
                                    host=destination_host, retries=settings.command_retries, allow_failure=True,
                                    timeout_seconds=settings.command_timeout_seconds, operation="mooncake_receiver_cleanup")

        if not payload and last_error:
            raise RuntimeError(f"mooncake sender failed after retries: {last_error}")
        return TransferResult(
            success=bool(payload.get("success", False)),
            backend=str(payload.get("backend", self.name)),
            bytes_transferred=int(payload.get("bytes_transferred", 0)),
            elapsed_seconds=float(payload.get("elapsed_seconds", 0.0)),
            throughput_bytes_per_second=float(payload.get("throughput_bytes_per_second", 0.0)),
            checksum_ok=bool(payload.get("checksum_ok", False)),
            resume_validation_ok=False,
            details=payload,
        )

    def verify(self, result: TransferResult) -> TransferResult:
        return result  # the mooncake agent verifies its own manifest on commit

    def sync_agent_to(self, host: str) -> None:
        self._executor.run(f"mkdir -p {shlex.quote(str(self._settings.remote_root))}", host=host, retries=self._settings.command_retries)
        self._executor.copy(self._agent_path, f"{host}:{self._settings.remote_root}/", retries=self._settings.command_retries)

    # -- private helpers, ported near-verbatim ---------------------------------
    def _reserve_control_port(self, destination_host: str, preferred_port: int) -> int:
        for offset in range(100):
            port = preferred_port + offset
            script = (
                "import socket,sys; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); rc=0\n"
                "try: s.bind(('0.0.0.0', int(sys.argv[1])))\nexcept OSError: rc=1\ns.close(); raise SystemExit(rc)"
            )
            command = f"python3 -c {shlex.quote(script)} {port}"
            result = self._executor.run(command, host=destination_host, retries=self._settings.command_retries, allow_failure=True)
            if result.ok:
                return port
        raise RuntimeError(f"unable to reserve receiver control port starting at {preferred_port}")

    def _receiver_command(self, destination_ckpt: str, destination_tmp: str, pid_path: str, log_path: str, session: str, control_port: int) -> str:
        config = self._config
        chunk_bytes = config.chunk_mb * 1024 * 1024
        parts = [
            f"python3 {self._agent_path}",
            "--mode receiver",
            f"--metadata-server {shlex.quote(config.metadata_server)}",
            f"--protocol {shlex.quote(config.protocol)}",
            f"--local-server-name {shlex.quote(config.destination.server_name)}",
            f"--peer-server-name {shlex.quote(config.source.server_name)}",
            f"--chunk-bytes {chunk_bytes}",
            f"--max-inflight {config.max_inflight}",
            f"--destination-dir {shlex.quote(destination_ckpt)}",
            f"--destination-tmp-dir {shlex.quote(destination_tmp)}",
            f"--session-id {shlex.quote(session)}",
            "--control-host 0.0.0.0",
            f"--control-port {control_port}",
        ]
        launch = f"nohup {' '.join(parts)} > {shlex.quote(log_path)} 2>&1 & echo $! > {shlex.quote(pid_path)}"
        return f"{_mooncake_runtime_prefix(config)} bash -lc {shlex.quote(launch)}"

    def _sender_command(self, source_ckpt: str, destination_ckpt: str, destination_tmp: str, control_host: str, session: str, control_port: int) -> str:
        config = self._config
        chunk_bytes = config.chunk_mb * 1024 * 1024
        parts = [
            f"python3 {self._agent_path}",
            "--mode sender",
            f"--metadata-server {shlex.quote(config.metadata_server)}",
            f"--protocol {shlex.quote(config.protocol)}",
            f"--local-server-name {shlex.quote(config.source.server_name)}",
            f"--peer-server-name {shlex.quote(config.destination.server_name)}",
            f"--chunk-bytes {chunk_bytes}",
            f"--max-inflight {config.max_inflight}",
            f"--source-dir {shlex.quote(source_ckpt)}",
            f"--destination-dir {shlex.quote(destination_ckpt)}",
            f"--destination-tmp-dir {shlex.quote(destination_tmp)}",
            f"--session-id {shlex.quote(session)}",
            f"--control-host {shlex.quote(control_host)}",
            f"--control-port {control_port}",
        ]
        return f"{_mooncake_runtime_prefix(config)} {' '.join(parts)}"

    def _wait_for_receiver(self, source_host: str, destination_host: str, control_host: str, control_port: int, pid_path: str, log_path: str) -> None:
        probe = (
            f"python3 -c \"import socket; "
            f"s=socket.create_connection(({control_host!r}, {control_port}), timeout=2); "
            "s.sendall(b'{\\\"command\\\": \\\"hello\\\"}\\n'); "
            "r=s.makefile('rb').readline().decode('utf-8'); s.close(); "
            "raise SystemExit(0 if r.strip() else 2)\""
        )
        for _ in range(20):
            result = self._executor.run(probe, host=source_host, retries=self._settings.command_retries, allow_failure=True,
                                         timeout_seconds=self._settings.command_timeout_seconds, operation="mooncake_receiver_probe")
            if result.ok:
                return
            alive = self._executor.run(f"test -f {pid_path} && kill -0 $(cat {pid_path})", host=destination_host,
                                        retries=self._settings.command_retries, allow_failure=True,
                                        timeout_seconds=self._settings.command_timeout_seconds, operation="mooncake_receiver_alive")
            if not alive.ok:
                excerpt = self._receiver_log_excerpt(destination_host, log_path)
                raise RuntimeError(f"mooncake receiver process exited before becoming reachable at {control_host}:{control_port}; log={excerpt}")
            time.sleep(1)
        excerpt = self._receiver_log_excerpt(destination_host, log_path)
        raise RuntimeError(f"mooncake receiver not reachable at {control_host}:{control_port}; log={excerpt}")

    def _receiver_log_excerpt(self, destination_host: str, log_path: str) -> str:
        command = (
            "python3 -c \"from pathlib import Path; p=Path(%r); "
            "text=p.read_text(errors='replace') if p.exists() else 'log file missing'; "
            "lines=text.splitlines(); print(' | '.join(lines[-20:]))\"" % log_path
        )
        result = self._executor.run(command, host=destination_host, retries=self._settings.command_retries, allow_failure=True,
                                     timeout_seconds=self._settings.command_timeout_seconds, operation="mooncake_receiver_log")
        return result.stdout.strip() or "no log output"


def _mooncake_runtime_prefix(config: MooncakeTransferConfig) -> str:
    if config.protocol.strip().lower() != "tcp":
        return "env"
    return "env MC_FORCE_TCP=1 MC_TE_FILTERS=__no_such_rdma_device__"


def _receiver_control_host(config: MooncakeTransferConfig) -> str:
    host = config.destination.control_host.strip()
    if host and host not in {"0.0.0.0", "127.0.0.1", "localhost"}:
        return host
    return config.destination.server_name.split(":", 1)[0]


def _mooncake_retryable_failure(error_text: str, receiver_excerpt: str) -> bool:
    text = f"{error_text} {receiver_excerpt}".lower()
    return any(p in text for p in ["address already in use", "connection refused", "not reachable", "exited before becoming reachable"])


def _parse_sender_result(stdout: str) -> Dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    try:
        payload = json.loads(lines[-1]) if lines else {}
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"mooncake sender returned non-JSON output: {stdout}") from exc
    if not payload:
        raise RuntimeError("mooncake sender produced no JSON result")
    return payload
