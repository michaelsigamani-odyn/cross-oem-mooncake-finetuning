"""scp-based checkpoint transfer, hopping through the local machine
running the pipeline as an intermediate stage (matches the original
behavior: source -> local /tmp stage -> destination). Manifest
verification shells out to scripts/validation/checkpoint_io.py on both
ends -- that script is a standalone workload artifact (ask #2/#5), not
duplicated logic here.
"""
import json
import shlex
import time
from pathlib import Path
from typing import Any, Dict, List

from ..config.models import PortabilitySettings
from ..execution.base import Executor
from .base import TransferBackend, TransferResult


class ScpTransferBackend(TransferBackend):
    name = "scp"

    def __init__(self, executor: Executor, settings: PortabilitySettings, remote_checkpoint_io_path: str):
        self._executor = executor
        self._settings = settings
        self._checkpoint_io_path = remote_checkpoint_io_path

    def transfer(self, *, source_host: str, source_path: str, destination_host: str, destination_path: str) -> TransferResult:
        settings = self._settings
        local_stage = Path("/tmp") / settings.run_id
        checkpoint_name = Path(source_path).name
        local_stage.mkdir(parents=True, exist_ok=True)
        local_checkpoint_dir = local_stage / checkpoint_name

        start = time.perf_counter()
        if local_checkpoint_dir.exists():
            self._executor.run(f"rm -rf {shlex.quote(str(local_checkpoint_dir))}")
        self._executor.run(
            f"rm -rf {shlex.quote(destination_path)}",
            host=destination_host, retries=settings.command_retries, allow_failure=True,
            timeout_seconds=settings.transfer_timeout_seconds, operation="checkpoint_clean_destination",
        )
        self._executor.copy(
            f"{source_host}:{source_path}", str(local_stage), recursive=True, retries=settings.command_retries,
        )
        self._executor.copy(
            str(local_checkpoint_dir), f"{destination_host}:{Path(destination_path).parent}/", recursive=True,
            retries=settings.command_retries,
        )
        elapsed = time.perf_counter() - start

        verified = self._verify_manifests(source_host, source_path, destination_host, destination_path)
        bytes_transferred = int(verified["total_bytes"])
        return TransferResult(
            success=bool(verified["checksum_ok"]),
            backend=self.name,
            bytes_transferred=bytes_transferred,
            elapsed_seconds=elapsed,
            throughput_bytes_per_second=0.0 if elapsed <= 0 else bytes_transferred / elapsed,
            checksum_ok=bool(verified["checksum_ok"]),
            resume_validation_ok=False,
            details=verified,
        )

    def verify(self, result: TransferResult) -> TransferResult:
        # scp verification happens inline during transfer() via manifests;
        # nothing further to do post-hoc.
        return result

    def _verify_manifests(self, source_host: str, source_path: str, destination_host: str, destination_path: str) -> Dict[str, Any]:
        settings = self._settings
        stamp = f"{settings.run_id}_{settings.checkpoint_step}_{int(time.time())}"
        source_manifest_remote = f"/tmp/source_manifest_{stamp}.json"
        destination_manifest_remote = f"/tmp/destination_manifest_{stamp}.json"
        source_manifest_local = Path("/tmp") / f"source_manifest_{stamp}.json"
        destination_manifest_local = Path("/tmp") / f"destination_manifest_{stamp}.json"

        self._executor.run(
            f"python3 {self._checkpoint_io_path} --mode manifest --checkpoint-a {source_path} --output {source_manifest_remote}",
            host=source_host, retries=settings.command_retries, timeout_seconds=settings.transfer_timeout_seconds, operation="manifest_source",
        )
        self._executor.run(
            f"python3 {self._checkpoint_io_path} --mode manifest --checkpoint-a {destination_path} --output {destination_manifest_remote}",
            host=destination_host, retries=settings.command_retries, timeout_seconds=settings.transfer_timeout_seconds, operation="manifest_destination",
        )
        self._executor.copy(f"{source_host}:{source_manifest_remote}", str(source_manifest_local), retries=settings.command_retries)
        self._executor.copy(f"{destination_host}:{destination_manifest_remote}", str(destination_manifest_local), retries=settings.command_retries)

        source_manifest = json.loads(source_manifest_local.read_text())
        destination_manifest = json.loads(destination_manifest_local.read_text())
        mismatches = _strict_manifest_mismatches(source_manifest, destination_manifest)
        return {
            "checksum_ok": not mismatches,
            "mismatches": mismatches,
            "file_count": int(source_manifest.get("file_count", 0)),
            "total_bytes": int(source_manifest.get("total_bytes", 0)),
        }


def _as_int(value: Any):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _strict_manifest_mismatches(source_manifest: Dict[str, Any], destination_manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    destination_files = {item["relative_path"]: item for item in destination_manifest.get("files", [])}
    mismatches: List[Dict[str, Any]] = []
    for source_item in source_manifest.get("files", []):
        relative_path = source_item["relative_path"]
        destination_item = destination_files.get(relative_path)
        if destination_item is None:
            mismatches.append({"relative_path": relative_path, "error": "missing"})
            continue
        source_size, destination_size = _as_int(source_item.get("size_bytes")), _as_int(destination_item.get("size_bytes"))
        source_sha, destination_sha = source_item.get("sha256"), destination_item.get("sha256")
        if source_size is None or destination_size is None or not source_sha or not destination_sha:
            if str(source_item.get("suffix", "")) != str(destination_item.get("suffix", "")):
                mismatches.append({"relative_path": relative_path, "error": "suffix_mismatch"})
            continue
        if source_size != destination_size or str(source_sha) != str(destination_sha):
            mismatches.append({"relative_path": relative_path, "source_size": source_size, "destination_size": destination_size})
    source_paths = {item["relative_path"] for item in source_manifest.get("files", [])}
    for relative_path in sorted(p for p in destination_files if p not in source_paths):
        mismatches.append({"relative_path": relative_path, "error": "unexpected"})
    return mismatches
