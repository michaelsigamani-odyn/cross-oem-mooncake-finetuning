"""Filesystem-backed dataset/artifact handling. Direct port of
`sync_asset_to_host` / `save_asset_from_host` / `output_asset_paths`
(original lines 550-609), now behind the DatasetProvider/ArtifactStore
interfaces and using the injected Executor instead of free ssh_cmd calls.
"""
from pathlib import Path
from typing import List

from ..execution.base import Executor
from ..remote_paths import resolve_remote_root
from .base import ArtifactStore, DatasetProvider


class LocalFilesystemDatasetProvider(DatasetProvider):
    def __init__(self, executor: Executor, local_root: str, remote_root: str, command_retries: int = 3):
        self._executor = executor
        self._local_root = Path(local_root)
        self._remote_root = remote_root
        self._retries = command_retries

    def local_path(self, dataset_path: str) -> Path:
        return self._local_root / dataset_path

    def sync_to_host(self, dataset_path: str, host: str) -> str:
        local_path = self.local_path(dataset_path)
        remote_root = resolve_remote_root(self._executor, host, self._remote_root, self._retries)
        remote_path = f"{remote_root}/{dataset_path}"
        if not local_path.exists():
            return remote_path  # assume already present remotely; nothing to push
        parent = f"{remote_root}/{Path(dataset_path).parent}" if not local_path.is_dir() else f"{remote_root}/{dataset_path}"
        self._executor.run(f"mkdir -p {parent}", host=host, retries=self._retries)
        self._executor.copy(str(local_path), f"{host}:{parent}", recursive=True, retries=self._retries)
        return remote_path


class LocalFilesystemArtifactStore(ArtifactStore):
    def __init__(self, executor: Executor, local_root: str, remote_root: str, run_id: str, command_retries: int = 3):
        self._executor = executor
        self._local_root = Path(local_root)
        self._remote_root = remote_root
        self._run_id = run_id
        self._retries = command_retries

    def local_path_for(self, asset_key: str) -> Path:
        return self._local_root / "artifacts" / self._run_id / asset_key

    def save_from_host(self, host: str, remote_path: str, asset_key: str) -> Path:
        local_target = self.local_path_for(asset_key)
        local_target.parent.mkdir(parents=True, exist_ok=True)
        if not self._executor.exists(host, remote_path):
            return local_target
        recursive = self._executor.run(f"test -d {remote_path}", host=host, allow_failure=True).ok
        target = str(local_target.parent if recursive else local_target)
        self._executor.copy(f"{host}:{remote_path}", target, recursive=recursive, retries=self._retries)
        return local_target

    def list_saved_assets(self) -> List[str]:
        run_dir = self._local_root / "artifacts" / self._run_id
        if not run_dir.exists():
            return []
        return sorted(str(p.relative_to(run_dir)) for p in run_dir.rglob("*") if p.is_file())


def default_output_asset_paths(run_id: str, final_step: int, checkpoint_step: int, extra_asset_paths: List[str]) -> List[str]:
    """Ported from `output_asset_paths` (original lines 589-606): the
    canonical set of files this pipeline produces and should materialize
    as assets, regardless of which host produced them."""
    target_run = f"nvidia_run_{run_id}"
    source_run = f"amd_run_{run_id}"
    defaults = [
        f"{source_run}/run_summary.json",
        f"{target_run}/run_summary.json",
        f"{target_run}/resume_validation_proof.json",
        f"{target_run}/checkpoint-{final_step}/trainer_state.json",
        f"{source_run}/checkpoint-{checkpoint_step}/checkpoint_meta.json",
    ]
    return list(dict.fromkeys([*extra_asset_paths, *defaults]))
