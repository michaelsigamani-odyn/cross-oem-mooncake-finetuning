"""Data abstraction (ask #5): input datasets and output logs/checkpoints
are accessed through these two interfaces instead of raw path-string
concatenation scattered through the pipeline (the original had
`local_asset_path` / `remote_asset_path` / `normalized_asset_path` /
`remote_dataset_path` / `sync_paths` / `save_asset_from_host` /
`output_asset_paths` as free functions operating on plain strings,
original lines 524-609).

Swapping "local filesystem" for "S3" or "a shared NFS mount" means writing
one new implementation of each interface -- assets and workloads never
construct paths themselves.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List


class DatasetProvider(ABC):
    """Makes a named dataset available on a remote host before training."""

    @abstractmethod
    def sync_to_host(self, dataset_path: str, host: str) -> str:
        """Ensure `dataset_path` exists on `host`; return the path to use
        there (may differ from the local path)."""

    @abstractmethod
    def local_path(self, dataset_path: str) -> Path:
        ...


class ArtifactStore(ABC):
    """Pulls workload outputs (checkpoints, run summaries, logs) back from
    a remote host into durable local storage, and lists what's there."""

    @abstractmethod
    def save_from_host(self, host: str, remote_path: str, asset_key: str) -> Path:
        ...

    @abstractmethod
    def local_path_for(self, asset_key: str) -> Path:
        ...

    @abstractmethod
    def list_saved_assets(self) -> List[str]:
        ...
