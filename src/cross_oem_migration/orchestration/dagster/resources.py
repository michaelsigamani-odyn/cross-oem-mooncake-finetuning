"""Dagster resources: this file (and assets/, checks.py, definitions.py)
is the ONLY part of the codebase allowed to `import dagster` (ask #3).
Everything a resource does is delegate to the framework-agnostic classes
in config/, execution/, hardware/, transfer/, workloads/, data/ -- so
those packages stay importable, testable, and reusable from a plain
script, a notebook, or a different orchestrator entirely.
"""
from pathlib import Path

from dagster import ConfigurableResource

from ...config import PortabilitySettings, load_settings
from ...data.filesystem import LocalFilesystemArtifactStore, LocalFilesystemDatasetProvider
from ...data.metrics_db import RunMetricsDatabase
from ...execution.ssh import SSHExecutor
from ...hardware import get_adapter
from ...transfer import get_backend


class SettingsResource(ConfigurableResource):
    """Loads configs/run.json + configs/machines.json once per run.
    Wrapping `load_settings()` in a resource (rather than calling it at
    import time, as the original module-level-ish pattern effectively
    did) means tests can inject a different config_dir per test."""

    def get(self) -> PortabilitySettings:
        return load_settings()


class ExecutorResource(ConfigurableResource):
    def get(self, settings: PortabilitySettings) -> SSHExecutor:
        return SSHExecutor(ssh_password=settings.ssh_password, connect_timeout_seconds=settings.ssh_connect_timeout_seconds)


class HardwareResource(ConfigurableResource):
    """Thin pass-through to hardware.get_adapter -- exists as a resource
    so assets don't import hardware.registry directly, keeping the
    dependency direction one-way (orchestration depends on hardware, not
    the reverse)."""

    def adapter_for(self, vendor: str):
        return get_adapter(vendor)


class TransferResource(ConfigurableResource):
    def get(self, settings: PortabilitySettings, executor: SSHExecutor):
        return get_backend(settings.transfer.backend, executor, settings, settings.remote_root)


class DataResource(ConfigurableResource):
    def dataset_provider(self, settings: PortabilitySettings, executor: SSHExecutor) -> LocalFilesystemDatasetProvider:
        return LocalFilesystemDatasetProvider(executor, settings.local_root, settings.remote_root, settings.command_retries)

    def artifact_store(self, settings: PortabilitySettings, executor: SSHExecutor) -> LocalFilesystemArtifactStore:
        return LocalFilesystemArtifactStore(executor, settings.local_root, settings.remote_root, settings.run_id, settings.command_retries)

    def metrics_db(self, settings: PortabilitySettings) -> RunMetricsDatabase:
        return RunMetricsDatabase(Path(settings.local_root) / "artifacts" / "run_metrics.sqlite")
