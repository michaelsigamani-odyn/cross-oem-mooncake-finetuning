"""Publishes every output artifact (checkpoints, telemetry, transfer test
results, the report) from the remote hosts into local `artifacts/<run_id>/`,
and attaches two asset checks. Ports `save_assets` +
`saved_assets_exist_check` + `saved_assets_supported_type_check` (original
lines 3470-3525).

Difference from the original: the original emitted ad hoc
`AssetMaterialization` events by hand inside plain `@op`s, with the asset
checks (`@asset_check`) pointed at a manually-constructed `AssetKey` that
wasn't otherwise produced by any `@asset`. That's why asset checks likely
weren't rendering attached to anything sensible in the UI. Here,
`published_manifest` is a real `@asset`, and the two checks are declared
against it directly via `check_specs`/`@asset_check(asset=published_manifest)`,
which is what makes them show up as pass/fail badges on that asset node.
"""
from pathlib import Path
from typing import Any, Dict, List

from dagster import AssetCheckResult, AssetExecutionContext, MetadataValue, asset, asset_check

from ....data.filesystem import LocalFilesystemArtifactStore, default_output_asset_paths
from ....remote_paths import resolve_remote_root
from ..resources import DataResource, ExecutorResource, SettingsResource
from .reporting import cross_oem_report

SUPPORTED_DATA_TYPES = {"directory", "json", "jsonl", "csv", "parquet", "arrow", "sqlite", "db", "txt"}


def _infer_data_type(path: Path) -> str:
    if path.is_dir():
        return "directory"
    suffix = path.suffix.lower()
    return suffix.lstrip(".") if suffix else "unknown"


def _asset_paths(settings) -> List[str]:
    return default_output_asset_paths(settings.run_id, settings.final_step, settings.checkpoint_step, settings.asset_paths)


@asset(group_name="published", deps=[cross_oem_report], description="Copies every output artifact from the remote hosts into local artifacts/<run_id>/.")
def published_manifest(context: AssetExecutionContext, settings: SettingsResource, executor: ExecutorResource, data: DataResource) -> Dict[str, Any]:
    cfg = settings.get()
    exec_ = executor.get(cfg)
    store: LocalFilesystemArtifactStore = data.artifact_store(cfg, exec_)
    source_root = resolve_remote_root(exec_, cfg.source_host, cfg.remote_root, cfg.command_retries)
    target_root = resolve_remote_root(exec_, cfg.target_host, cfg.remote_root, cfg.command_retries)

    saved_assets = []
    for asset_path in _asset_paths(cfg):
        local_path = store.save_from_host(cfg.target_host, f"{target_root}/{asset_path}", asset_path)
        if not local_path.exists():
            local_path = store.save_from_host(cfg.source_host, f"{source_root}/{asset_path}", asset_path)
        data_type = _infer_data_type(local_path) if local_path.exists() else "missing"
        saved_assets.append({"asset_path": asset_path, "saved_path": str(local_path), "data_type": data_type})

    manifest = {"run_id": cfg.run_id, "saved_assets": saved_assets}
    context.add_output_metadata({
        "saved_assets": MetadataValue.json(saved_assets),
        "artifact_root": MetadataValue.path(str(Path(cfg.local_root) / "artifacts" / cfg.run_id)),
    })
    return manifest


@asset_check(asset=published_manifest, name="saved_assets_exist", description="Every declared output asset was actually copied down and is non-empty.")
def saved_assets_exist_check(settings: SettingsResource, published_manifest: Dict[str, Any]) -> AssetCheckResult:
    cfg = settings.get()
    missing = [item["asset_path"] for item in published_manifest["saved_assets"] if item["data_type"] == "missing"]
    empty = [
        item["asset_path"] for item in published_manifest["saved_assets"]
        if item["data_type"] not in ("missing", "directory") and Path(item["saved_path"]).exists() and Path(item["saved_path"]).stat().st_size == 0
    ]
    return AssetCheckResult(
        passed=not missing and not empty,
        metadata={"missing_paths": MetadataValue.json(missing), "empty_files": MetadataValue.json(empty)},
    )


@asset_check(asset=published_manifest, name="saved_assets_supported_type", description="Every saved output asset has a recognized, queryable data type.")
def saved_assets_supported_type_check(published_manifest: Dict[str, Any]) -> AssetCheckResult:
    typed = [item for item in published_manifest["saved_assets"] if item["data_type"] != "missing"]
    unsupported = [item for item in typed if item["data_type"] not in SUPPORTED_DATA_TYPES]
    return AssetCheckResult(
        passed=not unsupported,
        metadata={"unsupported_assets": MetadataValue.json(unsupported), "supported_types": MetadataValue.json(sorted(SUPPORTED_DATA_TYPES))},
    )
