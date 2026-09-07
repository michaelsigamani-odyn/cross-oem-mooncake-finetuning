"""Dagster entry point. This is the ONLY file `workspace.yaml` /
`dagster_cloud.yaml` should reference (`dagster dev -f
src/cross_oem_migration/orchestration/dagster/definitions.py`, or install
the package and use its module path -- see pyproject.toml). Everything it
assembles is defined elsewhere; this file's only job is wiring."""
from dagster import Definitions

from .assets import ALL_ASSET_CHECKS, ALL_ASSETS
from .jobs import cross_oem_migration_job
from .resources import DataResource, ExecutorResource, HardwareResource, SettingsResource, TransferResource

defs = Definitions(
    assets=ALL_ASSETS,
    asset_checks=ALL_ASSET_CHECKS,
    jobs=[cross_oem_migration_job],
    resources={
        "settings": SettingsResource(),
        "executor": ExecutorResource(),
        "hardware": HardwareResource(),
        "transfer": TransferResource(),
        "data": DataResource(),
    },
)
