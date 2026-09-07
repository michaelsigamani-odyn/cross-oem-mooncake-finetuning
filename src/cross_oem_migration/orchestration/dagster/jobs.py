"""Asset job -- the equivalent of the original @job finetuning_sequential_job
(lines 3527-3541), but derived automatically from the asset graph via
`define_asset_job` rather than hand-wired op-to-op dependencies. Dagster
computes the execution order from `deps=[...]` on each asset."""
from dagster import define_asset_job

from .assets import ALL_ASSETS

cross_oem_migration_job = define_asset_job(
    name="cross_oem_migration_job",
    selection=ALL_ASSETS,
    description="End-to-end cross-OEM checkpoint migration: preflight -> train -> transfer -> resume -> validate -> report -> publish.",
)
