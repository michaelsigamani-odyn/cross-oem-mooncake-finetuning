from .preflight import source_gpu_preflight, target_gpu_preflight
from .published import published_manifest, saved_assets_exist_check, saved_assets_supported_type_check
from .reporting import cross_oem_report, run_metrics_db
from .resume import resume_validation, resumed_checkpoint
from .training import source_checkpoint
from .transfer import transferred_checkpoint

ALL_ASSETS = [
    source_gpu_preflight,
    target_gpu_preflight,
    source_checkpoint,
    transferred_checkpoint,
    resumed_checkpoint,
    resume_validation,
    cross_oem_report,
    run_metrics_db,
    published_manifest,
]

ALL_ASSET_CHECKS = [
    saved_assets_exist_check,
    saved_assets_supported_type_check,
]

__all__ = ["ALL_ASSETS", "ALL_ASSET_CHECKS"]
