"""Reporting assets. Ports the pure-math core of `build_cross_oem_report`
(original lines 3289-3469) via reporting.metrics -- all the energy/cost
formulas and their independent cross-checks live there, dependency-free
and unit-testable. This file's job is purely: pull the two run summaries,
call the pure functions, and materialize the result -- including a
SQLite-backed `run_metrics_db` asset so run history is queryable, not
just a JSON blob per run (this addresses "assets... including databases"
from the request; see data/metrics_db.py for why SQLite).
"""
import json
from datetime import datetime, timezone
from typing import Any, Dict

from dagster import AssetExecutionContext, MetadataValue, asset

from ....reporting.metrics import as_float, end_to_end_summary, formula_validation, migration_to_first_step_seconds
from ..resources import DataResource, ExecutorResource, SettingsResource
from .resume import resume_validation


@asset(group_name="reporting", deps=[resume_validation], description="Cross-OEM migration report: end-to-end throughput/energy/cost with independent formula cross-checks.")
def cross_oem_report(context: AssetExecutionContext, settings: SettingsResource, executor: ExecutorResource) -> Dict[str, Any]:
    cfg = settings.get()
    exec_ = executor.get(cfg)

    source = json.loads(exec_.run(f"cat {cfg.remote_root}/amd_run_{cfg.run_id}/run_summary.json", host=cfg.source_host).stdout)
    target = json.loads(exec_.run(f"cat {cfg.remote_root}/nvidia_run_{cfg.run_id}/run_summary.json", host=cfg.target_host).stdout)
    proof = json.loads(exec_.run(f"cat {cfg.remote_root}/nvidia_run_{cfg.run_id}/resume_validation_proof.json", host=cfg.target_host).stdout)

    migration = {
        "checkpoint_save_seconds": source.get("checkpoint_save_seconds"),
        "checkpoint_load_seconds": target.get("checkpoint_load_seconds"),
        "resume_to_first_step_seconds": target.get("resume_to_first_step_seconds"),
    }
    migration["migration_to_first_step_seconds"] = migration_to_first_step_seconds(migration)
    total_training = (as_float(source.get("training_runtime_seconds")) or 0) + (as_float(target.get("training_runtime_seconds")) or 0)
    migration["migration_overhead_percent"] = (
        None if total_training <= 0 or migration["migration_to_first_step_seconds"] is None
        else migration["migration_to_first_step_seconds"] / total_training * 100.0
    )

    end_to_end = end_to_end_summary(source, target, migration)
    validation = formula_validation(source, target, migration, end_to_end)

    report = {
        "run_id": cfg.run_id, "source": source, "target": target,
        "migration": migration, "end_to_end": end_to_end,
        "formula_validation": validation, "resume_validation_proof": proof,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    context.add_output_metadata({
        "formula_checks_passed": validation["passed"],
        "wall_clock_tokens_per_second": end_to_end.get("wall_clock_tokens_per_second"),
        "gpu_energy_kwh": end_to_end.get("gpu_energy_kwh"),
        "preview": MetadataValue.json({"backend": end_to_end.get("backend"), "steps_executed": end_to_end.get("steps_executed")}),
    })
    if not validation["passed"]:
        failing = [c["name"] for c in validation["checks"] if not c["passed"]]
        raise RuntimeError(f"Cross-OEM report failed formula cross-checks: {failing}")
    return report


@asset(group_name="reporting", deps=[cross_oem_report], description="Queryable SQLite table of run metrics across all runs (see data/metrics_db.py).")
def run_metrics_db(context: AssetExecutionContext, settings: SettingsResource, data: DataResource, cross_oem_report: Dict[str, Any]) -> Dict[str, Any]:
    """Takes the in-memory `cross_oem_report` asset (Dagster passes it in
    directly -- no need to re-fetch it from a remote host) and records one
    row per run into the SQLite metrics database. This is what makes the
    "database" asset in the UI actually reflect a *history* across runs,
    not just the latest one."""
    cfg = settings.get()
    db = data.metrics_db(cfg)

    end_to_end = cross_oem_report.get("end_to_end", {})
    row = {
        "source_host": cfg.source_host,
        "target_host": cfg.target_host,
        "status": "complete" if cross_oem_report.get("formula_validation", {}).get("passed") else "formula_check_failed",
        "checkpoint_step": cfg.checkpoint_step,
        "final_step": cfg.final_step,
        "final_loss": cross_oem_report.get("target", {}).get("final_loss"),
        "tokens_per_second": end_to_end.get("wall_clock_tokens_per_second"),
        "gpu_energy_kwh": end_to_end.get("gpu_energy_kwh"),
        "energy_cost_gbp": (cross_oem_report.get("source", {}).get("energy_cost_gbp") or 0)
        + (cross_oem_report.get("target", {}).get("energy_cost_gbp") or 0),
    }
    recorded_at = cross_oem_report.get("generated_at") or datetime.now(timezone.utc).isoformat()
    db.record(cfg.run_id, row, recorded_at=recorded_at)

    context.add_output_metadata({
        "row_count": db.row_count(),
        "db_path": MetadataValue.path(str(db._db_path)),
        "latest_row": MetadataValue.json(row),
    })
    return {"row_count": db.row_count()}
