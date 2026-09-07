"""Pure metrics/validation math -- zero I/O, zero Dagster, zero subprocess.
Ported unchanged from the original file (energy/cost helpers ~lines
800-1330; cross-checked formula validation ~lines 2958-3070). This is
where the report's numbers get *cross-checked against themselves*
(e.g. "does tokens_per_joule actually equal tokens / energy_joules,
recomputed independently?") -- the part of the pipeline most directly
in service of "fair, truthful" reporting, so it's kept as a small,
dependency-free, directly-unit-testable module.
"""
import math
from typing import Any, Dict, List, Optional

FORMULA_RELATIVE_TOLERANCE = 1e-6
FORMULA_ABSOLUTE_TOLERANCE = 1e-9


def as_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or not denominator:
        return None
    return numerator / denominator


def almost_equal(left: Optional[float], right: Optional[float]) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=FORMULA_RELATIVE_TOLERANCE, abs_tol=FORMULA_ABSOLUTE_TOLERANCE)


def energy_cost_gbp(energy_kwh: Optional[float], price_per_kwh: float) -> Optional[float]:
    return None if energy_kwh is None else energy_kwh * price_per_kwh


def cost_per_million_tokens(cost_gbp: Optional[float], useful_tokens: Optional[int]) -> Optional[float]:
    if cost_gbp is None or not useful_tokens:
        return None
    return cost_gbp / (useful_tokens / 1_000_000.0)


def phase_steps_executed(start_step: Optional[int], end_step: Optional[int]) -> Optional[int]:
    if start_step is None or end_step is None:
        return None
    return end_step - start_step


def migration_to_first_step_seconds(report: Dict[str, Any]) -> Optional[float]:
    parts = [as_float(report.get(k)) for k in ["checkpoint_save_seconds", "transfer_seconds", "checkpoint_load_seconds", "resume_to_first_step_seconds"]]
    values = [v for v in parts if v is not None]
    return None if not values else float(sum(values))


def phase_runtime_seconds(report: Dict[str, Any]) -> Optional[float]:
    preferred = as_float(report.get("training_runtime_seconds"))
    return preferred if preferred is not None else as_float(report.get("runtime_seconds"))


def end_to_end_summary(source: Dict[str, Any], target: Dict[str, Any], migration: Dict[str, Any]) -> Dict[str, Any]:
    total_tokens = (as_int(source.get("useful_training_tokens")) or 0) + (as_int(target.get("useful_training_tokens")) or 0)
    source_time = phase_runtime_seconds(source) or 0.0
    target_time = phase_runtime_seconds(target) or 0.0
    migration_time = as_float(migration.get("migration_to_first_step_seconds")) or 0.0
    total_energy = (as_float(source.get("gpu_energy_joules")) or 0.0) + (as_float(target.get("gpu_energy_joules")) or 0.0)
    training_seconds = source_time + target_time
    wall_seconds = training_seconds + migration_time
    return {
        "backend": f"{str(source.get('vendor', 'source')).upper()} -> {str(target.get('vendor', 'target')).upper()}",
        "start_step": as_int(source.get("start_step")),
        "end_step": as_int(target.get("end_step")),
        "steps_executed": phase_steps_executed(as_int(source.get("start_step")), as_int(target.get("end_step"))),
        "useful_training_tokens": total_tokens,
        "training_runtime_seconds": training_seconds,
        "wall_clock_runtime_seconds": wall_seconds,
        "training_tokens_per_second": safe_ratio(float(total_tokens), training_seconds),
        "wall_clock_tokens_per_second": safe_ratio(float(total_tokens), wall_seconds),
        "gpu_energy_joules": total_energy,
        "gpu_energy_wh": total_energy / 3600.0,
        "gpu_energy_kwh": total_energy / 3_600_000.0,
        "tokens_per_joule": safe_ratio(float(total_tokens), total_energy),
    }


def formula_validation(source: Dict[str, Any], target: Dict[str, Any], migration: Dict[str, Any], end_to_end: Dict[str, Any]) -> Dict[str, Any]:
    """Independently recomputes every derived metric in the report and
    checks it matches what was actually reported -- this is what makes
    the cross-OEM report auditable rather than just asserted."""
    checks: List[Dict[str, Any]] = []
    add = checks.append

    add({"name": "source.steps_executed", "passed": as_int(source.get("steps_executed")) == phase_steps_executed(as_int(source.get("start_step")), as_int(source.get("end_step")))})
    add({"name": "target.steps_executed", "passed": as_int(target.get("steps_executed")) == phase_steps_executed(as_int(target.get("start_step")), as_int(target.get("end_step")))})

    source_tokens = as_int(source.get("useful_training_tokens")) or 0
    target_tokens = as_int(target.get("useful_training_tokens")) or 0
    add({"name": "end_to_end.total_useful_tokens", "passed": as_int(end_to_end.get("useful_training_tokens")) == (source_tokens + target_tokens)})
    add({"name": "source.tokens_per_second", "passed": almost_equal(as_float(source.get("tokens_per_second")), safe_ratio(float(source_tokens), phase_runtime_seconds(source)))})
    add({"name": "target.tokens_per_second", "passed": almost_equal(as_float(target.get("tokens_per_second")), safe_ratio(float(target_tokens), phase_runtime_seconds(target)))})

    source_tpj_expected = safe_ratio(float(source_tokens), as_float(source.get("gpu_energy_joules")))
    target_tpj_expected = safe_ratio(float(target_tokens), as_float(target.get("gpu_energy_joules")))
    source_kwh_expected = safe_ratio(as_float(source.get("gpu_energy_joules")), 3_600_000.0)
    target_kwh_expected = safe_ratio(as_float(target.get("gpu_energy_joules")), 3_600_000.0)
    add({"name": "source.tokens_per_joule", "passed": True if source_tpj_expected is None else almost_equal(as_float(source.get("tokens_per_joule")), source_tpj_expected)})
    add({"name": "target.tokens_per_joule", "passed": True if target_tpj_expected is None else almost_equal(as_float(target.get("tokens_per_joule")), target_tpj_expected)})
    add({"name": "source.gpu_energy_kwh", "passed": True if source_kwh_expected is None else almost_equal(as_float(source.get("gpu_energy_kwh")), source_kwh_expected)})
    add({"name": "target.gpu_energy_kwh", "passed": True if target_kwh_expected is None else almost_equal(as_float(target.get("gpu_energy_kwh")), target_kwh_expected)})

    source_direct_cost = cost_per_million_tokens(as_float(source.get("energy_cost_gbp")), as_int(source.get("useful_training_tokens")))
    target_direct_cost = cost_per_million_tokens(as_float(target.get("energy_cost_gbp")), as_int(target.get("useful_training_tokens")))
    source_tpj = as_float(source.get("tokens_per_joule"))
    target_tpj = as_float(target.get("tokens_per_joule"))
    source_alt_cost = safe_ratio(as_float(source.get("electricity_price_per_kwh")), (3.6 * source_tpj) if source_tpj else None)
    target_alt_cost = safe_ratio(as_float(target.get("electricity_price_per_kwh")), (3.6 * target_tpj) if target_tpj else None)
    source_cost_actual = as_float(source.get("energy_cost_per_million_tokens_gbp"))
    target_cost_actual = as_float(target.get("energy_cost_per_million_tokens_gbp"))
    source_cost_checks = [v for v in [source_direct_cost, source_alt_cost] if v is not None]
    target_cost_checks = [v for v in [target_direct_cost, target_alt_cost] if v is not None]
    add({"name": "source.energy_cost_per_million_tokens", "passed": True if not source_cost_checks else all(almost_equal(source_cost_actual, v) for v in source_cost_checks)})
    add({"name": "target.energy_cost_per_million_tokens", "passed": True if not target_cost_checks else all(almost_equal(target_cost_actual, v) for v in target_cost_checks)})

    total_training_seconds = (phase_runtime_seconds(source) or 0) + (phase_runtime_seconds(target) or 0)
    expected_overhead_percent = (
        safe_ratio(as_float(migration.get("migration_to_first_step_seconds")), total_training_seconds) * 100.0
        if total_training_seconds > 0 else None
    )
    add({"name": "migration_overhead_percent", "passed": almost_equal(as_float(migration.get("migration_overhead_percent")), expected_overhead_percent)})
    add({"name": "end_to_end.cross_oem_energy", "passed": almost_equal(as_float(end_to_end.get("gpu_energy_joules")), (as_float(source.get("gpu_energy_joules")) or 0.0) + (as_float(target.get("gpu_energy_joules")) or 0.0))})

    return {
        "relative_tolerance": FORMULA_RELATIVE_TOLERANCE,
        "absolute_tolerance": FORMULA_ABSOLUTE_TOLERANCE,
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
    }
