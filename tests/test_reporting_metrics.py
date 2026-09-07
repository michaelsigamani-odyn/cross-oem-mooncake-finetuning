from cross_oem_migration.reporting.metrics import (
    almost_equal,
    cost_per_million_tokens,
    end_to_end_summary,
    formula_validation,
    safe_ratio,
)


def test_safe_ratio_handles_zero_denominator():
    assert safe_ratio(10.0, 0) is None
    assert safe_ratio(10.0, None) is None
    assert safe_ratio(10.0, 5.0) == 2.0


def test_cost_per_million_tokens():
    assert cost_per_million_tokens(1.0, 1_000_000) == 1.0
    assert cost_per_million_tokens(1.0, 0) is None
    assert cost_per_million_tokens(None, 100) is None


def test_end_to_end_summary_sums_tokens_and_energy():
    source = {"useful_training_tokens": 100, "training_runtime_seconds": 10.0, "gpu_energy_joules": 1000.0, "start_step": 0, "vendor": "amd"}
    target = {"useful_training_tokens": 200, "training_runtime_seconds": 20.0, "gpu_energy_joules": 2000.0, "end_step": 30, "vendor": "nvidia"}
    result = end_to_end_summary(source, target, {"migration_to_first_step_seconds": 5.0})
    assert result["useful_training_tokens"] == 300
    assert result["training_runtime_seconds"] == 30.0
    assert result["wall_clock_runtime_seconds"] == 35.0
    assert result["gpu_energy_joules"] == 3000.0
    assert result["backend"] == "AMD -> NVIDIA"


def test_formula_validation_passes_for_internally_consistent_report():
    source = {
        "start_step": 0, "end_step": 10, "steps_executed": 10, "useful_training_tokens": 1000,
        "training_runtime_seconds": 10.0, "tokens_per_second": 100.0, "gpu_energy_joules": 3600.0,
        "gpu_energy_kwh": 0.001, "tokens_per_joule": 1000 / 3600.0, "electricity_price_per_kwh": 0.27,
        "energy_cost_gbp": 0.00027, "energy_cost_per_million_tokens_gbp": (0.00027 / (1000 / 1_000_000.0)),
    }
    target = dict(source, start_step=10, end_step=20)
    migration = {"migration_to_first_step_seconds": 2.0, "migration_overhead_percent": 2.0 / 20.0 * 100.0}
    end_to_end = end_to_end_summary(source, target, migration)
    result = formula_validation(source, target, migration, end_to_end)
    failing = [c["name"] for c in result["checks"] if not c["passed"]]
    assert result["passed"], f"unexpected failing checks: {failing}"


def test_almost_equal_treats_none_as_only_equal_to_none():
    assert almost_equal(None, None) is True
    assert almost_equal(1.0, None) is False
    assert almost_equal(None, 1.0) is False


def test_formula_validation_uses_runtime_seconds_when_training_runtime_missing():
    source = {
        "start_step": 0, "end_step": 10, "steps_executed": 10, "useful_training_tokens": 1000,
        "training_runtime_seconds": None, "runtime_seconds": 10.0, "tokens_per_second": 100.0,
        "gpu_energy_joules": 3600.0, "gpu_energy_kwh": 0.001, "tokens_per_joule": 1000 / 3600.0,
        "electricity_price_per_kwh": 0.27, "energy_cost_gbp": 0.00027,
        "energy_cost_per_million_tokens_gbp": (0.00027 / (1000 / 1_000_000.0)),
    }
    target = dict(source, start_step=10, end_step=20)
    migration = {"migration_to_first_step_seconds": 2.0, "migration_overhead_percent": 2.0 / 20.0 * 100.0}
    end_to_end = end_to_end_summary(source, target, migration)
    result = formula_validation(source, target, migration, end_to_end)
    assert result["passed"]
