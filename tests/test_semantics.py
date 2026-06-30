from __future__ import annotations

import numpy as np

from generation_agent.planner import SeriesPlan
from generation_agent.semantic_types import AnomalyOverrides
from test_helpers import local_generate_from_plan
from generation_agent.semantic_validators import validate_and_repair
from generation_agent.semantic_transforms import apply_semantic_process
from generation_agent.context_compactor import compact_validation_for_metadata


def _generate(semantic_type: str, semantic_config: dict, constraints: dict):
    plan = SeriesPlan(
        domain="test",
        generator_type="cyclic_signal",
        baseline=10.0,
        daily_amplitude=2.0,
        noise_sigma=0.5,
        semantic_type=semantic_type,
        semantic_config=semantic_config,
        output_constraints=constraints,
    )
    return local_generate_from_plan(plan, length=96, seed=7)


def test_cumulative_is_monotonic_and_matches_increments():
    frame = _generate(
        "cumulative",
        {"initial_value": 100.0, "allow_negative_increment": False},
        {"nonnegative": True, "monotonic": "nondecreasing"},
    )
    assert np.all(frame["value"].diff().dropna() >= 0)
    assert np.isclose(frame["value"].iloc[-1], 100.0 + frame["increment"].sum(), atol=1e-3)
    assert frame.attrs["validation_report"]["passed"]


def test_stock_flow_conservation():
    frame = _generate(
        "stock_flow",
        {"initial_value": 500.0, "inflow_scale": 1.0, "outflow_scale": 0.8},
        {"nonnegative": True, "conservation": True},
    )
    assert {"inflow", "outflow", "net_flow"}.issubset(frame.columns)
    assert frame.attrs["validation_report"]["checks"]["stock_flow_balance"]


def test_stock_flow_validator_preserves_valid_gross_flows():
    plan = SeriesPlan(
        semantic_type="stock_flow",
        semantic_config={"initial_value": 10.0},
        output_constraints={"nonnegative": True, "conservation": True},
    )
    columns = {
        "inflow": np.asarray([5.0, 5.0, 1.0]),
        "outflow": np.asarray([3.0, 3.0, 2.0]),
    }
    values, report = validate_and_repair(
        np.asarray([12.0, 14.0, 13.0]),
        plan,
        columns,
    )
    assert np.allclose(values, [12.0, 14.0, 13.0])
    assert np.any((columns["inflow"] > 0) & (columns["outflow"] > 0))
    assert "recomputed_stock_flow_columns" not in report["repairs_applied"]
    assert report["checks"]["stock_flow_balance"]
    assert np.allclose(columns["net_flow"], columns["inflow"] - columns["outflow"])


def test_random_walk_uses_steps():
    frame = _generate(
        "random_walk",
        {"initial_value": 100.0, "drift": 0.1, "volatility": 1.0},
        {"nonnegative": True},
    )
    assert "step" in frame
    assert np.isclose(frame["value"].iloc[-1], 100.0 + frame["step"].sum(), atol=1e-3)


def test_random_walk_steps_do_not_inherit_periodic_base_curve():
    plan = SeriesPlan(
        semantic_type="random_walk",
        semantic_config={"initial_value": 100.0, "drift": 0.0, "volatility": 1.0},
        output_constraints={"nonnegative": True},
    )
    base = np.tile(np.sin(np.linspace(0, 2 * np.pi, 24, endpoint=False)), 4)
    _values, _flags, columns = apply_semantic_process(base, plan, np.random.default_rng(7))
    steps = columns["step"]
    corr = np.corrcoef(steps[:-24], steps[24:])[0, 1]
    assert abs(corr) < 0.5


def test_decay_recovery_moves_toward_equilibrium():
    frame = _generate(
        "decay_recovery",
        {"equilibrium": 5.0, "impulse": 20.0, "recovery_rate": 0.2},
        {"nonnegative": True},
    )
    assert abs(frame["value"].iloc[-1] - 5.0) < abs(frame["value"].iloc[0] - 5.0)


def test_saturation_growth_is_bounded_and_monotonic():
    frame = _generate(
        "saturation_growth",
        {"initial_value": 10.0, "capacity": 100.0, "growth_rate": 0.08},
        {"nonnegative": True, "upper_bound": 100.0, "monotonic": "nondecreasing"},
    )
    assert np.all(frame["value"].diff().dropna() >= 0)
    assert frame["value"].max() <= 100.0


def test_regime_switching_outputs_state():
    frame = _generate(
        "regime_switching",
        {"states": [0.0, 0.5, 1.0], "transition_probability": 0.3},
        {"nonnegative": True},
    )
    assert "state" in frame
    assert frame["state"].nunique() > 1


def test_multivariate_lag_outputs_driver():
    frame = _generate(
        "multivariate_lag",
        {"driver_name": "temperature", "lag": 3, "coefficient": 0.7, "residual_sigma": 0.1},
        {"nonnegative": True},
    )
    assert {"temperature", "lagged_driver"}.issubset(frame.columns)
    assert np.allclose(frame["lagged_driver"].iloc[3:].to_numpy(), frame["temperature"].iloc[:-3].to_numpy())


def test_explicit_multivariate_plan_skips_legacy_multivariate_semantic_columns():
    plan = SeriesPlan(
        domain="power",
        generator_type="cyclic_signal",
        baseline=100.0,
        daily_amplitude=10.0,
        noise_sigma=0.1,
        semantic_type="multivariate_lag",
        semantic_config={"driver_name": "legacy_driver", "lag": 3},
        variables=[
            {"name": "load", "role": "target", "unit": "kW"},
            {"name": "temperature", "role": "driver", "unit": "degC", "generator_type": "smooth_environmental"},
        ],
        relationships=[
            {
                "source": "temperature",
                "target": "load",
                "effect": "positive_lagged_effect",
                "operator": "linear_lag",
                "lag": 2,
                "coefficient": 5.0,
            }
        ],
    )
    frame = local_generate_from_plan(plan, length=72, seed=9)
    assert "temperature" in frame.columns
    assert "legacy_driver" not in frame.columns
    assert "lagged_driver" not in frame.columns
    assert frame.attrs["validation_report"]["checks"]["multivariate_relationships"]


def test_anomaly_override_can_force_on_and_off():
    plan = SeriesPlan(
        semantic_type="cumulative",
        semantic_config={"initial_value": 0.0},
        output_constraints={"monotonic": "nondecreasing"},
        anomaly_enabled=False,
        anomaly_target="increment",
    )
    enabled = local_generate_from_plan(
        plan,
        length=48,
        seed=3,
        anomaly_overrides=AnomalyOverrides(enabled=True, severity="high"),
    )
    disabled = local_generate_from_plan(
        plan,
        length=48,
        seed=3,
        anomaly_overrides=AnomalyOverrides(enabled=False),
    )
    assert enabled["anomaly"].sum() > 0
    assert disabled["anomaly"].sum() == 0
    assert np.all(np.diff(enabled["value"].to_numpy()) >= 0.0)


def test_cumulative_repair_keeps_increment_identity_after_clipping():
    plan = SeriesPlan(
        semantic_type="cumulative",
        semantic_config={"initial_value": 0.0, "allow_negative_increment": False},
        output_constraints={"upper_bound": 12.0, "monotonic": "nondecreasing"},
    )
    columns = {"increment": np.asarray([5.0, 5.0, 5.0, 5.0])}
    values, report = validate_and_repair(
        np.asarray([5.0, 10.0, 15.0, 20.0]),
        plan,
        columns,
    )
    assert np.allclose(values, 0.0 + np.cumsum(columns["increment"]))
    assert report["checks"]["cumulative_identity"]
    assert report["checks"]["upper_bound"]


def test_validator_reports_raw_and_repaired_status_separately():
    plan = SeriesPlan(output_constraints={"upper_bound": 10.0})
    values, report = validate_and_repair(np.asarray([5.0, 12.0]), plan, {})
    assert report["raw_passed"] is False
    assert report["repaired_passed"] is True
    assert report["passed"] is True
    assert "clipped_to_upper_bound" in report["repairs_applied"]
    assert report["repair_warnings"][0]["type"] == "post_generation_repair"
    assert np.all(values <= 10.0)
    compact = compact_validation_for_metadata(report)
    assert compact["raw_passed"] is False
    assert compact["repaired_passed"] is True
    assert compact["repairs_applied"] == ["clipped_to_upper_bound"]
    assert compact["repair_warnings"][0]["severity"] == "soft_warning"


def test_stock_flow_repair_keeps_flow_conservation_after_clipping():
    plan = SeriesPlan(
        semantic_type="stock_flow",
        semantic_config={"initial_value": 10.0},
        output_constraints={"upper_bound": 12.0, "conservation": True, "nonnegative": True},
    )
    columns = {
        "inflow": np.asarray([5.0, 5.0, 5.0]),
        "outflow": np.asarray([0.0, 0.0, 0.0]),
    }
    values, report = validate_and_repair(
        np.asarray([15.0, 20.0, 25.0]),
        plan,
        columns,
    )
    expected = 10.0 + np.cumsum(columns["inflow"] - columns["outflow"])
    assert np.allclose(values, expected)
    assert report["checks"]["stock_flow_balance"]
    assert report["checks"]["upper_bound"]
    assert report["passed"] is False
    assert report["repaired_passed"] is True
    assert report["critical_repairs"] == ["recomputed_stock_flow_columns"]
    compact = compact_validation_for_metadata(report)
    assert compact["critical_repairs"] == ["recomputed_stock_flow_columns"]
