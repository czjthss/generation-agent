from __future__ import annotations

import numpy as np

from generation_agent.planner import SeriesPlan
from generation_agent.semantic_types import AnomalyOverrides
from generation_agent.agent import GenerationAgent


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
    return GenerationAgent(model=None).generate_from_plan(plan, length=96, seed=7)


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


def test_random_walk_uses_steps():
    frame = _generate(
        "random_walk",
        {"initial_value": 100.0, "drift": 0.1, "volatility": 1.0},
        {"nonnegative": True},
    )
    assert "step" in frame
    assert np.isclose(frame["value"].iloc[-1], 100.0 + frame["step"].sum(), atol=1e-3)


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


def test_anomaly_override_can_force_on_and_off():
    plan = SeriesPlan(
        semantic_type="cumulative",
        semantic_config={"initial_value": 0.0},
        output_constraints={"monotonic": "nondecreasing"},
        anomaly_enabled=False,
        anomaly_target="increment",
    )
    agent = GenerationAgent(model=None)
    enabled = agent.generate_from_plan(
        plan,
        length=48,
        seed=3,
        anomaly_overrides=AnomalyOverrides(enabled=True, severity="high"),
    )
    disabled = agent.generate_from_plan(
        plan,
        length=48,
        seed=3,
        anomaly_overrides=AnomalyOverrides(enabled=False),
    )
    assert enabled["anomaly"].sum() > 0
    assert disabled["anomaly"].sum() == 0
    assert np.all(np.diff(enabled["value"].to_numpy()) >= 0.0)
