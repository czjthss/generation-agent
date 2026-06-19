from __future__ import annotations

from typing import Any

import numpy as np

from .features import AnomalyConfig, add_anomalies
from .planner import SeriesPlan


def _anomaly_config(plan: SeriesPlan) -> AnomalyConfig:
    return AnomalyConfig(
        enabled=plan.anomaly_enabled,
        count=plan.anomaly_count,
        magnitude=plan.anomaly_magnitude,
        width=plan.anomaly_width,
        kind=plan.anomaly_kind,
        severity=plan.anomaly_severity,
        direction="both",
    )


def _standardize(values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    if std < 1e-9:
        return np.zeros_like(values, dtype=float)
    return (values - float(np.mean(values))) / std


def _inject_if_target(
    values: np.ndarray,
    plan: SeriesPlan,
    rng: np.random.Generator,
    targets: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    if plan.anomaly_target not in targets:
        return values.copy(), np.zeros(len(values), dtype=int)
    return add_anomalies(values, rng, _anomaly_config(plan))


def apply_semantic_process(
    base_values: np.ndarray,
    plan: SeriesPlan,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    semantic_type = plan.semantic_type
    params = plan.semantic_config
    length = len(base_values)
    columns: dict[str, np.ndarray] = {}

    if semantic_type == "cumulative":
        allow_negative = bool(params.get("allow_negative_increment", False))
        increments = base_values.copy()
        if not allow_negative:
            increments = np.maximum(increments, 0.0)
        increments, flags = _inject_if_target(increments, plan, rng, {"increment", "flow"})
        if not allow_negative:
            increments = np.maximum(increments, 0.0)
        initial = float(params.get("initial_value", 0.0))
        values = initial + np.cumsum(increments)
        columns["increment"] = increments
        return values, flags, columns

    if semantic_type == "stock_flow":
        scale = max(float(np.mean(np.abs(base_values))) * 0.08, 1.0)
        inflow = rng.gamma(2.0, scale / 2.0, size=length) * float(params.get("inflow_scale", 1.0))
        outflow = rng.gamma(2.0, scale / 2.0, size=length) * float(params.get("outflow_scale", 0.85))
        flags = np.zeros(length, dtype=int)
        if plan.anomaly_target in {"flow", "inflow"}:
            inflow, flags = add_anomalies(inflow, rng, _anomaly_config(plan))
            inflow = np.maximum(inflow, 0.0)
        elif plan.anomaly_target == "outflow":
            outflow, flags = add_anomalies(outflow, rng, _anomaly_config(plan))
            outflow = np.maximum(outflow, 0.0)
        initial = float(params.get("initial_value", max(plan.baseline, 0.0)))
        values = np.empty(length, dtype=float)
        previous = initial
        for i in range(length):
            previous = previous + inflow[i] - outflow[i]
            if plan.output_constraints.get("nonnegative", True):
                previous = max(previous, 0.0)
            values[i] = previous
        columns.update({"inflow": inflow, "outflow": outflow, "net_flow": inflow - outflow})
        return values, flags, columns

    if semantic_type == "regime_switching":
        states = np.asarray(params.get("states", [0.4, 1.0, 1.5]), dtype=float)
        if states.size == 0:
            states = np.asarray([1.0])
        probability = float(params.get("transition_probability", 0.05))
        state_index = np.zeros(length, dtype=int)
        current = int(params.get("initial_state", min(1, len(states) - 1)))
        for i in range(length):
            if i and rng.random() < probability:
                choices = [j for j in range(len(states)) if j != current]
                if choices:
                    current = int(rng.choice(choices))
            state_index[i] = current
        flags = np.zeros(length, dtype=int)
        if plan.anomaly_enabled and plan.anomaly_target == "state" and length:
            positions = rng.choice(length, size=min(plan.anomaly_count, length), replace=False)
            fault_state = int(np.argmin(states))
            for pos in positions:
                end = min(length, pos + max(1, plan.anomaly_width))
                state_index[pos:end] = fault_state
                flags[pos:end] = 1
        values = base_values * states[state_index]
        columns["state"] = state_index.astype(float)
        return values, flags, columns

    if semantic_type == "random_walk":
        volatility = float(params.get("volatility", max(plan.noise_sigma, 1.0)))
        drift = float(params.get("drift", plan.trend_slope / max(length, 1)))
        steps = drift + volatility * _standardize(base_values)
        steps, flags = _inject_if_target(steps, plan, rng, {"step", "increment"})
        initial = float(params.get("initial_value", max(plan.baseline, 1.0)))
        values = initial + np.cumsum(steps)
        columns["step"] = steps
        return values, flags, columns

    if semantic_type == "decay_recovery":
        equilibrium = float(params.get("equilibrium", plan.baseline))
        rate = min(max(float(params.get("recovery_rate", 0.12)), 0.0), 1.0)
        impulses = np.zeros(length, dtype=float)
        impulses[0] = float(params.get("impulse", plan.daily_amplitude or plan.baseline))
        impulses, flags = _inject_if_target(impulses, plan, rng, {"impulse", "value"})
        values = np.empty(length, dtype=float)
        previous = equilibrium + impulses[0]
        for i in range(length):
            if i:
                previous = previous + rate * (equilibrium - previous) + impulses[i]
            values[i] = previous
        columns["impulse"] = impulses
        return values, flags, columns

    if semantic_type == "saturation_growth":
        capacity = max(float(params.get("capacity", max(plan.baseline * 10.0, 100.0))), 1e-6)
        rate = max(float(params.get("growth_rate", 0.06)), 0.0)
        initial = min(max(float(params.get("initial_value", max(plan.baseline, 1.0))), 1e-6), capacity)
        growth_rate = np.full(length, rate, dtype=float)
        growth_rate, flags = _inject_if_target(growth_rate, plan, rng, {"growth_rate"})
        growth_rate = np.maximum(growth_rate, 0.0)
        values = np.empty(length, dtype=float)
        previous = initial
        for i in range(length):
            previous += growth_rate[i] * previous * (1.0 - previous / capacity)
            previous = min(max(previous, 0.0), capacity)
            values[i] = previous
        columns["growth_rate"] = growth_rate
        return values, flags, columns

    if semantic_type == "multivariate_lag":
        driver = base_values.copy()
        driver, flags = _inject_if_target(driver, plan, rng, {"driver"})
        lag = max(0, int(params.get("lag", 2)))
        coefficient = float(params.get("coefficient", 0.7))
        lagged = np.roll(driver, lag)
        if lag:
            lagged[:lag] = driver[0]
        residual_scale = max(float(params.get("residual_sigma", plan.noise_sigma)), 0.0)
        values = plan.baseline + coefficient * lagged + rng.normal(0.0, residual_scale, size=length)
        columns[str(params.get("driver_name", "driver"))] = driver
        columns["lagged_driver"] = lagged
        return values, flags, columns

    values, flags = _inject_if_target(base_values, plan, rng, {"value"})
    return values, flags, columns
