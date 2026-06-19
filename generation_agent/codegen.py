from __future__ import annotations

import json
import textwrap

from .planner import SeriesPlan


def render_generator_code(
    plan: SeriesPlan,
    length: int = 168,
    freq: str = "h",
    start: str = "2026-07-01 00:00:00",
    seed: int | None = 42,
) -> str:
    plan_json = json.dumps(plan.to_dict(), ensure_ascii=False, indent=2)
    template = '''\
"""Standalone generator script produced by generation_agent."""

import json

import numpy as np
import pandas as pd


PLAN = json.loads(r"""__PLAN_JSON__""")


def linear_trend(length, slope):
    return slope * np.linspace(0.0, 1.0, length)


def hour_of_day(length):
    return np.arange(length, dtype=float) % 24.0


def daily_load_cycle(length, amplitude=1.0):
    hour = hour_of_day(length)
    morning = np.exp(-0.5 * ((hour - 9.0) / 2.2) ** 2)
    afternoon = 1.15 * np.exp(-0.5 * ((hour - 15.0) / 3.0) ** 2)
    evening = 0.55 * np.exp(-0.5 * ((hour - 20.0) / 2.4) ** 2)
    night_dip = -0.45 * np.exp(-0.5 * ((hour - 3.0) / 2.6) ** 2)
    shape = morning + afternoon + evening + night_dip
    return amplitude * (shape - shape.mean()) / (shape.std() + 1e-9)


def working_day_gate(length, weekend_factor=0.72):
    day = (np.arange(length) // 24) % 7
    gate = np.ones(length)
    gate[(day == 5) | (day == 6)] = weekend_factor
    return gate


def heat_index_effect(length, amplitude=1.0):
    hour = hour_of_day(length)
    shape = np.exp(-0.5 * ((hour - 14.0) / 4.2) ** 2)
    shape += 0.35 * np.exp(-0.5 * ((hour - 22.0) / 4.5) ** 2)
    return amplitude * (shape - shape.mean()) / (shape.std() + 1e-9)


def generate_precipitation(length, rng):
    params = PLAN.get("domain_params", {})
    values = np.zeros(length, dtype=float)
    anomaly = np.zeros(length, dtype=int)
    event_probability = float(params.get("event_probability", 0.18))
    mean_duration = max(1.0, float(params.get("mean_duration", 5.0)))
    shape = max(0.3, float(params.get("intensity_shape", 1.4)))
    scale = max(0.1, float(params.get("intensity_scale", 5.0)))
    dry_spell_bias = float(params.get("dry_spell_bias", 0.68))
    storm_probability = float(params.get("storm_probability", 0.08))
    storm_multiplier = max(1.0, float(params.get("storm_multiplier", 3.0)))
    cursor = 0
    while cursor < length:
        if rng.random() < dry_spell_bias:
            cursor += int(rng.integers(3, 18))
            continue
        if rng.random() > event_probability:
            cursor += 1
            continue
        duration = int(max(1, rng.geometric(1.0 / mean_duration)))
        end = min(length, cursor + duration)
        is_storm = rng.random() < storm_probability
        event_scale = scale * (storm_multiplier if is_storm else 1.0)
        raw = rng.gamma(shape=shape, scale=event_scale, size=end - cursor)
        envelope = np.sin(np.linspace(0.15, np.pi - 0.15, end - cursor))
        values[cursor:end] += raw * np.maximum(envelope, 0.15)
        if is_storm:
            anomaly[cursor:end] = 1
        cursor = end + int(rng.integers(1, 10))
    drizzle = (values > 0) & (rng.random(length) < 0.18)
    values[drizzle] *= rng.uniform(0.15, 0.55, size=drizzle.sum())
    return values, anomaly


def generate_solar(length, rng):
    params = PLAN.get("domain_params", {})
    hour = hour_of_day(length)
    sunrise = float(params.get("sunrise_hour", 6.0))
    sunset = float(params.get("sunset_hour", 19.0))
    daylight = (hour >= sunrise) & (hour <= sunset)
    phase = (hour - sunrise) / max(1.0, sunset - sunrise)
    envelope = np.where(daylight, np.sin(np.pi * np.clip(phase, 0.0, 1.0)) ** 1.7, 0.0)
    day_index = np.arange(length) // 24
    day_factor = rng.uniform(0.82, 1.08, size=int(day_index.max()) + 1 if length else 1)
    values = PLAN["daily_amplitude"] * envelope * day_factor[day_index]
    cursor = 0
    while cursor < length:
        if daylight[cursor] and rng.random() < float(params.get("cloud_probability", 0.18)):
            end = min(length, cursor + int(rng.integers(1, 5)))
            drop = rng.uniform(float(params.get("cloud_drop_min", 0.35)), float(params.get("cloud_drop_max", 0.8)))
            values[cursor:end] *= drop
        cursor += 1
    values += rng.normal(0.0, PLAN["noise_sigma"], size=length) * daylight
    return np.maximum(values, 0.0), np.zeros(length, dtype=int)


def generate_temperature(length, rng):
    params = PLAN.get("domain_params", {})
    hour = hour_of_day(length)
    peak_hour = float(params.get("peak_hour", 15.0))
    target = PLAN["baseline"] + PLAN["daily_amplitude"] * np.sin(2 * np.pi * (hour - peak_hour + 6) / 24)
    target += linear_trend(length, PLAN["trend_slope"])
    target += rng.normal(0.0, PLAN["noise_sigma"], size=length)
    values = target.copy()
    inertia = float(params.get("inertia", 0.88))
    for i in range(1, length):
        values[i] = inertia * values[i - 1] + (1.0 - inertia) * target[i]
    return values, np.zeros(length, dtype=int)


def generate_count_process(length, rng):
    params = PLAN.get("domain_params", {})
    hour = hour_of_day(length)
    morning = np.exp(-0.5 * ((hour - float(params.get("morning_peak", 8.0))) / 2.0) ** 2)
    evening = np.exp(-0.5 * ((hour - float(params.get("evening_peak", 18.0))) / 2.8) ** 2)
    rate = PLAN["baseline"] + PLAN["daily_amplitude"] * (0.8 * morning + evening)
    if PLAN["weekly_enabled"]:
        rate *= working_day_gate(length, weekend_factor=0.78)
    rate += linear_trend(length, PLAN["trend_slope"])
    rate = np.maximum(rate, 0.1)
    overdispersion = max(1.0, float(params.get("overdispersion", 1.35)))
    values = rng.negative_binomial(np.maximum(rate / (overdispersion - 1 + 1e-9), 1.0), 1 / overdispersion)
    return values.astype(float), np.zeros(length, dtype=int)


def generate_bounded_utilization(length, rng):
    params = PLAN.get("domain_params", {})
    values = np.full(length, float(PLAN["baseline"]))
    values += daily_load_cycle(length, amplitude=PLAN["daily_amplitude"])
    if PLAN["weekly_enabled"]:
        values += PLAN["weekly_amplitude"] * np.sin(2 * np.pi * np.arange(length) / (24 * 7) - 0.8)
    values += rng.normal(0.0, PLAN["noise_sigma"], size=length)
    batch_hour = float(params.get("batch_hour", 2.0))
    for i, hour in enumerate(hour_of_day(length)):
        if abs(hour - batch_hour) < 0.5 and rng.random() < float(params.get("batch_probability", 0.35)):
            values[i : min(length, i + 2)] += rng.uniform(8.0, 24.0)
    return np.clip(values, 0.0, float(params.get("upper_bound", 100.0))), np.zeros(length, dtype=int)


def generate_cyclic_signal(length, rng):
    values = np.full(length, float(PLAN["baseline"]))
    values += linear_trend(length, PLAN["trend_slope"])
    values += daily_load_cycle(length, amplitude=PLAN["daily_amplitude"])
    if PLAN["weekly_enabled"]:
        values += PLAN["weekly_amplitude"] * np.sin(2 * np.pi * np.arange(length) / (24 * 7) - 0.8)
        values *= working_day_gate(length)
    if PLAN["seasonal_amplitude"]:
        values += PLAN["seasonal_amplitude"] * np.sin(2 * np.pi * np.arange(length) / max(length, 24 * 30) + 0.5)
    if PLAN["heat_effect"]:
        values += heat_index_effect(length, amplitude=PLAN["heat_effect"])
    values += rng.normal(0.0, PLAN["noise_sigma"], size=length)
    return values, np.zeros(length, dtype=int)


def add_anomalies(values, rng):
    anomaly = np.zeros(len(values), dtype=int)
    if not PLAN.get("anomaly_enabled", False):
        return values, anomaly
    count = min(int(PLAN.get("anomaly_count", 0)), len(values))
    if count <= 0:
        return values, anomaly
    values = values.copy()
    width = max(1, int(PLAN.get("anomaly_width", 1)))
    positions = rng.choice(len(values), size=count, replace=False)
    scale = np.std(values) or 1.0
    for pos in positions:
        start_pos = max(0, pos - width // 2)
        end_pos = min(len(values), start_pos + width)
        direction = "both"
        sign = 1.0 if direction == "positive" else -1.0 if direction == "negative" else rng.choice([-1.0, 1.0])
        delta = sign * PLAN.get("anomaly_magnitude", 3.0) * scale
        if PLAN.get("anomaly_kind") == "drop":
            delta = -abs(delta)
        elif PLAN.get("anomaly_kind") == "positive_spike":
            delta = abs(delta)
        values[start_pos:end_pos] += delta
        anomaly[start_pos:end_pos] = 1
    return values, anomaly


def standardize(values):
    scale = np.std(values)
    return np.zeros_like(values) if scale < 1e-9 else (values - np.mean(values)) / scale


def inject_for_target(values, rng, targets):
    if PLAN.get("anomaly_target", "value") not in targets:
        return values.copy(), np.zeros(len(values), dtype=int)
    return add_anomalies(values, rng)


def apply_semantic_process(base_values, rng):
    semantic_type = PLAN.get("semantic_type", "instantaneous")
    params = PLAN.get("semantic_config", {})
    constraints = PLAN.get("output_constraints", {})
    length = len(base_values)
    columns = {}

    if semantic_type == "cumulative":
        allow_negative = bool(params.get("allow_negative_increment", False))
        increments = base_values.copy()
        if not allow_negative:
            increments = np.maximum(increments, 0.0)
        increments, flags = inject_for_target(increments, rng, {"increment", "flow"})
        if not allow_negative:
            increments = np.maximum(increments, 0.0)
        values = float(params.get("initial_value", 0.0)) + np.cumsum(increments)
        columns["increment"] = increments
    elif semantic_type == "stock_flow":
        scale = max(float(np.mean(np.abs(base_values))) * 0.08, 1.0)
        inflow = rng.gamma(2.0, scale / 2.0, size=length) * float(params.get("inflow_scale", 1.0))
        outflow = rng.gamma(2.0, scale / 2.0, size=length) * float(params.get("outflow_scale", 0.85))
        flags = np.zeros(length, dtype=int)
        if PLAN.get("anomaly_target") in {"flow", "inflow"}:
            inflow, flags = add_anomalies(inflow, rng)
            inflow = np.maximum(inflow, 0.0)
        elif PLAN.get("anomaly_target") == "outflow":
            outflow, flags = add_anomalies(outflow, rng)
            outflow = np.maximum(outflow, 0.0)
        previous = float(params.get("initial_value", max(PLAN["baseline"], 0.0)))
        values = np.empty(length)
        for i in range(length):
            previous += inflow[i] - outflow[i]
            if constraints.get("nonnegative", True):
                previous = max(previous, 0.0)
            values[i] = previous
        columns.update({"inflow": inflow, "outflow": outflow, "net_flow": inflow - outflow})
    elif semantic_type == "regime_switching":
        states = np.asarray(params.get("states", [0.4, 1.0, 1.5]), dtype=float)
        probability = float(params.get("transition_probability", 0.05))
        state_index = np.zeros(length, dtype=int)
        current = int(params.get("initial_state", min(1, len(states) - 1)))
        for i in range(length):
            if i and rng.random() < probability and len(states) > 1:
                current = int(rng.choice([j for j in range(len(states)) if j != current]))
            state_index[i] = current
        flags = np.zeros(length, dtype=int)
        if PLAN.get("anomaly_enabled") and PLAN.get("anomaly_target") == "state":
            for pos in rng.choice(length, size=min(int(PLAN.get("anomaly_count", 0)), length), replace=False):
                end = min(length, pos + max(1, int(PLAN.get("anomaly_width", 1))))
                state_index[pos:end] = int(np.argmin(states))
                flags[pos:end] = 1
        values = base_values * states[state_index]
        columns["state"] = state_index
    elif semantic_type == "random_walk":
        steps = float(params.get("drift", PLAN["trend_slope"] / max(length, 1)))
        steps = steps + float(params.get("volatility", max(PLAN["noise_sigma"], 1.0))) * standardize(base_values)
        steps, flags = inject_for_target(steps, rng, {"step", "increment"})
        values = float(params.get("initial_value", max(PLAN["baseline"], 1.0))) + np.cumsum(steps)
        columns["step"] = steps
    elif semantic_type == "decay_recovery":
        equilibrium = float(params.get("equilibrium", PLAN["baseline"]))
        rate = np.clip(float(params.get("recovery_rate", 0.12)), 0.0, 1.0)
        impulses = np.zeros(length)
        impulses[0] = float(params.get("impulse", PLAN["daily_amplitude"] or PLAN["baseline"]))
        impulses, flags = inject_for_target(impulses, rng, {"impulse", "value"})
        values = np.empty(length)
        previous = equilibrium + impulses[0]
        for i in range(length):
            if i:
                previous += rate * (equilibrium - previous) + impulses[i]
            values[i] = previous
        columns["impulse"] = impulses
    elif semantic_type == "saturation_growth":
        capacity = max(float(params.get("capacity", max(PLAN["baseline"] * 10.0, 100.0))), 1e-6)
        growth_rate = np.full(length, max(float(params.get("growth_rate", 0.06)), 0.0))
        growth_rate, flags = inject_for_target(growth_rate, rng, {"growth_rate"})
        growth_rate = np.maximum(growth_rate, 0.0)
        previous = np.clip(float(params.get("initial_value", max(PLAN["baseline"], 1.0))), 1e-6, capacity)
        values = np.empty(length)
        for i in range(length):
            previous += growth_rate[i] * previous * (1.0 - previous / capacity)
            previous = np.clip(previous, 0.0, capacity)
            values[i] = previous
        columns["growth_rate"] = growth_rate
    elif semantic_type == "multivariate_lag":
        driver, flags = inject_for_target(base_values, rng, {"driver"})
        lag = max(0, int(params.get("lag", 2)))
        lagged = np.roll(driver, lag)
        if lag:
            lagged[:lag] = driver[0]
        values = PLAN["baseline"] + float(params.get("coefficient", 0.7)) * lagged
        values += rng.normal(0.0, float(params.get("residual_sigma", PLAN["noise_sigma"])), size=length)
        columns[str(params.get("driver_name", "driver"))] = driver
        columns["lagged_driver"] = lagged
    else:
        values, flags = inject_for_target(base_values, rng, {"value"})

    lower = constraints.get("lower_bound", PLAN.get("lower_bound"))
    if constraints.get("nonnegative") and lower is None:
        lower = 0.0
    if lower is not None:
        values = np.maximum(values, float(lower))
    if constraints.get("upper_bound") is not None:
        values = np.minimum(values, float(constraints["upper_bound"]))
    if constraints.get("integer"):
        values = np.round(values)
    return values, flags, columns


def generate(length=__LENGTH__, freq=__FREQ__, start=__START__, seed=__SEED__):
    rng = np.random.default_rng(seed)
    generator_type = PLAN.get("generator_type", "cyclic_signal")
    if generator_type == "intermittent_event":
        values, anomaly = generate_precipitation(length, rng)
    elif generator_type == "daylight_envelope":
        values, anomaly = generate_solar(length, rng)
    elif generator_type == "smooth_environmental":
        values, anomaly = generate_temperature(length, rng)
    elif generator_type == "count_process":
        values, anomaly = generate_count_process(length, rng)
    elif generator_type == "bounded_utilization":
        values, anomaly = generate_bounded_utilization(length, rng)
    else:
        values, anomaly = generate_cyclic_signal(length, rng)

    values, extra_anomaly, semantic_columns = apply_semantic_process(values, rng)
    anomaly = np.maximum(anomaly, extra_anomaly)
    payload = {
        "timestamp": pd.date_range(start=start, periods=length, freq=freq),
        "value": values.round(4),
        "anomaly": anomaly,
    }
    for name, column in semantic_columns.items():
        payload[name] = np.asarray(column).round(4)
    payload.update({
        "unit": PLAN["unit"],
        "domain": PLAN["domain"],
        "generator_type": generator_type,
        "semantic_type": PLAN.get("semantic_type", "instantaneous"),
    })
    return pd.DataFrame(payload)


if __name__ == "__main__":
    df = generate()
    df.to_csv("generated_timeseries.csv", index=False)
    print(df.head(12).to_string(index=False))
'''
    return (
        textwrap.dedent(template)
        .replace("__PLAN_JSON__", plan_json)
        .replace("__LENGTH__", repr(length))
        .replace("__FREQ__", repr(freq))
        .replace("__START__", repr(start))
        .replace("__SEED__", repr(seed))
    )
