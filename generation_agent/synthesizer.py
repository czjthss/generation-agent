from __future__ import annotations

import pandas as pd
import numpy as np

from .features import (
    daily_load_cycle,
    gaussian_noise,
    heat_index_effect,
    linear_trend,
    sinusoidal_cycle,
    working_day_gate,
)
from .planner import SeriesPlan
from .semantic_transforms import apply_semantic_process
from .semantic_validators import validate_and_repair
from .component_workflow import synthesize_component_base


def _hour_of_day(length: int) -> np.ndarray:
    return np.arange(length, dtype=float) % 24.0


def _generate_precipitation(plan: SeriesPlan, length: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    params = plan.domain_params
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


def _generate_solar(plan: SeriesPlan, length: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    params = plan.domain_params
    hour = _hour_of_day(length)
    sunrise = float(params.get("sunrise_hour", 6.0))
    sunset = float(params.get("sunset_hour", 19.0))
    daylight = (hour >= sunrise) & (hour <= sunset)
    phase = (hour - sunrise) / max(1.0, sunset - sunrise)
    envelope = np.where(daylight, np.sin(np.pi * np.clip(phase, 0.0, 1.0)) ** 1.7, 0.0)
    day_index = np.arange(length) // 24
    day_factor = rng.uniform(0.82, 1.08, size=int(day_index.max()) + 1 if length else 1)
    values = plan.daily_amplitude * envelope * day_factor[day_index]

    cloud_probability = float(params.get("cloud_probability", 0.18))
    anomaly = np.zeros(length, dtype=int)
    cursor = 0
    while cursor < length:
        if daylight[cursor] and rng.random() < cloud_probability:
            duration = int(rng.integers(1, 5))
            end = min(length, cursor + duration)
            drop = rng.uniform(float(params.get("cloud_drop_min", 0.35)), float(params.get("cloud_drop_max", 0.8)))
            values[cursor:end] *= drop
        cursor += 1
    values += rng.normal(0.0, plan.noise_sigma, size=length) * daylight
    return np.maximum(values, 0.0), anomaly


def _generate_temperature(plan: SeriesPlan, length: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    params = plan.domain_params
    hour = _hour_of_day(length)
    peak_hour = float(params.get("peak_hour", 15.0))
    target = plan.baseline + plan.daily_amplitude * np.sin(2 * np.pi * (hour - peak_hour + 6) / 24)
    target += linear_trend(length, plan.trend_slope)
    target += rng.normal(0.0, plan.noise_sigma, size=length)
    values = target.copy()
    inertia = float(params.get("inertia", 0.88))
    for i in range(1, length):
        values[i] = inertia * values[i - 1] + (1.0 - inertia) * target[i]
    return values, np.zeros(length, dtype=int)


def _generate_count_process(plan: SeriesPlan, length: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    params = plan.domain_params
    hour = _hour_of_day(length)
    morning = np.exp(-0.5 * ((hour - float(params.get("morning_peak", 8.0))) / 2.0) ** 2)
    evening = np.exp(-0.5 * ((hour - float(params.get("evening_peak", 18.0))) / 2.8) ** 2)
    rate = plan.baseline + plan.daily_amplitude * (0.8 * morning + evening)
    if plan.weekly_enabled:
        rate *= working_day_gate(length, weekend_factor=0.78)
    rate += linear_trend(length, plan.trend_slope)
    rate = np.maximum(rate, 0.1)
    overdispersion = max(1.0, float(params.get("overdispersion", 1.35)))
    values = rng.negative_binomial(np.maximum(rate / (overdispersion - 1 + 1e-9), 1.0), 1 / overdispersion)
    return values.astype(float), np.zeros(length, dtype=int)


def _generate_bounded_utilization(plan: SeriesPlan, length: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    params = plan.domain_params
    values = np.full(length, float(plan.baseline), dtype=float)
    values += daily_load_cycle(length, amplitude=plan.daily_amplitude, phase=float(params.get("daily_phase", 0.0)))
    if plan.weekly_enabled:
        values += sinusoidal_cycle(length, period=24 * 7, amplitude=plan.weekly_amplitude, phase=-0.8)
    values += rng.normal(0.0, plan.noise_sigma, size=length)
    batch_hour = float(params.get("batch_hour", 2.0))
    for i, hour in enumerate(_hour_of_day(length)):
        if abs(hour - batch_hour) < 0.5 and rng.random() < float(params.get("batch_probability", 0.35)):
            values[i : min(length, i + 2)] += rng.uniform(8.0, 24.0)
    upper = float(params.get("upper_bound", 100.0))
    return np.clip(values, 0.0, upper), np.zeros(length, dtype=int)


def _generate_cyclic_signal(plan: SeriesPlan, length: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    params = plan.domain_params
    values = np.full(length, float(plan.baseline), dtype=float)
    values += linear_trend(length, plan.trend_slope)
    values += daily_load_cycle(length, amplitude=plan.daily_amplitude, phase=float(params.get("daily_phase", 0.0)))

    if plan.weekly_enabled:
        values += sinusoidal_cycle(length, period=24 * 7, amplitude=plan.weekly_amplitude, phase=-0.8)
        values *= working_day_gate(length, weekend_factor=float(params.get("weekend_factor", 0.72)))
    if plan.seasonal_amplitude:
        values += sinusoidal_cycle(length, period=max(length, 24 * 30), amplitude=plan.seasonal_amplitude, phase=0.5)
    if plan.heat_effect:
        values += heat_index_effect(length, amplitude=plan.heat_effect)

    values += gaussian_noise(length, rng, plan.noise_sigma)
    return values, np.zeros(length, dtype=int)


def synthesize_series(
    plan: SeriesPlan,
    length: int = 168,
    freq: str = "h",
    start: str = "2026-07-01 00:00:00",
    seed: int | None = 42,
) -> pd.DataFrame:
    if length <= 0:
        raise ValueError("length must be positive")

    rng = np.random.default_rng(seed)
    component_workflow = plan.metadata.get("component_workflow")
    component_artifacts = None
    if isinstance(component_workflow, dict):
        try:
            from .component_workflow import ComponentWorkflow, InputProfile, MechanismComponent, VariableProfile

            workflow = ComponentWorkflow(
                input_profile=InputProfile(**component_workflow["input_profile"]),
                variable_profile=VariableProfile(**component_workflow["variable_profile"]),
                components=[
                    MechanismComponent(**item)
                    for item in component_workflow.get("components", [])
                    if isinstance(item, dict)
                ],
                composition=component_workflow.get("composition", {}),
                final_transform=component_workflow.get("final_transform", "identity"),
                anomaly_target=component_workflow.get("anomaly_target", plan.anomaly_target),
                validator_rules=component_workflow.get("validator_rules", []),
                agent_trace=component_workflow.get("agent_trace", {}),
            )
            values, anomaly, component_artifacts = synthesize_component_base(plan, workflow, length, rng)
        except Exception as exc:
            plan.metadata["component_workflow_error"] = f"{exc.__class__.__name__}: {exc}"
            component_artifacts = None

    if component_artifacts is None:
        if plan.generator_type == "intermittent_event":
            values, anomaly = _generate_precipitation(plan, length, rng)
        elif plan.generator_type == "daylight_envelope":
            values, anomaly = _generate_solar(plan, length, rng)
        elif plan.generator_type == "smooth_environmental":
            values, anomaly = _generate_temperature(plan, length, rng)
        elif plan.generator_type == "count_process":
            values, anomaly = _generate_count_process(plan, length, rng)
        elif plan.generator_type == "bounded_utilization":
            values, anomaly = _generate_bounded_utilization(plan, length, rng)
        else:
            values, anomaly = _generate_cyclic_signal(plan, length, rng)

    values, semantic_anomaly, semantic_columns = apply_semantic_process(values, plan, rng)
    anomaly = np.maximum(anomaly, semantic_anomaly)
    values, validation = validate_and_repair(values, plan, semantic_columns)

    index = pd.date_range(start=start, periods=length, freq=freq)
    payload = {
        "timestamp": index,
        "value": values.round(4),
        "anomaly": anomaly.astype(int),
    }
    for name, column in semantic_columns.items():
        payload[name] = np.asarray(column).round(4)
    payload.update(
        {
            "unit": plan.unit,
            "domain": plan.domain,
            "generator_type": plan.generator_type,
            "semantic_type": plan.semantic_type,
        }
    )
    frame = pd.DataFrame(payload)
    frame.attrs["validation_report"] = validation
    if component_artifacts is not None:
        frame.attrs["component_workflow"] = component_artifacts["workflow"]
        frame.attrs["component_report"] = component_artifacts["component_report"]
        frame.attrs["component_stats"] = component_artifacts["component_stats"]
    return frame
