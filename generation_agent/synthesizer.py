from __future__ import annotations

import pandas as pd
import numpy as np

from .features import (
    AnomalyConfig,
    add_anomalies,
    gaussian_noise,
    linear_trend,
)
from .planner import SeriesPlan
from .semantic_transforms import apply_semantic_process
from .semantic_validators import validate_and_repair
from .component_workflow import synthesize_component_base
from .time_features import (
    TimeContext,
    build_time_context,
    daily_load_shape,
    heat_index_effect_from_time,
    hours_to_steps,
    seasonal_cycle_from_time,
    weekly_cycle_from_time,
    working_day_gate_from_time,
)


def _hour_of_day(length: int, context: TimeContext | None = None) -> np.ndarray:
    if context is not None:
        return context.hour_of_day
    return np.arange(length, dtype=float) % 24.0


def _generate_precipitation(plan: SeriesPlan, length: int, rng: np.random.Generator, context: TimeContext) -> tuple[np.ndarray, np.ndarray]:
    params = plan.domain_params
    values = np.zeros(length, dtype=float)
    anomaly = np.zeros(length, dtype=int)
    event_probability = float(params.get("event_probability", 0.18))
    mean_duration = max(1.0, float(params.get("mean_duration", 5.0)))
    mean_duration_steps = max(1.0, mean_duration * context.steps_per_hour)
    shape = max(0.3, float(params.get("intensity_shape", 1.4)))
    scale = max(0.1, float(params.get("intensity_scale", 5.0)))
    dry_spell_bias = float(params.get("dry_spell_bias", 0.68))
    storm_probability = float(params.get("storm_probability", 0.08))
    storm_multiplier = max(1.0, float(params.get("storm_multiplier", 3.0)))

    cursor = 0
    while cursor < length:
        if rng.random() < dry_spell_bias:
            cursor += int(rng.integers(hours_to_steps(context, 3), hours_to_steps(context, 18) + 1))
            continue
        if rng.random() > event_probability:
            cursor += 1
            continue

        duration = int(max(1, rng.geometric(1.0 / mean_duration_steps)))
        end = min(length, cursor + duration)
        is_storm = rng.random() < storm_probability
        event_scale = scale * (storm_multiplier if is_storm else 1.0)
        raw = rng.gamma(shape=shape, scale=event_scale, size=end - cursor)
        envelope = np.sin(np.linspace(0.15, np.pi - 0.15, end - cursor))
        values[cursor:end] += raw * np.maximum(envelope, 0.15)
        if is_storm:
            anomaly[cursor:end] = 1
        cursor = end + int(rng.integers(hours_to_steps(context, 1), hours_to_steps(context, 10) + 1))

    drizzle = (values > 0) & (rng.random(length) < 0.18)
    values[drizzle] *= rng.uniform(0.15, 0.55, size=drizzle.sum())
    return values, anomaly


def _generate_solar(plan: SeriesPlan, length: int, rng: np.random.Generator, context: TimeContext) -> tuple[np.ndarray, np.ndarray]:
    params = plan.domain_params
    hour = _hour_of_day(length, context)
    sunrise = float(params.get("sunrise_hour", 6.0))
    sunset = float(params.get("sunset_hour", 19.0))
    day_index = context.day_code
    day_factor = rng.uniform(0.82, 1.08, size=int(day_index.max()) + 1 if length else 1)
    if context.has_intraday_resolution:
        daylight = (hour >= sunrise) & (hour <= sunset)
        phase = (hour - sunrise) / max(1.0, sunset - sunrise)
        envelope = np.where(daylight, np.sin(np.pi * np.clip(phase, 0.0, 1.0)) ** 1.7, 0.0)
        values = plan.daily_amplitude * envelope * day_factor[day_index]
        noise_gate = daylight
    else:
        daylight_hours = max(sunset - sunrise, 0.0)
        aggregation_days = max(context.step_hours / 24.0, 1.0)
        seasonal = 0.92 + 0.16 * np.maximum(
            seasonal_cycle_from_time(context, amplitude=1.0, period_days=365.25, phase=-0.2),
            -0.5,
        )
        values = plan.daily_amplitude * (daylight_hours / 12.0) * aggregation_days * seasonal * day_factor[day_index]
        daylight = np.ones(length, dtype=bool)
        noise_gate = np.ones(length, dtype=bool)

    cloud_probability = float(params.get("cloud_probability", 0.18))
    anomaly = np.zeros(length, dtype=int)
    cursor = 0
    while cursor < length:
        if daylight[cursor] and rng.random() < cloud_probability:
            duration = int(rng.integers(hours_to_steps(context, 1), hours_to_steps(context, 5) + 1))
            end = min(length, cursor + duration)
            drop = rng.uniform(float(params.get("cloud_drop_min", 0.35)), float(params.get("cloud_drop_max", 0.8)))
            values[cursor:end] *= drop
        cursor += 1
    values += rng.normal(0.0, plan.noise_sigma, size=length) * noise_gate
    return np.maximum(values, 0.0), anomaly


def _generate_temperature(plan: SeriesPlan, length: int, rng: np.random.Generator, context: TimeContext) -> tuple[np.ndarray, np.ndarray]:
    params = plan.domain_params
    hour = _hour_of_day(length, context)
    peak_hour = float(params.get("peak_hour", 15.0))
    if context.has_intraday_resolution:
        cycle = plan.daily_amplitude * np.sin(2 * np.pi * (hour - peak_hour + 6) / 24)
    else:
        cycle = np.zeros(length, dtype=float)
    target = plan.baseline + cycle
    target += linear_trend(length, plan.trend_slope)
    target += rng.normal(0.0, plan.noise_sigma, size=length)
    values = target.copy()
    inertia = float(params.get("inertia", 0.88))
    for i in range(1, length):
        values[i] = inertia * values[i - 1] + (1.0 - inertia) * target[i]
    return values, np.zeros(length, dtype=int)


def _generate_count_process(plan: SeriesPlan, length: int, rng: np.random.Generator, context: TimeContext) -> tuple[np.ndarray, np.ndarray]:
    params = plan.domain_params
    hour = _hour_of_day(length, context)
    if context.has_intraday_resolution:
        morning = np.exp(-0.5 * ((hour - float(params.get("morning_peak", 8.0))) / 2.0) ** 2)
        evening = np.exp(-0.5 * ((hour - float(params.get("evening_peak", 18.0))) / 2.8) ** 2)
    else:
        morning = np.zeros(length, dtype=float)
        evening = np.zeros(length, dtype=float)
    rate = plan.baseline + plan.daily_amplitude * (0.8 * morning + evening)
    if plan.weekly_enabled:
        rate *= working_day_gate_from_time(context, weekend_factor=0.78)
    rate += linear_trend(length, plan.trend_slope)
    rate = np.maximum(rate, 0.1)
    overdispersion = max(1.0, float(params.get("overdispersion", 1.35)))
    values = rng.negative_binomial(np.maximum(rate / (overdispersion - 1 + 1e-9), 1.0), 1 / overdispersion)
    return values.astype(float), np.zeros(length, dtype=int)


def _generate_bounded_utilization(plan: SeriesPlan, length: int, rng: np.random.Generator, context: TimeContext) -> tuple[np.ndarray, np.ndarray]:
    params = plan.domain_params
    values = np.full(length, float(plan.baseline), dtype=float)
    values += daily_load_shape(context, amplitude=plan.daily_amplitude, phase=float(params.get("daily_phase", 0.0)))
    if plan.weekly_enabled:
        values += weekly_cycle_from_time(context, amplitude=plan.weekly_amplitude, phase=-0.8)
    values += rng.normal(0.0, plan.noise_sigma, size=length)
    batch_hour = float(params.get("batch_hour", 2.0))
    duration = hours_to_steps(context, 2)
    for i, hour in enumerate(_hour_of_day(length, context)):
        if abs(hour - batch_hour) < 0.5 and rng.random() < float(params.get("batch_probability", 0.35)):
            values[i : min(length, i + duration)] += rng.uniform(8.0, 24.0)
    upper = float(params.get("upper_bound", 100.0))
    return np.clip(values, 0.0, upper), np.zeros(length, dtype=int)


def _generate_cyclic_signal(plan: SeriesPlan, length: int, rng: np.random.Generator, context: TimeContext) -> tuple[np.ndarray, np.ndarray]:
    params = plan.domain_params
    values = np.full(length, float(plan.baseline), dtype=float)
    values += linear_trend(length, plan.trend_slope)
    values += daily_load_shape(context, amplitude=plan.daily_amplitude, phase=float(params.get("daily_phase", 0.0)))
    if {"shift_start_hour", "shift_end_hour", "shift_amplitude"}.issubset(params):
        start_hour = float(params.get("shift_start_hour", 8.0))
        end_hour = float(params.get("shift_end_hour", 18.0))
        if start_hour <= end_hour:
            in_shift = (context.hour_of_day >= start_hour) & (context.hour_of_day <= end_hour)
        else:
            in_shift = (context.hour_of_day >= start_hour) | (context.hour_of_day <= end_hour)
        shift_gate = working_day_gate_from_time(
            context,
            weekend_factor=float(params.get("shift_weekend_factor", params.get("weekend_factor", 0.72))),
        )
        values += float(params.get("shift_amplitude", 0.0)) * in_shift.astype(float) * shift_gate

    if plan.weekly_enabled:
        values += weekly_cycle_from_time(context, amplitude=plan.weekly_amplitude, phase=-0.8)
        values *= working_day_gate_from_time(context, weekend_factor=float(params.get("weekend_factor", 0.72)))
    if plan.seasonal_amplitude:
        values += seasonal_cycle_from_time(context, amplitude=plan.seasonal_amplitude, period_days=30.0, phase=0.5)
    if plan.heat_effect:
        values += heat_index_effect_from_time(context, amplitude=plan.heat_effect)

    values += gaussian_noise(length, rng, plan.noise_sigma)
    return values, np.zeros(length, dtype=int)


def _lag1_correlation(values: np.ndarray) -> float:
    if len(values) < 3 or float(np.std(values)) <= 1e-9:
        return 0.0
    corr = np.corrcoef(values[:-1], values[1:])[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


def _clip_reference_tail(values: np.ndarray, plan: SeriesPlan, quantiles: dict) -> np.ndarray:
    caps = [float(quantiles[key]) for key in ("p999", "p99") if key in quantiles]
    if not caps:
        return values
    cap = max(caps)
    if cap <= 0.0:
        return values
    clipped = np.minimum(values, cap)
    plan.metadata.setdefault("reference_target_execution", {})["tail_cap"] = {
        "cap": cap,
        "method": "p999_or_p99_clip",
    }
    return clipped


def _calibrate_reference_distribution(values: np.ndarray, plan: SeriesPlan, quantiles: dict) -> np.ndarray:
    required = ("p05", "p50", "p95")
    if not all(key in quantiles for key in required):
        return _clip_reference_tail(values, plan, quantiles)
    is_sparse_event = (
        plan.generator_type == "intermittent_event"
        or float(plan.metadata.get("reference_executable_targets", {}).get("event_run_length", {}).get("zero_ratio", 0.0)) >= 0.2
    )
    if is_sparse_event:
        positive = values > 0.0
        if not np.any(positive):
            return values
        current_p95 = float(np.quantile(values[positive], 0.95))
        target_p95 = float(quantiles["p95"])
        if current_p95 > 1e-9 and target_p95 > 0.0:
            values = values.copy()
            values[positive] *= target_p95 / current_p95
        return _clip_reference_tail(values, plan, quantiles)
    current_p05, current_p50, current_p95 = np.quantile(values, [0.05, 0.5, 0.95])
    current_span = float(current_p95 - current_p05)
    target_span = float(quantiles["p95"]) - float(quantiles["p05"])
    if current_span <= 1e-9 or target_span <= 0.0:
        return values
    calibrated = (values - current_p50) * (target_span / current_span) + float(quantiles["p50"])
    lower = plan.lower_bound
    if plan.output_constraints.get("nonnegative") and lower is None:
        lower = 0.0
    if lower is not None:
        calibrated = np.maximum(calibrated, float(lower))
    upper = plan.output_constraints.get("upper_bound")
    if upper is not None:
        calibrated = np.minimum(calibrated, float(upper))
    return _clip_reference_tail(calibrated, plan, quantiles)


def _reference_acf_filter(values: np.ndarray, target: float, current: float) -> tuple[np.ndarray, float, str]:
    if len(values) < 3:
        return values, 0.0, "skipped_short_series"
    if target >= 0.0:
        alpha = float(np.clip(target - current, 0.05, 0.85))
        filtered = values.astype(float, copy=True)
        for index in range(1, len(filtered)):
            filtered[index] = alpha * filtered[index - 1] + (1.0 - alpha) * values[index]
        return filtered, alpha, "ar_smoothing"
    alpha = float(np.clip(abs(target - current), 0.05, 0.85))
    filtered = values.astype(float, copy=True)
    centered = values - float(np.mean(values))
    for index in range(1, len(filtered)):
        centered_value = -alpha * (filtered[index - 1] - float(np.mean(values))) + (1.0 - alpha) * centered[index]
        filtered[index] = float(np.mean(values)) + centered_value
    return filtered, alpha, "negative_ar_alternating_filter"


def _smooth_positive_runs(values: np.ndarray, target: float, current: float) -> tuple[np.ndarray, dict[str, float | str] | None]:
    positive = values > 0.0
    if not np.any(positive):
        return values, None
    alpha = float(np.clip(target - max(current, 0.0), 0.05, 0.85))
    if alpha <= 0.0:
        return values, None
    filtered = values.astype(float, copy=True)
    start = None
    for index, active in enumerate(positive):
        if active and start is None:
            start = index
        if start is not None and (not active or index == len(positive) - 1):
            end = index + 1 if active and index == len(positive) - 1 else index
            for run_index in range(start + 1, end):
                filtered[run_index] = alpha * filtered[run_index - 1] + (1.0 - alpha) * values[run_index]
            start = None
    filtered[~positive] = 0.0
    return filtered, {"alpha": alpha, "method": "event_internal_intensity_smoothing"}


def _apply_reference_acf_target(values: np.ndarray, plan: SeriesPlan, targets: dict) -> tuple[np.ndarray, dict[str, float | str] | None]:
    target = None
    acf_targets = targets.get("acf_targets") if isinstance(targets, dict) else None
    if isinstance(acf_targets, dict):
        for key in ("1", 1):
            if key in acf_targets:
                target = acf_targets[key]
                break
    if target is None:
        target = plan.metadata.get("reference_targets", {}).get("lag1_correlation")
    if target is None:
        return values, None
    target = float(np.clip(float(target), -0.98, 0.98))
    current = _lag1_correlation(values)
    if abs(target - current) <= 0.05 or len(values) < 3:
        plan.metadata.setdefault("reference_target_execution", {})["acf_lag1"] = {
            "target": target,
            "before": current,
            "after": current,
            "method": "skipped_already_close",
        }
        return values, None
    if target >= 0.0 and target < current:
        plan.metadata.setdefault("reference_target_execution", {})["acf_lag1"] = {
            "target": target,
            "before": current,
            "after": current,
            "method": "skipped_decrease_not_applied",
            "reason": "positive_acf_decorrelation_filter_not_enabled",
        }
        return values, None
    if plan.generator_type == "intermittent_event":
        if target < 0.0:
            plan.metadata.setdefault("reference_target_execution", {})["acf_lag1"] = {
                "target": target,
                "before": current,
                "after": current,
                "method": "skipped_negative_acf_for_sparse_event",
            }
            return values, None
        filtered, info = _smooth_positive_runs(values, target, current)
        plan.metadata.setdefault("reference_target_execution", {})["acf_lag1"] = {
            "target": target,
            "before": current,
            "after": _lag1_correlation(filtered),
            "method": "event_internal_intensity_smoothing",
            "zero_mask_preserved": True,
            "alpha": info["alpha"] if info else 0.0,
        }
        return filtered, info
    filtered, alpha, method = _reference_acf_filter(values, target, current)
    plan.metadata.setdefault("reference_target_execution", {})["acf_lag1"] = {
        "target": target,
        "before": current,
        "after": _lag1_correlation(filtered),
        "method": method,
        "alpha": alpha,
    }
    return filtered, {"alpha": alpha, "method": method}


def _propagate_anomaly_mask_for_reference_filter(
    anomaly: np.ndarray | None,
    filter_info: dict[str, float | str] | None,
) -> np.ndarray | None:
    if anomaly is None or filter_info is None or not np.any(anomaly):
        return anomaly
    alpha = float(filter_info.get("alpha", 0.0))
    if alpha <= 0.0:
        return anomaly
    width = int(np.clip(np.ceil(np.log(0.05) / np.log(max(alpha, 1e-6))), 1, 48))
    propagated = anomaly.astype(int, copy=True)
    starts = np.flatnonzero(anomaly.astype(bool))
    for start in starts:
        propagated[start : min(len(propagated), start + width + 1)] = 1
    return propagated


def _run_lengths(mask: np.ndarray, value: bool) -> list[int]:
    runs: list[int] = []
    current = 0
    for item in mask.astype(bool):
        if bool(item) == value:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def _deterministic_run_mask(length: int, zero_ratio: float, zero_run: float, nonzero_run: float) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=bool)
    target_positive = float(np.clip(1.0 - zero_ratio, 0.0, 1.0))
    if target_positive <= 1e-9:
        return np.zeros(length, dtype=bool)
    if target_positive >= 1.0 - 1e-9:
        return np.ones(length, dtype=bool)
    on = max(1, int(round(nonzero_run if nonzero_run > 0 else max(1.0, target_positive * 12.0))))
    off_from_ratio = on * zero_ratio / max(target_positive, 1e-9)
    off = max(1, int(round(zero_run if zero_run > 0 else off_from_ratio)))
    mask = np.zeros(length, dtype=bool)
    cursor = 0
    while cursor < length:
        cursor += off
        if cursor >= length:
            break
        end = min(length, cursor + on)
        mask[cursor:end] = True
        cursor = end
    desired = int(round(target_positive * length))
    current = int(mask.sum())
    if current < desired:
        candidates = np.flatnonzero(~mask)
        if candidates.size:
            scores = np.sin(candidates * 12.9898 + length * 0.123)
            chosen = candidates[np.argsort(scores)[-min(desired - current, candidates.size):]]
            mask[chosen] = True
    elif current > desired:
        candidates = np.flatnonzero(mask)
        if candidates.size:
            scores = np.sin(candidates * 78.233 + length * 0.456)
            chosen = candidates[np.argsort(scores)[: min(current - desired, candidates.size)]]
            mask[chosen] = False
    return mask


def _apply_reference_event_run_target(values: np.ndarray, plan: SeriesPlan, targets: dict) -> np.ndarray:
    event_target = targets.get("event_run_length") if isinstance(targets, dict) else None
    if not isinstance(event_target, dict):
        return values
    zero_ratio = float(event_target.get("zero_ratio", 0.0) or 0.0)
    if plan.generator_type != "intermittent_event" and zero_ratio < 0.2:
        return values
    current_zero = float(np.mean(values <= 0.0)) if len(values) else 0.0
    target_zero = float(np.clip(zero_ratio, 0.0, 0.995))
    nonzero_run = float(event_target.get("median_nonzero_run", 0.0) or 0.0)
    zero_run = float(event_target.get("median_zero_run", 0.0) or 0.0)
    if abs(current_zero - target_zero) <= 0.04 and nonzero_run <= 0 and zero_run <= 0:
        plan.metadata.setdefault("reference_target_execution", {})["event_run_length"] = {
            "method": "skipped_already_close",
            "target_zero_ratio": target_zero,
            "before_zero_ratio": current_zero,
            "after_zero_ratio": current_zero,
        }
        return values
    mask = _deterministic_run_mask(len(values), target_zero, zero_run, nonzero_run)
    positives = values[values > 0.0]
    if positives.size == 0:
        positives = np.array([max(float(plan.domain_params.get("intensity_scale", plan.baseline or 1.0)), 1.0)])
    result = np.zeros_like(values, dtype=float)
    positions = np.flatnonzero(mask)
    if positions.size:
        tiled = np.resize(np.sort(positives), positions.size)
        phase = np.sin(np.linspace(0.15, np.pi - 0.15, positions.size))
        result[positions] = np.maximum(tiled * np.maximum(phase, 0.2), 0.0)
    after_zero = float(np.mean(result <= 0.0)) if len(result) else 0.0
    nonzero_runs = _run_lengths(result > 0.0, True)
    zero_runs = _run_lengths(result > 0.0, False)
    plan.metadata.setdefault("reference_target_execution", {})["event_run_length"] = {
        "method": "deterministic_run_length_mask",
        "target_zero_ratio": target_zero,
        "before_zero_ratio": current_zero,
        "after_zero_ratio": after_zero,
        "target_median_nonzero_run": nonzero_run,
        "after_median_nonzero_run": float(np.median(nonzero_runs)) if nonzero_runs else 0.0,
        "target_median_zero_run": zero_run,
        "after_median_zero_run": float(np.median(zero_runs)) if zero_runs else 0.0,
    }
    return result


def _apply_reference_segment_schedule(values: np.ndarray, plan: SeriesPlan, targets: dict) -> np.ndarray:
    schedule = targets.get("segment_schedule") if isinstance(targets, dict) else None
    if not isinstance(schedule, dict):
        return values
    segments = schedule.get("segments")
    if not isinstance(segments, list) or len(segments) < 2 or len(values) < 2:
        return values
    adjusted = values.astype(float, copy=True)
    chunks = np.array_split(np.arange(len(values)), len(segments))
    applied = []
    for indices, segment in zip(chunks, segments):
        if len(indices) == 0 or not isinstance(segment, dict):
            continue
        current = adjusted[indices]
        target_mean = segment.get("mean")
        target_std = segment.get("std")
        if target_mean is None and target_std is None:
            continue
        local = current.copy()
        if target_std is not None and float(np.std(local)) > 1e-9:
            local = (local - float(np.mean(local))) * (max(float(target_std), 0.0) / float(np.std(local))) + float(np.mean(local))
        if target_mean is not None:
            local = local + (float(target_mean) - float(np.mean(local)))
        if plan.output_constraints.get("nonnegative") or plan.lower_bound == 0:
            local = np.maximum(local, 0.0)
        adjusted[indices] = local
        applied.append({
            "segment": int(segment.get("segment", len(applied))),
            "target_mean": None if target_mean is None else float(target_mean),
            "target_std": None if target_std is None else float(target_std),
        })
    if applied:
        plan.metadata.setdefault("reference_target_execution", {})["segment_schedule"] = {
            "method": "piecewise_segment_mean_std_calibration",
            "segments_applied": applied,
        }
    return adjusted


def _apply_reference_executable_targets(
    values: np.ndarray,
    plan: SeriesPlan,
    anomaly: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if plan.semantic_type != "instantaneous":
        return values, anomaly
    targets = plan.metadata.get("reference_executable_targets", {})
    if not isinstance(targets, dict):
        return values, anomaly
    quantiles = targets.get("distribution_quantiles") or plan.domain_params.get("reference_distribution", {}).get("quantile_grid")
    adjusted = values
    adjusted = _apply_reference_event_run_target(adjusted, plan, targets)
    adjusted = _apply_reference_segment_schedule(adjusted, plan, targets)
    if isinstance(quantiles, dict):
        adjusted = _calibrate_reference_distribution(adjusted, plan, quantiles)
        plan.metadata.setdefault("reference_target_execution", {})["distribution_quantiles"] = {
            "method": "quantile_calibration_with_tail_cap"
        }
    adjusted, filter_info = _apply_reference_acf_target(adjusted, plan, targets)
    anomaly = _propagate_anomaly_mask_for_reference_filter(anomaly, filter_info)
    if filter_info is not None and anomaly is not None:
        plan.metadata.setdefault("reference_target_execution", {})["propagated_anomaly_effect"] = {
            "active_points": int(np.sum(anomaly)),
            "method": "acf_filter_forward_mask_expansion",
        }
    if isinstance(quantiles, dict):
        adjusted = _calibrate_reference_distribution(adjusted, plan, quantiles)
    return adjusted, anomaly


def _safe_column_name(name: str, fallback: str) -> str:
    import re

    cleaned = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff]+", "_", str(name).strip())
    cleaned = cleaned.strip("_")
    return cleaned or fallback


def _primary_variable_name(plan: SeriesPlan) -> str:
    for item in plan.variables or []:
        role = str(item.get("role", "")).lower()
        name = str(item.get("name", "")).strip()
        if name and role in {"target", "primary", "observable", "output"}:
            return _safe_column_name(name, "value")
    if plan.variables:
        return _safe_column_name(str(plan.variables[0].get("name", "value")), "value")
    return "value"


def _variable_generator_hint(name: str, variable: dict, plan: SeriesPlan) -> str:
    text = " ".join(
        str(part).lower()
        for part in (
            name,
            variable.get("role", ""),
            variable.get("semantic_hint", ""),
            variable.get("value_support", ""),
        )
    )
    if any(key in text for key in ("temp", "temperature")):
        return "temperature"
    if any(key in text for key in ("rain", "precip")):
        return "rain"
    if any(key in text for key in ("solar", "pv", "irradiance")):
        return "solar"
    if any(key in text for key in ("traffic", "order", "request", "count")):
        return "count"
    if any(key in text for key in ("schedule", "shift", "state", "open")):
        return "state"
    if any(key in text for key in ("humidity",)):
        return "smooth"
    if any(key in text for key in ("cpu", "memory", "util", "usage")):
        return "bounded"
    return str(variable.get("generator_type") or plan.generator_type or "cyclic_signal")


def _generate_variable_series(
    name: str,
    variable: dict,
    plan: SeriesPlan,
    length: int,
    rng: np.random.Generator,
    context: TimeContext,
) -> np.ndarray:
    hint = _variable_generator_hint(name, variable, plan)
    role = str(variable.get("role", "")).lower()
    profile = variable.get("profile", {}) if isinstance(variable.get("profile"), dict) else {}

    local = SeriesPlan.from_dict(plan.to_dict())
    if profile:
        local.baseline = float(profile.get("p50", profile.get("mean", local.baseline)))
        spread = max(float(profile.get("p99", local.baseline)) - float(profile.get("p01", local.baseline)), 0.0)
        local.daily_amplitude = max(spread / 4.0, 0.0)
        local.noise_sigma = max(float(profile.get("std", local.noise_sigma)) * 0.25, 1e-9)
        if float(profile.get("nonnegative_ratio", 0.0)) >= 0.999:
            local.lower_bound = 0.0

    if hint == "temperature":
        local.generator_type = "smooth_environmental"
        local.baseline = float(profile.get("p50", 28.0)) if profile else 28.0
        local.daily_amplitude = (
            float(np.clip((float(profile.get("p99", local.baseline)) - float(profile.get("p01", local.baseline))) / 4.0, 2.0, 8.0))
            if profile
            else 4.0
        )
        local.weekly_enabled = False
        local.weekly_amplitude = 0.0
        local.trend_slope = 0.0
        local.noise_sigma = (
            float(np.clip(float(profile.get("std", 2.0)) * 0.15, 0.25, 1.5))
            if profile
            else 0.5
        )
        local.domain_params = {**local.domain_params, "inertia": 0.88, "peak_hour": 15.0}
        values, _ = _generate_temperature(local, length, rng, context)
    elif hint == "rain":
        local.generator_type = "intermittent_event"
        local.domain_params = {**plan.domain_params, **local.domain_params}
        if profile:
            local.domain_params["event_probability"] = float(np.clip(1.0 - float(profile.get("zero_ratio", 0.7)), 0.01, 0.95))
            local.domain_params["intensity_scale"] = max(float(profile.get("mean", 2.0)), 0.1)
        values, _ = _generate_precipitation(local, length, rng, context)
    elif hint == "solar":
        local.generator_type = "daylight_envelope"
        local.daily_amplitude = max(float(profile.get("p99", 600.0)) if profile else 600.0, 1.0)
        values, _ = _generate_solar(local, length, rng, context)
    elif hint == "count":
        local.generator_type = "count_process"
        local.baseline = max(float(profile.get("p50", local.baseline)) if profile else local.baseline, 1.0)
        if not profile:
            local.daily_amplitude = max(min(local.daily_amplitude, local.baseline * 0.8), 1.0)
            local.noise_sigma = max(min(local.noise_sigma, local.baseline * 0.15), 0.25)
        values, _ = _generate_count_process(local, length, rng, context)
    elif hint == "state":
        gate = working_day_gate_from_time(context, weekend_factor=0.2)
        hour = _hour_of_day(length, context)
        active = ((hour >= 8) & (hour <= 18)).astype(float)
        values = (gate > 0.5).astype(float) * active
        if "state" in name.lower():
            values = np.where(values > 0, 1.0, 0.0)
    elif hint == "bounded":
        local.generator_type = "bounded_utilization"
        if not profile:
            local.baseline = float(np.clip(local.baseline, 35.0, 70.0))
            local.daily_amplitude = float(np.clip(local.daily_amplitude, 5.0, 18.0))
            local.noise_sigma = float(np.clip(local.noise_sigma, 0.5, 4.0))
        values, _ = _generate_bounded_utilization(local, length, rng, context)
    elif hint == "smooth":
        local.generator_type = "smooth_environmental"
        if not profile:
            local.baseline = 60.0
            local.daily_amplitude = 6.0
            local.weekly_enabled = False
            local.weekly_amplitude = 0.0
            local.trend_slope = 0.0
            local.noise_sigma = 1.0
            local.domain_params = {**local.domain_params, "inertia": 0.9, "peak_hour": 14.0}
        values, _ = _generate_temperature(local, length, rng, context)
    else:
        values, _ = _generate_cyclic_signal(local, length, rng, context)

    if role == "target":
        return values
    return np.asarray(values, dtype=float)


def _lagged(values: np.ndarray, lag: int) -> np.ndarray:
    lag = max(0, int(lag))
    if lag == 0 or len(values) == 0:
        return values.copy()
    result = np.roll(values, lag)
    result[:lag] = values[0]
    return result


def _standardized_effect(values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    if std < 1e-9:
        return np.zeros_like(values, dtype=float)
    return (values - float(np.mean(values))) / std


def _trigger_mask(driver: np.ndarray, threshold: float, trigger_op: str | None = None) -> np.ndarray:
    op = str(trigger_op or "gte").lower()
    if op in {"eq", "=", "=="}:
        tolerance = max(1e-6, abs(float(threshold)) * 1e-6)
        return np.isclose(driver, threshold, rtol=1e-6, atol=tolerance)
    if op in {"lte", "le", "<="}:
        return driver <= threshold
    if op in {"lt", "<"}:
        return driver < threshold
    if op in {"gt", ">"}:
        return driver > threshold
    return driver >= threshold


def _trigger_starts(active: np.ndarray) -> np.ndarray:
    if len(active) == 0:
        return np.array([], dtype=int)
    return np.flatnonzero(active & np.r_[False, ~active[:-1]])


def _relationship_effect(driver: np.ndarray, target: np.ndarray, relationship: dict, sign: float, coefficient: float) -> tuple[np.ndarray, str]:
    operator = str(relationship.get("operator") or relationship.get("relationship_operator") or relationship.get("type") or "linear_lag").lower()
    if operator in {"linear", "linear_lag", "lagged_linear"}:
        effect = sign * coefficient * _standardized_effect(driver)
    elif operator == "threshold":
        threshold = float(relationship.get("threshold", float(np.mean(driver))))
        effect = np.where(driver >= threshold, sign * coefficient, 0.0)
    elif operator == "saturation":
        midpoint = float(relationship.get("midpoint", float(np.mean(driver))))
        steepness = max(float(relationship.get("steepness", 1.0)), 1e-6)
        scaled = (driver - midpoint) / (float(np.std(driver)) * steepness + 1e-9)
        effect = sign * coefficient * (1.0 / (1.0 + np.exp(-scaled)) - 0.5) * 2.0
    elif operator == "state_gate":
        threshold = float(relationship.get("threshold", float(np.mean(driver))))
        inactive = float(relationship.get("inactive_factor", 0.0))
        active = float(relationship.get("active_factor", 1.0))
        effect = target * (np.where(driver >= threshold, active, inactive) - 1.0)
    elif operator == "event_trigger":
        threshold = float(relationship.get("threshold", float(np.quantile(driver, 0.9))))
        trigger_op = str(relationship.get("trigger_op", "gte"))
        width = max(1, int(relationship.get("width", 3)))
        effect = np.zeros_like(driver, dtype=float)
        starts = _trigger_starts(_trigger_mask(driver, threshold, trigger_op))
        for start in starts:
            end = min(len(effect), start + width)
            effect[start:end] += sign * coefficient * np.exp(-np.linspace(0.0, 2.5, end - start))
    elif operator == "piecewise":
        threshold = float(relationship.get("threshold", float(np.mean(driver))))
        low_slope = float(relationship.get("low_slope", 0.25))
        high_slope = float(relationship.get("high_slope", 1.0))
        centered = _standardized_effect(driver - threshold)
        effect = sign * coefficient * np.where(driver < threshold, low_slope * centered, high_slope * centered)
    else:
        operator = "linear_lag"
        effect = sign * coefficient * _standardized_effect(driver)
    if relationship.get("threshold") is not None and operator not in {"threshold", "piecewise", "state_gate", "event_trigger"}:
        effect = np.where(driver >= float(relationship["threshold"]), effect, 0.0)
    return np.asarray(effect, dtype=float), operator


def _multivariate_anomaly_config(plan: SeriesPlan) -> AnomalyConfig:
    return AnomalyConfig(
        enabled=plan.anomaly_enabled,
        count=plan.anomaly_count,
        magnitude=plan.anomaly_magnitude,
        width=plan.anomaly_width,
        kind=plan.anomaly_kind,
        severity=plan.anomaly_severity,
        direction="both",
    )


def _relationship_names(relationship: dict) -> tuple[str, str]:
    return (
        _safe_column_name(str(relationship.get("source") or relationship.get("source_variable") or ""), ""),
        _safe_column_name(str(relationship.get("target") or relationship.get("target_variable") or ""), ""),
    )


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or len(right) < 3:
        return 0.0
    if float(np.std(left)) < 1e-9 or float(np.std(right)) < 1e-9:
        return 0.0
    corr = float(np.corrcoef(left, right)[0, 1])
    return corr if np.isfinite(corr) else 0.0


def _apply_multivariate_plan(
    frame_payload: dict[str, np.ndarray],
    values: np.ndarray,
    anomaly: np.ndarray,
    plan: SeriesPlan,
    length: int,
    rng: np.random.Generator,
    context: TimeContext,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    if not plan.variables or len(plan.variables) <= 1:
        return values, anomaly, {}, {"enabled": False, "reason": "single_variable_plan"}

    primary = _primary_variable_name(plan)
    variable_columns: dict[str, np.ndarray] = {}
    for index, variable in enumerate(plan.variables):
        raw_name = str(variable.get("name", f"variable_{index + 1}"))
        name = _safe_column_name(raw_name, f"variable_{index + 1}")
        role = str(variable.get("role", "")).lower()
        if name == primary:
            variable_columns[name] = values.copy()
        else:
            variable_columns[name] = _generate_variable_series(name, variable, plan, length, rng, context)

    if primary not in variable_columns:
        variable_columns[primary] = values.copy()

    driver_event_report: dict[str, object] = {"enabled": False}
    if plan.anomaly_enabled and plan.anomaly_target == "driver":
        driver_name = str(plan.semantic_config.get("driver_name", "") or "")
        driver = _safe_column_name(driver_name, "") if driver_name else ""
        if not driver:
            for relationship in plan.relationships or []:
                candidate, target = _relationship_names(relationship)
                if candidate and target in {primary, "value"}:
                    driver = candidate
                    break
        if not driver:
            for name, variable in zip(variable_columns, plan.variables):
                if name != primary and str(variable.get("role", "")).lower() in {"driver", "exogenous", "input"}:
                    driver = name
                    break
        if driver in variable_columns:
            variable_columns[driver], driver_flags = add_anomalies(
                variable_columns[driver],
                rng,
                _multivariate_anomaly_config(plan),
            )
            anomaly = np.maximum(anomaly, driver_flags)
            driver_event_report = {
                "enabled": True,
                "driver": driver,
                "active_points": int(driver_flags.sum()),
                "observed_events": int(np.sum((driver_flags == 1) & (np.r_[0, driver_flags[:-1]] == 0))),
            }

    target = variable_columns[primary].copy()
    relation_reports: list[dict[str, object]] = []
    target_scale = max(float(np.std(target)), max(abs(float(np.mean(target))) * 0.08, 1.0))
    for relationship in plan.relationships or []:
        if not isinstance(relationship, dict):
            continue
        source, rel_target = _relationship_names(relationship)
        if not source or source not in variable_columns:
            continue
        if rel_target not in {primary, "value", ""}:
            continue
        lag = int(relationship.get("lag", relationship.get("lag_steps", 0)) or 0)
        effect_text = str(relationship.get("effect", relationship.get("relation", ""))).lower()
        sign = -1.0 if any(key in effect_text for key in ("negative", "drop", "decrease", "reduce")) else 1.0
        coefficient = relationship.get("coefficient")
        if coefficient is None:
            coefficient = 0.18 * target_scale
        coefficient = float(coefficient)
        driver = _lagged(variable_columns[source], lag)
        effect, operator = _relationship_effect(driver, target, relationship, sign, coefficient)
        target += effect
        relation_reports.append(
            {
                "source": source,
                "target": primary,
                "effect": effect_text or ("positive_lagged_effect" if sign > 0 else "negative_lagged_effect"),
                "operator": operator,
                "lag": lag,
                "coefficient": coefficient,
                "applied": True,
            }
        )

    variable_columns[primary] = target
    for name, column in variable_columns.items():
        frame_payload[name] = np.asarray(column, dtype=float)

    return target, anomaly, variable_columns, {
        "enabled": True,
        "primary_target": primary,
        "variables": [
            {
                "name": _safe_column_name(str(item.get("name", f"variable_{i + 1}")), f"variable_{i + 1}"),
                "role": item.get("role", "driver" if i else "target"),
                "unit": item.get("unit", "unknown"),
            }
            for i, item in enumerate(plan.variables)
        ],
        "relationships_applied": relation_reports,
        "driver_anomaly": driver_event_report,
    }


def _relationship_audit(frame_payload: dict[str, np.ndarray], plan: SeriesPlan) -> dict[str, object]:
    primary = _primary_variable_name(plan)
    target = np.asarray(frame_payload.get(primary, frame_payload.get("value", [])), dtype=float)
    reports: list[dict[str, object]] = []
    if len(target) < 8:
        return {"relationships": reports, "passed": True}
    for relationship in plan.relationships or []:
        if not isinstance(relationship, dict):
            continue
        source, rel_target = _relationship_names(relationship)
        if source not in frame_payload or rel_target not in {primary, "value", ""}:
            continue
        driver = np.asarray(frame_payload[source], dtype=float)
        expected_sign = -1 if "negative" in str(relationship.get("effect", "")).lower() else 1
        operator = str(relationship.get("operator") or relationship.get("relationship_operator") or relationship.get("type") or "linear_lag").lower()
        if operator == "event_trigger":
            threshold = float(relationship.get("threshold", float(np.quantile(driver, 0.9))))
            driver_for_corr = _trigger_mask(
                driver,
                threshold,
                str(relationship.get("trigger_op", "gte")),
            ).astype(float)
        else:
            driver_for_corr = driver
        best_lag, best_corr = 0, 0.0
        for lag in range(0, min(24, len(target) // 4) + 1):
            left = driver_for_corr[: len(driver_for_corr) - lag] if lag else driver_for_corr
            right = target[lag:] if lag else target
            if len(left) < 8 or float(np.std(left)) < 1e-9 or float(np.std(right)) < 1e-9:
                continue
            corr = float(np.corrcoef(left, right)[0, 1])
            if np.isfinite(corr) and abs(corr) > abs(best_corr):
                best_lag, best_corr = lag, corr
        sign_ok = best_corr == 0.0 or np.sign(best_corr) == expected_sign
        operator_check: dict[str, object] = {"operator": operator, "passed": True}
        if operator in {"threshold", "piecewise", "state_gate", "event_trigger"}:
            threshold = float(relationship.get("threshold", float(np.mean(driver))))
            trigger_op = str(relationship.get("trigger_op", "gte"))
            active = (
                _trigger_mask(driver, threshold, trigger_op)
                if operator == "event_trigger"
                else driver >= threshold
            )
            inactive = ~active
            if active.any() and inactive.any():
                active_mean = float(np.mean(target[active]))
                inactive_mean = float(np.mean(target[inactive]))
                direction_ok = (active_mean - inactive_mean) * expected_sign >= -1e-6
                operator_check.update(
                    {
                        "active_target_mean": active_mean,
                        "inactive_target_mean": inactive_mean,
                        "trigger_op": trigger_op if operator == "event_trigger" else "gte",
                        "direction_ok": bool(direction_ok),
                        "passed": bool(direction_ok),
                    }
                )
            else:
                operator_check.update({"passed": False, "reason": "threshold_does_not_split_driver_states"})
        elif operator == "saturation":
            midpoint = float(relationship.get("midpoint", float(np.mean(driver))))
            low = driver < midpoint
            high = driver >= midpoint
            if low.sum() >= 4 and high.sum() >= 4:
                low_slope = abs(_safe_corr(driver[low], target[low]))
                high_slope = abs(_safe_corr(driver[high], target[high]))
                operator_check.update(
                    {
                        "low_region_abs_corr": float(low_slope),
                        "high_region_abs_corr": float(high_slope),
                        "passed": bool(max(low_slope, high_slope) >= 0.03),
                    }
                )
        requires_operator_check = operator in {"threshold", "piecewise", "state_gate", "event_trigger", "saturation"}
        relation_passed = bool(
            sign_ok
            and (
                bool(operator_check["passed"])
                if requires_operator_check
                else abs(best_corr) >= 0.05
            )
        )
        reports.append(
            {
                "source": source,
                "target": primary,
                "expected_effect": relationship.get("effect", relationship.get("relation", "")),
                "operator": operator,
                "observed_best_lag": int(best_lag),
                "observed_correlation": float(best_corr),
                "sign_ok": bool(sign_ok),
                "operator_check": operator_check,
                "passed": relation_passed,
            }
        )
    return {"relationships": reports, "passed": all(item["passed"] for item in reports) if reports else True}


def _semantic_plan_for_execution(plan: SeriesPlan) -> SeriesPlan:
    if plan.semantic_type == "multivariate_lag" and (plan.variables or plan.relationships):
        payload = plan.to_dict()
        payload["semantic_type"] = "instantaneous"
        payload["semantic_config"] = {
            **dict(plan.semantic_config),
            "disabled_reason": "explicit_variables_relationships_drive_multivariate_generation",
        }
        return SeriesPlan.from_dict(payload)
    return plan


def _component_report_passed(report: dict[str, object]) -> bool:
    if not isinstance(report, dict) or not report:
        return True
    if report.get("status") == "REVISE":
        return False
    return all(
        not isinstance(component, dict) or component.get("status") != "REVISE"
        for component in report.get("components", [])
    )


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
    time_context = build_time_context(length, freq, start)
    component_workflow = plan.metadata.get("component_workflow")
    component_artifacts = None
    component_workflow_warning = None
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
            values, anomaly, component_artifacts = synthesize_component_base(
                plan, workflow, length, rng, freq=freq, start=start
            )
        except Exception as exc:
            component_workflow_warning = f"{exc.__class__.__name__}: {exc}"
            plan.metadata["component_workflow_error"] = component_workflow_warning
            if plan.metadata.get("cost_mode") == "strict" or component_workflow.get("strict"):
                raise RuntimeError(
                    f"Component workflow failed in strict mode: {component_workflow_warning}"
                ) from exc
            component_artifacts = None

    if component_artifacts is None:
        if plan.generator_type == "intermittent_event":
            values, anomaly = _generate_precipitation(plan, length, rng, time_context)
        elif plan.generator_type == "daylight_envelope":
            values, anomaly = _generate_solar(plan, length, rng, time_context)
        elif plan.generator_type == "smooth_environmental":
            values, anomaly = _generate_temperature(plan, length, rng, time_context)
        elif plan.generator_type == "count_process":
            values, anomaly = _generate_count_process(plan, length, rng, time_context)
        elif plan.generator_type == "bounded_utilization":
            values, anomaly = _generate_bounded_utilization(plan, length, rng, time_context)
        else:
            values, anomaly = _generate_cyclic_signal(plan, length, rng, time_context)

    semantic_plan = _semantic_plan_for_execution(plan)
    values, semantic_anomaly, semantic_columns = apply_semantic_process(values, semantic_plan, rng)
    anomaly = np.maximum(anomaly, semantic_anomaly)
    multivariate_columns: dict[str, np.ndarray] = {}
    multivariate_report: dict[str, object] = {"enabled": False}
    values, anomaly, multivariate_columns, multivariate_report = _apply_multivariate_plan(
        {},
        values,
        anomaly,
        plan,
        length,
        rng,
        time_context,
    )
    values, anomaly = _apply_reference_executable_targets(values, plan, anomaly)
    anomaly_flags = anomaly.astype(int)
    anomaly_starts = int(
        np.sum((anomaly_flags == 1) & (np.r_[0, anomaly_flags[:-1]] == 0))
    )
    anomaly_execution = {
        "enabled": bool(plan.anomaly_enabled),
        "requested_count": int(plan.anomaly_count),
        "active_points": int(anomaly_flags.sum()),
        "observed_events": anomaly_starts,
        "observed_fraction": float(anomaly_flags.mean()) if length else 0.0,
        "target": plan.anomaly_target,
        "kind": plan.anomaly_kind,
    }
    if component_artifacts is not None:
        report = component_artifacts.get("component_report", {})
        for component in report.get("components", []):
            if component.get("name") != "anomaly_intervention":
                continue
            component["stats"] = {
                "active_points": anomaly_execution["active_points"],
                "observed_events": anomaly_execution["observed_events"],
                "observed_fraction": anomaly_execution["observed_fraction"],
            }
            component.setdefault("checks", {})["execution"] = bool(
                anomaly_execution["active_points"]
            ) or not (plan.anomaly_enabled and plan.anomaly_count > 0)
            component["status"] = (
                "PASS" if all(component["checks"].values()) else "REVISE"
            )
    values, validation = validate_and_repair(values, plan, semantic_columns)
    if component_artifacts is not None:
        component_report = component_artifacts.get("component_report", {})
        component_passed = _component_report_passed(component_report)
        validation.setdefault("checks", {})["component_quality_report"] = component_passed
        if not component_passed:
            validation["passed"] = False
            validation.setdefault("warnings", []).append(
                {
                    "type": "component_quality_report",
                    "severity": "hard_warning",
                    "message": "Component workflow quality report requested revision.",
                    "issues": component_report.get("issues", []),
                }
            )
    if multivariate_report.get("enabled"):
        primary_name = _primary_variable_name(plan)
        multivariate_columns[primary_name] = values
        audit_payload = {"value": values, **multivariate_columns}
        relationship_audit = _relationship_audit(audit_payload, plan)
        validation.setdefault("checks", {})["multivariate_relationships"] = bool(
            relationship_audit.get("passed", True)
        )
        validation["passed"] = bool(validation.get("passed", True)) and bool(
            relationship_audit.get("passed", True)
        )
        multivariate_report["relationship_audit"] = relationship_audit
    if component_workflow_warning:
        validation.setdefault("warnings", []).append(
            {
                "type": "component_workflow_fallback",
                "severity": "hard_warning",
                "message": component_workflow_warning,
                "fallback": "legacy_generator",
            }
        )
        validation.setdefault("checks", {})["component_workflow_executed"] = False
        validation["passed"] = False

    payload = {
        "timestamp": time_context.index,
        "value": values.round(4),
        "anomaly": anomaly.astype(int),
    }
    for name, column in semantic_columns.items():
        payload[name] = np.asarray(column).round(4)
    for name, column in multivariate_columns.items():
        if name in payload:
            continue
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
    frame.attrs["anomaly_execution"] = anomaly_execution
    frame.attrs["multivariate_report"] = multivariate_report
    if component_artifacts is not None:
        frame.attrs["component_workflow"] = component_artifacts["workflow"]
        frame.attrs["component_report"] = component_artifacts["component_report"]
        frame.attrs["component_stats"] = component_artifacts["component_stats"]
    return frame
