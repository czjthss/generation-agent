from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

import numpy as np
import pandas as pd

from .dataset_scenario_agent import DatasetScenario
from .planner import SeriesPlan


DIVERSITY_STRENGTHS = {"off", "low", "medium", "high"}


def validate_diversity_strength(value: str | None) -> str:
    strength = (value or "medium").strip().lower()
    if strength not in DIVERSITY_STRENGTHS:
        raise ValueError("diversity_strength must be off, low, medium, or high")
    return strength


def _scale_for_strength(strength: str) -> float:
    return {"off": 0.0, "low": 0.12, "medium": 0.28, "high": 0.45}[strength]


def _positive(value: float, minimum: float = 0.0) -> float:
    return float(max(minimum, value))


def _bounded_probability(value: float, minimum: float = 0.0, maximum: float = 0.95) -> float:
    return float(min(maximum, max(minimum, value)))


def _multiply(value: float, rng: np.random.Generator, scale: float, minimum: float = 0.0) -> float:
    if scale <= 0:
        return float(value)
    return _positive(float(value) * float(rng.lognormal(mean=0.0, sigma=scale)), minimum)


def diversify_plan(
    plan: SeriesPlan,
    scenario: DatasetScenario,
    series_index: int,
    variant_index: int,
    seed: int | None,
    strength: str = "medium",
) -> SeriesPlan:
    """Create a dataset-only variant of a plan without changing its semantic contract."""
    strength = validate_diversity_strength(strength)
    scale = _scale_for_strength(strength)
    if scale == 0.0:
        return plan

    rng_seed = None if seed is None else int(seed) + series_index * 1009 + variant_index * 9176
    rng = np.random.default_rng(rng_seed)
    params = dict(plan.domain_params)
    metadata = dict(plan.metadata)
    mechanism_variant = "parameter_perturbation"
    metadata.update(
        {
            "dataset_diversity": {
                "strength": strength,
                "scenario_description": scenario.description,
                "scenario_diversity_axis": scenario.diversity_axis,
                "variant_index": variant_index,
            }
        }
    )

    baseline = _multiply(plan.baseline, rng, scale, minimum=0.0)
    daily_amplitude = _multiply(plan.daily_amplitude, rng, scale * 1.2, minimum=0.0)
    weekly_amplitude = _multiply(plan.weekly_amplitude, rng, scale * 1.2, minimum=0.0)
    seasonal_amplitude = _multiply(plan.seasonal_amplitude, rng, scale * 1.4, minimum=0.0)
    heat_effect = _multiply(plan.heat_effect, rng, scale * 1.4, minimum=0.0)
    noise_sigma = _multiply(plan.noise_sigma, rng, scale * 1.5, minimum=0.0)
    anomaly_enabled = bool(plan.anomaly_enabled)
    anomaly_count = plan.anomaly_count
    anomaly_width = plan.anomaly_width
    anomaly_magnitude = plan.anomaly_magnitude
    trend_slope = float(plan.trend_slope)
    if trend_slope:
        trend_slope *= float(rng.uniform(1.0 - scale, 1.0 + scale))
    elif rng.random() < 0.3 * scale:
        trend_slope = float(rng.normal(0.0, max(1.0, baseline * 0.03 * scale)))

    if plan.generator_type == "intermittent_event":
        if strength in {"medium", "high"} and rng.random() < 0.55:
            mechanism_variant = str(rng.choice(["convective_bursts", "long_dry_spells", "persistent_light_events"]))
            if mechanism_variant == "convective_bursts":
                params["storm_probability"] = max(float(params.get("storm_probability", 0.08)), 0.22)
                params["storm_multiplier"] = max(float(params.get("storm_multiplier", 3.0)), 4.5)
                params["mean_duration"] = max(1.0, float(params.get("mean_duration", 5.0)) * 0.65)
            elif mechanism_variant == "long_dry_spells":
                params["dry_spell_bias"] = min(0.95, max(float(params.get("dry_spell_bias", 0.6)), 0.78))
                params["event_probability"] = min(float(params.get("event_probability", 0.18)), 0.12)
            else:
                params["storm_probability"] = min(float(params.get("storm_probability", 0.08)), 0.04)
                params["mean_duration"] = max(float(params.get("mean_duration", 5.0)), 8.0)
        params["event_probability"] = _bounded_probability(
            float(params.get("event_probability", 0.18)) + rng.normal(0.0, 0.12 * scale),
            0.01,
            0.75,
        )
        params["dry_spell_bias"] = _bounded_probability(
            float(params.get("dry_spell_bias", 0.6)) + rng.normal(0.0, 0.18 * scale),
            0.05,
            0.95,
        )
        params["mean_duration"] = _multiply(float(params.get("mean_duration", 5.0)), rng, scale, 1.0)
        params["intensity_shape"] = _multiply(float(params.get("intensity_shape", 1.4)), rng, scale, 0.3)
        params["intensity_scale"] = _multiply(float(params.get("intensity_scale", 5.0)), rng, scale * 1.3, 0.1)
        params["storm_probability"] = _bounded_probability(
            float(params.get("storm_probability", 0.08)) + rng.normal(0.0, 0.08 * scale),
            0.0,
            0.55,
        )
        params["storm_multiplier"] = _multiply(float(params.get("storm_multiplier", 3.0)), rng, scale, 1.0)
    elif plan.generator_type == "daylight_envelope":
        if strength in {"medium", "high"} and rng.random() < 0.5:
            mechanism_variant = str(rng.choice(["clear_sky", "broken_clouds", "persistent_cloud_cover"]))
            if mechanism_variant == "clear_sky":
                params["cloud_probability"] = min(float(params.get("cloud_probability", 0.18)), 0.06)
            elif mechanism_variant == "persistent_cloud_cover":
                params["cloud_probability"] = max(float(params.get("cloud_probability", 0.18)), 0.45)
                params["cloud_drop_min"] = min(float(params.get("cloud_drop_min", 0.35)), 0.2)
        sunrise_shift = float(rng.normal(0.0, 1.2 * scale))
        sunset_shift = float(rng.normal(0.0, 1.2 * scale))
        params["sunrise_hour"] = float(np.clip(float(params.get("sunrise_hour", 6.0)) + sunrise_shift, 4.0, 9.0))
        params["sunset_hour"] = float(np.clip(float(params.get("sunset_hour", 19.0)) + sunset_shift, 15.0, 22.0))
        if params["sunset_hour"] <= params["sunrise_hour"] + 6.0:
            params["sunset_hour"] = params["sunrise_hour"] + 6.0
        params["cloud_probability"] = _bounded_probability(
            float(params.get("cloud_probability", 0.18)) + rng.normal(0.0, 0.18 * scale),
            0.0,
            0.8,
        )
        params["cloud_drop_min"] = float(np.clip(float(params.get("cloud_drop_min", 0.35)) + rng.normal(0, 0.12 * scale), 0.05, 0.8))
        params["cloud_drop_max"] = float(np.clip(float(params.get("cloud_drop_max", 0.8)) + rng.normal(0, 0.12 * scale), params["cloud_drop_min"], 0.98))
    elif plan.generator_type == "smooth_environmental":
        params["inertia"] = float(np.clip(float(params.get("inertia", 0.88)) + rng.normal(0.0, 0.18 * scale), 0.35, 0.98))
        params["peak_hour"] = float((float(params.get("peak_hour", 15.0)) + rng.normal(0.0, 4.0 * scale)) % 24)
    elif plan.generator_type == "count_process":
        if strength == "high" and rng.random() < 0.45:
            mechanism_variant = str(rng.choice(["commute_double_peak", "event_day_single_peak", "flat_background"]))
            if mechanism_variant == "event_day_single_peak":
                params["morning_peak"] = float(rng.uniform(10.0, 16.0))
                params["evening_peak"] = params["morning_peak"]
            elif mechanism_variant == "flat_background":
                daily_amplitude *= 0.35
        params["morning_peak"] = float((float(params.get("morning_peak", 8.0)) + rng.normal(0.0, 3.0 * scale)) % 24)
        params["evening_peak"] = float((float(params.get("evening_peak", 18.0)) + rng.normal(0.0, 3.0 * scale)) % 24)
        params["overdispersion"] = _multiply(float(params.get("overdispersion", 1.35)), rng, scale, 1.0)
    elif plan.generator_type == "bounded_utilization":
        if strength in {"medium", "high"} and rng.random() < 0.5:
            mechanism_variant = str(rng.choice(["batch_heavy", "interactive_workload", "near_capacity"]))
            if mechanism_variant == "batch_heavy":
                params["batch_probability"] = max(float(params.get("batch_probability", 0.35)), 0.65)
            elif mechanism_variant == "near_capacity":
                baseline = max(baseline, float(params.get("upper_bound", 100.0)) * 0.72)
        params["batch_hour"] = float((float(params.get("batch_hour", 2.0)) + rng.normal(0.0, 5.0 * scale)) % 24)
        params["batch_probability"] = _bounded_probability(
            float(params.get("batch_probability", 0.35)) + rng.normal(0.0, 0.25 * scale),
            0.0,
            0.9,
        )
        params["upper_bound"] = _multiply(float(params.get("upper_bound", 100.0)), rng, scale * 0.4, 1.0)
    else:
        if strength in {"medium", "high"}:
            mechanism_variant = str(
                rng.choice(
                    [
                        "single_operating_window",
                        "double_operating_window",
                        "near_continuous_operation",
                        "driver_response_dominated",
                        "scheduled_event_pulses",
                        "strong_calendar_gate",
                        "phase_shifted_operation",
                        "trend_dominated",
                        "seasonal_dominated",
                    ]
                )
            )
            if mechanism_variant == "single_operating_window":
                params["shift_start_hour"] = float(rng.uniform(7.0, 10.0))
                params["shift_end_hour"] = float(rng.uniform(15.0, 20.0))
                params["shift_amplitude"] = max(daily_amplitude * 0.7, baseline * 0.10)
                params["weekend_factor"] = float(np.clip(rng.normal(0.50, 0.12), 0.15, 0.85))
            elif mechanism_variant == "double_operating_window":
                params["shift_start_hour"] = float(rng.uniform(5.5, 8.0))
                params["shift_end_hour"] = float(rng.uniform(20.0, 23.5))
                params["shift_amplitude"] = max(daily_amplitude, baseline * 0.16)
                params["weekend_factor"] = float(np.clip(rng.normal(0.68, 0.10), 0.35, 0.95))
            elif mechanism_variant == "near_continuous_operation":
                daily_amplitude *= 0.35
                weekly_amplitude *= 0.25
                baseline *= 1.12
                params["weekend_factor"] = float(np.clip(rng.normal(0.92, 0.05), 0.75, 1.05))
            elif mechanism_variant == "driver_response_dominated":
                heat_effect = max(heat_effect, baseline * float(rng.uniform(0.06, 0.20)))
                seasonal_amplitude = max(seasonal_amplitude, baseline * float(rng.uniform(0.03, 0.12)))
            elif mechanism_variant == "scheduled_event_pulses":
                params["batch_hour"] = float(rng.uniform(0.0, 23.0))
                params["batch_probability"] = float(np.clip(rng.normal(0.35, 0.18), 0.05, 0.8))
                params["shift_amplitude"] = max(params.get("shift_amplitude", 0.0), baseline * float(rng.uniform(0.04, 0.12)))
            elif mechanism_variant == "strong_calendar_gate":
                params["weekend_factor"] = float(np.clip(rng.normal(0.45, 0.12), 0.15, 0.8))
            elif mechanism_variant == "trend_dominated":
                daily_amplitude *= 0.45
                trend_slope = float(rng.normal(0.0, max(1.0, baseline * 0.18)))
            elif mechanism_variant == "seasonal_dominated":
                daily_amplitude *= 0.55
                seasonal_amplitude = max(seasonal_amplitude, baseline * float(rng.uniform(0.08, 0.25)))
        params["daily_phase"] = float(params.get("daily_phase", rng.uniform(-6.0, 6.0) * scale))
        params["weekend_factor"] = float(
            params.get("weekend_factor", np.clip(rng.normal(0.72, 0.25 * scale), 0.35, 1.05))
        )

    if plan.semantic_type == "cumulative":
        params["daily_phase"] = float(params.get("daily_phase", rng.uniform(-4.0, 4.0) * scale))
    if plan.anomaly_enabled and plan.anomaly_count > 0:
        anomaly_count = max(1, int(round(plan.anomaly_count * rng.uniform(0.75, 1.35))))
        anomaly_width = max(1, int(round(plan.anomaly_width * rng.uniform(0.75, 1.8))))
        anomaly_magnitude = _multiply(plan.anomaly_magnitude, rng, scale * 0.7, 0.1)
    metadata.setdefault("dataset_diversity", {})["mechanism_variant"] = mechanism_variant

    return replace(
        plan,
        baseline=baseline,
        trend_slope=trend_slope,
        daily_amplitude=daily_amplitude,
        weekly_enabled=bool(plan.weekly_enabled or (strength == "high" and rng.random() < 0.25)),
        weekly_amplitude=weekly_amplitude,
        seasonal_amplitude=seasonal_amplitude,
        heat_effect=heat_effect,
        noise_sigma=noise_sigma,
        anomaly_enabled=anomaly_enabled,
        anomaly_count=anomaly_count,
        anomaly_magnitude=anomaly_magnitude,
        anomaly_width=anomaly_width,
        domain_params=params,
        metadata=metadata,
    )


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or len(right) < 3:
        return 0.0
    if np.std(left) < 1e-9 or np.std(right) < 1e-9:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _sample_values(values: np.ndarray, max_points: int = 1024) -> np.ndarray:
    if len(values) <= max_points:
        return values.astype(float)
    positions = np.linspace(0, len(values) - 1, max_points).round().astype(int)
    return values[positions].astype(float)


def shape_signature(frame: pd.DataFrame, max_points: int = 1024) -> dict[str, Any]:
    values = frame["value"].astype(float).to_numpy()
    sample = _sample_values(values, max_points=max_points)
    finite = sample[np.isfinite(sample)]
    if len(finite) == 0:
        finite = np.array([0.0], dtype=float)
    diffs = np.diff(finite) if len(finite) > 1 else np.array([0.0])
    centered = finite - float(np.mean(finite))
    scaled = centered / (float(np.std(centered)) + 1e-9)
    x = np.linspace(0.0, 1.0, len(finite))
    fft = np.abs(np.fft.rfft(scaled)) if len(finite) > 3 else np.array([0.0])
    spectral_strength = 0.0
    if len(fft) > 2 and float(np.sum(fft[1:])) > 1e-9:
        spectral_strength = float(np.max(fft[1:]) / np.sum(fft[1:]))

    quantiles = np.quantile(finite, [0.05, 0.25, 0.5, 0.75, 0.95])
    feature_vector = np.array(
        [
            float(np.mean(finite)),
            float(np.std(finite)),
            float(np.min(finite)),
            float(np.max(finite)),
            float(np.mean(values == 0.0)),
            float(np.mean(frame["anomaly"].to_numpy() > 0)) if "anomaly" in frame else 0.0,
            *[float(item) for item in quantiles],
            _safe_corr(finite[:-1], finite[1:]) if len(finite) > 2 else 0.0,
            _safe_corr(finite[:-24], finite[24:]) if len(finite) > 48 else 0.0,
            _safe_corr(x, finite) if len(finite) > 2 else 0.0,
            float(np.mean(np.abs(diffs))),
            float(np.std(diffs)),
            spectral_strength,
        ],
        dtype=float,
    )
    return {
        "sample": scaled.astype(float),
        "features": feature_vector,
        "summary": {
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "zero_fraction": float(np.mean(values == 0.0)),
            "lag1_autocorrelation": _safe_corr(finite[:-1], finite[1:]) if len(finite) > 2 else 0.0,
            "lag24_autocorrelation": _safe_corr(finite[:-24], finite[24:]) if len(finite) > 48 else 0.0,
            "trend_correlation": _safe_corr(x, finite) if len(finite) > 2 else 0.0,
            "spectral_strength": spectral_strength,
        },
    }


def signature_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_features = np.asarray(left["features"], dtype=float)
    right_features = np.asarray(right["features"], dtype=float)
    feature_scale = np.maximum(np.abs(left_features), np.abs(right_features))
    feature_scale[feature_scale < 1.0] = 1.0
    normalized_distance = float(np.linalg.norm((left_features - right_features) / feature_scale))
    feature_similarity = math.exp(-normalized_distance / 3.0)

    left_sample = np.asarray(left["sample"], dtype=float)
    right_sample = np.asarray(right["sample"], dtype=float)
    size = min(len(left_sample), len(right_sample))
    if size >= 3:
        sample_similarity = abs(_safe_corr(left_sample[:size], right_sample[:size]))
    else:
        sample_similarity = 0.0
    return float(0.55 * feature_similarity + 0.45 * sample_similarity)


def max_similarity(signature: dict[str, Any], accepted: list[dict[str, Any]]) -> dict[str, Any]:
    if not accepted:
        return {"max_similarity": 0.0, "nearest_series_id": None, "accepted": True}
    scores = [signature_similarity(signature, item["signature"]) for item in accepted]
    best_index = int(np.argmax(scores))
    return {
        "max_similarity": float(scores[best_index]),
        "nearest_series_id": accepted[best_index]["series_id"],
        "accepted": True,
    }
