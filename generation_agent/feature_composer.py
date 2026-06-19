from __future__ import annotations

from .planner import SeriesPlan


GENERATOR_TYPES = {
    "cyclic_signal",
    "intermittent_event",
    "daylight_envelope",
    "smooth_environmental",
    "count_process",
    "bounded_utilization",
}


def build_feature_plan(
    description: str,
    domain: str,
    unit: str,
    generator_type: str,
    baseline: float,
    trend_slope: float = 0.0,
    daily_amplitude: float = 0.0,
    weekly_enabled: bool = False,
    weekly_amplitude: float = 0.0,
    seasonal_amplitude: float = 0.0,
    heat_effect: float = 0.0,
    noise_sigma: float = 1.0,
    lower_bound: float | None = 0.0,
    anomaly_count: int = 0,
    anomaly_magnitude: float = 3.0,
    anomaly_width: int = 1,
    anomaly_kind: str = "spike",
    event_probability: float = 0.18,
    mean_duration: float = 5.0,
    intensity_shape: float = 1.4,
    intensity_scale: float = 5.0,
    dry_spell_bias: float = 0.5,
    storm_probability: float = 0.08,
    storm_multiplier: float = 3.0,
    sunrise_hour: float = 6.0,
    sunset_hour: float = 19.0,
    cloud_probability: float = 0.18,
    cloud_drop_min: float = 0.35,
    cloud_drop_max: float = 0.8,
    inertia: float = 0.88,
    peak_hour: float = 15.0,
    morning_peak: float = 8.0,
    evening_peak: float = 18.0,
    overdispersion: float = 1.35,
    upper_bound: float = 100.0,
    batch_hour: float = 2.0,
    batch_probability: float = 0.0,
    semantic_type: str = "instantaneous",
    semantic_config: dict | None = None,
    output_constraints: dict | None = None,
    variables: list[dict] | None = None,
    relationships: list[dict] | None = None,
    anomaly_enabled: bool = False,
    anomaly_severity: str = "medium",
    anomaly_target: str = "value",
    rationale: str = "",
) -> SeriesPlan:
    if generator_type not in GENERATOR_TYPES:
        generator_type = "cyclic_signal"

    domain_params = {}
    if generator_type == "intermittent_event":
        domain_params = {
            "event_probability": event_probability,
            "mean_duration": mean_duration,
            "intensity_shape": intensity_shape,
            "intensity_scale": intensity_scale,
            "dry_spell_bias": dry_spell_bias,
            "storm_probability": storm_probability,
            "storm_multiplier": storm_multiplier,
        }
    elif generator_type == "daylight_envelope":
        domain_params = {
            "sunrise_hour": sunrise_hour,
            "sunset_hour": sunset_hour,
            "cloud_probability": cloud_probability,
            "cloud_drop_min": cloud_drop_min,
            "cloud_drop_max": cloud_drop_max,
        }
    elif generator_type == "smooth_environmental":
        domain_params = {"inertia": inertia, "peak_hour": peak_hour}
    elif generator_type == "count_process":
        domain_params = {
            "morning_peak": morning_peak,
            "evening_peak": evening_peak,
            "overdispersion": overdispersion,
        }
    elif generator_type == "bounded_utilization":
        domain_params = {
            "upper_bound": upper_bound,
            "batch_hour": batch_hour,
            "batch_probability": batch_probability,
        }

    return SeriesPlan(
        domain=domain,
        generator_type=generator_type,
        unit=unit,
        baseline=baseline,
        trend_slope=trend_slope,
        daily_amplitude=daily_amplitude,
        weekly_enabled=weekly_enabled,
        weekly_amplitude=weekly_amplitude,
        seasonal_amplitude=seasonal_amplitude,
        heat_effect=heat_effect,
        noise_sigma=noise_sigma,
        anomaly_count=anomaly_count,
        anomaly_magnitude=anomaly_magnitude,
        anomaly_width=anomaly_width,
        anomaly_kind=anomaly_kind,
        lower_bound=lower_bound,
        domain_params=domain_params,
        semantic_type=semantic_type,
        semantic_config=semantic_config or {},
        output_constraints=output_constraints or {},
        variables=variables or [],
        relationships=relationships or [],
        anomaly_enabled=anomaly_enabled,
        anomaly_severity=anomaly_severity,
        anomaly_target=anomaly_target,
        metadata={
            "planner": "feature_composer",
            "description": description,
            "feature_rationale": rationale,
        },
    )
