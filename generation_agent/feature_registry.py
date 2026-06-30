from __future__ import annotations

from copy import deepcopy
from typing import Any


FEATURE_REGISTRY: dict[str, dict[str, Any]] = {
    "baseline": {"value_roles": ["additive_level"], "params": {"level": "float"}},
    "linear_trend": {"value_roles": ["additive_deviation"], "params": {"slope": "float"}},
    "cyclic_signal": {"value_roles": ["additive_deviation"], "params": {"amplitude": "float", "phase": "float"}},
    "working_day_shift": {"value_roles": ["additive_positive_component"], "params": {"start_hour": "float", "end_hour": "float", "amplitude": "float", "weekend_factor": "float"}},
    "weekly_gate": {"value_roles": ["additive_deviation", "multiplicative_envelope"], "params": {"amplitude": "float", "weekend_factor": "float"}},
    "seasonal_cycle": {"value_roles": ["additive_deviation"], "params": {"amplitude": "float", "period_days": "float", "phase": "float"}},
    "heat_index_effect": {"value_roles": ["additive_deviation", "additive_positive_component"], "params": {"amplitude": "float"}},
    "gaussian_peak": {"value_roles": ["additive_positive_component"], "params": {"center": "float", "width": "float", "amplitude": "float"}},
    "scheduled_pulse": {"value_roles": ["additive_positive_component"], "params": {"hour": "float", "probability": "float", "amplitude": "float", "width": "float"}},
    "event_mask": {"value_roles": ["mask"], "params": {"event_probability": "float", "mean_duration": "float"}},
    "gamma_intensity": {"value_roles": ["multiplicative_positive_component"], "params": {"shape": "float", "scale": "float"}},
    "storm_multiplier": {"value_roles": ["multiplicative_boost"], "params": {"probability": "float", "multiplier": "float"}},
    "daylight_envelope": {"value_roles": ["multiplicative_envelope"], "params": {"sunrise_hour": "float", "sunset_hour": "float", "amplitude": "float"}},
    "cloud_drop": {"value_roles": ["multiplicative_attenuation", "multiplicative_factor"], "params": {"probability": "float", "drop_min": "float", "drop_max": "float"}},
    "inertia_filter": {"value_roles": ["dynamic_filter"], "params": {"inertia": "float"}},
    "negative_binomial": {"value_roles": ["sampling_distribution"], "params": {"overdispersion": "float"}},
    "noise": {"value_roles": ["additive_deviation"], "params": {"sigma": "float"}},
    "anomaly_strategy": {"value_roles": ["bounded_additive_intervention"], "params": {"enabled": "bool", "count": "int", "kind": "string", "width": "int", "target": "string", "severity": "string"}},
}

MECHANISM_TEMPLATES: dict[str, dict[str, Any]] = {
    "level_trend_noise": {"components": ["baseline", "linear_trend", "noise"], "use_when": "persistent continuous processes"},
    "calendar_operating_process": {"components": ["baseline", "working_day_shift", "weekly_gate", "scheduled_pulse", "noise"], "use_when": "activity changes by operating windows or calendars"},
    "sparse_event_process": {"components": ["event_mask", "gamma_intensity", "storm_multiplier"], "use_when": "zero-inflated events with durations and intensities"},
    "daylight_limited_process": {"components": ["daylight_envelope", "cloud_drop", "noise"], "use_when": "availability is limited by daylight or a physical window"},
    "smooth_inertial_process": {"components": ["baseline", "cyclic_signal", "inertia_filter", "noise"], "use_when": "continuous variables with persistence"},
    "count_arrival_process": {"components": ["baseline", "gaussian_peak", "negative_binomial"], "use_when": "nonnegative discrete arrivals"},
    "bounded_capacity_process": {"components": ["baseline", "cyclic_signal", "scheduled_pulse", "noise"], "use_when": "bounded utilization or capacity ratios"},
    "driver_response_process": {"components": ["baseline", "heat_index_effect", "seasonal_cycle", "noise"], "use_when": "external drivers change target magnitude"},
}


def feature_capability_manifest() -> dict[str, Any]:
    return {
        "feature_families": deepcopy(FEATURE_REGISTRY),
        "mechanism_templates": deepcopy(MECHANISM_TEMPLATES),
        "relationship_operators": ["linear_lag", "threshold", "piecewise", "saturation", "state_gate", "event_trigger"],
        "component_contract": {
            "required_fields": ["name", "role", "component_semantic", "value_role", "sign_or_bounds", "time_scale_behavior", "statistical_shape", "feature_family", "params"],
            "rule": "feature_family must be in feature_families unless intentionally unsupported for review failure",
        },
    }
