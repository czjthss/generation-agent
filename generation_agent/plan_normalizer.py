from __future__ import annotations

from dataclasses import replace
from typing import Any

from .planner import SeriesPlan


EXECUTABLE_CONSTRAINT_KEYS = {
    "lower_bound",
    "upper_bound",
    "nonnegative",
    "monotonic",
    "integer",
    "conservation",
}

VALID_GENERATORS = {
    "cyclic_signal",
    "intermittent_event",
    "daylight_envelope",
    "smooth_environmental",
    "count_process",
    "bounded_utilization",
}

VALID_SEMANTICS = {
    "instantaneous",
    "cumulative",
    "stock_flow",
    "regime_switching",
    "random_walk",
    "decay_recovery",
    "saturation_growth",
    "multivariate_lag",
}

SEMANTIC_CONFIG_KEYS = {
    "instantaneous": set(),
    "cumulative": {"initial_value", "allow_negative_increment"},
    "stock_flow": {"initial_value", "inflow_scale", "outflow_scale"},
    "regime_switching": {"states", "transition_probability", "initial_state"},
    "random_walk": {
        "initial_value",
        "drift",
        "volatility",
        "innovation_distribution",
        "innovation_df",
        "innovation_ar",
        "innovation_autocorrelation",
        "regime_volatility",
        "volatility_switch_probability",
        "high_volatility_multiplier",
    },
    "decay_recovery": {"equilibrium", "recovery_rate", "impulse"},
    "saturation_growth": {"initial_value", "capacity", "growth_rate"},
    "multivariate_lag": {"lag", "coefficient", "residual_sigma", "driver_name"},
}

EXECUTABLE_DOMAIN_PARAM_KEYS = {
    "batch_hour",
    "batch_probability",
    "cloud_drop_max",
    "cloud_drop_min",
    "cloud_probability",
    "daily_phase",
    "dry_spell_bias",
    "evening_peak",
    "event_probability",
    "inertia",
    "intensity_scale",
    "intensity_shape",
    "mean_duration",
    "morning_peak",
    "overdispersion",
    "peak_hour",
    "shift_amplitude",
    "shift_end_hour",
    "shift_start_hour",
    "shift_weekend_factor",
    "storm_multiplier",
    "storm_probability",
    "sunrise_hour",
    "sunset_hour",
    "upper_bound",
    "weekend_factor",
}




def _coerce_float(value: Any, default: float) -> float:
    try:
        if isinstance(value, bool):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        if isinstance(value, bool):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "on"}:
            return True
        if normalized in {"false", "no", "0", "off"}:
            return False
    return default


def _coerce_numeric_states(value: Any, default: list[float]) -> tuple[list[float], bool]:
    if not isinstance(value, (list, tuple)) or not value:
        return list(default), True
    numeric: list[float] = []
    converted_labels = False
    for index, item in enumerate(value):
        try:
            if isinstance(item, bool):
                raise ValueError
            numeric.append(float(item))
        except (TypeError, ValueError):
            converted_labels = True
            if len(value) == 1:
                numeric.append(1.0)
            else:
                numeric.append(0.7 + 0.6 * index / max(1, len(value) - 1))
    numeric = [max(0.0, item) for item in numeric]
    if not numeric:
        return list(default), True
    return numeric, converted_labels


def _sanitize_semantic_config(
    semantic_type: str,
    semantic_config: dict[str, Any],
    defaults: dict[str, Any],
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    sanitized = dict(semantic_config)
    notes: list[str] = []
    moved: dict[str, Any] = {}

    if semantic_type == "cumulative":
        sanitized["initial_value"] = _coerce_float(sanitized.get("initial_value"), float(defaults.get("initial_value", 0.0)))
        sanitized["allow_negative_increment"] = _coerce_bool(
            sanitized.get("allow_negative_increment"),
            bool(defaults.get("allow_negative_increment", False)),
        )
    elif semantic_type == "stock_flow":
        for key in ("initial_value", "inflow_scale", "outflow_scale"):
            sanitized[key] = _coerce_float(sanitized.get(key), float(defaults.get(key, 1.0)))
    elif semantic_type == "regime_switching":
        states, converted = _coerce_numeric_states(
            sanitized.get("states"), list(defaults.get("states", [0.4, 1.0, 1.5]))
        )
        if converted:
            moved["states"] = sanitized.get("states")
            notes.append("coerced_regime_state_labels_to_numeric_levels")
        sanitized["states"] = states
        sanitized["transition_probability"] = min(
            max(_coerce_float(sanitized.get("transition_probability"), float(defaults.get("transition_probability", 0.05))), 0.0),
            1.0,
        )
        initial = _coerce_int(sanitized.get("initial_state"), int(defaults.get("initial_state", min(1, len(states) - 1))))
        sanitized["initial_state"] = min(max(initial, 0), max(0, len(states) - 1))
    elif semantic_type == "random_walk":
        for key in ("initial_value", "drift", "volatility"):
            sanitized[key] = _coerce_float(sanitized.get(key), float(defaults.get(key, 0.0)))
        sanitized["volatility"] = max(sanitized["volatility"], 1e-6)
    elif semantic_type == "decay_recovery":
        for key in ("equilibrium", "recovery_rate", "impulse"):
            sanitized[key] = _coerce_float(sanitized.get(key), float(defaults.get(key, 0.0)))
        sanitized["recovery_rate"] = min(max(sanitized["recovery_rate"], 0.0), 1.0)
    elif semantic_type == "saturation_growth":
        for key in ("initial_value", "capacity", "growth_rate"):
            sanitized[key] = _coerce_float(sanitized.get(key), float(defaults.get(key, 1.0)))
        sanitized["capacity"] = max(sanitized["capacity"], 1e-6)
        sanitized["initial_value"] = min(max(sanitized["initial_value"], 1e-6), sanitized["capacity"])
        sanitized["growth_rate"] = max(sanitized["growth_rate"], 0.0)
    elif semantic_type == "multivariate_lag":
        sanitized["lag"] = max(0, _coerce_int(sanitized.get("lag"), int(defaults.get("lag", 2))))
        sanitized["coefficient"] = _coerce_float(sanitized.get("coefficient"), float(defaults.get("coefficient", 0.7)))
        sanitized["residual_sigma"] = max(
            _coerce_float(sanitized.get("residual_sigma"), float(defaults.get("residual_sigma", 0.0))),
            0.0,
        )
        driver = sanitized.get("driver_name", defaults.get("driver_name", "driver"))
        sanitized["driver_name"] = str(driver or "driver")
    return sanitized, notes, moved

def _default_semantic_config(semantic_type: str, plan: SeriesPlan) -> dict[str, Any]:
    defaults: dict[str, dict[str, Any]] = {
        "cumulative": {"initial_value": 0.0, "allow_negative_increment": False},
        "stock_flow": {
            "initial_value": max(float(plan.baseline), 0.0),
            "inflow_scale": 1.0,
            "outflow_scale": 0.85,
        },
        "regime_switching": {
            "states": [0.4, 1.0, 1.5],
            "transition_probability": 0.05,
            "initial_state": 1,
        },
        "random_walk": {
            "initial_value": max(float(plan.baseline), 1.0),
            "drift": float(plan.trend_slope),
            "volatility": max(float(plan.noise_sigma), 1e-6),
        },
        "decay_recovery": {
            "equilibrium": float(plan.baseline),
            "recovery_rate": 0.12,
            "impulse": float(plan.daily_amplitude or plan.baseline),
        },
        "saturation_growth": {
            "initial_value": max(float(plan.baseline), 1.0),
            "capacity": max(float(plan.baseline) * 10.0, 100.0),
            "growth_rate": 0.06,
        },
        "multivariate_lag": {
            "lag": 2,
            "coefficient": 0.7,
            "residual_sigma": max(float(plan.noise_sigma), 0.0),
            "driver_name": "driver",
        },
    }
    return defaults.get(semantic_type, {})




def _strong_domain_prior(description: str) -> SeriesPlan | None:
    try:
        from .planner import heuristic_plan

        prior = heuristic_plan(description)
    except Exception:
        return None
    if prior.domain == "generic" and prior.generator_type == "cyclic_signal":
        return None
    return prior


def _should_apply_domain_prior(plan: SeriesPlan, prior: SeriesPlan) -> bool:
    """Only use heuristic priors as a last-resort executable fallback.

    The LLM owns domain and mechanism decisions. Heuristic priors are retained as
    compatibility guards, but they should not overwrite a valid LLM plan merely
    because a keyword matched a known domain.
    """
    return plan.generator_type not in VALID_GENERATORS

def normalize_plan_for_execution(plan: SeriesPlan, description: str) -> SeriesPlan:
    """Prune non-executable LLM plan details while preserving domain intent.

    The LLM may describe business constraints or process details that are useful as
    rationale but not executable by the local numerical kernel. Keep those details in
    metadata, and expose only supported mathematical constraints to Reflection,
    generation, validation, and repair.
    """
    metadata = dict(plan.metadata)
    normalizations: list[str] = []

    domain_prior = _strong_domain_prior(description)
    prior_overrides: dict[str, Any] = {}
    if domain_prior is not None:
        metadata["domain_prior_reference"] = {
            "suggested_generator_type": domain_prior.generator_type,
            "suggested_domain": domain_prior.domain,
            "applied": bool(_should_apply_domain_prior(plan, domain_prior)),
            "policy": "heuristic priors warn or rescue unsupported generators; they do not override valid LLM plans",
        }
    if domain_prior is not None and _should_apply_domain_prior(plan, domain_prior):
        prior_overrides = {
            "domain": domain_prior.domain,
            "generator_type": domain_prior.generator_type,
            "unit": domain_prior.unit,
            "baseline": domain_prior.baseline,
            "trend_slope": domain_prior.trend_slope,
            "daily_amplitude": domain_prior.daily_amplitude,
            "weekly_enabled": domain_prior.weekly_enabled,
            "weekly_amplitude": domain_prior.weekly_amplitude,
            "seasonal_amplitude": domain_prior.seasonal_amplitude,
            "heat_effect": domain_prior.heat_effect,
            "noise_sigma": domain_prior.noise_sigma,
            "lower_bound": domain_prior.lower_bound,
            "domain_params": dict(domain_prior.domain_params or {}),
        }
        metadata["domain_prior_correction"] = {
            "from_generator_type": plan.generator_type,
            "to_generator_type": domain_prior.generator_type,
            "domain": domain_prior.domain,
            "reason": "LLM requested an unsupported generator; heuristic prior selected the closest executable family",
        }
        normalizations.append("applied_domain_prior_for_unsupported_generator")
        plan = replace(plan, **prior_overrides)

    raw_domain_params = dict(plan.domain_params or {})
    domain_params = {
        key: value
        for key, value in raw_domain_params.items()
        if key in EXECUTABLE_DOMAIN_PARAM_KEYS
    }
    unsupported_domain_params = {
        key: value
        for key, value in raw_domain_params.items()
        if key not in EXECUTABLE_DOMAIN_PARAM_KEYS
    }
    if unsupported_domain_params:
        metadata["non_executable_domain_details"] = unsupported_domain_params
        normalizations.append("moved_non_executable_domain_params_to_metadata")

    raw_constraints = dict(plan.output_constraints or {})
    executable_constraints = {
        key: value for key, value in raw_constraints.items() if key in EXECUTABLE_CONSTRAINT_KEYS
    }
    dropped_constraints = {
        key: value for key, value in raw_constraints.items() if key not in EXECUTABLE_CONSTRAINT_KEYS
    }
    if dropped_constraints:
        metadata["non_executable_business_constraints"] = dropped_constraints
        normalizations.append("moved_non_executable_output_constraints_to_metadata")

    generator_type = plan.generator_type
    semantic_type = plan.semantic_type
    semantic_config = dict(plan.semantic_config or {})

    if generator_type not in VALID_GENERATORS:
        metadata["requested_generator_type"] = generator_type
        generator_type = "cyclic_signal"
        normalizations.append("replaced_unsupported_generator_with_cyclic_signal")
    if semantic_type not in VALID_SEMANTICS:
        metadata["requested_semantic_type"] = semantic_type
        semantic_type = "instantaneous"
        semantic_config = {}
        normalizations.append("replaced_unsupported_semantic_with_instantaneous")

    allowed_semantic_keys = SEMANTIC_CONFIG_KEYS.get(semantic_type, set())
    unsupported_semantic_config = {
        key: value
        for key, value in semantic_config.items()
        if key not in allowed_semantic_keys
    }
    semantic_config = {
        key: value
        for key, value in semantic_config.items()
        if key in allowed_semantic_keys
    }
    if unsupported_semantic_config:
        metadata["non_executable_semantic_details"] = unsupported_semantic_config
        normalizations.append("moved_non_executable_semantic_config_to_metadata")

    defaults = _default_semantic_config(semantic_type, plan)
    missing_semantic_fields = []
    for key, value in defaults.items():
        if semantic_config.get(key) in (None, [], ""):
            semantic_config[key] = value
            missing_semantic_fields.append(key)
    if missing_semantic_fields:
        metadata["semantic_defaults_applied"] = missing_semantic_fields
        normalizations.append("completed_executable_semantic_config")

    semantic_config, semantic_type_notes, moved_semantic_values = _sanitize_semantic_config(
        semantic_type, semantic_config, defaults
    )
    if semantic_type_notes:
        normalizations.extend(semantic_type_notes)
    if moved_semantic_values:
        metadata.setdefault("non_executable_semantic_details", {}).update(moved_semantic_values)

    if semantic_type == "instantaneous" and semantic_config:
        metadata["non_executable_semantic_details"] = semantic_config
        semantic_config = {}
        normalizations.append("removed_semantic_config_from_instantaneous_output")

    if semantic_type == "cumulative":
        executable_constraints.setdefault("nonnegative", True)
        if "upper_bound" in executable_constraints:
            metadata.setdefault("non_executable_business_constraints", {})[
                "cumulative_upper_bound"
            ] = executable_constraints.pop("upper_bound")
            normalizations.append("removed_upper_bound_from_cumulative_output")
        if not semantic_config.get("allow_negative_increment", False):
            executable_constraints.setdefault("monotonic", "nondecreasing")
    elif semantic_type == "stock_flow":
        executable_constraints.setdefault("nonnegative", True)
        executable_constraints.setdefault("conservation", True)
    elif executable_constraints.get("conservation"):
        metadata.setdefault("non_executable_business_constraints", {})["conservation"] = (
            executable_constraints.pop("conservation")
        )
        normalizations.append("limited_conservation_constraint_to_stock_flow")

    if generator_type in {"intermittent_event", "count_process", "daylight_envelope"}:
        executable_constraints.setdefault("nonnegative", True)
    if generator_type == "count_process":
        executable_constraints.setdefault("integer", True)
    if generator_type == "bounded_utilization":
        executable_constraints.setdefault("lower_bound", 0.0)
        executable_constraints.setdefault(
            "upper_bound", float(domain_params.get("upper_bound", 100.0))
        )

    monotonic = executable_constraints.get("monotonic")
    if monotonic not in (None, True, "nondecreasing", "increasing"):
        metadata.setdefault("non_executable_business_constraints", {})["monotonic"] = (
            executable_constraints.pop("monotonic")
        )
        normalizations.append("moved_unsupported_monotonic_mode_to_metadata")

    if semantic_type != "cumulative" and "upper_bound" in domain_params and "upper_bound" not in executable_constraints:
        executable_constraints["upper_bound"] = domain_params["upper_bound"]
        normalizations.append("promoted_domain_upper_bound_to_output_constraints")
    if plan.lower_bound is not None and "lower_bound" not in executable_constraints:
        executable_constraints["lower_bound"] = plan.lower_bound

    if executable_constraints.get("nonnegative"):
        executable_constraints["lower_bound"] = max(
            float(executable_constraints.get("lower_bound", 0.0)), 0.0
        )
    lower = executable_constraints.get("lower_bound")
    upper = executable_constraints.get("upper_bound")
    if lower is not None and upper is not None and float(upper) < float(lower):
        metadata.setdefault("non_executable_business_constraints", {})[
            "invalid_upper_bound"
        ] = upper
        executable_constraints.pop("upper_bound")
        normalizations.append("removed_upper_bound_below_lower_bound")

    if normalizations:
        metadata["execution_normalization"] = normalizations
    return replace(
        plan,
        generator_type=generator_type,
        semantic_type=semantic_type,
        semantic_config=semantic_config,
        domain_params=domain_params,
        output_constraints=executable_constraints,
        metadata=metadata,
    )
