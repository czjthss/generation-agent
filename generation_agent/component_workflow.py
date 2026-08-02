from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .features import (
    gaussian_noise,
    linear_trend,
)
from .planner import SeriesPlan
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


@dataclass
class InputProfile:
    description: str
    length: int
    freq: str
    start: str
    reference_profile: dict[str, Any] | None = None
    generation_mode: str = "sequence"


@dataclass
class VariableProfile:
    name: str
    domain: str
    unit: str
    variable_semantic: str
    value_kind: str
    constraints: list[str] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class MechanismComponent:
    name: str
    role: str
    component_semantic: str
    value_role: str
    sign_constraint: str
    time_scale_behavior: str
    statistical_shape: str
    feature_family: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentWorkflow:
    input_profile: InputProfile
    variable_profile: VariableProfile
    components: list[MechanismComponent]
    composition: dict[str, Any]
    final_transform: str
    anomaly_target: str
    validator_rules: list[str]
    agent_trace: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_profile": asdict(self.input_profile),
            "variable_profile": asdict(self.variable_profile),
            "components": [asdict(component) for component in self.components],
            "composition": self.composition,
            "final_transform": self.final_transform,
            "anomaly_target": self.anomaly_target,
            "validator_rules": self.validator_rules,
            "agent_trace": self.agent_trace,
        }


def _sampling_level(freq: str) -> str:
    value = freq.lower()
    if "min" in value or value.endswith("s"):
        return "sub_hourly"
    if value in {"h", "1h"} or value.endswith("h"):
        return "hourly"
    if value in {"d", "1d"} or value.endswith("d"):
        return "daily"
    if value.startswith("w"):
        return "weekly"
    if value.startswith("m") or value in {"ms", "me"}:
        return "monthly"
    return "unknown"


def _value_kind(plan: SeriesPlan) -> str:
    if plan.semantic_type == "cumulative":
        return "cumulative_nonnegative"
    if plan.semantic_type == "stock_flow":
        return "stock_balance"
    if plan.generator_type == "count_process":
        return "nonnegative_count"
    if plan.generator_type == "bounded_utilization":
        return "bounded_ratio"
    if plan.generator_type == "intermittent_event":
        return "zero_inflated_nonnegative"
    return "continuous"


def _variable_constraints(plan: SeriesPlan) -> list[str]:
    constraints: list[str] = []
    if plan.lower_bound is not None or plan.output_constraints.get("nonnegative"):
        constraints.append("lower_bound")
    if plan.output_constraints.get("nonnegative") or plan.lower_bound == 0:
        constraints.append("nonnegative")
    if "upper_bound" in plan.output_constraints or "upper_bound" in plan.domain_params:
        constraints.append("upper_bound")
    if plan.output_constraints.get("monotonic"):
        constraints.append(f"monotonic_{plan.output_constraints['monotonic']}")
    if plan.output_constraints.get("conservation"):
        constraints.append("conservation")
    return constraints


def _base_variable_profile(plan: SeriesPlan) -> VariableProfile:
    return VariableProfile(
        name=str(plan.metadata.get("observable", plan.domain or "value")),
        domain=plan.domain,
        unit=plan.unit,
        variable_semantic=plan.semantic_type,
        value_kind=_value_kind(plan),
        constraints=_variable_constraints(plan),
        relationships=list(plan.relationships),
    )


def _component(
    name: str,
    role: str,
    component_semantic: str,
    value_role: str,
    sign_constraint: str,
    time_scale_behavior: str,
    statistical_shape: str,
    feature_family: str,
    **params: Any,
) -> MechanismComponent:
    return MechanismComponent(
        name=name,
        role=role,
        component_semantic=component_semantic,
        value_role=value_role,
        sign_constraint=sign_constraint,
        time_scale_behavior=time_scale_behavior,
        statistical_shape=statistical_shape,
        feature_family=feature_family,
        params={key: value for key, value in params.items() if value is not None},
    )



def _anomaly_component(plan: SeriesPlan) -> MechanismComponent | None:
    if not (plan.anomaly_enabled or plan.anomaly_count > 0):
        return None
    return _component(
        "anomaly_intervention",
        "LLM-selected abnormal intervention applied by the local numerical kernel",
        "sparse_operational_anomaly",
        "bounded_additive_intervention",
        "signed",
        "short abnormal windows",
        "sparse_spike_drop_or_shift",
        "anomaly_strategy",
        enabled=plan.anomaly_enabled,
        count=plan.anomaly_count,
        kind=plan.anomaly_kind,
        width=plan.anomaly_width,
        target=plan.anomaly_target,
        severity=plan.anomaly_severity,
    )


def _append_common_components(components: list[MechanismComponent], plan: SeriesPlan) -> list[MechanismComponent]:
    anomaly = _anomaly_component(plan)
    if anomaly is not None and not any(component.name == anomaly.name for component in components):
        components = [*components, anomaly]
    return components


def _feature_family_from_llm_component(component: dict[str, Any], plan: SeriesPlan) -> str:
    explicit = str(component.get("feature_family", "")).strip()
    if explicit:
        return explicit
    text = " ".join(
        str(component.get(key, "")).lower()
        for key in ("name", "role", "component_semantic", "time_scale_behavior", "statistical_shape")
    )
    if any(token in text for token in ("daylight", "solar", "pv", "irradiance", "sun")):
        return "daylight_envelope"
    if any(token in text for token in ("cloud", "attenuation", "shading")):
        return "cloud_drop"
    if any(token in text for token in ("event", "arrival", "rain", "precip", "sparse", "storm")):
        return "event_mask" if "intensity" not in text else "gamma_intensity"
    if any(token in text for token in ("shift", "operating window", "workday", "calendar")):
        return "working_day_shift"
    if any(token in text for token in ("weekly", "weekend")):
        return "weekly_gate"
    if any(token in text for token in ("season", "annual")):
        return "seasonal_cycle"
    if any(token in text for token in ("heat", "temperature response", "cooling")):
        return "heat_index_effect"
    if any(token in text for token in ("peak", "rush")):
        return "gaussian_peak"
    if any(token in text for token in ("cycle", "periodic", "diurnal", "daily")):
        return "cyclic_signal"
    if any(token in text for token in ("trend", "drift")):
        return "linear_trend"
    if any(token in text for token in ("noise", "residual", "jitter")):
        return "noise"
    if any(token in text for token in ("baseline", "level", "base")):
        return "baseline"
    return "unknown_component"


def _component_value_role(component: dict[str, Any]) -> str:
    explicit = str(component.get("value_role", "")).strip()
    if explicit:
        return explicit
    text = f"{component.get('role', '')} {component.get('component_semantic', '')}".lower()
    if any(token in text for token in ("cloud", "attenuation", "shading")):
        return "multiplicative_attenuation"
    if any(token in text for token in ("mask", "gate", "envelope")):
        return "multiplicative_envelope"
    if any(token in text for token in ("noise", "residual", "deviation")):
        return "additive_deviation"
    if any(token in text for token in ("baseline", "base", "level")):
        return "additive_level"
    return "additive_positive_component"


def _llm_component_workflow(
    plan: SeriesPlan,
    input_profile: InputProfile,
    variable_profile: VariableProfile,
    composition: dict[str, Any],
    final_transform: str,
) -> ComponentWorkflow | None:
    mechanism = plan.metadata.get("mechanism_planning_agent")
    if not isinstance(mechanism, dict):
        return None
    targets = mechanism.get("target_variables")
    if not isinstance(targets, list) or not targets:
        return None
    target = next((item for item in targets if isinstance(item, dict) and item.get("components")), targets[0])
    raw_components = target.get("components") if isinstance(target, dict) else None
    if not isinstance(raw_components, list) or not raw_components:
        return None
    components: list[MechanismComponent] = []
    for index, item in enumerate(raw_components):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or f"llm_component_{index + 1}")
        components.append(
            _component(
                name=name,
                role=str(item.get("role", "LLM-planned mechanism component")),
                component_semantic=str(item.get("component_semantic", "mechanism_component")),
                value_role=_component_value_role(item),
                sign_constraint=str(item.get("sign_or_bounds") or item.get("sign_constraint") or "signed"),
                time_scale_behavior=str(item.get("time_scale_behavior", "unspecified")),
                statistical_shape=str(item.get("statistical_shape", "unspecified")),
                feature_family=_feature_family_from_llm_component(item, plan),
                **(item.get("params", {}) if isinstance(item.get("params"), dict) else {}),
            )
        )
    if not components:
        return None
    target_composition = target.get("composition") if isinstance(target.get("composition"), dict) else {}
    if target_composition.get("operator"):
        composition = {**composition, "operator": target_composition.get("operator")}
    if target_composition.get("final_transform"):
        final_transform = str(target_composition.get("final_transform"))
    components = _append_common_components(components, plan)
    return ComponentWorkflow(
        input_profile=input_profile,
        variable_profile=variable_profile,
        components=components,
        composition=composition,
        final_transform=final_transform,
        anomaly_target=plan.anomaly_target,
        validator_rules=sorted(set([*_variable_constraints(plan), "llm_component_dsl_executed"])),
        agent_trace={
            "requirement_understanding_agent": "provided request, variables, time configuration, and constraints",
            "mechanism_planning_agent": "provided mechanism components consumed by the executable component DSL",
            "parameter_compiler_agent": "compiled the mechanism components into feature-family execution parameters",
            "component_source": "llm_mechanism_plan",
        },
    )


def build_component_workflow(
    description: str,
    plan: SeriesPlan,
    length: int,
    freq: str,
    start: str,
    reference_profile: dict[str, Any] | None = None,
    generation_mode: str = "sequence",
) -> ComponentWorkflow:
    """Build the compact multi-agent state used by the new generation workflow.

    The upstream LLM agents enrich the SeriesPlan before this step. This layer makes
    their component workflow explicit, auditable, and executable by deterministic
    numerical feature generators.
    """
    input_profile = InputProfile(
        description=description,
        length=length,
        freq=freq,
        start=start,
        reference_profile=reference_profile,
        generation_mode=generation_mode,
    )
    variable_profile = _base_variable_profile(plan)
    sampling = _sampling_level(freq)
    components: list[MechanismComponent] = []
    composition: dict[str, Any] = {"operator": "add", "clip": {}}
    final_transform = "identity" if plan.semantic_type == "instantaneous" else plan.semantic_type

    if plan.lower_bound is not None:
        composition["clip"]["lower"] = plan.lower_bound
    if "upper_bound" in plan.domain_params:
        composition["clip"]["upper"] = plan.domain_params["upper_bound"]
    if "upper_bound" in plan.output_constraints:
        composition["clip"]["upper"] = plan.output_constraints["upper_bound"]

    llm_workflow = _llm_component_workflow(
        plan,
        input_profile=input_profile,
        variable_profile=variable_profile,
        composition=composition,
        final_transform=final_transform,
    )
    if llm_workflow is not None:
        return llm_workflow

    params = plan.domain_params
    if plan.generator_type == "intermittent_event":
        components = [
            _component(
                "dry_spell_state",
                "controls gaps between event clusters",
                "state_gate",
                "mask",
                "nonnegative",
                "event clusters visible at the selected sampling level",
                "sparse_binary_runs",
                "dry_spell_process",
                dry_spell_bias=params.get("dry_spell_bias", 0.5),
            ),
            _component(
                "event_arrival",
                "activates rainfall or sparse event periods",
                "event_mask",
                "mask",
                "nonnegative",
                "intermittent arrivals",
                "zero_inflated",
                "event_mask",
                event_probability=params.get("event_probability", 0.18),
                mean_duration=params.get("mean_duration", 5.0),
            ),
            _component(
                "event_intensity",
                "sets event magnitude",
                "positive_intensity",
                "multiplicative_positive_component",
                "nonnegative",
                "right-skewed event values",
                "right_skewed_long_tail",
                "gamma_intensity",
                shape=params.get("intensity_shape", 1.4),
                scale=params.get("intensity_scale", 5.0),
            ),
            _component(
                "storm_boost",
                "adds rare high-intensity events",
                "sparse_extreme_event",
                "multiplicative_boost",
                "nonnegative",
                "rare bursts",
                "spiky_tail",
                "storm_multiplier",
                probability=params.get("storm_probability", 0.08),
                multiplier=params.get("storm_multiplier", 3.0),
            ),
        ]
        composition = {"operator": "event_mask_times_intensity", "clip": {"lower": 0.0}}
    elif plan.generator_type == "daylight_envelope":
        components = [
            _component(
                "daylight_envelope",
                "physical availability window",
                "availability_mask",
                "multiplicative_envelope",
                "nonnegative",
                "zero outside daylight hours" if sampling in {"hourly", "sub_hourly"} else "aggregated daylight effect",
                "bounded_daily_arch",
                "daylight_envelope",
                sunrise_hour=params.get("sunrise_hour", 6.0),
                sunset_hour=params.get("sunset_hour", 19.0),
                amplitude=plan.daily_amplitude,
            ),
            _component(
                "cloud_attenuation",
                "weather-driven production drop",
                "random_attenuation",
                "multiplicative_factor",
                "nonnegative",
                "short cloudy intervals",
                "occasional_downward_pulses",
                "cloud_drop",
                probability=params.get("cloud_probability", 0.18),
                drop_min=params.get("cloud_drop_min", 0.35),
                drop_max=params.get("cloud_drop_max", 0.8),
            ),
            _component(
                "measurement_noise",
                "small observation noise",
                "signed_noise",
                "additive_deviation",
                "signed",
                "local variation",
                "zero_mean_noise",
                "noise",
                sigma=plan.noise_sigma,
            ),
        ]
        composition = {"operator": "multiply_then_add_noise", "clip": {"lower": 0.0}}
    elif plan.generator_type == "smooth_environmental":
        components = [
            _component(
                "baseline_level",
                "typical environmental level",
                "baseline_level",
                "additive_level",
                "signed" if plan.lower_bound is None else "nonnegative",
                "persistent level",
                "low_frequency",
                "baseline",
                level=plan.baseline,
            ),
            _component(
                "diurnal_target",
                "daily environmental forcing",
                "periodic_target",
                "additive_deviation",
                "signed",
                "daily cycle visible at hourly resolution",
                "smooth_periodic",
                "cyclic_signal",
                period=24,
                amplitude=plan.daily_amplitude,
                peak_hour=params.get("peak_hour", 15.0),
            ),
            _component(
                "thermal_inertia",
                "slows changes between observations",
                "autoregressive_smoothing",
                "dynamic_filter",
                "signed",
                "smooth persistence",
                "high_autocorrelation",
                "inertia_filter",
                inertia=params.get("inertia", 0.88),
            ),
            _component(
                "weather_noise",
                "small local perturbation",
                "signed_noise",
                "additive_deviation",
                "signed",
                "short-term variation",
                "low_variance_noise",
                "noise",
                sigma=plan.noise_sigma,
            ),
        ]
    elif plan.generator_type == "count_process":
        components = [
            _component(
                "base_rate",
                "ordinary demand arrival rate",
                "count_rate",
                "additive_positive_component",
                "nonnegative",
                "persistent demand",
                "nonnegative_rate",
                "baseline",
                level=plan.baseline,
            ),
            _component(
                "morning_peak",
                "morning demand concentration",
                "peak_rate",
                "additive_positive_component",
                "nonnegative",
                "within-day peak",
                "localized_peak",
                "gaussian_peak",
                center=params.get("morning_peak", 8.0),
                amplitude=plan.daily_amplitude * 0.8,
            ),
            _component(
                "evening_peak",
                "evening demand concentration",
                "peak_rate",
                "additive_positive_component",
                "nonnegative",
                "within-day peak",
                "localized_peak",
                "gaussian_peak",
                center=params.get("evening_peak", 18.0),
                amplitude=plan.daily_amplitude,
            ),
            _component(
                "overdispersed_sampling",
                "integer count realization",
                "count_noise",
                "sampling_distribution",
                "nonnegative",
                "random count draw",
                "overdispersed_count",
                "negative_binomial",
                overdispersion=params.get("overdispersion", 1.35),
            ),
        ]
        composition = {"operator": "rate_then_count_sample", "clip": {"lower": 0.0}}
    elif plan.generator_type == "bounded_utilization":
        components = [
            _component(
                "background_utilization",
                "normal background usage",
                "baseline_level",
                "additive_positive_component",
                "nonnegative",
                "persistent level",
                "bounded_level",
                "baseline",
                level=plan.baseline,
            ),
            _component(
                "workload_cycle",
                "regular workload pattern",
                "periodic_deviation",
                "additive_deviation",
                "signed",
                "daily utilization rhythm",
                "cyclic",
                "cyclic_signal",
                period=24,
                amplitude=plan.daily_amplitude,
                phase=params.get("daily_phase", 0.0),
            ),
            _component(
                "batch_job",
                "scheduled job spikes",
                "scheduled_event",
                "additive_positive_component",
                "nonnegative",
                "short batch windows",
                "sparse_positive_pulses",
                "scheduled_pulse",
                hour=params.get("batch_hour", 2.0),
                probability=params.get("batch_probability", 0.35),
            ),
            _component(
                "utilization_noise",
                "random usage jitter",
                "signed_noise",
                "additive_deviation",
                "signed",
                "short-term variation",
                "zero_mean_noise",
                "noise",
                sigma=plan.noise_sigma,
            ),
        ]
        composition = {"operator": "add_then_clip", "clip": {"lower": 0.0, "upper": params.get("upper_bound", 100.0)}}
    else:
        components = [
            _component(
                "baseline_level",
                "persistent minimum or central level",
                "baseline_level",
                "additive_level",
                "nonnegative" if plan.lower_bound is not None and plan.lower_bound >= 0 else "signed",
                "persistent level",
                "low_frequency",
                "baseline",
                level=plan.baseline,
            ),
            _component(
                "trend",
                "slow directional change",
                "trend_component",
                "additive_deviation",
                "signed",
                "low-frequency trend",
                "monotone_or_flat_drift",
                "linear_trend",
                slope=plan.trend_slope,
            ),
            _component(
                "daily_cycle",
                "within-day operating rhythm",
                "periodic_deviation",
                "additive_deviation",
                "signed",
                "visible if hourly/sub-hourly",
                "cyclic",
                "cyclic_signal",
                period=24,
                amplitude=plan.daily_amplitude,
                phase=params.get("daily_phase", 0.0),
            ),
            _component(
                "weekly_gate",
                "workday/weekend effect",
                "calendar_gate",
                "multiplicative_or_additive_calendar_effect",
                "signed",
                "weekly calendar structure",
                "piecewise_calendar",
                "weekly_gate",
                enabled=plan.weekly_enabled,
                amplitude=plan.weekly_amplitude,
                weekend_factor=params.get("weekend_factor", 0.72),
            ),
            _component(
                "weather_response",
                "temperature or season driven demand",
                "threshold_or_seasonal_response",
                "additive_deviation",
                "signed",
                "seasonal/afternoon response",
                "smooth_response",
                "heat_index_effect",
                amplitude=plan.heat_effect,
            ),
            _component(
                "operating_window",
                "optional calendar or shift operating window when the plan provides operating hours",
                "calendar_operating_window",
                "additive_positive_component",
                "nonnegative",
                "time-of-day or workday-gated activity",
                "piecewise_positive_plateau",
                "working_day_shift",
                start_hour=params.get("shift_start_hour"),
                end_hour=params.get("shift_end_hour"),
                amplitude=params.get("shift_amplitude"),
                weekend_factor=params.get("shift_weekend_factor", params.get("weekend_factor")),
            ) if {"shift_start_hour", "shift_end_hour", "shift_amplitude"}.issubset(params) else None,
            _component(
                "scheduled_pulse",
                "optional recurring pulse process when the plan provides a pulse hour",
                "scheduled_event",
                "additive_positive_component",
                "nonnegative",
                "short recurring windows",
                "sparse_positive_pulses",
                "scheduled_pulse",
                hour=params.get("batch_hour"),
                probability=params.get("batch_probability"),
            ) if "batch_hour" in params or "batch_probability" in params else None,
            _component(
                "residual_noise",
                "unexplained local variation",
                "signed_noise",
                "additive_deviation",
                "signed",
                "short-term variation",
                "zero_mean_noise",
                "noise",
                sigma=plan.noise_sigma,
            ),
        ]
        if plan.seasonal_amplitude:
            components.insert(
                4,
                _component(
                    "seasonal_cycle",
                    "longer seasonal variation",
                    "periodic_deviation",
                    "additive_deviation",
                    "signed",
                    "visible only when length spans enough horizon",
                    "slow_cycle",
                    "seasonal_cycle",
                    amplitude=plan.seasonal_amplitude,
                ),
            )
    validator_rules = list(variable_profile.constraints)
    validator_rules.extend(["component_contribution_reasonable", "no_component_dominates_unless_declared"])
    if plan.generator_type == "intermittent_event":
        validator_rules.extend(["zero_inflated", "event_intensity_nonnegative"])
    if plan.generator_type == "smooth_environmental":
        validator_rules.append("smooth_autocorrelated")
    if plan.generator_type == "bounded_utilization":
        validator_rules.append("bounded_utilization")
    components = [component for component in components if component is not None]
    components = _append_common_components(components, plan)

    return ComponentWorkflow(
        input_profile=input_profile,
        variable_profile=variable_profile,
        components=components,
        composition=composition,
        final_transform=final_transform,
        anomaly_target=plan.anomaly_target,
        validator_rules=sorted(set(validator_rules)),
        agent_trace={
            "requirement_understanding_agent": "understood request, reference priors, variables, time configuration, and constraints",
            "mechanism_planning_agent": "decomposed variables into mechanism components and quality expectations",
            "parameter_compiler_agent": "mapped mechanisms into executable feature families, semantics, anomalies, and validators",
        },
    )


def _hour(length: int, context: TimeContext | None = None) -> np.ndarray:
    if context is not None:
        return context.hour_of_day
    return np.arange(length, dtype=float) % 24.0


def _component_stats(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        finite = np.array([0.0])
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "zero_fraction": float(np.mean(np.asarray(values) == 0.0)),
    }


def _component_is_optional(component: MechanismComponent) -> bool:
    text = f"{component.name} {component.role} {component.component_semantic} {component.value_role}".lower()
    return any(token in text for token in ("optional", "noise", "residual", "anomaly", "jitter"))


def _component_is_mandatory(component: MechanismComponent) -> bool:
    text = f"{component.name} {component.role} {component.component_semantic} {component.value_role}".lower()
    return any(token in text for token in ("mandatory", "required", "essential", "must"))


def _quality_report(
    component_values: dict[str, np.ndarray],
    workflow: ComponentWorkflow,
    composed: np.ndarray,
) -> dict[str, Any]:
    total_std = float(np.std(composed)) or 1.0
    component_reports = []
    issues: list[str] = []
    is_llm_dsl = workflow.agent_trace.get("component_source") == "llm_mechanism_plan"
    feature_families = {str(component.feature_family).lower() for component in workflow.components}
    for component in workflow.components:
        values = component_values.get(component.name, np.zeros_like(composed))
        stats = _component_stats(values)
        contribution = float(np.std(values) / total_std) if total_std else 0.0
        checks = {
            "semantic": True,
            "time_behavior": True,
            "statistical_shape": True,
            "contribution": contribution <= 3.0 or component.value_role in {"additive_level", "multiplicative_envelope"},
            "executed": True,
        }
        family = str(component.feature_family).lower()
        if family == "unknown_component":
            checks["executed"] = False
            issues.append(f"{component.name} uses an unsupported feature family")
        if family == "storm_multiplier" and not {"event_mask", "gamma_intensity"}.issubset(feature_families):
            checks["executed"] = False
            issues.append(f"{component.name} has no event mask and intensity to multiply")
        if component.value_role == "multiplicative_attenuation" or (
            is_llm_dsl and family == "cloud_drop"
        ):
            factor = 1.0 + np.asarray(values, dtype=float)
            if np.any(factor < -1e-6) or np.any(factor > 1.0 + 1e-6):
                checks["semantic"] = False
                issues.append(f"{component.name} violates multiplicative attenuation factor bounds")
        elif family == "cloud_drop" and component.value_role == "multiplicative_factor":
            if stats["max"] > 1e-6:
                checks["semantic"] = False
                issues.append(f"{component.name} should be a nonpositive attenuation contribution")
        elif component.sign_constraint == "nonnegative" and stats["min"] < -1e-6:
            checks["semantic"] = False
            issues.append(f"{component.name} violates nonnegative component semantics")
        if (
            family not in {"noise", "gaussian_noise", "residual_noise", "anomaly_strategy"}
            and not _component_is_optional(component)
            and (is_llm_dsl or _component_is_mandatory(component))
            and abs(stats["mean"]) <= 1e-9
            and stats["std"] <= 1e-9
        ):
            checks["executed"] = False
            issues.append(f"{component.name} has zero contribution despite being a required component")
        if component.component_semantic in {"event_mask", "state_gate"} and stats["zero_fraction"] < 0.1:
            checks["statistical_shape"] = False
            issues.append(f"{component.name} is not sparse enough for its event semantics")
        if not checks["contribution"]:
            issues.append(f"{component.name} dominates the composed series")
        component_reports.append(
            {
                "name": component.name,
                "feature_family": component.feature_family,
                "component_semantic": component.component_semantic,
                "stats": stats,
                "std_share": contribution,
                "checks": checks,
                "status": "PASS" if all(checks.values()) else "REVISE",
            }
        )
    return {
        "status": "PASS" if not issues else "REVISE",
        "issues": issues,
        "components": component_reports,
    }


def _clip_bounds(workflow: ComponentWorkflow) -> tuple[float | None, float | None]:
    clip = workflow.composition.get("clip", {}) if isinstance(workflow.composition, dict) else {}
    lower = clip.get("lower")
    upper = clip.get("upper")
    return (
        float(lower) if lower is not None else None,
        float(upper) if upper is not None else None,
    )


def _is_additive_workflow(workflow: ComponentWorkflow) -> bool:
    operator = str(workflow.composition.get("operator", "")).lower()
    if "add" in operator or "sum" in operator:
        return True
    additive_roles = {
        "additive_level",
        "additive_deviation",
        "additive_positive_component",
        "bounded_additive_intervention",
    }
    return any(component.value_role in additive_roles for component in workflow.components)


def _enforce_additive_component_bounds(
    values: np.ndarray,
    component_values: dict[str, np.ndarray],
    workflow: ComponentWorkflow,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Budget additive components against declared bounds before validation.

    This is intentionally domain-neutral: any additive process with declared lower
    or upper support uses the same mechanism, so quality fixes do not depend on a
    single prompt or domain name.
    """
    if not _is_additive_workflow(workflow):
        return values, component_values
    lower, upper = _clip_bounds(workflow)
    if lower is None and upper is None:
        return values, component_values

    adjusted = {name: array.astype(float, copy=True) for name, array in component_values.items()}
    components_by_name = {component.name: component for component in workflow.components}
    base_names = [
        name
        for name, component in components_by_name.items()
        if name in adjusted and component.value_role == "additive_level"
    ]
    positive_names = [
        name
        for name, component in components_by_name.items()
        if name in adjusted and component.value_role == "additive_positive_component"
    ]
    signed_names = [
        name
        for name, component in components_by_name.items()
        if name in adjusted and component.value_role in {"additive_deviation", "bounded_additive_intervention"}
    ]

    if not base_names:
        return values, component_values

    base = np.sum([adjusted[name] for name in base_names], axis=0)
    positive = np.sum([np.maximum(adjusted[name], 0.0) for name in positive_names], axis=0) if positive_names else np.zeros_like(values)
    if upper is not None and positive_names:
        headroom = np.maximum(float(upper) - base, 1e-9)
        overload = positive > headroom
        if np.any(overload):
            scale = np.ones_like(values, dtype=float)
            scale[overload] = headroom[overload] / np.maximum(positive[overload], 1e-9)
            for name in positive_names:
                adjusted[name] = np.maximum(adjusted[name], 0.0) * scale
            positive = np.sum([adjusted[name] for name in positive_names], axis=0)

    signed = np.sum([adjusted[name] for name in signed_names], axis=0) if signed_names else np.zeros_like(values)
    if lower is not None:
        signed = np.maximum(signed, float(lower) - (base + positive))
    if upper is not None:
        signed = np.minimum(signed, float(upper) - (base + positive))
    if signed_names:
        original_signed = np.sum([adjusted[name] for name in signed_names], axis=0)
        delta = signed - original_signed
        target = next((name for name in signed_names if "noise" in name or "residual" in name), signed_names[-1])
        adjusted[target] = adjusted[target] + delta

    recomposed = np.sum(list(adjusted.values()), axis=0) if adjusted else values
    if lower is not None:
        recomposed = np.maximum(recomposed, float(lower))
    if upper is not None:
        recomposed = np.minimum(recomposed, float(upper))
    return recomposed, adjusted


def _component_array_from_family(
    component: MechanismComponent,
    plan: SeriesPlan,
    length: int,
    rng: np.random.Generator,
    context: TimeContext,
) -> np.ndarray:
    params = {**plan.domain_params, **component.params}
    family = str(component.feature_family).lower()
    hour = context.hour_of_day
    if family in {"baseline", "base_level", "level"}:
        return np.full(length, float(params.get("level", plan.baseline)), dtype=float)
    if family in {"linear_trend", "trend"}:
        return linear_trend(length, float(params.get("slope", plan.trend_slope)))
    if family in {"cyclic_signal", "positive_daily_cycle", "daily_cycle"}:
        return daily_load_shape(
            context,
            amplitude=float(params.get("amplitude", plan.daily_amplitude)),
            phase=float(params.get("phase", params.get("daily_phase", 0.0))),
        )
    if family == "working_day_shift":
        start_hour = float(params.get("start_hour", params.get("shift_start_hour", 8.0)))
        end_hour = float(params.get("end_hour", params.get("shift_end_hour", 18.0)))
        if start_hour <= end_hour:
            in_window = (hour >= start_hour) & (hour <= end_hour)
        else:
            in_window = (hour >= start_hour) | (hour <= end_hour)
        gate = working_day_gate_from_time(
            context,
            weekend_factor=float(params.get("weekend_factor", params.get("shift_weekend_factor", 0.72))),
        )
        return float(params.get("amplitude", params.get("shift_amplitude", plan.daily_amplitude))) * in_window.astype(float) * gate
    if family == "weekly_gate":
        return weekly_cycle_from_time(
            context,
            amplitude=float(params.get("amplitude", plan.weekly_amplitude)),
            phase=float(params.get("phase", -0.8)),
        )
    if family == "seasonal_cycle":
        return seasonal_cycle_from_time(
            context,
            amplitude=float(params.get("amplitude", plan.seasonal_amplitude or plan.daily_amplitude)),
            period_days=float(params.get("period_days", 365.25)),
            phase=float(params.get("phase", 0.5)),
        )
    if family == "heat_index_effect":
        return heat_index_effect_from_time(context, amplitude=float(params.get("amplitude", plan.heat_effect)))
    if family == "gaussian_peak":
        center = float(params.get("center", params.get("peak_hour", 12.0)))
        width = max(float(params.get("width", 2.5)), 1e-6)
        return float(params.get("amplitude", plan.daily_amplitude)) * np.exp(-0.5 * ((hour - center) / width) ** 2)
    if family == "daylight_envelope":
        sunrise = float(params.get("sunrise_hour", 6.0))
        sunset = float(params.get("sunset_hour", 19.0))
        if context.has_intraday_resolution:
            daylight = (hour >= sunrise) & (hour <= sunset)
            phase = (hour - sunrise) / max(1.0, sunset - sunrise)
            return float(params.get("amplitude", plan.daily_amplitude)) * np.where(
                daylight,
                np.sin(np.pi * np.clip(phase, 0.0, 1.0)) ** 1.7,
                0.0,
            )
        daylight_hours = max(sunset - sunrise, 0.0)
        aggregation_days = max(context.step_hours / 24.0, 1.0)
        return np.full(
            length,
            float(params.get("amplitude", plan.daily_amplitude)) * (daylight_hours / 12.0) * aggregation_days,
            dtype=float,
        )
    if family == "event_mask":
        probability = float(params.get("event_probability", 0.18))
        return (rng.random(length) < probability).astype(float)
    if family == "gamma_intensity":
        return rng.gamma(
            shape=max(float(params.get("shape", params.get("intensity_shape", 1.4))), 0.3),
            scale=max(float(params.get("scale", params.get("intensity_scale", 5.0))), 0.1),
            size=length,
        )
    if family == "storm_multiplier":
        probability = float(params.get("probability", params.get("storm_probability", 0.08)))
        multiplier = max(float(params.get("multiplier", params.get("storm_multiplier", 3.0))), 1.0)
        return np.where(rng.random(length) < probability, multiplier - 1.0, 0.0)
    if family == "cloud_drop":
        probability = float(params.get("probability", params.get("cloud_probability", 0.18)))
        drop_min = float(params.get("drop_min", params.get("cloud_drop_min", 0.35)))
        drop_max = float(params.get("drop_max", params.get("cloud_drop_max", 0.8)))
        active = rng.random(length) < probability
        drops = rng.uniform(drop_min, drop_max, size=length)
        return np.where(active, drops - 1.0, 0.0)
    if family == "scheduled_pulse":
        center = float(params.get("hour", params.get("batch_hour", 2.0)))
        probability = float(params.get("probability", params.get("batch_probability", 0.35)))
        amplitude = float(params.get("amplitude", max(plan.daily_amplitude, plan.baseline * 0.08)))
        width = max(float(params.get("width", 1.5)), 0.25)
        pulse_shape = np.exp(-0.5 * ((hour - center) / width) ** 2) if context.has_intraday_resolution else np.ones(length)
        active = rng.random(length) < probability
        return amplitude * pulse_shape * active.astype(float)
    if family in {"noise", "gaussian_noise", "residual_noise"}:
        return gaussian_noise(length, rng, float(params.get("sigma", plan.noise_sigma)))
    if family == "anomaly_strategy":
        return np.zeros(length, dtype=float)
    return np.zeros(length, dtype=float)


def _is_bounded_factor(values: np.ndarray, *, upper: float = 1.0) -> bool:
    finite = values[np.isfinite(values)]
    return bool(finite.size == 0 or (float(np.min(finite)) >= -1e-9 and float(np.max(finite)) <= upper + 1e-9))


def _execute_llm_component_dsl(
    plan: SeriesPlan,
    workflow: ComponentWorkflow,
    length: int,
    rng: np.random.Generator,
    context: TimeContext,
) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    if workflow.agent_trace.get("component_source") != "llm_mechanism_plan":
        return None
    component_values = {
        component.name: _component_array_from_family(component, plan, length, rng, context)
        for component in workflow.components
    }
    additive = np.zeros(length, dtype=float)
    envelope = np.ones(length, dtype=float)
    multiplicative_base = np.zeros(length, dtype=float)
    has_multiplicative_base = False
    event_mask = None
    event_intensity = None
    storm_factor = np.ones(length, dtype=float)
    storm_components: list[tuple[str, np.ndarray]] = []
    for component in workflow.components:
        values = component_values[component.name]
        family = str(component.feature_family).lower()
        if family == "event_mask":
            event_mask = values
        elif family == "gamma_intensity":
            event_intensity = values
        elif family == "storm_multiplier":
            storm_components.append((component.name, np.maximum(values, 0.0)))
        elif family == "cloud_drop":
            envelope *= np.clip(1.0 + values, 0.0, 1.0)
        elif component.value_role == "multiplicative_envelope":
            if _is_bounded_factor(values):
                envelope *= np.maximum(values, 0.0)
            else:
                multiplicative_base += np.maximum(values, 0.0)
                has_multiplicative_base = True
        elif component.value_role == "multiplicative_attenuation":
            envelope *= np.clip(1.0 + values, 0.0, 1.0)
        else:
            additive += values
    if storm_components and event_mask is not None:
        active_event = (event_mask > 0.0).astype(float)
        for name, values in storm_components:
            event_scoped = values * active_event
            component_values[name] = event_scoped
            storm_factor *= 1.0 + event_scoped
    if event_mask is not None and event_intensity is not None:
        additive += event_mask * event_intensity * storm_factor
    values = (multiplicative_base if has_multiplicative_base else np.zeros(length, dtype=float)) * envelope + additive
    lower, upper = _clip_bounds(workflow)
    if lower is not None:
        values = np.maximum(values, lower)
    if upper is not None:
        values = np.minimum(values, upper)
    return values, component_values


def synthesize_component_base(
    plan: SeriesPlan,
    workflow: ComponentWorkflow,
    length: int,
    rng: np.random.Generator,
    freq: str | None = None,
    start: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate the pre-semantic base process from component features."""
    anomaly = np.zeros(length, dtype=int)
    component_values: dict[str, np.ndarray] = {}
    params = plan.domain_params
    context = build_time_context(
        length,
        freq or workflow.input_profile.freq,
        start or workflow.input_profile.start,
    )
    hour = _hour(length, context)
    component_names = {component.name for component in workflow.components}

    dsl_result = _execute_llm_component_dsl(plan, workflow, length, rng, context)
    if dsl_result is not None:
        values, component_values = dsl_result
    elif plan.generator_type == "intermittent_event":
        mask = np.zeros(length, dtype=float)
        intensity = np.zeros(length, dtype=float)
        storm_boost = np.ones(length, dtype=float)
        cursor = 0
        dry_spell_bias = float(params.get("dry_spell_bias", 0.5))
        event_probability = float(params.get("event_probability", 0.18))
        mean_duration = max(1.0, float(params.get("mean_duration", 5.0)))
        shape = max(0.3, float(params.get("intensity_shape", 1.4)))
        scale = max(0.1, float(params.get("intensity_scale", 5.0)))
        storm_probability = float(params.get("storm_probability", 0.08))
        storm_multiplier = max(1.0, float(params.get("storm_multiplier", 3.0)))
        mean_duration_steps = max(1.0, mean_duration * context.steps_per_hour)
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
            mask[cursor:end] = 1.0
            storm_boost[cursor:end] = storm_multiplier if is_storm else 1.0
            raw = rng.gamma(shape=shape, scale=scale, size=end - cursor)
            envelope = np.sin(np.linspace(0.15, np.pi - 0.15, end - cursor))
            intensity[cursor:end] = raw * np.maximum(envelope, 0.15)
            if is_storm:
                anomaly[cursor:end] = 1
            cursor = end + int(rng.integers(hours_to_steps(context, 1), hours_to_steps(context, 10) + 1))
        values = mask * intensity * storm_boost
        drizzle = (values > 0) & (rng.random(length) < 0.18)
        values[drizzle] *= rng.uniform(0.15, 0.55, size=drizzle.sum())
        component_values = {
            "dry_spell_state": 1.0 - mask,
            "event_arrival": mask,
            "event_intensity": intensity,
            "storm_boost": storm_boost - 1.0,
        }
    elif plan.generator_type == "daylight_envelope":
        sunrise = float(params.get("sunrise_hour", 6.0))
        sunset = float(params.get("sunset_hour", 19.0))
        day_index = context.day_code
        day_factor = rng.uniform(0.82, 1.08, size=int(day_index.max()) + 1 if length else 1)
        if context.has_intraday_resolution:
            daylight = (hour >= sunrise) & (hour <= sunset)
            phase = (hour - sunrise) / max(1.0, sunset - sunrise)
            envelope = np.where(daylight, np.sin(np.pi * np.clip(phase, 0.0, 1.0)) ** 1.7, 0.0)
            daylight_power = plan.daily_amplitude * envelope * day_factor[day_index]
            noise_gate = daylight
        else:
            daylight_hours = max(sunset - sunrise, 0.0)
            aggregation_days = max(context.step_hours / 24.0, 1.0)
            seasonal = 0.92 + 0.16 * np.maximum(
                seasonal_cycle_from_time(context, amplitude=1.0, period_days=365.25, phase=-0.2),
                -0.5,
            )
            daylight_power = plan.daily_amplitude * (daylight_hours / 12.0) * aggregation_days
            daylight_power = daylight_power * seasonal * day_factor[day_index]
            daylight = np.ones(length, dtype=bool)
            noise_gate = np.ones(length, dtype=bool)
        attenuation = np.ones(length, dtype=float)
        cursor = 0
        while cursor < length:
            if daylight[cursor] and rng.random() < float(params.get("cloud_probability", 0.18)):
                duration = int(rng.integers(hours_to_steps(context, 1), hours_to_steps(context, 5) + 1))
                end = min(length, cursor + duration)
                attenuation[cursor:end] = rng.uniform(
                    float(params.get("cloud_drop_min", 0.35)),
                    float(params.get("cloud_drop_max", 0.8)),
                )
            cursor += 1
        noise = rng.normal(0.0, plan.noise_sigma, size=length) * noise_gate
        values = np.maximum(daylight_power * attenuation + noise, 0.0)
        component_values = {
            "daylight_envelope": daylight_power,
            "cloud_attenuation": daylight_power * (attenuation - 1.0),
            "measurement_noise": noise,
        }
    elif plan.generator_type == "smooth_environmental":
        peak_hour = float(params.get("peak_hour", 15.0))
        baseline = np.full(length, float(plan.baseline), dtype=float)
        cycle = (
            plan.daily_amplitude * np.sin(2 * np.pi * (hour - peak_hour + 6) / 24)
            if context.has_intraday_resolution
            else np.zeros(length, dtype=float)
        )
        trend = linear_trend(length, plan.trend_slope)
        noise = rng.normal(0.0, plan.noise_sigma, size=length)
        target = baseline + cycle + trend + noise
        values = target.copy()
        inertia = float(params.get("inertia", 0.88))
        for i in range(1, length):
            values[i] = inertia * values[i - 1] + (1.0 - inertia) * target[i]
        component_values = {
            "baseline_level": baseline,
            "diurnal_target": cycle + trend,
            "thermal_inertia": values - target,
            "weather_noise": noise,
        }
    elif plan.generator_type == "count_process":
        if context.has_intraday_resolution:
            morning = np.exp(-0.5 * ((hour - float(params.get("morning_peak", 8.0))) / 2.0) ** 2)
            evening = np.exp(-0.5 * ((hour - float(params.get("evening_peak", 18.0))) / 2.8) ** 2)
        else:
            morning = np.zeros(length, dtype=float)
            evening = np.zeros(length, dtype=float)
        base_rate = np.full(length, plan.baseline, dtype=float)
        morning_rate = plan.daily_amplitude * 0.8 * morning
        evening_rate = plan.daily_amplitude * evening
        rate = base_rate + morning_rate + evening_rate
        if plan.weekly_enabled:
            rate *= working_day_gate_from_time(context, weekend_factor=0.78)
        rate += linear_trend(length, plan.trend_slope)
        rate = np.maximum(rate, 0.1)
        overdispersion = max(1.0, float(params.get("overdispersion", 1.35)))
        values = rng.negative_binomial(np.maximum(rate / (overdispersion - 1 + 1e-9), 1.0), 1 / overdispersion).astype(float)
        component_values = {
            "base_rate": base_rate,
            "morning_peak": morning_rate,
            "evening_peak": evening_rate,
            "overdispersed_sampling": values - rate,
        }
    elif plan.generator_type == "bounded_utilization":
        background = np.full(length, float(plan.baseline), dtype=float)
        workload = daily_load_shape(context, amplitude=plan.daily_amplitude, phase=float(params.get("daily_phase", 0.0)))
        weekly = weekly_cycle_from_time(context, amplitude=plan.weekly_amplitude, phase=-0.8) if plan.weekly_enabled else np.zeros(length)
        noise = rng.normal(0.0, plan.noise_sigma, size=length)
        batch = np.zeros(length, dtype=float)
        batch_hour = float(params.get("batch_hour", 2.0))
        batch_duration = hours_to_steps(context, 2)
        for i, item_hour in enumerate(hour):
            if abs(item_hour - batch_hour) < 0.5 and rng.random() < float(params.get("batch_probability", 0.35)):
                batch[i : min(length, i + batch_duration)] += rng.uniform(8.0, 24.0)
        values = background + workload + weekly + batch + noise
        values = np.clip(values, 0.0, float(params.get("upper_bound", 100.0)))
        component_values = {
            "background_utilization": background,
            "workload_cycle": workload + weekly,
            "batch_job": batch,
            "utilization_noise": noise,
        }
    else:
        baseline = np.full(length, float(plan.baseline), dtype=float)
        trend = linear_trend(length, plan.trend_slope)
        daily = daily_load_shape(context, amplitude=plan.daily_amplitude, phase=float(params.get("daily_phase", 0.0)))
        weekly = np.zeros(length, dtype=float)
        if plan.weekly_enabled:
            weekly = weekly_cycle_from_time(context, amplitude=plan.weekly_amplitude, phase=-0.8)
            weekly += baseline * (working_day_gate_from_time(context, weekend_factor=float(params.get("weekend_factor", 0.72))) - 1.0)
        seasonal = (
            seasonal_cycle_from_time(context, amplitude=plan.seasonal_amplitude, period_days=30.0, phase=0.5)
            if plan.seasonal_amplitude
            else np.zeros(length)
        )
        heat = heat_index_effect_from_time(context, amplitude=plan.heat_effect) if plan.heat_effect else np.zeros(length)
        noise = gaussian_noise(length, rng, plan.noise_sigma)
        values = baseline + trend + daily + weekly + seasonal + heat + noise
        component_values = {
            "baseline_level": baseline,
            "trend": trend,
            "daily_cycle": daily,
            "weekly_gate": weekly,
            "seasonal_cycle": seasonal,
            "weather_response": heat,
            "residual_noise": noise,
        }

    for component in workflow.components:
        component_values.setdefault(component.name, np.zeros(length, dtype=float))
    values, component_values = _enforce_additive_component_bounds(values, component_values, workflow)
    report = _quality_report(component_values, workflow, values)
    return values, anomaly, {
        "workflow": workflow.to_dict(),
        "component_report": report,
        "component_stats": {name: _component_stats(values) for name, values in component_values.items()},
    }
