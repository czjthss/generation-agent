from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

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

    return ComponentWorkflow(
        input_profile=input_profile,
        variable_profile=variable_profile,
        components=components,
        composition=composition,
        final_transform=final_transform,
        anomaly_target=plan.anomaly_target,
        validator_rules=sorted(set(validator_rules)),
        agent_trace={
            "input_profile_agent": "constructed input/reference/time context",
            "scenario_variable_agent": "defined variable semantics, constraints, and relationships",
            "component_mechanism_agent": "decomposed the variable into mechanism components",
            "component_feature_planning_agent": "mapped components to executable feature families",
        },
    )


def _hour(length: int) -> np.ndarray:
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


def _quality_report(
    component_values: dict[str, np.ndarray],
    workflow: ComponentWorkflow,
    composed: np.ndarray,
) -> dict[str, Any]:
    total_std = float(np.std(composed)) or 1.0
    component_reports = []
    issues: list[str] = []
    for component in workflow.components:
        values = component_values.get(component.name, np.zeros_like(composed))
        stats = _component_stats(values)
        contribution = float(np.std(values) / total_std) if total_std else 0.0
        checks = {
            "semantic": True,
            "time_behavior": True,
            "statistical_shape": True,
            "contribution": contribution <= 3.0 or component.value_role in {"additive_level", "multiplicative_envelope"},
        }
        if component.sign_constraint == "nonnegative" and stats["min"] < -1e-6:
            checks["semantic"] = False
            issues.append(f"{component.name} violates nonnegative component semantics")
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


def synthesize_component_base(
    plan: SeriesPlan,
    workflow: ComponentWorkflow,
    length: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate the pre-semantic base process from component features."""
    anomaly = np.zeros(length, dtype=int)
    component_values: dict[str, np.ndarray] = {}
    params = plan.domain_params
    hour = _hour(length)

    if plan.generator_type == "intermittent_event":
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
            mask[cursor:end] = 1.0
            storm_boost[cursor:end] = storm_multiplier if is_storm else 1.0
            raw = rng.gamma(shape=shape, scale=scale, size=end - cursor)
            envelope = np.sin(np.linspace(0.15, np.pi - 0.15, end - cursor))
            intensity[cursor:end] = raw * np.maximum(envelope, 0.15)
            if is_storm:
                anomaly[cursor:end] = 1
            cursor = end + int(rng.integers(1, 10))
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
        daylight = (hour >= sunrise) & (hour <= sunset)
        phase = (hour - sunrise) / max(1.0, sunset - sunrise)
        envelope = np.where(daylight, np.sin(np.pi * np.clip(phase, 0.0, 1.0)) ** 1.7, 0.0)
        day_index = np.arange(length) // 24
        day_factor = rng.uniform(0.82, 1.08, size=int(day_index.max()) + 1 if length else 1)
        daylight_power = plan.daily_amplitude * envelope * day_factor[day_index]
        attenuation = np.ones(length, dtype=float)
        cursor = 0
        while cursor < length:
            if daylight[cursor] and rng.random() < float(params.get("cloud_probability", 0.18)):
                duration = int(rng.integers(1, 5))
                end = min(length, cursor + duration)
                attenuation[cursor:end] = rng.uniform(
                    float(params.get("cloud_drop_min", 0.35)),
                    float(params.get("cloud_drop_max", 0.8)),
                )
            cursor += 1
        noise = rng.normal(0.0, plan.noise_sigma, size=length) * daylight
        values = np.maximum(daylight_power * attenuation + noise, 0.0)
        component_values = {
            "daylight_envelope": daylight_power,
            "cloud_attenuation": daylight_power * (attenuation - 1.0),
            "measurement_noise": noise,
        }
    elif plan.generator_type == "smooth_environmental":
        peak_hour = float(params.get("peak_hour", 15.0))
        baseline = np.full(length, float(plan.baseline), dtype=float)
        cycle = plan.daily_amplitude * np.sin(2 * np.pi * (hour - peak_hour + 6) / 24)
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
        morning = np.exp(-0.5 * ((hour - float(params.get("morning_peak", 8.0))) / 2.0) ** 2)
        evening = np.exp(-0.5 * ((hour - float(params.get("evening_peak", 18.0))) / 2.8) ** 2)
        base_rate = np.full(length, plan.baseline, dtype=float)
        morning_rate = plan.daily_amplitude * 0.8 * morning
        evening_rate = plan.daily_amplitude * evening
        rate = base_rate + morning_rate + evening_rate
        if plan.weekly_enabled:
            rate *= working_day_gate(length, weekend_factor=0.78)
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
        workload = daily_load_cycle(length, amplitude=plan.daily_amplitude, phase=float(params.get("daily_phase", 0.0)))
        weekly = sinusoidal_cycle(length, period=24 * 7, amplitude=plan.weekly_amplitude, phase=-0.8) if plan.weekly_enabled else np.zeros(length)
        noise = rng.normal(0.0, plan.noise_sigma, size=length)
        batch = np.zeros(length, dtype=float)
        batch_hour = float(params.get("batch_hour", 2.0))
        for i, item_hour in enumerate(hour):
            if abs(item_hour - batch_hour) < 0.5 and rng.random() < float(params.get("batch_probability", 0.35)):
                batch[i : min(length, i + 2)] += rng.uniform(8.0, 24.0)
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
        daily = daily_load_cycle(length, amplitude=plan.daily_amplitude, phase=float(params.get("daily_phase", 0.0)))
        weekly = np.zeros(length, dtype=float)
        if plan.weekly_enabled:
            weekly = sinusoidal_cycle(length, period=24 * 7, amplitude=plan.weekly_amplitude, phase=-0.8)
            weekly += baseline * (working_day_gate(length, weekend_factor=float(params.get("weekend_factor", 0.72))) - 1.0)
        seasonal = (
            sinusoidal_cycle(length, period=max(length, 24 * 30), amplitude=plan.seasonal_amplitude, phase=0.5)
            if plan.seasonal_amplitude
            else np.zeros(length)
        )
        heat = heat_index_effect(length, amplitude=plan.heat_effect) if plan.heat_effect else np.zeros(length)
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

    report = _quality_report(component_values, workflow, values)
    return values, anomaly, {
        "workflow": workflow.to_dict(),
        "component_report": report,
        "component_stats": {name: _component_stats(values) for name, values in component_values.items()},
    }
