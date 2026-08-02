from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
import re
from typing import Any

from .env import load_project_env


load_project_env()


DEFAULT_LLM_MODEL = "gpt-5.5"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


@dataclass
class SeriesPlan:
    domain: str = "generic"
    generator_type: str = "cyclic_signal"
    unit: str = "value"
    baseline: float = 100.0
    trend_slope: float = 0.0
    daily_amplitude: float = 10.0
    weekly_enabled: bool = False
    weekly_amplitude: float = 3.0
    seasonal_amplitude: float = 0.0
    heat_effect: float = 0.0
    noise_sigma: float = 2.0
    anomaly_count: int = 0
    anomaly_magnitude: float = 3.0
    anomaly_width: int = 1
    anomaly_kind: str = "spike"
    lower_bound: float | None = 0.0
    domain_params: dict[str, Any] = field(default_factory=dict)
    semantic_type: str = "instantaneous"
    semantic_config: dict[str, Any] = field(default_factory=dict)
    output_constraints: dict[str, Any] = field(default_factory=dict)
    variables: list[dict[str, Any]] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)
    anomaly_enabled: bool = False
    anomaly_severity: str = "medium"
    anomaly_target: str = "value"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid_semantics = {
            "instantaneous",
            "cumulative",
            "stock_flow",
            "regime_switching",
            "random_walk",
            "decay_recovery",
            "saturation_growth",
            "multivariate_lag",
        }
        if self.semantic_type not in valid_semantics:
            self.semantic_type = "instantaneous"
        if self.semantic_type == "cumulative":
            self.semantic_config.setdefault("initial_value", 0.0)
            self.semantic_config.setdefault("allow_negative_increment", False)
            self.output_constraints.setdefault("nonnegative", True)
            if not self.semantic_config["allow_negative_increment"]:
                self.output_constraints.setdefault("monotonic", "nondecreasing")
            if self.anomaly_target == "value":
                self.anomaly_target = "increment"
        elif self.semantic_type == "stock_flow":
            self.output_constraints.setdefault("nonnegative", True)
            self.output_constraints.setdefault("conservation", True)
            if self.anomaly_target == "value":
                self.anomaly_target = "flow"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SeriesPlan":
        valid = {field.name for field in cls.__dataclass_fields__.values()}
        clean = {key: value for key, value in payload.items() if key in valid}
        if "anomaly_enabled" not in clean and int(clean.get("anomaly_count", 0)) > 0:
            clean["anomaly_enabled"] = True
        return cls(**clean)


def heuristic_plan(description: str) -> SeriesPlan:
    text = description.lower()
    try:
        from .domain_rules import match_rule, rule_to_plan

        custom_rule = match_rule(description)
    except Exception:
        custom_rule = None
    if custom_rule is not None:
        return rule_to_plan(custom_rule, description)

    plan = SeriesPlan(metadata={"planner": "heuristic", "description": description})

    if any(k in text for k in ["cumulative", "accumulated", "total-to-date", "running total"]):
        plan.semantic_type = "cumulative"
        plan.semantic_config = {"initial_value": 0.0, "allow_negative_increment": False}
        plan.output_constraints = {"nonnegative": True, "monotonic": "nondecreasing"}
        plan.anomaly_target = "increment"
    elif any(k in text for k in ["inventory", "stock", "balance", "reservoir", "battery", "soc"]):
        plan.semantic_type = "stock_flow"
        plan.semantic_config = {
            "initial_value": 1000.0,
            "inflow_scale": 1.0,
            "outflow_scale": 0.85,
        }
        plan.output_constraints = {"nonnegative": True, "conservation": True}
        plan.anomaly_target = "flow"
    elif any(k in text for k in ["random walk", "stock price", "exchange rate", "asset price"]):
        plan.semantic_type = "random_walk"
        plan.semantic_config = {"initial_value": 100.0, "drift": 0.02, "volatility": 1.0}
        plan.output_constraints = {"nonnegative": True}
        plan.anomaly_target = "step"
    elif any(k in text for k in ["decay", "recovery", "drug concentration", "fault recovery", "cooling"]):
        plan.semantic_type = "decay_recovery"
        plan.semantic_config = {"equilibrium": 0.0, "recovery_rate": 0.12, "impulse": 20.0}
        plan.output_constraints = {"nonnegative": True}
        plan.anomaly_target = "impulse"
    elif any(k in text for k in ["saturation growth", "market penetration", "user growth", "logistic", "gompertz"]):
        plan.semantic_type = "saturation_growth"
        plan.semantic_config = {"initial_value": 10.0, "capacity": 1000.0, "growth_rate": 0.06}
        plan.output_constraints = {"nonnegative": True, "monotonic": "nondecreasing"}
        plan.anomaly_target = "growth_rate"
    elif any(k in text for k in ["regime", "state switching", "operating state", "fault state"]):
        plan.semantic_type = "regime_switching"
        plan.semantic_config = {
            "states": [0.4, 1.0, 1.5],
            "transition_probability": 0.05,
        }
        plan.output_constraints = {"nonnegative": True}
        plan.anomaly_target = "state"
    elif any(k in text for k in ["lag", "influence", "related variables", "multivariate"]):
        plan.semantic_type = "multivariate_lag"
        plan.semantic_config = {"driver_name": "driver", "lag": 2, "coefficient": 0.7}
        plan.relationships = [{"source": "driver", "target": "value", "lag": 2, "coefficient": 0.7}]
        plan.anomaly_target = "driver"

    if any(k in text for k in ["rain", "rainfall", "precipitation", "storm", "drizzle"]):
        plan.domain = "precipitation"
        plan.generator_type = "intermittent_event"
        plan.unit = "mm"
        plan.baseline = 0.0
        plan.trend_slope = 0.0
        plan.daily_amplitude = 0.0
        plan.weekly_enabled = False
        plan.weekly_amplitude = 0.0
        plan.seasonal_amplitude = 0.0
        plan.heat_effect = 0.0
        plan.noise_sigma = 0.0
        plan.lower_bound = 0.0
        plan.domain_params = {
            "event_probability": 0.22,
            "mean_duration": 5.0,
            "intensity_shape": 1.4,
            "intensity_scale": 5.0,
            "dry_spell_bias": 0.48,
            "storm_probability": 0.08,
            "storm_multiplier": 3.0,
        }
        plan.metadata["domain_knowledge"] = "precipitation is generated as sparse storm events, not a smooth daily cycle"
    elif any(k in text for k in ["solar", "photovoltaic", "pv power", "irradiance", "sunlight"]):
        plan.domain = "solar_power"
        plan.generator_type = "daylight_envelope"
        plan.unit = "kW"
        plan.baseline = 0.0
        plan.daily_amplitude = 600.0
        plan.noise_sigma = 20.0
        plan.lower_bound = 0.0
        plan.domain_params = {
            "sunrise_hour": 6.0,
            "sunset_hour": 19.0,
            "cloud_probability": 0.18,
            "cloud_drop_min": 0.35,
            "cloud_drop_max": 0.8,
        }
        plan.metadata["domain_knowledge"] = "solar power follows daylight envelope and is zero at night"
    elif any(k in text for k in ["temperature", "air temperature", "room temperature", "cold chain"]):
        plan.domain = "temperature"
        plan.generator_type = "smooth_environmental"
        plan.unit = "°C"
        plan.baseline = 26.0
        plan.daily_amplitude = 4.0
        plan.noise_sigma = 0.4
        plan.lower_bound = None
        plan.domain_params = {"inertia": 0.88, "peak_hour": 15.0}
        plan.metadata["domain_knowledge"] = "temperature changes smoothly with inertia instead of abrupt independent noise"
    elif any(k in text for k in ["sales", "revenue", "turnover", "gmv"]):
        plan.domain = "sales"
        plan.generator_type = "cyclic_signal"
        plan.unit = "CNY"
        plan.baseline = 5000.0
        plan.daily_amplitude = 1800.0
        plan.weekly_enabled = True
        plan.weekly_amplitude = 700.0
        plan.noise_sigma = 350.0
        plan.lower_bound = 0.0
        plan.metadata["domain_knowledge"] = "sales amount is generated as nonnegative period revenue before optional accumulation"
    elif any(k in text for k in ["traffic", "vehicle flow", "passenger flow", "metro", "orders", "requests", "qps", "api"]):
        plan.domain = "demand_count"
        plan.generator_type = "count_process"
        plan.unit = "count"
        plan.baseline = 220.0
        plan.daily_amplitude = 90.0
        plan.weekly_enabled = True
        plan.weekly_amplitude = 25.0
        plan.noise_sigma = 10.0
        plan.lower_bound = 0.0
        plan.domain_params = {"morning_peak": 8.0, "evening_peak": 18.0, "overdispersion": 1.35}
        plan.metadata["domain_knowledge"] = "count demand uses peaks and over-dispersed nonnegative counts"
    elif any(k in text for k in ["server", "cpu", "memory", "database", "gateway"]):
        plan.domain = "server_metric"
        plan.generator_type = "bounded_utilization"
        plan.unit = "%"
        plan.baseline = 42.0
        plan.daily_amplitude = 18.0
        plan.weekly_enabled = True
        plan.weekly_amplitude = 4.0
        plan.noise_sigma = 4.0
        plan.lower_bound = 0.0
        plan.domain_params = {"upper_bound": 100.0, "batch_hour": 2.0, "batch_probability": 0.35}
        plan.metadata["domain_knowledge"] = "server utilization is bounded and can include batch jobs or bursts"
    elif any(k in text for k in ["electric", "electricity", "load", "power"]):
        plan.domain = "electric_load"
        plan.generator_type = "cyclic_signal"
        plan.unit = "kW"
        plan.baseline = 900.0
        plan.daily_amplitude = 135.0
        plan.weekly_enabled = True
        plan.weekly_amplitude = 35.0
        plan.noise_sigma = 22.0
        plan.trend_slope = 20.0
    if any(k in text for k in ["south china", "guangdong", "guangxi", "fujian", "hainan"]):
        if plan.domain == "precipitation":
            plan.domain_params["event_probability"] = max(plan.domain_params.get("event_probability", 0.18), 0.24)
            plan.domain_params["dry_spell_bias"] = min(plan.domain_params.get("dry_spell_bias", 0.48), 0.42)
            plan.domain_params["storm_probability"] = max(plan.domain_params.get("storm_probability", 0.08), 0.12)
        elif plan.domain in {"electric_load", "generic"}:
            plan.heat_effect = max(plan.heat_effect, 85.0)
        plan.metadata["region"] = "south_china"
    if any(k in text for k in ["summer", "high temperature", "hot", "air conditioning"]):
        if plan.domain == "precipitation":
            plan.domain_params["event_probability"] = max(plan.domain_params.get("event_probability", 0.18), 0.28)
            plan.domain_params["dry_spell_bias"] = min(plan.domain_params.get("dry_spell_bias", 0.48), 0.36)
            plan.domain_params["intensity_scale"] = max(plan.domain_params.get("intensity_scale", 5.0), 7.0)
            plan.domain_params["storm_probability"] = max(plan.domain_params.get("storm_probability", 0.08), 0.15)
        elif plan.domain == "temperature":
            plan.baseline = max(plan.baseline, 31.0)
            plan.daily_amplitude = max(plan.daily_amplitude, 5.5)
        elif plan.domain == "solar_power":
            plan.daily_amplitude = max(plan.daily_amplitude, 680.0)
        else:
            plan.heat_effect = max(plan.heat_effect, 130.0)
            plan.seasonal_amplitude = max(plan.seasonal_amplitude, 30.0)
        plan.metadata["season"] = "summer"
    if any(k in text for k in ["storm", "heavy rainfall", "typhoon"]):
        plan.domain_params["storm_probability"] = max(plan.domain_params.get("storm_probability", 0.08), 0.28)
        plan.domain_params["storm_multiplier"] = max(plan.domain_params.get("storm_multiplier", 3.0), 5.0)
    if any(k in text for k in ["light rain", "drizzle"]):
        plan.domain_params["intensity_scale"] = min(plan.domain_params.get("intensity_scale", 5.0), 2.0)
        plan.domain_params["storm_probability"] = 0.02
    if any(k in text for k in ["anomaly", "fault", "spike", "surge", "drop"]):
        plan.anomaly_enabled = True
        plan.anomaly_count = 4
        plan.anomaly_magnitude = 3.5
        plan.anomaly_width = 2
        if "drop" in text:
            plan.anomaly_kind = "drop"
        elif "surge" in text or "spike" in text:
            plan.anomaly_kind = "positive_spike"
    if any(k in text for k in ["decline", "decrease", "decay"]):
        plan.trend_slope = -abs(plan.trend_slope or 10.0)
    if any(k in text for k in ["growth", "increase", "rise"]):
        plan.trend_slope = abs(plan.trend_slope or 10.0)

    metric_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(kw|mw|kwh|°c|℃|mm|%)", text)
    if metric_match:
        value = float(metric_match.group(1))
        unit = metric_match.group(2)
        if unit == "mw":
            value *= 1000.0
            unit = "kW"
        plan.baseline = value
        plan.unit = unit

    return plan


def plan_from_description(
    description: str,
    model: str | None = None,
    reference_profile: dict[str, Any] | None = None,
    cost_mode: str = "balanced",
) -> SeriesPlan:
    def finalize(plan: SeriesPlan) -> SeriesPlan:
        from .reference_profiler import apply_reference_priors

        return apply_reference_priors(plan, reference_profile, "structure")

    if not model:
        raise RuntimeError("LLM model is required; no-LLM generation mode has been removed")
    from .dependency_check import ensure_llm_dependencies

    ensure_llm_dependencies()
    from .langchain_agent import plan_with_langchain_tools

    planned = plan_with_langchain_tools(
        description,
        model=model,
        reference_profile=reference_profile,
        cost_mode=cost_mode,
    )
    if planned is None:
        raise RuntimeError("LLM agent planning returned no valid plan")
    return finalize(planned)
