from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from .planner import SeriesPlan


SEMANTIC_TYPES = {
    "instantaneous",
    "cumulative",
    "stock_flow",
    "regime_switching",
    "random_walk",
    "decay_recovery",
    "saturation_growth",
    "multivariate_lag",
}

SEVERITY_PRESETS = {
    "low": {"count": 1, "magnitude": 1.5, "width": 1},
    "medium": {"count": 3, "magnitude": 3.0, "width": 2},
    "high": {"count": 6, "magnitude": 5.0, "width": 4},
}

ANOMALY_TARGETS_BY_SEMANTIC = {
    "instantaneous": ("value",),
    "cumulative": ("increment", "flow"),
    "stock_flow": ("flow", "inflow", "outflow"),
    "regime_switching": ("state",),
    "random_walk": ("step", "increment"),
    "decay_recovery": ("impulse", "value"),
    "saturation_growth": ("growth_rate",),
    "multivariate_lag": ("driver",),
}

ANOMALY_KIND_ALIASES = {
    "level_shift": "shift",
    "temporary_outage": "drop",
    "dip": "drop",
    "negative_spike": "drop",
    "downward_spike": "drop",
    "mixed": "spike",
    "combined": "spike",
    "spike_drop": "spike",
    "spike_and_drop": "spike",
}

SUPPORTED_ANOMALY_KINDS = {"spike", "positive_spike", "drop", "shift"}


@dataclass(frozen=True)
class AnomalyOverrides:
    enabled: bool | None = None
    severity: str | None = None


@dataclass(frozen=True)
class AnomalyStrategy:
    enabled: bool
    reason: str
    target: str = "value"
    kind: str = "spike"
    severity: str = "medium"
    count: int = 0
    width: int = 1
    magnitude: float = 3.0
    constraints_after_injection: tuple[str, ...] = ()
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_anomaly_strategy(
    plan: SeriesPlan,
    strategy: AnomalyStrategy,
    overrides: AnomalyOverrides | None = None,
) -> SeriesPlan:
    severity = strategy.severity if strategy.severity in SEVERITY_PRESETS else "medium"
    preset = SEVERITY_PRESETS[severity]
    enabled = strategy.enabled
    if overrides and overrides.enabled is not None:
        enabled = overrides.enabled
    if overrides and overrides.severity is not None:
        severity = overrides.severity if overrides.severity in SEVERITY_PRESETS else "medium"
        preset = SEVERITY_PRESETS[severity]

    count = max(0, int(strategy.count))
    width = max(1, int(strategy.width))
    magnitude = max(0.0, float(strategy.magnitude))
    if enabled and count == 0:
        count = preset["count"]
    if overrides and overrides.severity is not None:
        count = preset["count"] if enabled else 0
        width = preset["width"]
        magnitude = preset["magnitude"]
    if not enabled:
        count = 0

    supported_targets = ANOMALY_TARGETS_BY_SEMANTIC.get(
        plan.semantic_type, ("value",)
    )
    target = strategy.target if strategy.target in supported_targets else supported_targets[0]
    kind = ANOMALY_KIND_ALIASES.get(strategy.kind, strategy.kind)
    if kind not in SUPPORTED_ANOMALY_KINDS:
        kind = "spike"

    metadata = dict(plan.metadata)
    metadata["anomaly_strategy"] = {
        **strategy.to_dict(),
        "enabled": enabled,
        "severity": severity,
        "count": count,
        "width": width,
        "magnitude": magnitude,
        "target": target,
        "kind": kind,
        "normalization": {
            "requested_target": strategy.target,
            "requested_kind": strategy.kind,
            "target_changed": target != strategy.target,
            "kind_changed": kind != strategy.kind,
        },
        "user_override": {
            "enabled": overrides.enabled if overrides else None,
            "severity": overrides.severity if overrides else None,
        },
    }
    return replace(
        plan,
        anomaly_enabled=enabled,
        anomaly_severity=severity,
        anomaly_count=count,
        anomaly_magnitude=magnitude,
        anomaly_width=width,
        anomaly_kind=kind,
        anomaly_target=target,
        metadata=metadata,
    )


def apply_anomaly_overrides(plan: SeriesPlan, overrides: AnomalyOverrides | None) -> SeriesPlan:
    if overrides is None:
        return plan

    severity = overrides.severity or plan.anomaly_severity
    if severity not in SEVERITY_PRESETS:
        severity = "medium"
    preset = SEVERITY_PRESETS.get(severity, SEVERITY_PRESETS["medium"])
    enabled = plan.anomaly_enabled if overrides.enabled is None else overrides.enabled
    count = plan.anomaly_count
    magnitude = plan.anomaly_magnitude
    width = plan.anomaly_width

    if overrides.severity is not None:
        count = preset["count"]
        magnitude = preset["magnitude"]
        width = preset["width"]
    if enabled and count <= 0:
        count = preset["count"]

    return replace(
        plan,
        anomaly_enabled=enabled,
        anomaly_severity=severity,
        anomaly_count=count,
        anomaly_magnitude=magnitude,
        anomaly_width=width,
    )
