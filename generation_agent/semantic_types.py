from __future__ import annotations

from dataclasses import dataclass, replace

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


@dataclass(frozen=True)
class AnomalyOverrides:
    enabled: bool | None = None
    severity: str | None = None


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
