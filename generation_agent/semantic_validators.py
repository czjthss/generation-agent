from __future__ import annotations

from typing import Any

import numpy as np

from .planner import SeriesPlan


def validate_and_repair(
    values: np.ndarray,
    plan: SeriesPlan,
    columns: dict[str, np.ndarray],
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, dict[str, Any]]:
    result = values.astype(float, copy=True)
    constraints = dict(plan.output_constraints)
    repairs: list[str] = []
    checks: dict[str, bool] = {}

    lower = constraints.get("lower_bound", plan.lower_bound)
    upper = constraints.get("upper_bound")
    if constraints.get("nonnegative") and lower is None:
        lower = 0.0
    if lower is not None:
        lower = float(lower)
        if np.any(result < lower):
            result = np.maximum(result, lower)
            repairs.append("clipped_to_lower_bound")
        checks["lower_bound"] = bool(np.all(result >= lower - tolerance))
    if upper is not None:
        upper = float(upper)
        if np.any(result > upper):
            result = np.minimum(result, upper)
            repairs.append("clipped_to_upper_bound")
        checks["upper_bound"] = bool(np.all(result <= upper + tolerance))

    monotonic = constraints.get("monotonic")
    if monotonic == "nondecreasing":
        checks["monotonic_nondecreasing"] = bool(np.all(np.diff(result) >= -tolerance))
    elif monotonic == "nonincreasing":
        checks["monotonic_nonincreasing"] = bool(np.all(np.diff(result) <= tolerance))

    if constraints.get("integer"):
        if not np.allclose(result, np.round(result), atol=tolerance):
            result = np.round(result)
            repairs.append("rounded_to_integer")
        checks["integer"] = bool(np.allclose(result, np.round(result), atol=tolerance))

    if constraints.get("conservation") and {"inflow", "outflow"}.issubset(columns):
        initial = float(plan.semantic_config.get("initial_value", max(plan.baseline, 0.0)))
        expected = np.empty(len(result), dtype=float)
        previous = initial
        for i, net_flow in enumerate(columns["inflow"] - columns["outflow"]):
            previous += net_flow
            if constraints.get("nonnegative", True):
                previous = max(previous, 0.0)
            expected[i] = previous
        checks["stock_flow_balance"] = bool(np.allclose(result, expected, atol=1e-4))

    if plan.semantic_type == "cumulative" and "increment" in columns:
        initial = float(plan.semantic_config.get("initial_value", 0.0))
        expected = initial + np.cumsum(columns["increment"])
        checks["cumulative_identity"] = bool(np.allclose(result, expected, atol=1e-4))

    passed = all(checks.values()) if checks else True
    return result, {
        "passed": passed,
        "semantic_type": plan.semantic_type,
        "checks": checks,
        "repairs_applied": repairs,
    }
