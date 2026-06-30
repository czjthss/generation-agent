from __future__ import annotations

from typing import Any

import numpy as np

from .planner import SeriesPlan


def _constraint_checks(
    result: np.ndarray,
    plan: SeriesPlan,
    columns: dict[str, np.ndarray],
    lower: float | None,
    upper: float | None,
    tolerance: float,
) -> dict[str, bool]:
    constraints = dict(plan.output_constraints)
    checks: dict[str, bool] = {}
    if lower is not None:
        checks["lower_bound"] = bool(np.all(result >= lower - tolerance))
    if upper is not None:
        checks["upper_bound"] = bool(np.all(result <= upper + tolerance))

    monotonic = constraints.get("monotonic")
    if monotonic == "nondecreasing":
        checks["monotonic_nondecreasing"] = bool(np.all(np.diff(result) >= -tolerance))
    elif monotonic == "nonincreasing":
        checks["monotonic_nonincreasing"] = bool(np.all(np.diff(result) <= tolerance))

    if constraints.get("integer"):
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
    return checks


def _sync_cumulative_columns(
    result: np.ndarray,
    plan: SeriesPlan,
    columns: dict[str, np.ndarray],
    repairs: list[str],
    tolerance: float,
) -> np.ndarray:
    if plan.semantic_type != "cumulative" or "increment" not in columns:
        return result

    initial = float(plan.semantic_config.get("initial_value", 0.0))
    previous = np.r_[initial, result[:-1]]
    increments = result - previous
    if not bool(plan.semantic_config.get("allow_negative_increment", False)):
        if np.any(increments < -tolerance):
            increments = np.maximum(increments, 0.0)
        result = initial + np.cumsum(increments)
    changed = not np.allclose(columns["increment"], increments, atol=tolerance)
    columns["increment"] = increments
    if changed and "recomputed_cumulative_increment" not in repairs:
        repairs.append("recomputed_cumulative_increment")
    return result


def _sync_stock_flow_columns(
    result: np.ndarray,
    plan: SeriesPlan,
    columns: dict[str, np.ndarray],
    repairs: list[str],
    tolerance: float,
) -> None:
    if plan.semantic_type != "stock_flow" or not {"inflow", "outflow"}.issubset(columns):
        return

    initial = float(plan.semantic_config.get("initial_value", max(plan.baseline, 0.0)))
    old_inflow = np.asarray(columns["inflow"], dtype=float)
    old_outflow = np.asarray(columns["outflow"], dtype=float)
    expected = np.empty(len(result), dtype=float)
    previous = initial
    for index, net_flow in enumerate(old_inflow - old_outflow):
        previous += float(net_flow)
        if plan.output_constraints.get("nonnegative", True):
            previous = max(previous, 0.0)
        expected[index] = previous
    if np.allclose(result, expected, atol=max(tolerance, 1e-4)):
        columns["net_flow"] = old_inflow - old_outflow
        return

    inflow = np.zeros(len(result), dtype=float)
    outflow = np.zeros(len(result), dtype=float)
    previous = initial
    for index, value in enumerate(result):
        net_flow = float(value) - previous
        if net_flow >= 0.0:
            inflow[index] = net_flow
        else:
            outflow[index] = -net_flow
        previous = float(value)
    changed = not (
        np.allclose(old_inflow, inflow, atol=1e-6)
        and np.allclose(old_outflow, outflow, atol=1e-6)
    )
    columns["inflow"] = inflow
    columns["outflow"] = outflow
    columns["net_flow"] = inflow - outflow
    if changed and "recomputed_stock_flow_columns" not in repairs:
        repairs.append("recomputed_stock_flow_columns")


def validate_and_repair(
    values: np.ndarray,
    plan: SeriesPlan,
    columns: dict[str, np.ndarray],
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, dict[str, Any]]:
    result = values.astype(float, copy=True)
    constraints = dict(plan.output_constraints)
    repairs: list[str] = []

    lower = constraints.get("lower_bound", plan.lower_bound)
    upper = constraints.get("upper_bound")
    if constraints.get("nonnegative") and lower is None:
        lower = 0.0
    lower = float(lower) if lower is not None else None
    upper = float(upper) if upper is not None else None
    raw_columns = {name: np.asarray(column).copy() for name, column in columns.items()}
    raw_checks = _constraint_checks(result, plan, raw_columns, lower, upper, tolerance)
    raw_passed = all(raw_checks.values()) if raw_checks else True

    if lower is not None:
        if np.any(result < lower):
            result = np.maximum(result, lower)
            repairs.append("clipped_to_lower_bound")
    if upper is not None:
        if np.any(result > upper):
            result = np.minimum(result, upper)
            repairs.append("clipped_to_upper_bound")

    if constraints.get("integer"):
        if not np.allclose(result, np.round(result), atol=tolerance):
            result = np.round(result)
            repairs.append("rounded_to_integer")

    result = _sync_cumulative_columns(result, plan, columns, repairs, tolerance)
    _sync_stock_flow_columns(result, plan, columns, repairs, tolerance)

    checks = _constraint_checks(result, plan, columns, lower, upper, tolerance)
    repaired_passed = all(checks.values()) if checks else True
    critical_repairs = [
        repair
        for repair in repairs
        if repair in {"recomputed_cumulative_increment", "recomputed_stock_flow_columns"}
    ]
    repair_warnings = [
        {
            "type": "post_generation_repair",
            "severity": "hard_warning" if repair in critical_repairs else "soft_warning",
            "repair": repair,
        }
        for repair in repairs
    ]
    return result, {
        "passed": repaired_passed and not critical_repairs,
        "raw_passed": raw_passed,
        "repaired_passed": repaired_passed,
        "semantic_type": plan.semantic_type,
        "checks": {**checks, "critical_repairs_absent": not critical_repairs},
        "raw_checks": raw_checks,
        "repairs_applied": repairs,
        "repair_warnings": repair_warnings,
        "critical_repairs": critical_repairs,
    }
