from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REFERENCE_STRENGTHS = {"scale", "structure", "strict"}


@dataclass
class ReferenceProfile:
    source: str
    time_column: str | None
    value_column: str
    sample_count: int
    valid_count: int
    missing_ratio: float
    inferred_frequency: str | None
    statistics: dict[str, Any]
    dynamics: dict[str, Any]
    sparsity: dict[str, Any]
    periodicity: dict[str, Any]
    variables: list[dict[str, Any]] = None
    relationships: list[dict[str, Any]] = None
    distribution: dict[str, Any] = None
    nonstationarity: dict[str, Any] = None
    event_clusters: dict[str, Any] = None
    covariance: dict[str, Any] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _infer_columns(
    frame: pd.DataFrame,
    time_column: str | None,
    value_column: str | None,
) -> tuple[str | None, str]:
    if time_column is not None and time_column not in frame:
        raise ValueError(f"reference time column not found: {time_column}")
    if value_column is not None and value_column not in frame:
        raise ValueError(f"reference value column not found: {value_column}")

    if time_column is None:
        for candidate in ("time", "timestamp", "date", "datetime", "time_stamp"):
            if candidate in frame:
                time_column = candidate
                break

    if value_column is None:
        numeric = [name for name in frame.select_dtypes(include=[np.number]).columns if name != time_column]
        if not numeric:
            for name in frame.columns:
                if name == time_column:
                    continue
                converted = pd.to_numeric(frame[name], errors="coerce")
                if converted.notna().sum() >= max(3, len(frame) // 2):
                    numeric.append(name)
        if not numeric:
            raise ValueError("reference CSV has no usable numeric value column")
        value_column = "value" if "value" in numeric else numeric[0]
    return time_column, value_column


def _infer_frequency(frame: pd.DataFrame, time_column: str | None) -> str | None:
    if time_column is None:
        return None
    times = pd.to_datetime(frame[time_column], errors="coerce").dropna().drop_duplicates().sort_values()
    if len(times) < 3:
        return None
    try:
        inferred = pd.infer_freq(times)
    except ValueError:
        inferred = None
    if inferred:
        return str(inferred)
    deltas = times.diff().dropna()
    return str(deltas.median()) if not deltas.empty else None


def _run_lengths(mask: np.ndarray) -> list[int]:
    lengths: list[int] = []
    current = 0
    for flag in mask:
        if flag:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def _autocorrelations(values: np.ndarray, max_lag: int) -> dict[int, float]:
    centered = values - values.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 1e-12:
        return {}
    result: dict[int, float] = {}
    for lag in range(1, max_lag + 1):
        result[lag] = float(np.dot(centered[:-lag], centered[lag:]) / denominator)
    return result


def _numeric_columns(frame: pd.DataFrame, time_column: str | None) -> list[str]:
    columns: list[str] = []
    for name in frame.columns:
        if name == time_column:
            continue
        converted = pd.to_numeric(frame[name], errors="coerce")
        if converted.notna().sum() >= max(8, len(frame) // 3):
            columns.append(str(name))
    return columns


def _infer_unit_and_constraints(name: str, profile: dict[str, Any]) -> tuple[str, list[str]]:
    text = name.lower()
    unit = "unknown"
    constraints: list[str] = []
    if any(key in text for key in ("temp", "temperature")):
        unit = "degC"
        constraints.append("smooth_environmental")
    elif any(key in text for key in ("rain", "precip")):
        unit = "mm"
        constraints.extend(["nonnegative", "zero_inflated", "event_clustered"])
    elif any(key in text for key in ("load", "power", "electric", "kw")):
        unit = "kW"
        constraints.append("nonnegative")
    elif any(key in text for key in ("humidity", "rh")):
        unit = "%"
        constraints.extend(["bounded_0_100", "smooth_environmental"])
    elif any(key in text for key in ("cpu", "memory", "util", "usage")):
        unit = "%"
        constraints.append("bounded_0_100")
    elif any(key in text for key in ("sales", "revenue", "amount")):
        unit = "currency"
        constraints.append("nonnegative")
    if float(profile.get("nonnegative_ratio", 0.0)) >= 0.999:
        constraints.append("nonnegative")
    if float(profile.get("integer_ratio", 0.0)) >= 0.999:
        constraints.append("integer_like")
    if profile.get("min") is not None and profile.get("max") is not None:
        constraints.append(f"observed_range[{profile['min']:.6g},{profile['max']:.6g}]")
    return unit, sorted(set(constraints))


def _compact_column_profile(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {}
    std = float(finite.std(ddof=0))
    mean = float(finite.mean())
    zero_tolerance = max(1e-12, std * 1e-8)
    quantiles = np.quantile(finite, [0.01, 0.5, 0.99])
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": mean,
        "std": std,
        "p01": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p99": float(quantiles[2]),
        "zero_ratio": float(np.mean(np.abs(finite) <= zero_tolerance)),
        "integer_ratio": float(np.mean(np.isclose(finite, np.round(finite)))),
        "nonnegative_ratio": float(np.mean(finite >= 0.0)),
    }


def _segment_profile(values: np.ndarray, segments: int = 4) -> dict[str, Any]:
    chunks = [chunk for chunk in np.array_split(values, max(1, segments)) if len(chunk)]
    summaries = []
    for index, chunk in enumerate(chunks):
        x = np.arange(len(chunk), dtype=float)
        slope = float(np.polyfit(x, chunk, 1)[0]) if len(chunk) > 1 else 0.0
        summaries.append(
            {
                "segment": index,
                "mean": float(np.mean(chunk)),
                "std": float(np.std(chunk)),
                "slope_per_step": slope,
                "min": float(np.min(chunk)),
                "max": float(np.max(chunk)),
            }
        )
    means = np.array([item["mean"] for item in summaries], dtype=float) if summaries else np.array([0.0])
    return {
        "segments": summaries,
        "mean_range": float(np.max(means) - np.min(means)),
        "relative_mean_range": _finite(float((np.max(means) - np.min(means)) / (abs(float(np.mean(values))) + 1e-9))),
    }


def _event_cluster_profile(values: np.ndarray, zero_tolerance: float) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {}
    high_threshold = float(np.quantile(finite, 0.95))
    high_runs = _run_lengths(values >= high_threshold)
    nonzero_runs = _run_lengths(np.abs(values) > zero_tolerance)
    return {
        "high_value_threshold_p95": high_threshold,
        "high_value_run_count": len(high_runs),
        "median_high_value_run": float(np.median(high_runs)) if high_runs else 0.0,
        "max_high_value_run": int(max(high_runs)) if high_runs else 0,
        "nonzero_run_count": len(nonzero_runs),
        "median_nonzero_run": float(np.median(nonzero_runs)) if nonzero_runs else 0.0,
    }


def _covariance_profile(frame: pd.DataFrame, time_column: str | None, columns: list[str]) -> dict[str, Any]:
    selected = columns[:12]
    if len(selected) < 2:
        return {"columns": selected, "correlation_matrix": []}
    numeric = frame[selected].apply(pd.to_numeric, errors="coerce").interpolate(limit_direction="both").bfill().ffill()
    corr = numeric.corr().fillna(0.0).to_numpy(dtype=float)
    return {
        "columns": selected,
        "correlation_matrix": corr.round(4).tolist(),
        "strong_pairs": [
            {
                "left": selected[i],
                "right": selected[j],
                "correlation": float(corr[i, j]),
            }
            for i in range(len(selected))
            for j in range(i + 1, len(selected))
            if abs(float(corr[i, j])) >= 0.35
        ][:24],
    }


def _infer_relationship_operator(source_values: np.ndarray, target_values: np.ndarray, correlation: float) -> dict[str, Any]:
    finite_source = source_values[np.isfinite(source_values)]
    if len(finite_source) < 8:
        return {
            "operator": "linear_lag",
            "relationship_type": "lagged_statistical_dependency",
            "mechanism_inference": "heuristic_statistical",
            "mechanism_confidence": 0.35,
        }
    target_std = float(np.std(target_values)) + 1e-9
    rounded_source = np.round(source_values, 6)
    finite_rounded_source = np.round(finite_source, 6)
    unique_values, unique_counts = np.unique(finite_rounded_source, return_counts=True)
    unique_count = len(unique_values)
    if unique_count <= 4:
        rare_index = int(np.argmin(unique_counts))
        event_value = float(unique_values[rare_index])
        event_mask = rounded_source == event_value
        event_count = int(event_mask.sum())
        event_fraction = float(event_count / max(1, len(source_values)))
        padded = np.r_[False, event_mask, False]
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        run_lengths = changes[1::2] - changes[::2]
        max_run = int(run_lengths.max()) if len(run_lengths) else event_count
        if 0 < event_count <= max(3, int(0.12 * len(source_values))):
            target_high = float(np.mean(target_values[event_mask]))
            target_low = float(np.mean(target_values[~event_mask])) if np.any(~event_mask) else target_high
            if (
                event_fraction <= 0.12
                and max_run <= max(3, int(0.1 * len(source_values)))
                and abs(target_high - target_low) > 0.5 * target_std
            ):
                event_effect_sign = "positive" if target_high >= target_low else "negative"
                return {
                    "operator": "event_trigger",
                    "threshold": event_value,
                    "trigger_op": "eq",
                    "event_effect_sign": event_effect_sign,
                    "width": max(1, min(3, max_run)),
                    "relationship_type": "event_triggered_response",
                    "mechanism_inference": "heuristic_statistical",
                    "mechanism_confidence": 0.6,
                }
        threshold = float(np.median(finite_source))
        return {
            "operator": "state_gate",
            "threshold": threshold,
            "relationship_type": "state_dependent_effect",
            "mechanism_inference": "heuristic_statistical",
            "mechanism_confidence": 0.65,
        }
    threshold = float(np.median(finite_source))
    high = source_values >= threshold
    low = ~high
    minimum_split = max(4, int(0.25 * len(source_values)))
    if high.sum() >= minimum_split and low.sum() >= minimum_split:
        high_mean = float(np.mean(target_values[high]))
        low_mean = float(np.mean(target_values[low]))
        if abs(high_mean - low_mean) > 0.35 * target_std:
            return {
                "operator": "threshold",
                "threshold": threshold,
                "relationship_type": "threshold_response",
                "mechanism_inference": "heuristic_statistical",
                "mechanism_confidence": 0.6,
            }

    p95 = float(np.quantile(finite_source, 0.95))
    event_mask = source_values >= p95
    event_count = int(event_mask.sum())
    event_fraction = float(event_count / max(1, len(source_values)))
    if 0 < event_count <= max(3, int(0.12 * len(source_values))):
        padded = np.r_[False, event_mask, False]
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        run_lengths = changes[1::2] - changes[::2]
        max_run = int(run_lengths.max()) if len(run_lengths) else event_count
        if event_fraction <= 0.12 and max_run <= max(3, int(0.1 * len(source_values))):
            target_high = float(np.mean(target_values[event_mask]))
            target_low = float(np.mean(target_values[~event_mask])) if np.any(~event_mask) else target_high
            if abs(target_high - target_low) > 0.5 * target_std:
                event_effect_sign = "positive" if target_high >= target_low else "negative"
                return {
                    "operator": "event_trigger",
                    "threshold": p95,
                    "trigger_op": "gte",
                    "event_effect_sign": event_effect_sign,
                    "width": max(1, min(3, max_run)),
                    "relationship_type": "event_triggered_response",
                    "mechanism_inference": "heuristic_statistical",
                    "mechanism_confidence": 0.55,
                }
    p05 = float(np.quantile(finite_source, 0.05))
    low_event_mask = source_values <= p05
    low_event_count = int(low_event_mask.sum())
    low_event_fraction = float(low_event_count / max(1, len(source_values)))
    if 0 < low_event_count <= max(3, int(0.12 * len(source_values))):
        padded = np.r_[False, low_event_mask, False]
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        run_lengths = changes[1::2] - changes[::2]
        max_run = int(run_lengths.max()) if len(run_lengths) else low_event_count
        if low_event_fraction <= 0.12 and max_run <= max(3, int(0.1 * len(source_values))):
            target_high = float(np.mean(target_values[low_event_mask]))
            target_low = float(np.mean(target_values[~low_event_mask])) if np.any(~low_event_mask) else target_high
            if abs(target_high - target_low) > 0.5 * target_std:
                event_effect_sign = "positive" if target_high >= target_low else "negative"
                return {
                    "operator": "event_trigger",
                    "threshold": p05,
                    "trigger_op": "lte",
                    "event_effect_sign": event_effect_sign,
                    "width": max(1, min(3, max_run)),
                    "relationship_type": "event_triggered_response",
                    "mechanism_inference": "heuristic_statistical",
                    "mechanism_confidence": 0.55,
                }
    source_std = float(np.std(source_values)) + 1e-9
    if source_std > 1e-9:
        normalized_source = (source_values - float(np.mean(source_values))) / source_std
        saturating = np.tanh(normalized_source)
        if float(np.std(saturating)) > 1e-9:
            sat_corr = float(np.corrcoef(saturating, target_values)[0, 1])
            if np.isfinite(sat_corr) and abs(sat_corr) > abs(correlation) + 0.05:
                return {
                    "operator": "saturation",
                    "relationship_type": "saturating_response",
                    "mechanism_inference": "heuristic_statistical",
                    "mechanism_confidence": 0.45,
                }
    return {
        "operator": "linear_lag",
        "relationship_type": "lagged_statistical_dependency",
        "mechanism_inference": "heuristic_statistical",
        "mechanism_confidence": 0.4,
    }


def _profile_reference_variables(
    frame: pd.DataFrame,
    time_column: str | None,
    primary_value_column: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    numeric_columns = _numeric_columns(frame, time_column)
    if primary_value_column in numeric_columns:
        numeric_columns = [primary_value_column] + [
            name for name in numeric_columns if name != primary_value_column
        ]
    variables: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for index, name in enumerate(numeric_columns):
        series = pd.to_numeric(frame[name], errors="coerce").astype(float)
        valid = series.dropna().to_numpy(dtype=float)
        if len(valid) < 8:
            continue
        arrays[name] = series.interpolate(limit_direction="both").bfill().ffill().to_numpy(dtype=float)
        profile = _compact_column_profile(valid)
        unit, constraints = _infer_unit_and_constraints(name, profile)
        variables.append(
            {
                "name": str(name),
                "role": "target" if name == primary_value_column or index == 0 else "driver",
                "unit": unit,
                "constraints": constraints,
                "profile": profile,
            }
        )

    relationships: list[dict[str, Any]] = []
    names = [item["name"] for item in variables]
    primary = primary_value_column if primary_value_column in arrays else (names[0] if names else None)
    if primary:
        target = arrays[primary]
        target_std = float(np.std(target))
        for source in names:
            if source == primary or source not in arrays:
                continue
            source_values = arrays[source]
            source_std = float(np.std(source_values))
            if source_std < 1e-9 or target_std < 1e-9:
                continue
            best: tuple[int, float] | None = None
            max_lag = min(24, max(1, len(target) // 8))
            for lag in range(0, max_lag + 1):
                left = source_values[: len(source_values) - lag] if lag else source_values
                right = target[lag:] if lag else target
                if len(left) < 8 or len(right) < 8:
                    continue
                corr = float(np.corrcoef(left, right)[0, 1])
                if not np.isfinite(corr):
                    continue
                if best is None or abs(corr) > abs(best[1]):
                    best = (lag, corr)
            if best and abs(best[1]) >= 0.2:
                aligned_source = source_values[: len(source_values) - best[0]] if best[0] else source_values
                aligned_target = target[best[0]:] if best[0] else target
                mechanism = _infer_relationship_operator(aligned_source, aligned_target, best[1])
                if mechanism["operator"] == "event_trigger":
                    effect = (
                        "positive_event_effect"
                        if mechanism.get("event_effect_sign") == "positive"
                        else "negative_event_effect"
                    )
                else:
                    effect = "positive_lagged_effect" if best[1] >= 0 else "negative_lagged_effect"
                relationships.append(
                    {
                        "source": source,
                        "target": primary,
                        "effect": effect,
                        "operator": mechanism["operator"],
                        "lag": int(best[0]),
                        "correlation": float(best[1]),
                        "relationship_type": mechanism["relationship_type"],
                        "evidence_source": "reference_profile",
                        **{key: value for key, value in mechanism.items() if key not in {"operator", "relationship_type"}},
                    }
                )
    return variables, relationships


def profile_reference_frame(
    frame: pd.DataFrame,
    time_column: str | None = None,
    value_column: str | None = None,
    source: str = "dataframe",
    max_profile_points: int = 50_000,
) -> ReferenceProfile:
    if frame.empty:
        raise ValueError("reference data is empty")
    time_column, value_column = _infer_columns(frame, time_column, value_column)
    raw = pd.to_numeric(frame[value_column], errors="coerce")
    missing_ratio = float(raw.isna().mean())
    valid = raw.dropna().astype(float)
    if len(valid) < 8:
        raise ValueError("reference data needs at least 8 valid values")
    if len(valid) > max_profile_points:
        positions = np.linspace(0, len(valid) - 1, max_profile_points, dtype=int)
        valid = valid.iloc[positions]
    values = valid.to_numpy(dtype=float)

    mean = float(values.mean())
    std = float(values.std(ddof=0))
    quantiles = np.quantile(values, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    standardized = (values - mean) / std if std > 1e-12 else np.zeros_like(values)
    skewness = float(np.mean(standardized**3)) if std > 1e-12 else 0.0
    kurtosis = float(np.mean(standardized**4) - 3.0) if std > 1e-12 else 0.0

    x = np.arange(len(values), dtype=float)
    slope = float(np.polyfit(x, values, 1)[0]) if len(values) > 1 else 0.0
    differences = np.diff(values)
    diff_std = float(differences.std(ddof=0)) if len(differences) else 0.0
    diff_median_abs = float(np.median(np.abs(differences))) if len(differences) else 0.0
    lag_one = float(np.corrcoef(values[:-1], values[1:])[0, 1]) if std > 1e-12 else 0.0

    zero_tolerance = max(1e-12, std * 1e-8)
    zero_mask = np.abs(values) <= zero_tolerance
    nonzero_runs = _run_lengths(~zero_mask)
    zero_runs = _run_lengths(zero_mask)

    max_lag = min(336, max(1, len(values) // 3))
    acf = _autocorrelations(values, max_lag)
    peak_lags = sorted(acf, key=lambda lag: acf[lag], reverse=True)
    peak_lags = [lag for lag in peak_lags if lag > 1 and acf[lag] >= 0.2][:8]

    statistics = {
        "first_value": float(values[0]),
        "last_value": float(values[-1]),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": mean,
        "std": std,
        "coefficient_of_variation": _finite(std / abs(mean)) if abs(mean) > 1e-12 else None,
        "quantiles": {
            key: float(value)
            for key, value in zip(("p01", "p05", "p25", "p50", "p75", "p95", "p99"), quantiles)
        },
        "skewness": skewness,
        "excess_kurtosis": kurtosis,
        "integer_ratio": float(np.mean(np.isclose(values, np.round(values)))),
        "nonnegative_ratio": float(np.mean(values >= 0.0)),
        "positive_mean": float(values[values > zero_tolerance].mean()) if np.any(values > zero_tolerance) else 0.0,
    }
    distribution = {
        "quantile_grid": {
            key: float(value)
            for key, value in zip(
                ("p001", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "p999"),
                np.quantile(values, [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]),
            )
        },
        "upper_tail_fraction_above_p95": float(np.mean(values >= quantiles[5])),
        "lower_tail_fraction_below_p05": float(np.mean(values <= quantiles[1])),
    }
    dynamics = {
        "linear_slope_per_step": slope,
        "total_linear_change": slope * max(0, len(values) - 1),
        "mean_difference": float(differences.mean()) if len(differences) else 0.0,
        "difference_std": diff_std,
        "median_absolute_difference": diff_median_abs,
        "lag1_correlation": _finite(lag_one),
    }
    sparsity = {
        "zero_ratio": float(zero_mask.mean()),
        "nonzero_run_count": len(nonzero_runs),
        "median_nonzero_run": float(np.median(nonzero_runs)) if nonzero_runs else 0.0,
        "median_zero_run": float(np.median(zero_runs)) if zero_runs else 0.0,
    }
    periodicity = {
        "autocorrelation_peaks": [{"lag": lag, "correlation": acf[lag]} for lag in peak_lags],
        "selected_autocorrelations": {
            str(lag): acf[lag] for lag in (1, 2, 6, 12, 24, 48, 168) if lag in acf
        },
    }
    variables, relationships = _profile_reference_variables(frame, time_column, value_column)
    numeric_columns = _numeric_columns(frame, time_column)
    return ReferenceProfile(
        source=source,
        time_column=time_column,
        value_column=value_column,
        sample_count=len(frame),
        valid_count=len(values),
        missing_ratio=missing_ratio,
        inferred_frequency=_infer_frequency(frame, time_column),
        statistics=statistics,
        dynamics=dynamics,
        sparsity=sparsity,
        periodicity=periodicity,
        variables=variables,
        relationships=relationships,
        distribution=distribution,
        nonstationarity=_segment_profile(values),
        event_clusters=_event_cluster_profile(values, zero_tolerance),
        covariance=_covariance_profile(frame, time_column, numeric_columns),
    )


def profile_reference_csv(
    path: str | Path,
    time_column: str | None = None,
    value_column: str | None = None,
) -> ReferenceProfile:
    source = str(Path(path).expanduser().resolve())
    return profile_reference_frame(
        pd.read_csv(source),
        time_column=time_column,
        value_column=value_column,
        source=source,
    )


def profile_reference_arrow(
    path: str | Path,
    value_column: str | None = None,
) -> ReferenceProfile:
    from .compact_storage import read_series_arrow

    source = str(Path(path).expanduser().resolve())
    table, metadata = read_series_arrow(source)
    frame = table.to_pandas()
    time_column = None
    start = metadata.get("start")
    frequency = metadata.get("frequency")
    if start and frequency and len(frame):
        try:
            frame.insert(0, "timestamp", pd.date_range(start=start, periods=len(frame), freq=frequency))
            time_column = "timestamp"
        except Exception:
            time_column = None
    return profile_reference_frame(
        frame,
        time_column=time_column,
        value_column=value_column or ("value" if "value" in frame else None),
        source=source,
    )


def profile_reference_arrow_or_raise(path: str | Path) -> ReferenceProfile:
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".arrow":
        raise ValueError("Reference time series must be an .arrow file")
    return profile_reference_arrow(source)


def validate_reference_strength(strength: str) -> str:
    normalized = strength.lower()
    if normalized not in REFERENCE_STRENGTHS:
        raise ValueError(f"reference_strength must be one of {sorted(REFERENCE_STRENGTHS)}")
    return normalized


def apply_reference_priors(
    plan: Any,
    profile: ReferenceProfile | dict[str, Any] | None,
    strength: str = "structure",
) -> Any:
    if profile is None:
        return plan
    strength = validate_reference_strength(strength)
    payload = profile.to_dict() if isinstance(profile, ReferenceProfile) else profile
    statistics = payload["statistics"]
    dynamics = payload["dynamics"]
    sparsity = payload["sparsity"]
    periodicity = payload["periodicity"]
    quantiles = statistics["quantiles"]

    first = float(statistics["first_value"])
    median = float(quantiles["p50"])
    p01, p99 = float(quantiles["p01"]), float(quantiles["p99"])
    spread = max(float(quantiles["p95"]) - float(quantiles["p05"]), 0.0)
    diff_std = max(float(dynamics["difference_std"]), 0.0)

    if plan.semantic_type in {"cumulative", "stock_flow", "random_walk", "saturation_growth"}:
        plan.semantic_config["initial_value"] = first
    if plan.semantic_type == "cumulative":
        allow_negative = bool(plan.semantic_config.get("allow_negative_increment", False))
        mean_difference = float(dynamics.get("mean_difference", 0.0))
        plan.baseline = mean_difference if allow_negative else max(mean_difference, 0.0)
        plan.noise_sigma = diff_std
    elif plan.semantic_type == "random_walk":
        plan.semantic_config["drift"] = float(dynamics["linear_slope_per_step"])
        plan.semantic_config["volatility"] = diff_std
    elif plan.semantic_type == "saturation_growth":
        plan.semantic_config["capacity"] = max(p99, first)
    elif plan.semantic_type == "instantaneous":
        plan.baseline = median

    if plan.generator_type == "intermittent_event":
        plan.baseline = 0.0
        plan.noise_sigma = 0.0
        plan.domain_params["event_probability"] = float(
            np.clip(1.0 - float(sparsity["zero_ratio"]), 0.01, 0.95)
        )
        plan.domain_params["mean_duration"] = max(1.0, float(sparsity["median_nonzero_run"]))
        positive_mean = float(statistics.get("positive_mean", 0.0))
        if positive_mean > 0:
            plan.domain_params["intensity_scale"] = max(0.1, positive_mean / 1.4)
        event_clusters = payload.get("event_clusters") or {}
        if event_clusters:
            plan.domain_params["reference_event_cluster_profile"] = event_clusters
    else:
        plan.noise_sigma = max(diff_std / np.sqrt(2.0), 1e-9)
    if payload.get("distribution"):
        plan.domain_params["reference_distribution"] = payload["distribution"]
    if payload.get("nonstationarity"):
        plan.domain_params["reference_nonstationarity"] = payload["nonstationarity"]
    if payload.get("covariance"):
        plan.metadata["reference_covariance"] = payload["covariance"]

    if statistics["nonnegative_ratio"] >= 0.999:
        plan.lower_bound = 0.0
        plan.output_constraints.setdefault("nonnegative", True)
    elif plan.semantic_type == "instantaneous":
        plan.lower_bound = None

    if statistics["integer_ratio"] >= 0.999:
        plan.output_constraints.setdefault("integer", True)

    if strength in {"structure", "strict"} and plan.semantic_type == "instantaneous":
        plan.trend_slope = float(dynamics["total_linear_change"])
        selected = periodicity.get("selected_autocorrelations", {})
        inferred_frequency = str(payload.get("inferred_frequency") or "").lower()
        hourly = "hour" in inferred_frequency or inferred_frequency in {"h", "1h"}
        if hourly and float(selected.get("24", 0.0)) >= 0.35:
            plan.daily_amplitude = max(plan.daily_amplitude, spread / 2.0)
        if hourly and float(selected.get("168", 0.0)) >= 0.25:
            plan.weekly_enabled = True
            plan.weekly_amplitude = max(plan.weekly_amplitude, spread / 5.0)

    if strength == "strict" and plan.semantic_type == "instantaneous":
        plan.output_constraints.setdefault("lower_bound", p01)
        plan.output_constraints.setdefault("upper_bound", p99)

    reference_variables = payload.get("variables") or []
    if isinstance(reference_variables, list) and len(reference_variables) > 1:
        plan.variables = [
            {
                "name": str(item.get("name", f"variable_{index + 1}")),
                "role": str(item.get("role", "driver" if index else "target")),
                "unit": str(item.get("unit", "unknown")),
                "source": "reference_profile",
                "profile": item.get("profile", {}),
            }
            for index, item in enumerate(reference_variables)
            if isinstance(item, dict)
        ]
        plan.relationships = [
            item for item in payload.get("relationships", []) if isinstance(item, dict)
        ] or plan.relationships
        plan.metadata["multivariate_from_reference"] = True
        plan.metadata["primary_target"] = plan.variables[0]["name"] if plan.variables else "value"

    plan.metadata["reference_priors_applied"] = True
    plan.metadata["reference_strength"] = strength
    plan.metadata["reference_source"] = payload.get("source")
    plan.metadata["reference_targets"] = {
        "median": median,
        "std": float(statistics["std"]),
        "zero_ratio": float(sparsity["zero_ratio"]),
        "lag1_correlation": dynamics.get("lag1_correlation"),
        "total_linear_change": float(dynamics["total_linear_change"]),
    }
    plan.metadata["reference_executable_targets"] = {
        "distribution_quantiles": dict(payload.get("distribution", {}).get("quantile_grid", {})),
        "acf_targets": dict((periodicity.get("selected_autocorrelations") or {})),
        "event_run_length": {
            "median_nonzero_run": float(sparsity.get("median_nonzero_run", 0.0)),
            "median_zero_run": float(sparsity.get("median_zero_run", 0.0)),
            "zero_ratio": float(sparsity.get("zero_ratio", 0.0)),
        },
        "segment_schedule": payload.get("nonstationarity", {}),
    }
    return plan


def compare_to_reference(frame: pd.DataFrame, profile: ReferenceProfile | dict[str, Any]) -> dict[str, Any]:
    reference = profile.to_dict() if isinstance(profile, ReferenceProfile) else profile
    generated = profile_reference_frame(frame, value_column="value", source="generated").to_dict()
    ref_stats, gen_stats = reference["statistics"], generated["statistics"]
    ref_dyn, gen_dyn = reference["dynamics"], generated["dynamics"]
    ref_sparse, gen_sparse = reference["sparsity"], generated["sparsity"]

    def relative_error(left: float, right: float) -> float:
        return float(abs(left - right) / max(abs(left), 1e-9))

    return {
        "mean_relative_error": relative_error(float(ref_stats["mean"]), float(gen_stats["mean"])),
        "std_relative_error": relative_error(float(ref_stats["std"]), float(gen_stats["std"])),
        "zero_ratio_absolute_error": abs(float(ref_sparse["zero_ratio"]) - float(gen_sparse["zero_ratio"])),
        "lag1_absolute_error": abs(
            float(ref_dyn.get("lag1_correlation") or 0.0) - float(gen_dyn.get("lag1_correlation") or 0.0)
        ),
        "trend_relative_error": relative_error(
            float(ref_dyn["total_linear_change"]), float(gen_dyn["total_linear_change"])
        ),
        "generated_profile": generated,
    }


def profile_json(profile: ReferenceProfile | dict[str, Any]) -> str:
    payload = profile.to_dict() if isinstance(profile, ReferenceProfile) else profile
    return json.dumps(payload, ensure_ascii=False, indent=2)
