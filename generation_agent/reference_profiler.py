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
        raise ValueError("参考时间序列必须是 .arrow 文件")
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
    else:
        plan.noise_sigma = max(diff_std / np.sqrt(2.0), 1e-9)

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
