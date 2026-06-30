from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeContext:
    index: pd.DatetimeIndex
    elapsed_hours: np.ndarray
    elapsed_days: np.ndarray
    hour_of_day: np.ndarray
    day_of_week: np.ndarray
    day_code: np.ndarray
    step_hours: float
    steps_per_hour: float
    has_intraday_resolution: bool


def build_time_context(length: int, freq: str, start: str) -> TimeContext:
    index = pd.date_range(start=start, periods=length, freq=freq)
    if length <= 1:
        step_hours = 24.0 if str(freq).lower().startswith(("d", "w", "m")) else 1.0
    else:
        delta = (index[1] - index[0]).total_seconds() / 3600.0
        step_hours = float(abs(delta)) if delta else 1.0
    elapsed_hours = np.arange(length, dtype=float) * step_hours
    normalized_days = index.normalize()
    _, day_code = np.unique(normalized_days.astype("int64"), return_inverse=True)
    return TimeContext(
        index=index,
        elapsed_hours=elapsed_hours,
        elapsed_days=elapsed_hours / 24.0,
        hour_of_day=index.hour.to_numpy(dtype=float)
        + index.minute.to_numpy(dtype=float) / 60.0
        + index.second.to_numpy(dtype=float) / 3600.0,
        day_of_week=index.dayofweek.to_numpy(dtype=int),
        day_code=day_code.astype(int),
        step_hours=step_hours,
        steps_per_hour=float(1.0 / step_hours) if step_hours > 0 else 1.0,
        has_intraday_resolution=bool(step_hours < 24.0),
    )


def daily_load_shape(context: TimeContext, amplitude: float = 1.0, phase: float = 0.0) -> np.ndarray:
    if not context.has_intraday_resolution:
        return np.zeros(len(context.index), dtype=float)
    hour = (context.hour_of_day + phase) % 24.0
    morning = np.exp(-0.5 * ((hour - 9.0) / 2.2) ** 2)
    afternoon = 1.15 * np.exp(-0.5 * ((hour - 15.0) / 3.0) ** 2)
    evening = 0.55 * np.exp(-0.5 * ((hour - 20.0) / 2.4) ** 2)
    night_dip = -0.45 * np.exp(-0.5 * ((hour - 3.0) / 2.6) ** 2)
    shape = morning + afternoon + evening + night_dip
    shape = (shape - shape.mean()) / (shape.std() + 1e-9)
    return float(amplitude) * shape


def working_day_gate_from_time(context: TimeContext, weekend_factor: float = 0.72) -> np.ndarray:
    gate = np.ones(len(context.index), dtype=float)
    gate[(context.day_of_week == 5) | (context.day_of_week == 6)] = float(weekend_factor)
    return gate


def weekly_cycle_from_time(context: TimeContext, amplitude: float = 1.0, phase: float = -0.8) -> np.ndarray:
    return float(amplitude) * np.sin(2.0 * np.pi * context.elapsed_days / 7.0 + phase)


def seasonal_cycle_from_time(context: TimeContext, amplitude: float = 1.0, period_days: float = 30.0, phase: float = 0.5) -> np.ndarray:
    period = max(float(period_days), 1.0)
    return float(amplitude) * np.sin(2.0 * np.pi * context.elapsed_days / period + phase)


def heat_index_effect_from_time(context: TimeContext, amplitude: float = 1.0) -> np.ndarray:
    if not context.has_intraday_resolution:
        return np.zeros(len(context.index), dtype=float)
    hour = context.hour_of_day
    afternoon_heat = np.exp(-0.5 * ((hour - 14.0) / 4.2) ** 2)
    humid_night = 0.35 * np.exp(-0.5 * ((hour - 22.0) / 4.5) ** 2)
    shape = afternoon_heat + humid_night
    shape = (shape - shape.mean()) / (shape.std() + 1e-9)
    return float(amplitude) * shape


def hours_to_steps(context: TimeContext, hours: float, minimum: int = 1) -> int:
    return max(int(minimum), int(round(float(hours) * context.steps_per_hour)))
