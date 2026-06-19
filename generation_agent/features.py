from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AnomalyConfig:
    enabled: bool = False
    count: int = 0
    magnitude: float = 1.0
    width: int = 1
    kind: str = "spike"
    severity: str = "medium"
    direction: str = "both"


def linear_trend(length: int, slope: float, intercept: float = 0.0) -> np.ndarray:
    x = np.linspace(0.0, 1.0, length)
    return intercept + slope * x


def piecewise_trend(length: int, points: list[tuple[float, float]]) -> np.ndarray:
    if not points:
        return np.zeros(length)
    points = sorted((max(0.0, min(1.0, p)), v) for p, v in points)
    xs = np.linspace(0.0, 1.0, length)
    px = np.array([p for p, _ in points])
    py = np.array([v for _, v in points])
    return np.interp(xs, px, py)


def sinusoidal_cycle(length: int, period: float, amplitude: float, phase: float = 0.0) -> np.ndarray:
    x = np.arange(length, dtype=float)
    return amplitude * np.sin(2.0 * np.pi * x / period + phase)


def daily_load_cycle(length: int, period: int = 24, amplitude: float = 1.0, phase: float = 0.0) -> np.ndarray:
    hour = (np.arange(length, dtype=float) + phase) % period
    morning = np.exp(-0.5 * ((hour - 9.0) / 2.2) ** 2)
    afternoon = 1.15 * np.exp(-0.5 * ((hour - 15.0) / 3.0) ** 2)
    evening = 0.55 * np.exp(-0.5 * ((hour - 20.0) / 2.4) ** 2)
    night_dip = -0.45 * np.exp(-0.5 * ((hour - 3.0) / 2.6) ** 2)
    shape = morning + afternoon + evening + night_dip
    shape = (shape - shape.mean()) / (shape.std() + 1e-9)
    return amplitude * shape


def working_day_gate(length: int, period: int = 24, weekend_factor: float = 0.72) -> np.ndarray:
    day = (np.arange(length) // period) % 7
    gate = np.ones(length)
    gate[(day == 5) | (day == 6)] = weekend_factor
    return gate


def heat_index_effect(length: int, period: int = 24, amplitude: float = 1.0) -> np.ndarray:
    hour = np.arange(length) % period
    afternoon_heat = np.exp(-0.5 * ((hour - 14.0) / 4.2) ** 2)
    humid_night = 0.35 * np.exp(-0.5 * ((hour - 22.0) / 4.5) ** 2)
    shape = afternoon_heat + humid_night
    shape = (shape - shape.mean()) / (shape.std() + 1e-9)
    return amplitude * shape


def gaussian_noise(length: int, rng: np.random.Generator, sigma: float) -> np.ndarray:
    return rng.normal(0.0, sigma, size=length)


def add_anomalies(
    values: np.ndarray,
    rng: np.random.Generator,
    config: AnomalyConfig,
) -> tuple[np.ndarray, np.ndarray]:
    result = values.copy()
    flags = np.zeros(len(values), dtype=int)
    if not config.enabled or config.count <= 0 or len(values) == 0:
        return result, flags

    width = max(1, int(config.width))
    count = min(config.count, len(values))
    positions = rng.choice(len(values), size=count, replace=False)
    scale = np.std(values) or 1.0

    for pos in positions:
        start = max(0, pos - width // 2)
        end = min(len(values), start + width)
        if config.direction == "positive":
            sign = 1.0
        elif config.direction == "negative":
            sign = -1.0
        else:
            sign = rng.choice([-1.0, 1.0])
        if config.kind == "drop":
            delta = -abs(config.magnitude) * scale
        elif config.kind == "positive_spike":
            delta = abs(config.magnitude) * scale
        elif config.kind == "shift":
            delta = sign * config.magnitude * scale
        else:
            delta = sign * config.magnitude * scale
        result[start:end] += delta
        flags[start:end] = 1
    return result, flags
