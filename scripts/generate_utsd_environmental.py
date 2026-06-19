from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from generation_agent.planner import SeriesPlan


TOTAL_POINTS = 31_546_920
SEQUENCE_COUNT = 20
POINTS_PER_SEQUENCE = TOTAL_POINTS // SEQUENCE_COUNT


def _number(value, default: float, preferred_keys: tuple[str, ...] = ()) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in preferred_keys:
            candidate = value.get(key)
            if isinstance(candidate, (int, float)):
                return float(candidate)
        for candidate in value.values():
            if isinstance(candidate, (int, float)):
                return float(candidate)
    return float(default)


def _smooth_noise(rng: np.random.Generator, length: int, width: int) -> np.ndarray:
    noise = rng.normal(0.0, 1.0, length + width - 1)
    kernel = np.exp(-np.linspace(0.0, 5.0, width))
    kernel /= kernel.sum()
    return np.convolve(noise, kernel, mode="valid")


def _inject_events(
    values: np.ndarray,
    rng: np.random.Generator,
    count: int,
    magnitude: float,
    width: int,
    direction: str = "positive",
) -> np.ndarray:
    flags = np.zeros(len(values), dtype=np.uint8)
    if count <= 0:
        return flags
    positions = rng.choice(len(values), size=min(count, len(values)), replace=False)
    scale = float(np.std(values)) or 1.0
    for position in positions:
        event_width = max(1, int(rng.integers(max(1, width // 2), width + 1)))
        end = min(len(values), position + event_width)
        envelope = np.sin(np.linspace(0.15, np.pi - 0.15, end - position))
        sign = 1.0 if direction == "positive" else -1.0
        values[position:end] += sign * magnitude * scale * np.maximum(envelope, 0.15)
        flags[position:end] = 1
    return flags


def _rainfall(plan: SeriesPlan, length: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    params = {**plan.domain_params, **plan.semantic_config}
    values = np.zeros(length, dtype=np.float32)
    flags = np.zeros(length, dtype=np.uint8)
    dry_mean = float(params.get("dry_spell_mean", 1.0 / max(float(params.get("event_probability", 0.08)), 1e-4)))
    wet_mean = float(params.get("wet_spell_mean", params.get("mean_duration", 4.0)))
    shape = float(params.get("intensity_shape", 1.4))
    scale = float(params.get("intensity_scale", 4.0))
    storm_probability = float(params.get("storm_probability", 0.06))
    storm_multiplier = float(params.get("storm_multiplier", 3.5))

    cursor = 0
    while cursor < length:
        cursor += int(rng.geometric(1.0 / dry_mean))
        if cursor >= length:
            break
        duration = int(max(1, rng.geometric(1.0 / wet_mean)))
        end = min(length, cursor + duration)
        storm = rng.random() < storm_probability
        event_scale = scale * (storm_multiplier if storm else 1.0)
        raw = rng.gamma(shape, event_scale, end - cursor)
        envelope = np.sin(np.linspace(0.12, np.pi - 0.12, end - cursor))
        values[cursor:end] = (raw * np.maximum(envelope, 0.12)).astype(np.float32)
        if storm:
            flags[cursor:end] = 1
        cursor = end
    return values, flags


def _pm25(plan: SeriesPlan, length: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    params = {**plan.domain_params, **plan.semantic_config}
    index = np.arange(length, dtype=np.float64)
    smooth = _smooth_noise(
        rng,
        length,
        int(_number(params.get("correlation_width"), 64, ("width",))),
    )
    broad = _smooth_noise(
        rng,
        length,
        int(_number(params.get("regime_width"), 1200, ("width",))),
    )
    values = (
        plan.baseline
        + plan.noise_sigma * smooth
        + _number(
            params.get("regime_amplitude"),
            max(plan.baseline * 0.5, 10.0),
            ("amplitude",),
        )
        * np.tanh(broad)
        + plan.daily_amplitude
        * np.sin(
            2.0
            * np.pi
            * index
            / _number(params.get("short_cycle"), 240.0, ("period_index_units", "period"))
        )
        + plan.seasonal_amplitude
        * np.sin(
            2.0
            * np.pi
            * index
            / _number(params.get("long_cycle"), 8400.0, ("period_index_units", "period"))
            + 0.8
        )
    )
    values = np.maximum(values, 0.0).astype(np.float32)
    flags = _inject_events(
        values,
        rng,
        plan.anomaly_count if plan.anomaly_enabled else 0,
        plan.anomaly_magnitude,
        plan.anomaly_width,
        direction="positive",
    )
    return np.maximum(values, 0.0), flags


def _benzene(plan: SeriesPlan, length: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    params = {**plan.domain_params, **plan.semantic_config}
    index = np.arange(length, dtype=np.float64)
    background = _smooth_noise(
        rng,
        length,
        int(_number(params.get("correlation_width"), 48, ("width",))),
    )
    activity_cycle = _number(
        params.get("activity_cycle"),
        240.0,
        ("period_index_units", "cycle_period", "period"),
    )
    values = (
        plan.baseline
        + plan.noise_sigma * background
        + plan.daily_amplitude * np.maximum(
            np.sin(2.0 * np.pi * index / activity_cycle - 0.8),
            -0.35,
        )
    ).astype(np.float32)
    flags = _inject_events(
        values,
        rng,
        plan.anomaly_count if plan.anomaly_enabled else 0,
        plan.anomaly_magnitude,
        plan.anomaly_width,
        direction="positive",
    )
    sensor_dropout_count = int(params.get("sensor_dropout_count", 20))
    if sensor_dropout_count:
        positions = rng.choice(length, size=sensor_dropout_count, replace=False)
        for position in positions:
            end = min(length, position + int(rng.integers(2, 10)))
            values[position:end] *= rng.uniform(0.0, 0.15)
            flags[position:end] = 1
    return np.maximum(values, 0.0), flags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/data/czj/syn")
    parser.add_argument("--plans", required=True)
    parser.add_argument("--seed", type=int, default=20260615)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_plans = json.loads(Path(args.plans).read_text(encoding="utf-8"))
    plans = [SeriesPlan.from_dict(item) for item in raw_plans]
    if len(plans) != SEQUENCE_COUNT:
        raise ValueError(f"Expected {SEQUENCE_COUNT} LLM plans, got {len(plans)}")
    if any(not plan.metadata.get("synthetic_only") or plan.metadata.get("reference_data_used") for plan in plans):
        raise ValueError("Every plan must be marked synthetic_only with reference_data_used=false")
    variants: dict[str, int] = {}
    manifest = {
        "sequence_count": SEQUENCE_COUNT,
        "total_points": TOTAL_POINTS,
        "points_per_sequence": POINTS_PER_SEQUENCE,
        "storage": "independent float32 .npy long univariate sequences",
        "frequency_policy": "The UTSD freq field is not used to infer real sampling frequency.",
        "sequences": [],
    }

    for sequence_id, plan in enumerate(plans):
        if sequence_id < 7:
            domain = "AustraliaRainfall"
        elif sequence_id < 14:
            domain = "BeijingPM25Quality"
        else:
            domain = "BenzeneConcentration"
        variant = variants.get(domain, 0)
        variants[domain] = variant + 1
        plan.domain = domain
        stem = f"{sequence_id:02d}_{domain}_{variant:02d}"
        value_path = output_dir / f"{stem}.npy"
        anomaly_path = output_dir / f"{stem}_anomaly.npy"
        plan_path = output_dir / f"{stem}_plan.json"

        if value_path.exists() and anomaly_path.exists():
            values = np.load(value_path, mmap_mode="r", allow_pickle=False)
            anomaly = np.load(anomaly_path, mmap_mode="r", allow_pickle=False)
            if values.size != POINTS_PER_SEQUENCE or anomaly.size != POINTS_PER_SEQUENCE:
                raise ValueError(f"Incomplete checkpoint for {stem}")
            status = "reused"
        else:
            rng = np.random.default_rng(args.seed + sequence_id)
            if domain == "AustraliaRainfall":
                values, anomaly = _rainfall(plan, POINTS_PER_SEQUENCE, rng)
            elif domain == "BeijingPM25Quality":
                values, anomaly = _pm25(plan, POINTS_PER_SEQUENCE, rng)
            else:
                values, anomaly = _benzene(plan, POINTS_PER_SEQUENCE, rng)
            np.save(value_path, values.astype(np.float32, copy=False), allow_pickle=False)
            np.save(anomaly_path, anomaly.astype(np.uint8, copy=False), allow_pickle=False)
            status = "generated"

        if not plan_path.exists():
            plan_path.write_text(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["sequences"].append(
            {
                "id": sequence_id,
                "domain": domain,
                "variant": variant,
                "prompt": plan.metadata.get("description", ""),
                "value_file": value_path.name,
                "anomaly_file": anomaly_path.name,
                "plan_file": plan_path.name,
                "points": int(values.size),
                "dtype": str(values.dtype),
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "zero_ratio": float(np.mean(values == 0.0)),
                "anomaly_points": int(anomaly.sum()),
            }
        )
        print(
            f"[{sequence_id + 1:02d}/{SEQUENCE_COUNT}] {stem}: "
            f"{values.size:,} points ({status})",
            flush=True,
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {SEQUENCE_COUNT} sequences and {TOTAL_POINTS:,} points to {output_dir}")


if __name__ == "__main__":
    main()
