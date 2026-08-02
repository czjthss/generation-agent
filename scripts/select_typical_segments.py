from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _features(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(x))
    std = float(np.std(x))
    centered = x - mean
    skew = float(np.mean(centered**3) / (std**3 + 1e-12))
    lag1 = float(np.corrcoef(x[:-1], x[1:])[0, 1]) if std > 1e-12 else 1.0
    return np.array(
        [
            mean,
            std,
            float(np.quantile(x, 0.5)),
            float(np.quantile(x, 0.9)),
            float(np.quantile(x, 0.99)),
            float(np.mean(x == 0.0)),
            float(np.std(np.diff(x))),
            skew,
            lag1,
        ],
        dtype=np.float64,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/utsd_synthetic")
    parser.add_argument("--output-dir", default="outputs/utsd_synthetic_samples")
    parser.add_argument("--window", type=int, default=4096)
    parser.add_argument("--stride", type=int, default=16384)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))

    candidates: dict[str, list[dict]] = defaultdict(list)
    for item in manifest["sequences"]:
        values = np.load(input_dir / item["value_file"], mmap_mode="r", allow_pickle=False)
        anomaly = np.load(input_dir / item["anomaly_file"], mmap_mode="r", allow_pickle=False)
        for start in range(0, values.size - args.window + 1, args.stride):
            segment = np.asarray(values[start : start + args.window])
            candidates[item["domain"]].append(
                {
                    "item": item,
                    "start": start,
                    "features": _features(segment),
                    "anomaly_count": int(np.asarray(anomaly[start : start + args.window]).sum()),
                }
            )

    selected_manifest = {
        "selection": (
            "Candidate window with minimum robust feature distance to the domain median. "
            "Features: mean, std, median, q90, q99, zero ratio, difference std, skewness, lag-1 correlation."
        ),
        "window_points": args.window,
        "source": str(input_dir),
        "samples": [],
    }

    for domain, domain_candidates in candidates.items():
        matrix = np.vstack([candidate["features"] for candidate in domain_candidates])
        center = np.median(matrix, axis=0)
        scale = np.median(np.abs(matrix - center), axis=0)
        scale = np.where(scale < 1e-9, np.std(matrix, axis=0), scale)
        scale = np.where(scale < 1e-9, 1.0, scale)
        distances = np.sqrt(np.mean(((matrix - center) / scale) ** 2, axis=1))

        if domain == "AustraliaRainfall":
            zero_ratio = matrix[:, 5]
            valid = (zero_ratio > 0.55) & (zero_ratio < 0.95) & (matrix[:, 4] > 0.0)
            distances = np.where(valid, distances, np.inf)

        selected_index = int(np.argmin(distances))
        selected = domain_candidates[selected_index]
        item = selected["item"]
        start = selected["start"]
        values = np.load(input_dir / item["value_file"], mmap_mode="r", allow_pickle=False)
        anomaly = np.load(input_dir / item["anomaly_file"], mmap_mode="r", allow_pickle=False)
        segment = np.asarray(values[start : start + args.window], dtype=np.float32)
        flags = np.asarray(anomaly[start : start + args.window], dtype=np.uint8)

        csv_path = output_dir / f"{domain}_typical.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["index", "value", "anomaly"])
            writer.writerows((i, float(value), int(flag)) for i, (value, flag) in enumerate(zip(segment, flags)))
        np.save(output_dir / f"{domain}_typical.npy", segment, allow_pickle=False)

        feature_names = [
            "mean",
            "std",
            "median",
            "q90",
            "q99",
            "zero_ratio",
            "difference_std",
            "skewness",
            "lag1_correlation",
        ]
        statistics = dict(zip(feature_names, map(float, selected["features"])))
        selected_manifest["samples"].append(
            {
                "domain": domain,
                "source_file": item["value_file"],
                "source_start": start,
                "points": args.window,
                "csv_file": csv_path.name,
                "npy_file": f"{domain}_typical.npy",
                "anomaly_points": int(flags.sum()),
                "robust_distance": float(distances[selected_index]),
                "statistics": statistics,
            }
        )
        print(
            f"{domain}: {item['value_file']}[{start}:{start + args.window}], "
            f"distance={distances[selected_index]:.4f}",
            flush=True,
        )

    (output_dir / "samples_manifest.json").write_text(
        json.dumps(selected_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
