from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/utsd_synthetic")
    parser.add_argument("--output-dir", default="outputs/utsd_synthetic_compact")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    compact_manifest = {
        "sequence_count": manifest["sequence_count"],
        "total_points": manifest["total_points"],
        "encoding": "per-sequence uint8 linear quantization in compressed NPZ",
        "frequency_policy": manifest["frequency_policy"],
        "source_manifest": str(input_dir / "manifest.json"),
        "sequences": [],
    }

    for position, item in enumerate(manifest["sequences"], start=1):
        values = np.load(input_dir / item["value_file"], mmap_mode="r", allow_pickle=False)
        anomaly = np.load(input_dir / item["anomaly_file"], mmap_mode="r", allow_pickle=False)
        minimum = float(values.min())
        maximum = float(values.max())
        scale = (maximum - minimum) / 255.0 if maximum > minimum else 1.0
        quantized = np.rint((np.asarray(values) - minimum) / scale).clip(0, 255).astype(np.uint8)
        packed_anomaly = np.packbits(np.asarray(anomaly, dtype=np.uint8))
        output_name = Path(item["value_file"]).with_suffix(".npz").name
        np.savez_compressed(
            output_dir / output_name,
            values=quantized,
            anomaly_bits=packed_anomaly,
            value_min=np.float32(minimum),
            value_scale=np.float32(scale),
            point_count=np.int64(values.size),
        )
        compact_manifest["sequences"].append(
            {
                "id": item["id"],
                "domain": item["domain"],
                "file": output_name,
                "points": int(values.size),
                "decode": "value = uint8_value * value_scale + value_min",
                "anomaly_decode": "np.unpackbits(anomaly_bits)[:point_count]",
                "max_absolute_quantization_error": scale / 2.0,
            }
        )
        print(f"[{position:02d}/{manifest['sequence_count']}] {output_name}", flush=True)

    (output_dir / "manifest.json").write_text(
        json.dumps(compact_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
