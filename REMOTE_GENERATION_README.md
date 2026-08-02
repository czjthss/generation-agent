# Remote synthetic UTSD-style data

All sequences are synthetic. No AustraliaRainfall, BeijingPM25Quality, or
BenzeneConcentration source data was downloaded, read, fitted, or sampled.
The dataset names were used only as natural-language domain descriptions.

## Locations

- Agent code: `/path/to/generation-agent`
- LLM plans: `/path/to/generation-agent/utsd_llm_plans.json`
- Lossless float32 output: `outputs/utsd_synthetic`
- Compact quantized output: `outputs/utsd_synthetic_compact`

## Dataset shape

- 20 independent long univariate sequences
- 1,577,346 points per sequence
- 31,546,920 total points
- 7 AustraliaRainfall variants
- 7 BeijingPM25Quality variants
- 6 BenzeneConcentration variants

The UTSD `freq` field is not used to infer a physical sampling frequency.
Cycles in the plans are latent index-space structures only.

## Lossless format

Each sequence has:

- `<name>.npy`: float32 values
- `<name>_anomaly.npy`: uint8 anomaly flags
- `<name>_plan.json`: the corresponding LLM plan

Metadata and statistics are stored in `outputs/utsd_synthetic/manifest.json`.

## Compact format

Each `.npz` stores uint8 quantized values, packed anomaly bits, and decoding
parameters:

```python
import numpy as np

item = np.load("outputs/utsd_synthetic_compact/00_AustraliaRainfall_00.npz")
n = int(item["point_count"])
values = item["values"].astype(np.float32) * item["value_scale"] + item["value_min"]
anomaly = np.unpackbits(item["anomaly_bits"])[:n]
```

The compact representation preserves the sequence length but is lossy because
values are linearly quantized to uint8. Use the float32 files when exact
generated values are required.

## Reproduce

```bash
cd /path/to/generation-agent

PYTHONPATH=. .venv/bin/python -u scripts/generate_utsd_environmental.py \
  --plans utsd_llm_plans.json \
  --output-dir outputs/utsd_synthetic

.venv/bin/python -u scripts/compact_utsd_environmental.py \
  --input-dir outputs/utsd_synthetic \
  --output-dir outputs/utsd_synthetic_compact
```
