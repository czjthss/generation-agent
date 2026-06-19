from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc


ROW_METADATA_COLUMNS = {"timestamp", "unit", "domain", "generator_type", "semantic_type"}


def preview_points(frame: pd.DataFrame, max_points: int = 400) -> dict[str, Any]:
    if frame.empty:
        return {"points": [], "count": 0}
    count = min(max_points, len(frame))
    positions = np.unique(np.linspace(0, len(frame) - 1, count, dtype=int))
    subset = frame.iloc[positions]
    return {
        "count": int(len(frame)),
        "points": [
            {"time": str(row.timestamp), "value": float(row.value)}
            for row in subset[["timestamp", "value"]].itertuples(index=False)
        ],
    }


def compact_arrow_table(frame: pd.DataFrame) -> pa.Table:
    arrays: dict[str, pa.Array] = {}
    for name in frame.columns:
        if name in ROW_METADATA_COLUMNS:
            continue
        if name == "value":
            arrays[name] = pa.array(frame[name].astype("float32"), type=pa.float32())
        elif name == "anomaly":
            arrays[name] = pa.array(frame[name].astype("int8"), type=pa.int8())
        elif pd.api.types.is_numeric_dtype(frame[name]):
            arrays[name] = pa.array(frame[name].astype("float32"), type=pa.float32())
    return pa.table(arrays)


def write_series_arrow(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table = compact_arrow_table(frame)
    schema_metadata = {
        "generation_agent": json.dumps(metadata, ensure_ascii=False),
        "timestamp_policy": "start/frequency/length metadata; timestamp is not stored per row",
        "value_dtype": "float32",
    }
    table = table.replace_schema_metadata(
        {key.encode("utf-8"): value.encode("utf-8") for key, value in schema_metadata.items()}
    )
    with output.open("wb") as handle:
        with ipc.new_file(handle, table.schema) as writer:
            writer.write_table(table)
    return {
        "path": str(output),
        "rows": int(table.num_rows),
        "columns": table.column_names,
        "bytes": output.stat().st_size,
    }


def read_series_arrow(path: str | Path) -> tuple[pa.Table, dict[str, Any]]:
    with ipc.open_file(Path(path)) as reader:
        table = reader.read_all()
        raw = table.schema.metadata or {}
    payload = raw.get(b"generation_agent", b"{}").decode("utf-8")
    return table, json.loads(payload)
