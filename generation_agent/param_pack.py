from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .compact_storage import preview_points, write_series_arrow
from .context_compactor import compact_plan_for_metadata
from .planner import SeriesPlan
from .synthesizer import synthesize_series
from .workflow import LOCAL_KERNEL_NAME


PARAM_PACK_FORMAT = "generation-agent-param-pack"
PARAM_PACK_VERSION = "0.1"


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _source_hash() -> str:
    root = Path(__file__).resolve().parent
    names = (
        "planner.py",
        "synthesizer.py",
        "component_workflow.py",
        "semantic_transforms.py",
        "semantic_validators.py",
        "features.py",
    )
    digest = hashlib.sha256()
    for name in names:
        path = root / name
        if path.exists():
            digest.update(name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _pack_path(path: str | Path) -> Path:
    target = Path(path)
    text = str(target)
    if text.endswith(".syn.json.gz"):
        return target
    if target.suffix:
        return Path(text + ".syn.json.gz")
    return target.with_suffix(".syn.json.gz")


def build_param_pack(
    *,
    description: str,
    plan: SeriesPlan,
    length: int,
    freq: str,
    start: str,
    seed: int | None,
    metadata: dict[str, Any],
    frame: pd.DataFrame | None = None,
    preview_max_points: int = 400,
) -> dict[str, Any]:
    return {
        "format": PARAM_PACK_FORMAT,
        "version": PARAM_PACK_VERSION,
        "description": description,
        "length": int(length),
        "frequency": freq,
        "start": start,
        "seed": seed,
        "plan": plan.to_dict(),
        "plan_summary": compact_plan_for_metadata(plan.to_dict()),
        "metadata": metadata,
        "preview": preview_points(frame, max_points=preview_max_points) if frame is not None else None,
        "replay_contract": {
            "llm_required_for_materialize": False,
            "numeric_values_computed_by": LOCAL_KERNEL_NAME,
            "deterministic_inputs": [
                "plan",
                "length",
                "frequency",
                "start",
                "seed",
                "generator_source_hash",
            ],
        },
        "generator": {
            "name": LOCAL_KERNEL_NAME,
            "source_hash": _source_hash(),
        },
    }


def write_param_pack(
    path: str | Path,
    *,
    description: str,
    plan: SeriesPlan,
    length: int,
    freq: str,
    start: str,
    seed: int | None,
    metadata: dict[str, Any],
    frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    target = _pack_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = build_param_pack(
        description=description,
        plan=plan,
        length=length,
        freq=freq,
        start=start,
        seed=seed,
        metadata=metadata,
        frame=frame,
    )
    raw = _json_bytes(payload)
    with gzip.open(target, "wb") as handle:
        handle.write(raw)
    return {
        "path": str(target),
        "rows": int(length),
        "columns": [],
        "bytes": target.stat().st_size,
        "uncompressed_bytes": len(raw),
        "format": PARAM_PACK_FORMAT,
    }


def read_param_pack(path: str | Path) -> dict[str, Any]:
    with gzip.open(Path(path), "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("format") != PARAM_PACK_FORMAT:
        raise ValueError("Input is not a generation-agent parameter pack")
    if payload.get("version") != PARAM_PACK_VERSION:
        raise ValueError(f"Unsupported parameter pack version: {payload.get('version')}")
    if not isinstance(payload.get("plan"), dict):
        raise ValueError("Parameter pack does not contain a replayable plan")
    return payload


def materialize_param_pack(
    pack_path: str | Path,
    output_path: str | Path,
    *,
    verify_source_hash: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pack = read_param_pack(pack_path)
    expected_hash = str(pack.get("generator", {}).get("source_hash", ""))
    current_hash = _source_hash()
    if verify_source_hash and expected_hash and expected_hash != current_hash:
        raise ValueError(
            "Generator source hash differs from the parameter pack. "
            "Run without verify_source_hash to materialize with the current generator."
        )
    plan = SeriesPlan.from_dict(pack["plan"])
    length = int(pack["length"])
    freq = str(pack["frequency"])
    start = str(pack["start"])
    seed = pack.get("seed")
    seed = int(seed) if seed is not None else None
    frame = synthesize_series(plan, length=length, freq=freq, start=start, seed=seed)
    metadata = dict(pack.get("metadata") or {})
    metadata["materialized_from"] = {
        "path": str(pack_path),
        "format": PARAM_PACK_FORMAT,
        "source_hash_at_pack_time": expected_hash,
        "source_hash_at_materialize_time": current_hash,
        "source_hash_match": bool(expected_hash == current_hash),
    }
    storage = write_series_arrow(frame, output_path, metadata=metadata)
    return frame, storage
