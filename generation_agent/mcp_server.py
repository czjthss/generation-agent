from __future__ import annotations

import json
from pathlib import Path

from .agent import GenerationAgent, _generation_metadata
from .codegen import render_generator_code
from .compact_storage import write_series_arrow
from .dataset_generator import DatasetGenerator
from .param_pack import materialize_param_pack
from .planner import DEFAULT_LLM_MODEL, plan_from_description
from .reference_profiler import profile_reference_arrow_or_raise, profile_reference_csv
from .semantic_types import AnomalyOverrides

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover
    class FastMCP:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs) -> None:
            self.available = False

        def tool(self):
            def decorator(func):
                return func

            return decorator

        def run(self) -> None:
            raise RuntimeError("MCP server dependencies are not installed. Install with: pip install mcp")


mcp = FastMCP("time-series-generation-agent")


def _reference_payload(
    reference_arrow_path: str | None,
    reference_csv_path: str | None,
) -> dict | None:
    if reference_arrow_path and reference_csv_path:
        raise ValueError("Pass only one reference input: reference_arrow_path or reference_csv_path")
    if reference_arrow_path:
        return profile_reference_arrow_or_raise(reference_arrow_path).to_dict()
    if reference_csv_path:
        return profile_reference_csv(reference_csv_path).to_dict()
    else:
        return None


@mcp.tool()
def plan_time_series(
    description: str,
    model: str | None = DEFAULT_LLM_MODEL,
    reference_arrow_path: str | None = None,
    reference_csv_path: str | None = None,
    cost_mode: str = "balanced",
) -> str:
    """Convert a natural-language data request into a time-series synthesis plan."""
    plan = plan_from_description(
        description,
        model=model,
        reference_profile=_reference_payload(reference_arrow_path, reference_csv_path),
        cost_mode=cost_mode,
    )
    return json.dumps(plan.to_dict(), ensure_ascii=False, indent=2)


@mcp.tool()
def generate_time_series(
    description: str,
    length: int = 168,
    freq: str = "h",
    start: str = "2026-07-01 00:00:00",
    seed: int = 42,
    model: str | None = DEFAULT_LLM_MODEL,
    anomalies: str = "auto",
    anomaly_severity: str | None = None,
    output_path: str = "generated_timeseries.arrow",
    storage_mode: str = "arrow",
    reference_arrow_path: str | None = None,
    reference_csv_path: str | None = None,
    cost_mode: str = "balanced",
) -> str:
    """Generate a time series as an Arrow IPC file and return file metadata."""
    agent = GenerationAgent(model=model, cost_mode=cost_mode)
    enabled = None if anomalies == "auto" else anomalies == "on"
    overrides = AnomalyOverrides(
        enabled=enabled,
        severity=anomaly_severity,
    )
    df, _plan = agent.run_to_files(
        description,
        output=output_path,
        length=length,
        freq=freq,
        start=start,
        seed=seed,
        anomaly_overrides=overrides,
        reference_profile=_reference_payload(reference_arrow_path, reference_csv_path),
        storage_mode=storage_mode,
    )
    payload = {
        "output_path": df.attrs.get("output_path", str(Path(output_path).with_suffix(".arrow"))),
        "rows": int(len(df)),
        "storage_format": df.attrs.get("storage_format", "arrow_ipc"),
        "storage": df.attrs.get("storage", {}),
        "metadata_path": f"{df.attrs.get('output_path', str(Path(output_path).with_suffix('.arrow')))}.metadata.json",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


@mcp.tool()
def generate_time_series_code(
    description: str,
    length: int = 168,
    freq: str = "h",
    start: str = "2026-07-01 00:00:00",
    seed: int = 42,
    model: str | None = DEFAULT_LLM_MODEL,
    anomalies: str = "auto",
    anomaly_severity: str | None = None,
    reference_arrow_path: str | None = None,
    reference_csv_path: str | None = None,
    cost_mode: str = "balanced",
) -> str:
    """Generate replay Python code using the main local kernel."""
    from .semantic_types import apply_anomaly_overrides

    enabled = None if anomalies == "auto" else anomalies == "on"
    plan = apply_anomaly_overrides(
        plan_from_description(
            description,
            model=model,
            reference_profile=_reference_payload(reference_arrow_path, reference_csv_path),
            cost_mode=cost_mode,
        ),
        AnomalyOverrides(enabled=enabled, severity=anomaly_severity),
    )
    return render_generator_code(plan, length=length, freq=freq, start=start, seed=seed, model=model)


@mcp.tool()
def generate_time_series_dataset(
    domain: str,
    output_dir: str,
    series_count: int = 10,
    length: int = 168,
    freq: str = "h",
    start: str = "2026-07-01 00:00:00",
    seed: int = 42,
    model: str | None = DEFAULT_LLM_MODEL,
    reference_arrow_path: str | None = None,
    reference_csv_path: str | None = None,
    cost_mode: str = "balanced",
    storage_mode: str = "arrow",
    respect_scenario_frequency: bool = False,
) -> str:
    """Generate a multi-scenario synthetic dataset and return its manifest JSON."""
    profile = _reference_payload(reference_arrow_path, reference_csv_path)
    agent = GenerationAgent(model=model, cost_mode=cost_mode)
    manifest = DatasetGenerator(agent).generate_to_directory(
        domain=domain,
        output_dir=output_dir,
        series_count=series_count,
        length=length,
        freq=freq,
        start=start,
        seed=seed,
        reference_profile=profile,
        storage_mode=storage_mode,
        respect_scenario_frequency=respect_scenario_frequency,
    )
    return json.dumps(manifest, ensure_ascii=False, indent=2)


@mcp.tool()
def materialize_time_series_pack(
    pack_path: str,
    output_path: str = "generated_timeseries.arrow",
    verify_source_hash: bool = False,
) -> str:
    """Materialize a replayable .syn.json.gz parameter pack into an Arrow IPC file."""
    frame, storage = materialize_param_pack(
        pack_path,
        output_path,
        verify_source_hash=verify_source_hash,
    )
    payload = {
        "output_path": storage["path"],
        "rows": int(len(frame)),
        "storage_format": "arrow_ipc",
        "storage": storage,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)



if __name__ == "__main__":
    mcp.run()
