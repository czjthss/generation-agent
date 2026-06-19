from __future__ import annotations

import json

from .agent import GenerationAgent
from .codegen import render_generator_code
from .dataset_generator import DatasetGenerator
from .planner import DEFAULT_LLM_MODEL, plan_from_description
from .reference_profiler import profile_reference_arrow_or_raise
from .semantic_types import AnomalyOverrides

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Please install MCP first: pip install mcp") from exc


mcp = FastMCP("time-series-generation-agent")


def _reference_payload(
    reference_arrow_path: str | None,
) -> dict | None:
    if not reference_arrow_path:
        return None
    return profile_reference_arrow_or_raise(reference_arrow_path).to_dict()


@mcp.tool()
def plan_time_series(
    description: str,
    model: str | None = DEFAULT_LLM_MODEL,
    reference_arrow_path: str | None = None,
    reference_csv_path: str | None = None,
) -> str:
    """Convert a natural-language data request into a time-series synthesis plan."""
    plan = plan_from_description(
        description,
        model=model,
        reference_profile=_reference_payload(reference_arrow_path or reference_csv_path),
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
    reference_arrow_path: str | None = None,
    reference_csv_path: str | None = None,
) -> str:
    """Generate a time series as CSV text."""
    agent = GenerationAgent(model=model)
    enabled = None if anomalies == "auto" else anomalies == "on"
    overrides = AnomalyOverrides(
        enabled=enabled,
        severity=anomaly_severity,
    )
    df = agent.generate(
        description,
        length=length,
        freq=freq,
        start=start,
        seed=seed,
        anomaly_overrides=overrides,
        reference_profile=_reference_payload(reference_arrow_path or reference_csv_path),
    )
    return df.to_csv(index=False)


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
) -> str:
    """Generate standalone Python code that can recreate this type of time series."""
    from .semantic_types import apply_anomaly_overrides

    enabled = None if anomalies == "auto" else anomalies == "on"
    plan = apply_anomaly_overrides(
        plan_from_description(
            description,
            model=model,
            reference_profile=_reference_payload(reference_arrow_path or reference_csv_path),
        ),
        AnomalyOverrides(enabled=enabled, severity=anomaly_severity),
    )
    return render_generator_code(plan, length=length, freq=freq, start=start, seed=seed)


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
) -> str:
    """Generate a multi-scenario synthetic dataset and return its manifest JSON."""
    profile = _reference_payload(reference_arrow_path or reference_csv_path)
    agent = GenerationAgent(model=model)
    manifest = DatasetGenerator(agent).generate_to_directory(
        domain=domain,
        output_dir=output_dir,
        series_count=series_count,
        length=length,
        freq=freq,
        start=start,
        seed=seed,
        reference_profile=profile,
    )
    return json.dumps(manifest, ensure_ascii=False, indent=2)


@mcp.tool()
def synthesize_from_plan(
    plan_json: str,
    length: int = 168,
    freq: str = "h",
    start: str = "2026-07-01 00:00:00",
    anomalies: str = "auto",
    anomaly_severity: str | None = None,
) -> str:
    """Generate CSV from a plan returned by plan_time_series."""
    from .planner import SeriesPlan

    from .semantic_types import apply_anomaly_overrides

    enabled = None if anomalies == "auto" else anomalies == "on"
    plan = apply_anomaly_overrides(
        SeriesPlan.from_dict(json.loads(plan_json)),
        AnomalyOverrides(enabled=enabled, severity=anomaly_severity),
    )
    model = plan.metadata.get("model", DEFAULT_LLM_MODEL)
    df = GenerationAgent(model=model).generate_from_plan(
        plan, length=length, freq=freq, start=start
    )
    return df.to_csv(index=False)


if __name__ == "__main__":
    mcp.run()
