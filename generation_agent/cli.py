from __future__ import annotations

import argparse
import os
from pathlib import Path

from .agent import GenerationAgent
from .cost_policy import VALID_COST_MODES
from .dataset_generator import DatasetGenerator
from .planner import DEFAULT_LLM_MODEL
from .reference_profiler import profile_reference_arrow_or_raise
from .semantic_types import AnomalyOverrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic time series from one sentence.")
    parser.add_argument("description", nargs="?", help="Sequence description, or a broad domain in dataset mode.")
    parser.add_argument(
        "--materialize",
        default=None,
        help="Materialize a .syn.json.gz parameter pack into an Arrow file without calling the LLM.",
    )
    parser.add_argument(
        "--generation-mode",
        choices=["sequence", "dataset"],
        default="sequence",
        help="Generate one sequence or a multi-scenario dataset.",
    )
    parser.add_argument("--series-count", type=int, default=10, help="Number of series in dataset mode.")
    parser.add_argument(
        "--diversity-strength",
        choices=["off", "low", "medium", "high"],
        default="medium",
        help="Dataset-only parameter variation strength.",
    )
    parser.add_argument(
        "--respect-scenario-frequency",
        action="store_true",
        help="Dataset mode: use each scenario's normalized suggested frequency when valid.",
    )
    parser.add_argument("--output-dir", default="generated_dataset", help="Dataset output directory.")
    parser.add_argument(
        "--storage-mode",
        choices=["arrow", "param-pack"],
        default="arrow",
        help="Store generated values as Arrow or store a replayable compressed parameter pack.",
    )
    parser.add_argument("--length", type=int, default=168, help="Number of points to generate.")
    parser.add_argument("--freq", default="h", help="Pandas frequency, e.g. h, 30min, D.")
    parser.add_argument("--start", default="2026-07-01 00:00:00", help="Start timestamp.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--cost-mode",
        choices=sorted(VALID_COST_MODES),
        default=os.getenv("GENERATION_AGENT_COST_MODE", "balanced"),
        help="Control LLM context and retry budget: cheap, balanced, or strict.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GENERATION_AGENT_MODEL", DEFAULT_LLM_MODEL),
        help="OpenAI-compatible chat model for planning. OPENAI_API_KEY is required.",
    )
    parser.add_argument("--output", default="generated_timeseries.arrow", help="Arrow output path for sequence mode.")
    parser.add_argument("--reference", default=None, help="Optional reference time-series Arrow path.")
    parser.add_argument(
        "--save-trace",
        action="store_true",
        help="Write full planning and evaluation trace as .trace.json.gz. Default stores compact metadata only.",
    )
    parser.add_argument(
        "--anomalies",
        choices=["auto", "on", "off"],
        default="auto",
        help="Use the LLM decision, force anomaly injection on, or disable it.",
    )
    parser.add_argument("--anomaly-severity", choices=["low", "medium", "high"], default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.materialize:
        from .param_pack import materialize_param_pack

        frame, storage = materialize_param_pack(args.materialize, args.output)
        print(f"Materialized {len(frame)} rows to {Path(args.output).resolve()}")
        print("Storage format: Arrow IPC")
        print(f"Stored bytes: {storage['bytes']}")
        print(frame.head(12).to_string(index=False))
        return
    if not args.description:
        raise SystemExit("description is required unless --materialize is used")
    from .dependency_check import ensure_llm_dependencies

    try:
        ensure_llm_dependencies()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    agent = GenerationAgent(model=args.model, cost_mode=args.cost_mode)
    reference_profile = None
    if args.reference:
        reference_profile = profile_reference_arrow_or_raise(args.reference)
    enabled = None if args.anomalies == "auto" else args.anomalies == "on"
    anomaly_overrides = AnomalyOverrides(
        enabled=enabled,
        severity=args.anomaly_severity,
    )
    if args.generation_mode == "dataset":
        manifest = DatasetGenerator(agent).generate_to_directory(
            domain=args.description,
            output_dir=args.output_dir,
            series_count=args.series_count,
            length=args.length,
            freq=args.freq,
            start=args.start,
            seed=args.seed,
            anomaly_overrides=anomaly_overrides,
            reference_profile=reference_profile,
            diversity_strength=args.diversity_strength,
            save_trace=args.save_trace,
            storage_mode=args.storage_mode,
            respect_scenario_frequency=args.respect_scenario_frequency,
        )
        print(f"Wrote {manifest['series_count']} series to {Path(args.output_dir).resolve()}")
        print(f"Storage format: {manifest['storage_format']}")
        print(f"Scenario count: {manifest['scenario_count']}")
        print(f"Diversity: {manifest['diversity']}")
        if args.storage_mode == "arrow":
            print("Combined dataset file is not written; use manifest.json and per-series .arrow files.")
        else:
            print("Combined dataset file is not written; use manifest.json and per-series .syn.json.gz packs.")
        return

    df, _plan = agent.run_to_files(
        args.description,
        output=args.output,
        length=args.length,
        freq=args.freq,
        start=args.start,
        seed=args.seed,
        anomaly_overrides=anomaly_overrides,
        reference_profile=reference_profile.to_dict() if reference_profile else None,
        save_trace=args.save_trace,
        storage_mode=args.storage_mode,
    )
    output_path = Path(df.attrs.get("output_path", args.output)).resolve()
    storage = df.attrs.get("storage", {})
    print(f"Wrote {len(df)} rows to {output_path}")
    print(f"Storage format: {df.attrs.get('storage_format', 'arrow_ipc')}")
    if storage.get("bytes") is not None:
        print(f"Stored bytes: {storage['bytes']}")
    print(df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
