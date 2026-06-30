from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock
from typing import Any
from uuid import uuid4

from flask import Flask, abort, jsonify, render_template, request, send_file
import pandas as pd

from .agent import GenerationAgent
from .compact_storage import preview_points, write_series_arrow
from .context_compactor import (
    build_generation_trace,
    compact_component_report_for_agent,
    compact_component_workflow_for_agent,
    compact_plan_for_metadata,
    compact_quality_for_metadata,
    compact_validation_for_metadata,
    write_json_gz,
)
from .cost_policy import VALID_COST_MODES
from .dataset_generator import DatasetGenerator
from .planner import DEFAULT_LLM_MODEL
from .param_pack import write_param_pack
from .reference_profiler import profile_reference_arrow_or_raise
from .semantic_types import AnomalyOverrides
from .workflow import LLM_ROLES, LOCAL_KERNEL_NAME, WORKFLOW_STEPS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "web_outputs"
MAX_SERIES_COUNT = 10_000
MAX_LENGTH = 1_000_000
AVAILABLE_CHAT_MODELS = [
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v4-pro",
    "deepseek/deepseek-v3.2",
    "openai/gpt-5.5",
    "openai/gpt-5.4-pro",
    "openai/gpt-5.4",
    "openai/gpt-5.4-mini",
    "openai/gpt-5.2",
    "openai/gpt-5.1",
    "openai/gpt-5",
    "openai/gpt-5-mini",
    "openai/gpt-4.1",
    "openai/gpt-4.1-mini",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3-flash-preview",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
    "anthropic/claude-opus-4.5",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-haiku-4.5",
    "moonshotai/kimi-k2.6",
    "moonshotai/kimi-k2.5",
    "bailian/qwen3.7-max",
    "bailian/qwen3.7-plus",
    "bailian/qwen3.6-max-preview",
    "bailian/qwen3.6-plus",
    "bailian/qwen3.5-plus",
    "bailian/qwen-plus",
    "bailian/qwen-turbo",
    "volcengine/doubao-seed-2.0-pro",
    "volcengine/doubao-seed-2.0-mini",
    "x-ai/grok-4.1-fast",
    "z-ai/glm-5.2",
    "z-ai/glm-5.1",
    "z-ai/glm-4.7-flash:free",
]


@dataclass
class GenerationJob:
    id: str
    status: str = "queued"
    progress: int = 0
    message: str = "Job queued"
    files: list[Path] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "created_at": self.created_at,
            "result": self.result,
            "files": [
                {
                    "name": path.name,
                    "path": str(path),
                    "download_url": f"/api/jobs/{self.id}/files/{index}",
                }
                for index, path in enumerate(self.files)
            ],
        }


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, GenerationJob] = {}
        self._lock = Lock()

    def create(self) -> GenerationJob:
        job = GenerationJob(id=uuid4().hex)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> GenerationJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in changes.items():
                setattr(job, key, value)


def _as_int(payload: dict[str, Any], name: str, default: int) -> int:
    try:
        return int(payload.get(name, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


COMMON_FREQUENCIES = {"15min", "30min", "h", "D", "W", "MS", "ME"}


def _normalize_frequency(payload: dict[str, Any]) -> str:
    selected = str(payload.get("freq", "h")).strip() or "h"
    if selected == "custom":
        selected = str(payload.get("custom_freq", "")).strip()
        if not selected:
            raise ValueError("Custom frequency is required")
    elif selected not in COMMON_FREQUENCIES:
        raise ValueError("Invalid frequency parameter")

    try:
        pd.date_range(start="2026-01-01", periods=2, freq=selected)
    except Exception as exc:
        raise ValueError(f"Frequency is not a valid Pandas freq: {selected}") from exc
    return selected


def _normalize_start(payload: dict[str, Any]) -> str:
    start = str(payload.get("start", "2026-07-01T00:00")).strip()
    if not start:
        raise ValueError("Start time is required")
    try:
        parsed = pd.Timestamp(start)
    except Exception as exc:
        raise ValueError(f"Invalid start time format: {start}") from exc
    if pd.isna(parsed):
        raise ValueError(f"Invalid start time format: {start}")
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def _available_models(default_model: str | None = None) -> list[str]:
    models = list(AVAILABLE_CHAT_MODELS)
    if default_model and default_model not in models:
        models.insert(0, default_model)
    return models


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    description = str(payload.get("description", "")).strip()
    if not description:
        raise ValueError("Data description or domain is required")
    mode = str(payload.get("generation_mode", "sequence"))
    if mode not in {"sequence", "dataset"}:
        raise ValueError("generation_mode must be sequence or dataset")

    length = _as_int(payload, "length", 168)
    series_count = _as_int(payload, "series_count", 10)
    seed = _as_int(payload, "seed", 42)
    if not 1 <= length <= MAX_LENGTH:
        raise ValueError(f"length must be between 1 and {MAX_LENGTH}")
    if not 1 <= series_count <= MAX_SERIES_COUNT:
        raise ValueError(f"series_count must be between 1 and {MAX_SERIES_COUNT}")

    output_dir = Path(str(payload.get("output_dir", DEFAULT_OUTPUT_DIR)).strip()).expanduser().resolve()
    filename = Path(str(payload.get("filename", "generated_timeseries.arrow")).strip()).name
    if mode == "sequence" and not filename.lower().endswith(".arrow"):
        filename += ".arrow"

    anomalies = str(payload.get("anomalies", "auto"))
    severity = payload.get("anomaly_severity") or None
    if anomalies not in {"auto", "on", "off"}:
        raise ValueError("Invalid anomaly control parameter")
    if severity not in {None, "low", "medium", "high"}:
        raise ValueError("Invalid anomaly severity parameter")

    reference = str(payload.get("reference", "")).strip()
    reference_path = Path(reference).expanduser().resolve() if reference else None
    if reference_path and not reference_path.is_file():
        raise ValueError(f"Reference series does not exist: {reference_path}")
    if reference_path and reference_path.suffix.lower() != ".arrow":
        raise ValueError("Reference time series must be an Arrow file (.arrow)")
    diversity_strength = str(payload.get("diversity_strength", "medium")).strip()
    if diversity_strength not in {"off", "low", "medium", "high"}:
        raise ValueError("Invalid diversity strength parameter")
    default_model = os.getenv("GENERATION_AGENT_MODEL", DEFAULT_LLM_MODEL)
    model = str(payload.get("model", default_model)).strip() or default_model
    if model not in _available_models(default_model):
        raise ValueError("Invalid model parameter")
    cost_mode = str(payload.get("cost_mode", os.getenv("GENERATION_AGENT_COST_MODE", "balanced"))).strip()
    if cost_mode not in VALID_COST_MODES:
        raise ValueError("Invalid cost mode parameter")
    storage_mode = str(payload.get("storage_mode", "arrow")).strip()
    if storage_mode not in {"arrow", "param-pack"}:
        raise ValueError("Invalid storage mode parameter")
    respect_scenario_frequency = bool(payload.get("respect_scenario_frequency", False))

    return {
        "description": description,
        "generation_mode": mode,
        "series_count": series_count,
        "output_dir": output_dir,
        "filename": filename,
        "length": length,
        "freq": _normalize_frequency(payload),
        "start": _normalize_start(payload),
        "seed": seed,
        "model": model,
        "cost_mode": cost_mode,
        "storage_mode": storage_mode,
        "respect_scenario_frequency": respect_scenario_frequency,
        "reference_path": reference_path,
        "anomalies": anomalies,
        "anomaly_severity": severity,
        "diversity_strength": diversity_strength,
        "save_trace": bool(payload.get("save_trace", False)),
    }


def _run_generation(job_id: str, options: dict[str, Any], store: JobStore) -> None:
    try:
        store.update(job_id, status="running", progress=8, message="Preparing generation parameters")
        store.update(job_id, progress=12, message="Checking LLM dependencies")
        from .dependency_check import ensure_llm_dependencies

        ensure_llm_dependencies()
        reference_profile = None
        if options["reference_path"]:
            store.update(job_id, progress=16, message="Profiling reference time series")
            reference_profile = profile_reference_arrow_or_raise(options["reference_path"])

        enabled = None if options["anomalies"] == "auto" else options["anomalies"] == "on"
        overrides = AnomalyOverrides(enabled=enabled, severity=options["anomaly_severity"])
        agent = GenerationAgent(model=options["model"], cost_mode=options["cost_mode"])
        output_dir: Path = options["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)

        if options["generation_mode"] == "dataset":
            store.update(job_id, progress=24, message="Agent workflow is designing dataset scenarios and generating series")
            manifest = DatasetGenerator(agent).generate_to_directory(
                domain=options["description"], output_dir=output_dir,
                series_count=options["series_count"], length=options["length"],
                freq=options["freq"], start=options["start"], seed=options["seed"],
                anomaly_overrides=overrides, reference_profile=reference_profile,
                diversity_strength=options["diversity_strength"],
                save_trace=options["save_trace"],
                storage_mode=options["storage_mode"],
                respect_scenario_frequency=options["respect_scenario_frequency"],
            )
            files = sorted(output_dir.glob("*.arrow")) + sorted(output_dir.glob("*.syn.json.gz")) + [
                path for path in (output_dir / "manifest.json", output_dir / "scenarios.json") if path.exists()
            ]
            store.update(
                job_id, status="completed", progress=100,
                message=f"Generated {manifest['series_count']} series", files=files,
                result={"mode": "dataset", "output_dir": str(output_dir),
                        "storage_format": manifest["storage_format"],
                        "series_count": manifest["series_count"],
                        "scenario_count": manifest["scenario_count"],
                        "cost_mode": manifest.get("cost_mode"),
                        "length_per_series": manifest["length_per_series"],
                        "diversity": manifest.get("diversity", {}),
                        "dataset_file": "not written; see per-series Arrow files",
                        "preview": manifest.get("preview", {})},
            )
            return

        output_path = output_dir / options["filename"]
        if options["storage_mode"] == "param-pack" and not str(output_path).endswith(".syn.json.gz"):
            output_path = Path(str(output_path) + ".syn.json.gz")
        store.update(job_id, progress=25, message="Agent workflow is planning and generating one series")
        plan = agent.plan(
            options["description"],
            reference_profile=reference_profile.to_dict() if reference_profile else None,
        )
        frame = agent.generate_from_plan(
            plan, length=options["length"], freq=options["freq"], start=options["start"],
            seed=options["seed"], anomaly_overrides=overrides,
        )
        from .planner import SeriesPlan

        final_plan_payload = frame.attrs.get("final_plan")
        if isinstance(final_plan_payload, dict):
            plan = SeriesPlan.from_dict(final_plan_payload)
        metadata = {
                "series_id": "sequence_0001",
                "description": options["description"],
                "start": options["start"],
                "frequency": options["freq"],
                "length": options["length"],
                "cost_mode": options["cost_mode"],
                "unit": plan.unit,
                "domain": plan.domain,
                "generator_type": plan.generator_type,
                "semantic_type": plan.semantic_type,
                "plan_summary": compact_plan_for_metadata(plan.to_dict()),
                "anomaly_strategy": plan.metadata.get("anomaly_strategy", {}),
                "anomaly_execution": frame.attrs.get("anomaly_execution", {}),
                "multivariate_report": frame.attrs.get("multivariate_report", {}),
                "component_workflow": compact_component_workflow_for_agent(
                    frame.attrs.get("component_workflow", {})
                ),
                "component_report": compact_component_report_for_agent(
                    frame.attrs.get("component_report", {})
                ),
                "deterministic_validation": compact_validation_for_metadata(
                    frame.attrs.get("validation_report", {})
                ),
                "quality_evaluation": compact_quality_for_metadata(frame.attrs.get("series_audit", {})),
                "workflow_steps": WORKFLOW_STEPS,
                "value_generation": LOCAL_KERNEL_NAME,
                "llm_roles": LLM_ROLES,
                "trace": None,
                "storage_format": "arrow_ipc" if options["storage_mode"] == "arrow" else "generation_agent_param_pack",
                "data_policy": {
                    "llm_outputs_data_points": False,
                    "llm_directly_edits_data_points": False,
                    "numeric_values_computed_by": LOCAL_KERNEL_NAME,
                },
            }
        if options["save_trace"]:
            metadata["trace"] = write_json_gz(
                f"{output_path}.trace.json.gz",
                build_generation_trace(plan.to_dict(), frame.attrs),
            )
        if options["storage_mode"] == "arrow":
            storage = write_series_arrow(frame, output_path, metadata=metadata)
        else:
            storage = write_param_pack(
                output_path,
                description=options["description"],
                plan=plan,
                length=options["length"],
                freq=options["freq"],
                start=options["start"],
                seed=options["seed"],
                metadata=metadata,
                frame=frame,
            )
        store.update(
            job_id, status="completed", progress=100, message=f"Generated {len(frame)} data points",
            files=[output_path],
            result={"mode": "sequence", "output_dir": str(output_dir), "rows": len(frame),
                    "storage_format": metadata["storage_format"], "stored_bytes": storage["bytes"],
                    "domain": plan.domain, "unit": plan.unit,
                    "semantic_type": plan.semantic_type, "cost_mode": options["cost_mode"],
                    "output_file": str(output_path),
                    "preview": preview_points(frame)},
        )
    except Exception as exc:
        store.update(job_id, status="failed", progress=100, message="Generation failed",
                     error=f"{exc.__class__.__name__}: {exc}")


def _open_macos_dialog(kind: str) -> str:
    if kind == "directory":
        script = 'POSIX path of (choose folder with prompt "Select output folder")'
    elif kind == "arrow":
        script = 'POSIX path of (choose file with prompt "Select reference Arrow time series")'
    else:
        raise ValueError("unknown dialog kind")

    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode == 0:
        return str(Path(result.stdout.strip()).expanduser().resolve()) if result.stdout.strip() else ""
    if "-128" in result.stderr or "User canceled" in result.stderr:
        return ""
    raise RuntimeError(result.stderr.strip() or "macOS file picker failed to open")


def _open_native_dialog(kind: str) -> str:
    if sys.platform == "darwin":
        try:
            return _open_macos_dialog(kind)
        except FileNotFoundError:
            pass

    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError(f"Native file picker is unavailable: {exc}") from exc

    root = tk.Tk()
    root.withdraw()
    root.update()
    try:
        if kind == "directory":
            path = filedialog.askdirectory(title="Select output folder")
        elif kind == "arrow":
            path = filedialog.askopenfilename(
                title="Select reference Arrow time series",
                filetypes=[("Arrow IPC", "*.arrow"), ("All files", "*.*")],
            )
        else:
            raise ValueError("unknown dialog kind")
    finally:
        root.destroy()
    return str(Path(path).expanduser().resolve()) if path else ""


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates")
    store = JobStore()
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="generation-web")
    app.extensions["generation_job_store"] = store
    app.extensions["generation_executor"] = executor

    @app.get("/")
    def index() -> str:
        default_model = os.getenv("GENERATION_AGENT_MODEL", DEFAULT_LLM_MODEL)
        return render_template("index.html", default_output_dir=str(DEFAULT_OUTPUT_DIR),
                               default_model=default_model,
                               available_models=_available_models(default_model))

    @app.post("/api/generate")
    def generate() -> Any:
        try:
            options = _normalize_payload(request.get_json(silent=True) or {})
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        job = store.create()
        executor.submit(_run_generation, job.id, options, store)
        return jsonify(job.public_dict()), 202

    @app.post("/api/select-output-dir")
    def select_output_dir() -> Any:
        try:
            path = _open_native_dialog(kind="directory")
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500
        return jsonify({"path": path})

    @app.post("/api/select-reference-arrow")
    def select_reference_arrow() -> Any:
        try:
            path = _open_native_dialog(kind="arrow")
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500
        if path and Path(path).suffix.lower() != ".arrow":
            return jsonify({"error": "Please select an .arrow file"}), 400
        return jsonify({"path": path})

    @app.post("/api/upload-reference-arrow")
    def upload_reference_arrow() -> Any:
        upload = request.files.get("file")
        if upload is None or not upload.filename:
            return jsonify({"error": "Please select an Arrow file"}), 400
        filename = Path(upload.filename).name
        if Path(filename).suffix.lower() != ".arrow":
            return jsonify({"error": "Reference time series must be an Arrow file (.arrow)"}), 400
        upload_dir = DEFAULT_OUTPUT_DIR / "_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        target = upload_dir / f"{uuid4().hex}_{filename}"
        upload.save(target)
        return jsonify({"path": str(target.resolve())})

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id: str) -> Any:
        job = store.get(job_id)
        if job is None:
            abort(404)
        return jsonify(job.public_dict())

    @app.get("/api/jobs/<job_id>/files/<int:file_index>")
    def download(job_id: str, file_index: int) -> Any:
        job = store.get(job_id)
        if job is None or file_index < 0 or file_index >= len(job.files):
            abort(404)
        path = job.files[file_index]
        if not path.is_file():
            abort(404)
        return send_file(path, as_attachment=True, download_name=path.name)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the time-series generation web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    create_app().run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
