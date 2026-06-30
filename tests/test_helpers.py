from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from generation_agent.agent import _arrow_output_path, _generation_metadata, _param_pack_output_path
from generation_agent.compact_storage import write_series_arrow
from generation_agent.component_workflow import build_component_workflow
from generation_agent.cost_policy import get_cost_policy
from generation_agent.param_pack import write_param_pack
from generation_agent.planner import SeriesPlan, heuristic_plan
from generation_agent.semantic_types import AnomalyOverrides, apply_anomaly_overrides
from generation_agent.synthesizer import synthesize_series
from generation_agent.workflow import WORKFLOW_NAME


def local_generate_from_plan(
    plan: SeriesPlan,
    length: int = 168,
    freq: str = "h",
    start: str = "2026-07-01 00:00:00",
    seed: int | None = 42,
    anomaly_overrides: AnomalyOverrides | None = None,
    reference_profile: dict | None = None,
) -> pd.DataFrame:
    """Test helper for the local numerical kernel after an LLM plan has been compiled."""
    plan = apply_anomaly_overrides(plan, anomaly_overrides)
    description = str(plan.metadata.get("description", plan.domain))
    workflow = build_component_workflow(
        description,
        plan,
        length=length,
        freq=freq,
        start=start,
        reference_profile=reference_profile,
        generation_mode=str(plan.metadata.get("generation_mode", "sequence")),
    )
    plan.metadata["workflow"] = WORKFLOW_NAME
    plan.metadata["component_workflow"] = workflow.to_dict()
    if reference_profile:
        plan.metadata["reference_profile"] = reference_profile
    frame = synthesize_series(plan, length=length, freq=freq, start=start, seed=seed)
    frame.attrs.setdefault("series_audit", {"status": "PASS", "test_contract": "compiled_plan_local_kernel"})
    frame.attrs["final_plan"] = plan.to_dict()
    frame.attrs.setdefault("cost_mode", "test")
    return frame


class ReviewedFakeAgent:
    """Small fake for dataset/pack tests: planning and quality review are treated as completed."""

    def __init__(self, plan_factory=None, cost_mode: str = "balanced") -> None:
        self.model = "test-llm"
        self.cost_policy = get_cost_policy(cost_mode)
        self.plan_factory = plan_factory or (lambda description, reference_profile=None: heuristic_plan(description))

    def plan(self, description: str, reference_profile: dict | None = None) -> SeriesPlan:
        return self.plan_factory(description, reference_profile)

    def generate_from_plan(
        self,
        plan: SeriesPlan,
        length: int = 168,
        freq: str = "h",
        start: str = "2026-07-01 00:00:00",
        seed: int | None = 42,
        anomaly_overrides: AnomalyOverrides | None = None,
    ) -> pd.DataFrame:
        frame = local_generate_from_plan(
            plan,
            length=length,
            freq=freq,
            start=start,
            seed=seed,
            anomaly_overrides=anomaly_overrides,
            reference_profile=plan.metadata.get("reference_profile"),
        )
        frame.attrs["series_audit"] = {"status": "PASS", "llm_reviewed": True}
        frame.attrs["final_plan"] = plan.to_dict()
        frame.attrs["cost_mode"] = self.cost_policy.mode
        return frame

    def run_to_files(
        self,
        description: str,
        output: str | Path,
        length: int = 168,
        freq: str = "h",
        start: str = "2026-07-01 00:00:00",
        seed: int | None = 42,
        anomaly_overrides: AnomalyOverrides | None = None,
        reference_profile: dict | None = None,
        save_trace: bool = False,
        storage_mode: str = "arrow",
    ):
        if storage_mode not in {"arrow", "param-pack"}:
            raise ValueError("storage_mode must be 'arrow' or 'param-pack'")
        plan = self.plan(description, reference_profile)
        frame = self.generate_from_plan(
            plan,
            length=length,
            freq=freq,
            start=start,
            seed=seed,
            anomaly_overrides=anomaly_overrides,
        )
        output_path = _arrow_output_path(output) if storage_mode == "arrow" else _param_pack_output_path(output)
        metadata = _generation_metadata(
            description=description,
            plan=plan,
            frame=frame,
            length=length,
            freq=freq,
            start=start,
            seed=seed,
        )
        metadata["storage_format"] = "arrow_ipc" if storage_mode == "arrow" else "generation_agent_param_pack"
        if storage_mode == "arrow":
            storage = write_series_arrow(frame, output_path, metadata=metadata)
        else:
            storage = write_param_pack(
                output_path,
                description=description,
                plan=plan,
                length=length,
                freq=freq,
                start=start,
                seed=seed,
                metadata=metadata,
                frame=frame,
            )
        metadata["storage"] = storage
        frame.attrs["output_path"] = str(output_path)
        frame.attrs["storage"] = storage
        frame.attrs["storage_format"] = metadata["storage_format"]
        Path(f"{output_path}.metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return frame, plan
