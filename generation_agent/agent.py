from __future__ import annotations

from pathlib import Path
import os

import pandas as pd

from .planner import DEFAULT_LLM_MODEL, SeriesPlan, plan_from_description
from .semantic_types import AnomalyOverrides, apply_anomaly_overrides
from .synthesizer import synthesize_series
from .component_workflow import build_component_workflow


def _series_summary(frame: pd.DataFrame) -> dict:
    values = frame["value"].astype(float)
    sample_size = min(12, len(values))
    return {
        "count": int(len(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std(ddof=0)),
        "zero_fraction": float((values == 0).mean()),
        "integer_fraction": float((values.round() == values).mean()),
        "lag1_autocorrelation": float(values.autocorr(lag=1)) if len(values) > 2 else None,
        "anomaly_fraction": float(frame["anomaly"].mean()) if "anomaly" in frame else 0.0,
        "first_values": values.head(sample_size).round(6).tolist(),
        "middle_values": values.iloc[max(0, len(values) // 2 - sample_size // 2) :][:sample_size]
        .round(6)
        .tolist(),
        "last_values": values.tail(sample_size).round(6).tolist(),
    }


class GenerationAgent:
    """Agent facade: text request -> plan -> generated time series/code."""

    def __init__(self, model: str | None = DEFAULT_LLM_MODEL):
        self.model = model

    def plan(
        self,
        description: str,
        reference_profile: dict | None = None,
    ) -> SeriesPlan:
        return plan_from_description(
            description,
            model=self.model,
            reference_profile=reference_profile,
        )

    def generate(
        self,
        description: str,
        length: int = 168,
        freq: str = "h",
        start: str = "2026-07-01 00:00:00",
        seed: int | None = 42,
        anomaly_overrides: AnomalyOverrides | None = None,
        reference_profile: dict | None = None,
    ) -> pd.DataFrame:
        plan = apply_anomaly_overrides(
            self.plan(description, reference_profile), anomaly_overrides
        )
        return self._generate_with_review(
            description,
            plan,
            length=length,
            freq=freq,
            start=start,
            seed=seed,
            reference_profile=reference_profile,
        )

    def generate_from_plan(
        self,
        plan: SeriesPlan,
        length: int = 168,
        freq: str = "h",
        start: str = "2026-07-01 00:00:00",
        seed: int | None = 42,
        anomaly_overrides: AnomalyOverrides | None = None,
    ) -> pd.DataFrame:
        plan = apply_anomaly_overrides(plan, anomaly_overrides)
        description = str(plan.metadata.get("description", plan.domain))
        return self._generate_with_review(
            description,
            plan,
            length=length,
            freq=freq,
            start=start,
            seed=seed,
            reference_profile=plan.metadata.get("reference_profile"),
        )

    def _generate_with_review(
        self,
        description: str,
        plan: SeriesPlan,
        length: int,
        freq: str,
        start: str,
        seed: int | None,
        reference_profile: dict | None = None,
    ) -> pd.DataFrame:
        workflow = build_component_workflow(
            description,
            plan,
            length=length,
            freq=freq,
            start=start,
            reference_profile=reference_profile,
            generation_mode=str(plan.metadata.get("generation_mode", "sequence")),
        )
        plan.metadata["workflow"] = "component_agent_workflow"
        plan.metadata["component_workflow"] = workflow.to_dict()
        if reference_profile:
            plan.metadata["reference_profile"] = reference_profile
        frame = synthesize_series(plan, length=length, freq=freq, start=start, seed=seed)
        validation = frame.attrs.get("validation_report", {})
        audit: dict = {
            "status": "SKIPPED",
            "issues": [],
            "reason": "LLM output audit is unavailable",
        }
        if self.model and os.getenv("OPENAI_API_KEY"):
            try:
                from langchain_openai import ChatOpenAI

                from .langchain_agent import (
                    audit_generated_series,
                    revise_plan_from_series_audit,
                )
                from .planner import DEFAULT_OPENAI_BASE_URL

                llm = ChatOpenAI(
                    model=self.model,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url=os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
                    temperature=0,
                    max_completion_tokens=1200,
                    timeout=60,
                    max_retries=1,
                )
                audit = audit_generated_series(
                    llm,
                    description,
                    plan,
                    _series_summary(frame),
                    validation,
                    component_workflow=frame.attrs.get("component_workflow", {}),
                    component_report=frame.attrs.get("component_report", {}),
                )
                if audit.get("status") == "REGENERATE":
                    revised = revise_plan_from_series_audit(llm, description, plan, audit)
                    if revised is not None:
                        plan = revised
                        workflow = build_component_workflow(
                            description,
                            plan,
                            length=length,
                            freq=freq,
                            start=start,
                            reference_profile=reference_profile,
                            generation_mode=str(plan.metadata.get("generation_mode", "sequence")),
                        )
                        plan.metadata["workflow"] = "component_agent_workflow"
                        plan.metadata["component_workflow"] = workflow.to_dict()
                        frame = synthesize_series(
                            plan, length=length, freq=freq, start=start, seed=seed
                        )
                        validation = frame.attrs.get("validation_report", {})
                        audit = audit_generated_series(
                            llm,
                            description,
                            plan,
                            _series_summary(frame),
                            validation,
                            component_workflow=frame.attrs.get("component_workflow", {}),
                            component_report=frame.attrs.get("component_report", {}),
                        )
                        audit["regenerated_once"] = True
            except Exception as exc:
                audit = {
                    "status": "UNVERIFIED",
                    "issues": ["Series audit service failed."],
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                }
        from .domain_rules import commit_candidate_rule

        commit_candidate_rule(plan, validation, audit)
        frame.attrs["series_audit"] = audit
        frame.attrs["final_plan"] = plan.to_dict()
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
    ) -> tuple[pd.DataFrame, SeriesPlan]:
        plan = apply_anomaly_overrides(
            self.plan(description, reference_profile), anomaly_overrides
        )
        df = self.generate_from_plan(plan, length=length, freq=freq, start=start, seed=seed)
        final_plan_payload = df.attrs.get("final_plan")
        if isinstance(final_plan_payload, dict):
            plan = SeriesPlan.from_dict(final_plan_payload)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
        return df, plan
