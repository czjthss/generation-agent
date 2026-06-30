from __future__ import annotations

import json
from pathlib import Path
import os

import pandas as pd

from .compact_storage import write_series_arrow
from .context_compactor import (
    build_generation_trace,
    compact_component_report_for_agent,
    compact_component_workflow_for_agent,
    compact_plan_for_metadata,
    compact_quality_for_metadata,
    compact_series_summary_for_quality,
    compact_validation_for_metadata,
    write_json_gz,
)
from .planner import DEFAULT_LLM_MODEL, SeriesPlan, plan_from_description
from .param_pack import write_param_pack
from .cost_policy import get_cost_policy
from .semantic_types import (
    AnomalyOverrides,
    apply_anomaly_overrides,
    apply_anomaly_strategy,
)
from .synthesizer import synthesize_series
from .component_workflow import build_component_workflow
from .workflow import LLM_ROLES, LOCAL_KERNEL_NAME, WORKFLOW_NAME, WORKFLOW_STEPS


def _series_summary(frame: pd.DataFrame) -> dict:
    values = frame["value"].astype(float)
    sample_size = min(12, len(values))
    numeric_columns = [
        name
        for name in frame.select_dtypes(include="number").columns
        if name not in {"anomaly"}
    ]
    variable_summaries = {}
    for name in numeric_columns:
        column = frame[name].astype(float)
        variable_summaries[name] = {
            "minimum": float(column.min()),
            "maximum": float(column.max()),
            "mean": float(column.mean()),
            "standard_deviation": float(column.std(ddof=0)),
            "zero_fraction": float((column == 0).mean()),
            "lag1_autocorrelation": float(column.autocorr(lag=1)) if len(column) > 2 else None,
        }
    summary = {
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
        "variables": variable_summaries,
        "multivariate_report": frame.attrs.get("multivariate_report", {}),
    }
    return summary


def _generation_metadata(
    *,
    description: str,
    plan: SeriesPlan,
    frame: pd.DataFrame,
    length: int,
    freq: str,
    start: str,
    seed: int | None,
) -> dict:
    return {
        "description": description,
        "start": start,
        "frequency": freq,
        "length": length,
        "seed": seed,
        "model": plan.metadata.get("model"),
        "base_url": plan.metadata.get("base_url"),
        "cost_mode": plan.metadata.get("cost_mode") or frame.attrs.get("cost_mode"),
        "domain": plan.domain,
        "unit": plan.unit,
        "generator_type": plan.generator_type,
        "semantic_type": plan.semantic_type,
        "workflow_steps": WORKFLOW_STEPS,
        "value_generation": LOCAL_KERNEL_NAME,
        "llm_roles": LLM_ROLES,
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
        "series_summary": compact_series_summary_for_quality(_series_summary(frame)),
        "trace": None,
        "data_policy": {
            "llm_outputs_data_points": False,
            "llm_directly_edits_data_points": False,
            "numeric_values_computed_by": LOCAL_KERNEL_NAME,
        },
    }


def _arrow_output_path(output: str | Path) -> Path:
    path = Path(output)
    if path.suffix.lower() == ".arrow":
        return path
    if path.suffix:
        return path.with_suffix(".arrow")
    return path.with_suffix(".arrow")


def _param_pack_output_path(output: str | Path) -> Path:
    path = Path(output)
    text = str(path)
    if text.endswith(".syn.json.gz"):
        return path
    if path.suffix:
        return Path(text + ".syn.json.gz")
    return path.with_suffix(".syn.json.gz")


class GenerationAgent:
    """Agent facade: text request -> plan -> generated time series/code."""

    def __init__(self, model: str | None = DEFAULT_LLM_MODEL, cost_mode: str = "balanced"):
        if not model:
            raise RuntimeError("LLM model is required; local-only generation mode has been removed")
        self.model = model
        self.cost_policy = get_cost_policy(cost_mode)

    def plan(
        self,
        description: str,
        reference_profile: dict | None = None,
    ) -> SeriesPlan:
        return plan_from_description(
            description,
            model=self.model,
            reference_profile=reference_profile,
            cost_mode=self.cost_policy.mode,
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
        plan = self.plan(description, reference_profile)
        return self._generate_with_review(
            description,
            plan,
            length=length,
            freq=freq,
            start=start,
            seed=seed,
            anomaly_overrides=anomaly_overrides,
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
        description = str(plan.metadata.get("description", plan.domain))
        return self._generate_with_review(
            description,
            plan,
            length=length,
            freq=freq,
            start=start,
            seed=seed,
            anomaly_overrides=anomaly_overrides,
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
        anomaly_overrides: AnomalyOverrides | None = None,
        reference_profile: dict | None = None,
    ) -> pd.DataFrame:
        if not self.model:
            raise RuntimeError("LLM model is required; local-only generation mode has been removed")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for anomaly strategy and quality evaluation")
        from .dependency_check import ensure_llm_dependencies

        ensure_llm_dependencies()

        from .langchain_agent import decide_anomaly_strategy
        from .llm_client import create_chat_openai

        llm = create_chat_openai(
            model=self.model,
            temperature=0,
            max_completion_tokens=3000,
            timeout=60,
            max_retries=1,
        )
        anomaly_control = "auto"
        if anomaly_overrides and anomaly_overrides.enabled is not None:
            anomaly_control = "on" if anomaly_overrides.enabled else "off"
        strategy = decide_anomaly_strategy(
            llm,
            description,
            plan,
            anomaly_control=anomaly_control,
            severity_override=anomaly_overrides.severity if anomaly_overrides else None,
            attempts=self.cost_policy.anomaly_attempts,
        )
        plan = apply_anomaly_strategy(plan, strategy, anomaly_overrides)

        def generate_locally(current_plan: SeriesPlan) -> tuple[pd.DataFrame, dict]:
            workflow = build_component_workflow(
                description,
                current_plan,
                length=length,
                freq=freq,
                start=start,
                reference_profile=reference_profile,
                generation_mode=str(current_plan.metadata.get("generation_mode", "sequence")),
            )
            current_plan.metadata["workflow"] = WORKFLOW_NAME
            current_plan.metadata["component_workflow"] = workflow.to_dict()
            if reference_profile:
                current_plan.metadata["reference_profile"] = reference_profile
            generated = synthesize_series(
                current_plan, length=length, freq=freq, start=start, seed=seed
            )
            return generated, generated.attrs.get("validation_report", {})

        frame, validation = generate_locally(plan)
        from .langchain_agent import (
            QualityEvaluationError,
            audit_generated_series,
            revise_plan_from_series_audit,
        )

        audit = {}
        for regeneration_count in range(self.cost_policy.max_regenerations + 1):
            audit = audit_generated_series(
                llm,
                description,
                plan,
                _series_summary(frame),
                validation,
                component_workflow=frame.attrs.get("component_workflow", {}),
                component_report=frame.attrs.get("component_report", {}),
                anomaly_execution=frame.attrs.get("anomaly_execution", {}),
                attempts=self.cost_policy.quality_attempts,
                allow_json_repair=self.cost_policy.quality_json_repair,
            )
            audit["regeneration_count"] = regeneration_count
            if not validation.get("passed", False):
                audit["status"] = "REGENERATE"
                audit.setdefault("hard_failures", []).append(
                    "Deterministic mathematical validation failed."
                )
            if audit.get("status") in {"PASS", "PASS_WITH_WARNINGS"}:
                break
            if audit.get("status") == "UNVERIFIED":
                raise QualityEvaluationError(
                    "LLM quality evaluation could not be parsed or verified"
                )
            if regeneration_count >= self.cost_policy.max_regenerations:
                raise QualityEvaluationError(
                    f"Generated series still has hard quality failures after {self.cost_policy.max_regenerations} regenerations: "
                    + "; ".join(audit.get("hard_failures", audit.get("issues", [])))
                )
            revised = revise_plan_from_series_audit(
                llm,
                description,
                plan,
                audit,
                attempts=self.cost_policy.plan_attempts,
                direct_attempts=self.cost_policy.direct_revision_attempts,
            )
            if revised is None:
                raise QualityEvaluationError(
                    "Quality evaluation requested regeneration but no revised plan was produced"
                )
            # Preserve the independently reviewed anomaly strategy unless the audit
            # explicitly targets it, in which case ask the anomaly agent again.
            if str(audit.get("quality_evaluator", {}).get("revision_target")) == "anomaly":
                from .langchain_agent import decide_anomaly_strategy

                strategy = decide_anomaly_strategy(
                    llm,
                    description,
                    revised,
                    anomaly_control=(
                        "on" if plan.anomaly_enabled else "off"
                    ),
                    severity_override=plan.anomaly_severity,
                    attempts=self.cost_policy.anomaly_attempts,
                )
                plan = apply_anomaly_strategy(revised, strategy, anomaly_overrides)
            else:
                revised.metadata["anomaly_strategy"] = plan.metadata.get("anomaly_strategy", {})
                plan = apply_anomaly_strategy(
                    revised,
                    strategy,
                    anomaly_overrides,
                )
            frame, validation = generate_locally(plan)
        from .domain_rules import commit_candidate_rule

        commit_candidate_rule(
            plan,
            validation,
            {**audit, "status": "PASS" if audit.get("status") == "PASS_WITH_WARNINGS" else audit.get("status")},
        )
        frame.attrs["series_audit"] = audit
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
    ) -> tuple[pd.DataFrame, SeriesPlan]:
        if storage_mode not in {"arrow", "param-pack"}:
            raise ValueError("storage_mode must be 'arrow' or 'param-pack'")
        plan = self.plan(description, reference_profile)
        df = self.generate_from_plan(
            plan,
            length=length,
            freq=freq,
            start=start,
            seed=seed,
            anomaly_overrides=anomaly_overrides,
        )
        final_plan_payload = df.attrs.get("final_plan")
        if isinstance(final_plan_payload, dict):
            plan = SeriesPlan.from_dict(final_plan_payload)
        output_path = (
            _arrow_output_path(output)
            if storage_mode == "arrow"
            else _param_pack_output_path(output)
        )
        metadata = _generation_metadata(
            description=description,
            plan=plan,
            frame=df,
            length=length,
            freq=freq,
            start=start,
            seed=seed,
        )
        if save_trace:
            metadata["trace"] = write_json_gz(
                f"{output_path}.trace.json.gz",
                build_generation_trace(plan.to_dict(), df.attrs),
            )
        storage_format = "arrow_ipc" if storage_mode == "arrow" else "generation_agent_param_pack"
        metadata["storage_format"] = storage_format
        if storage_mode == "arrow":
            storage = write_series_arrow(df, output_path, metadata=metadata)
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
                frame=df,
            )
        metadata["storage"] = storage
        df.attrs["output_path"] = str(output_path)
        df.attrs["storage"] = storage
        df.attrs["storage_format"] = storage_format
        Path(f"{output_path}.metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return df, plan
