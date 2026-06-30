from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .agent import GenerationAgent
from .context_compactor import (
    build_generation_trace,
    compact_component_report_for_agent,
    compact_component_workflow_for_agent,
    compact_plan_for_metadata,
    compact_quality_for_metadata,
    compact_reference_profile_for_agent,
    compact_validation_for_metadata,
    write_json_gz,
)
from .compact_storage import preview_points, write_series_arrow
from .param_pack import write_param_pack
from .dataset_scenario_agent import DatasetScenarioDesigner
from .dataset_diversity import (
    diversify_plan,
    max_similarity,
    shape_signature,
    validate_diversity_strength,
)
from .reference_profiler import ReferenceProfile, compare_to_reference
from .semantic_types import AnomalyOverrides
from .workflow import LLM_ROLES, LOCAL_KERNEL_NAME, WORKFLOW_STEPS


FREQUENCY_ALIASES = {
    "hour": "h",
    "hourly": "h",
    "1h": "h",
    "one_hour": "h",
    "daily": "D",
    "day": "D",
    "1d": "D",
    "weekly": "W",
    "week": "W",
    "1w": "W",
    "monthly": "MS",
    "month": "MS",
    "1m": "MS",
    "quarterly": "QS",
    "quarter": "QS",
    "yearly": "YS",
    "annual": "YS",
    "annually": "YS",
    "15-minute": "15min",
    "15 minute": "15min",
    "15min": "15min",
    "30-minute": "30min",
    "30 minute": "30min",
    "30min": "30min",
}


def _normalize_scenario_frequency(raw_frequency: str) -> tuple[str | None, dict[str, Any] | None]:
    raw = str(raw_frequency or "").strip()
    if not raw or raw.lower() in {"unknown", "default", "same", "same_as_global", "global"}:
        return None, None
    key = raw.lower().replace("_", " ").replace("/", " ").strip()
    normalized = FREQUENCY_ALIASES.get(key, raw)
    try:
        pd.date_range(start="2026-01-01", periods=2, freq=normalized)
    except Exception:
        return None, {
            "type": "invalid_scenario_frequency",
            "suggested_frequency": raw,
            "policy": "global_frequency_fallback",
        }
    return normalized, None


def _lightweight_variant_audit(
    *,
    frame: Any,
    plan: Any,
    diversity_report: dict[str, Any],
    frequency_warning: dict[str, Any] | None,
    llm_reviewed: bool,
) -> dict[str, Any]:
    validation = frame.attrs.get("validation_report", {})
    anomaly = frame.attrs.get("anomaly_execution", {})
    issues: list[str] = []
    if not bool(validation.get("passed", True)):
        issues.append("deterministic_validation_failed")
    if validation.get("critical_repairs"):
        issues.append("critical_generation_repair")
    if not bool(validation.get("raw_passed", True)):
        issues.append("raw_generation_required_repair")
    if frequency_warning is not None:
        issues.append("scenario_frequency_policy_warning")
    if plan.anomaly_enabled and int(anomaly.get("active_points", 0)) == 0:
        issues.append("requested_anomaly_not_observed")
    if diversity_report.get("attempt", 1) > 1:
        issues.append("diversity_retry_used")
    max_similarity = diversity_report.get("max_similarity")
    if max_similarity is not None and float(max_similarity) >= float(diversity_report.get("similarity_threshold", 1.0)):
        issues.append("shape_similarity_above_threshold")
    return {
        "status": "PASS" if not issues else ("FAIL" if "critical_generation_repair" in issues else "WARN"),
        "llm_reviewed": bool(llm_reviewed),
        "issues": issues,
        "checks": {
            "raw_generation_passed": bool(validation.get("raw_passed", True)),
            "repaired_generation_passed": bool(validation.get("repaired_passed", validation.get("passed", True))),
            "critical_repairs_absent": not bool(validation.get("critical_repairs")),
            "frequency_policy_clean": frequency_warning is None,
            "anomaly_execution_consistent": not plan.anomaly_enabled or int(anomaly.get("active_points", 0)) > 0,
            "diversity_unique": "shape_similarity_above_threshold" not in issues,
        },
    }


class DatasetGenerator:
    def __init__(self, agent: GenerationAgent) -> None:
        self.agent = agent
        self.scenario_designer = DatasetScenarioDesigner(model=agent.model)

    def generate_to_directory(
        self,
        domain: str,
        output_dir: str | Path,
        series_count: int = 10,
        length: int = 168,
        freq: str = "h",
        start: str = "2026-07-01 00:00:00",
        seed: int | None = 42,
        anomaly_overrides: AnomalyOverrides | None = None,
        reference_profile: ReferenceProfile | dict[str, Any] | None = None,
        preview_max_points: int = 400,
        scenario_count: int | None = None,
        diversity_strength: str = "medium",
        diversity_check: bool = True,
        similarity_threshold: float = 0.96,
        max_diversity_retries: int = 2,
        save_trace: bool = False,
        storage_mode: str = "arrow",
        respect_scenario_frequency: bool = False,
    ) -> dict[str, Any]:
        if storage_mode not in {"arrow", "param-pack"}:
            raise ValueError("storage_mode must be 'arrow' or 'param-pack'")
        if series_count <= 0:
            raise ValueError("series_count must be positive")
        diversity_strength = validate_diversity_strength(diversity_strength)
        if scenario_count is None:
            scenario_count = min(series_count, 50)
        if scenario_count <= 0:
            raise ValueError("scenario_count must be positive")
        scenario_count = min(scenario_count, series_count)
        similarity_threshold = float(similarity_threshold)
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1]")
        policy_retry_limit = self.agent.cost_policy.dataset_max_diversity_retries
        if policy_retry_limit is not None:
            max_diversity_retries = min(max_diversity_retries, policy_retry_limit)
        max_diversity_retries = max(0, int(max_diversity_retries))
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        profile_payload = (
            reference_profile.to_dict() if isinstance(reference_profile, ReferenceProfile) else reference_profile
        )
        scenarios = self.scenario_designer.design(domain, scenario_count)
        (root / "scenarios.json").write_text(
            json.dumps([scenario.to_dict() for scenario in scenarios], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if profile_payload:
            (root / "reference_profile.json").write_text(
                json.dumps(compact_reference_profile_for_agent(profile_payload), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        entries: list[dict[str, Any]] = []
        skipped_entries: list[dict[str, Any]] = []
        accepted_signatures: list[dict[str, Any]] = []
        base_plan_cache: dict[int, Any] = {}
        preview: dict[str, Any] | None = None
        index = 0
        global_attempt = 0
        max_total_attempts = max(series_count * max(max_diversity_retries + 3, 5), series_count)
        while len(entries) < series_count and global_attempt < max_total_attempts:
            scenario = scenarios[index % len(scenarios)]
            scenario_index = index % len(scenarios)
            variant_index = index // len(scenarios)
            candidate_id = f"candidate_{global_attempt + 1:06d}"
            series_id = f"series_{len(entries) + 1:04d}"
            raw_scenario_frequency = str(getattr(scenario, "suggested_frequency", "") or "").strip()
            scenario_frequency, frequency_warning = _normalize_scenario_frequency(raw_scenario_frequency)
            effective_freq = freq
            if scenario_frequency and scenario_frequency != freq:
                if respect_scenario_frequency:
                    effective_freq = scenario_frequency
                else:
                    frequency_warning = {
                        "type": "scenario_frequency_ignored",
                        "suggested_frequency": raw_scenario_frequency,
                        "normalized_frequency": scenario_frequency,
                        "effective_frequency": freq,
                        "policy": "global_frequency_preserved",
                    }
            elif frequency_warning is not None:
                frequency_warning["effective_frequency"] = freq
            base_seed = None if seed is None else seed + global_attempt * 17
            if scenario_index not in base_plan_cache:
                base_plan_cache[scenario_index] = self.agent.plan(
                    scenario.description, reference_profile=profile_payload
                )
            base_plan = base_plan_cache[scenario_index]
            best: tuple[Any, Any, dict[str, Any], dict[str, Any], dict[str, Any], int, int | None] | None = None
            for attempt in range(max_diversity_retries + 1):
                attempt_seed = None if base_seed is None else base_seed + attempt * 100_003
                candidate_plan = diversify_plan(
                    base_plan,
                    scenario,
                    series_index=index,
                    variant_index=variant_index + attempt,
                    seed=seed,
                    strength=diversity_strength,
                )
                frame = self.agent.generate_from_plan(
                    candidate_plan,
                    length=length,
                    freq=effective_freq,
                    start=start,
                    seed=attempt_seed,
                    anomaly_overrides=anomaly_overrides,
                )
                final_plan_payload = frame.attrs.get("final_plan")
                if isinstance(final_plan_payload, dict):
                    from .planner import SeriesPlan

                    candidate_plan = SeriesPlan.from_dict(final_plan_payload)
                    base_plan_cache[scenario_index] = candidate_plan
                signature = shape_signature(frame)
                similarity = max_similarity(signature, accepted_signatures)
                is_unique = (
                    not diversity_check
                    or diversity_strength == "off"
                    or similarity["max_similarity"] < similarity_threshold
                    or not accepted_signatures
                )
                diversity_report = {
                    "enabled": bool(diversity_check),
                    "strength": diversity_strength,
                    "scenario_count": scenario_count,
                    "variant_index": variant_index,
                    "attempt": attempt + 1,
                    "max_attempts": max_diversity_retries + 1,
                    "similarity_threshold": similarity_threshold,
                    "max_similarity": similarity["max_similarity"],
                    "nearest_series_id": similarity["nearest_series_id"],
                    "accepted": bool(is_unique),
                    "shape_summary": signature["summary"],
                }
                candidate_audit = _lightweight_variant_audit(
                    frame=frame,
                    plan=candidate_plan,
                    diversity_report=diversity_report,
                    frequency_warning=frequency_warning,
                    llm_reviewed=True,
                )
                if candidate_audit["status"] == "FAIL":
                    diversity_report["accepted"] = False
                    best = (frame, candidate_plan, signature, diversity_report, candidate_audit, attempt + 1, attempt_seed)
                    continue
                best = (frame, candidate_plan, signature, diversity_report, candidate_audit, attempt + 1, attempt_seed)
                if is_unique:
                    break
            assert best is not None
            frame, plan, signature, diversity_report, variant_audit, attempts_used, series_seed = best
            if variant_audit["status"] == "FAIL":
                skipped_entries.append(
                    {
                        "series_id": candidate_id,
                        "description": scenario.description,
                        "scenario_index": index % len(scenarios),
                        "variant_index": variant_index,
                        "reason": "variant_audit_failed",
                        "variant_audit": variant_audit,
                        "attempts_used": attempts_used,
                    }
                )
                global_attempt += 1
                index += 1
                continue
            accepted_signatures.append({"series_id": series_id, "signature": signature})
            if preview is None:
                preview = preview_points(frame, max_points=preview_max_points)
                preview["series_id"] = series_id
                preview["description"] = scenario.description
            filename = f"{series_id}.arrow" if storage_mode == "arrow" else f"{series_id}.syn.json.gz"
            llm_reviewed = True
            metadata = {
                "series_id": series_id,
                "description": scenario.description,
                "scenario_index": index % len(scenarios),
                "variant_index": variant_index,
                "cost_mode": self.agent.cost_policy.mode,
                "llm_reviewed": llm_reviewed,
                "variant_audit": variant_audit,
                "start": start,
                "frequency": effective_freq,
                "requested_global_frequency": freq,
                "suggested_frequency": raw_scenario_frequency or None,
                "normalized_suggested_frequency": scenario_frequency,
                "frequency_warning": frequency_warning,
                "length": length,
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
                "storage_format": "arrow_ipc" if storage_mode == "arrow" else "generation_agent_param_pack",
                "data_policy": {
                    "llm_outputs_data_points": False,
                    "llm_directly_edits_data_points": False,
                    "numeric_values_computed_by": LOCAL_KERNEL_NAME,
                },
            }
            if save_trace:
                metadata["trace"] = write_json_gz(
                    root / f"{series_id}.trace.json.gz",
                    build_generation_trace(plan.to_dict(), frame.attrs),
                )
            if storage_mode == "arrow":
                storage = write_series_arrow(
                    frame,
                    root / filename,
                    metadata=metadata,
                )
            else:
                storage = write_param_pack(
                    root / filename,
                    description=scenario.description,
                    plan=plan,
                    length=length,
                    freq=effective_freq,
                    start=start,
                    seed=series_seed,
                    metadata=metadata,
                    frame=frame,
                )

            similarity = None
            if profile_payload:
                try:
                    similarity = compare_to_reference(frame, profile_payload)
                except Exception as exc:
                    similarity = {"error": f"{exc.__class__.__name__}: {exc}"}
            entries.append(
                {
                    "series_id": series_id,
                    "description": scenario.description,
                    "scenario_index": index % len(scenarios),
                    "variant_index": variant_index,
                    "rationale": scenario.rationale,
                    "tags": scenario.tags,
                    "observable": scenario.observable,
                    "unit": scenario.unit,
                    "suggested_frequency": scenario.suggested_frequency,
                    "normalized_suggested_frequency": scenario_frequency,
                    "effective_frequency": effective_freq,
                    "frequency_warning": frequency_warning,
                    "temporal_context": scenario.temporal_context,
                    "semantic_hint": scenario.semantic_hint,
                    "diversity_axis": scenario.diversity_axis,
                    "file": filename,
                    "storage": storage,
                    "seed": series_seed,
                    "diversity": diversity_report,
                    "diversity_retries": attempts_used - 1,
                    "plan_summary": metadata["plan_summary"],
                    "validation": metadata["deterministic_validation"],
                    "series_audit": metadata["quality_evaluation"],
                    "variant_audit": variant_audit,
                    "component_workflow": metadata["component_workflow"],
                    "component_report": metadata["component_report"],
                    "trace": metadata["trace"],
                    "reference_similarity": similarity,
                }
            )
            global_attempt += 1
            index += 1

        if len(entries) < series_count:
            raise RuntimeError(
                f"Dataset generation produced {len(entries)} accepted series but {series_count} were requested after {global_attempt} attempts"
            )

        manifest = {
            "mode": "dataset",
            "storage_format": "arrow_ipc" if storage_mode == "arrow" else "generation_agent_param_pack",
            "storage_mode": storage_mode,
            "domain": domain,
            "series_count": len(entries),
            "requested_series_count": series_count,
            "skipped_series_count": len(skipped_entries),
            "scenario_count": len(scenarios),
            "length_per_series": length,
            "frequency": freq,
            "frequency_policy": {
                "respect_scenario_frequency": bool(respect_scenario_frequency),
                "default": (
                    "scenario suggested_frequency is used per series"
                    if respect_scenario_frequency
                    else "global frequency is preserved; scenario suggested_frequency is recorded as metadata"
                ),
                "conflict_count": int(sum(1 for item in entries if item.get("frequency_warning"))),
            },
            "start": start,
            "diversity": {
                "strength": diversity_strength,
                "check_enabled": bool(diversity_check),
                "similarity_threshold": similarity_threshold,
                "max_retries_per_series": max_diversity_retries,
                "rejected_candidates": int(sum(item["diversity_retries"] for item in entries)),
                "mean_max_similarity": float(
                    sum(item["diversity"]["max_similarity"] for item in entries) / max(1, len(entries))
                ),
            },
            "cost_mode": self.agent.cost_policy.mode,
            "dataset_llm_policy": {
                "base_plan_cache": "one compact LLM plan per designed scenario; accepted variants still receive LLM anomaly and quality evaluation",
                "review_every_series": True,
                "cost_mode_effect": "cost mode may limit regeneration attempts, but accepted dataset series must pass LLM quality evaluation",
                "variant_policy": "all accepted series receive LLM quality review; failed variants are resampled until the requested count is met",
            },
            "timestamp_policy": "start/frequency/length stored in manifest and Arrow metadata; no per-row timestamp column",
            "row_metadata_policy": "unit/domain/generator_type/semantic_type stored in manifest and Arrow metadata; not repeated per row",
            "value_dtype": "float32",
            "workflow_steps": WORKFLOW_STEPS,
            "value_generation": LOCAL_KERNEL_NAME,
            "llm_roles": LLM_ROLES,
            "data_policy": {
                "llm_outputs_data_points": False,
                "llm_directly_edits_data_points": False,
                "numeric_values_computed_by": LOCAL_KERNEL_NAME,
            },
            "reference_strength": "structure" if profile_payload else None,
            "reference_source": profile_payload.get("source") if profile_payload else None,
            "dataset_file": None,
            "scenarios_file": "scenarios.json",
            "preview": preview or {"points": [], "count": 0},
            "series": entries,
            "skipped_series": skipped_entries,
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest
