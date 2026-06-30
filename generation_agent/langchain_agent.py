from __future__ import annotations

import json
import os
from typing import Any

from .env import load_project_env
from .cost_policy import (
    compact_plan_for_llm,
    compact_specialist_evidence,
    get_cost_policy,
    revision_plan_template,
)
from .context_compactor import (
    compact_capability_manifest_for_agent,
    compact_component_report_for_agent,
    compact_component_workflow_for_agent,
    compact_reference_profile_for_agent,
    compact_series_summary_for_quality,
)
from .planning_prompts import (
    ANOMALY_STRATEGY_PROMPT,
    MECHANISM_PLANNING_AGENT_PROMPT,
    PLAN_COMPILER_SYSTEM_PROMPT,
    QUALITY_EVALUATOR_PROMPT,
    REFLECTION_SYSTEM_PROMPT,
    REQUIREMENT_UNDERSTANDING_AGENT_PROMPT,
    SERIES_AUDITOR_PROMPT,
    build_compilation_input,
    build_mechanism_planning_input,
    build_requirement_understanding_input,
    build_reflection_input,
    build_revision_input,
)
from .planner import DEFAULT_LLM_MODEL, DEFAULT_OPENAI_BASE_URL, SeriesPlan
from .plan_normalizer import normalize_plan_for_execution
from .workflow import LOCAL_KERNEL_NAME, WORKFLOW_NAME, WORKFLOW_STEPS


load_project_env()


class PlanReviewError(RuntimeError):
    """Raised when a plan cannot pass the mandatory reflection gate."""


class AnomalyStrategyError(RuntimeError):
    """Raised when the mandatory LLM anomaly strategy cannot be obtained."""


class QualityEvaluationError(RuntimeError):
    """Raised when mandatory output quality evaluation cannot be completed."""


def _plan_json(plan: SeriesPlan, tool_name: str) -> str:
    plan.metadata["selected_tool"] = tool_name
    return json.dumps(plan.to_dict(), ensure_ascii=False, indent=2)


def _build_tools() -> list[Any]:
    from langchain_core.tools import tool

    def _parse_params(domain_params_json: str) -> dict[str, Any]:
        if not domain_params_json.strip():
            return {}
        try:
            payload = json.loads(domain_params_json)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _parse_list(payload_json: str) -> list[dict[str, Any]]:
        if not payload_json.strip():
            return []
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError:
            return []
        return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []

    def _parse_keywords(keywords_csv: str, description: str) -> list[str]:
        keywords = [item.strip() for item in keywords_csv.split(",") if item.strip()]
        if not keywords:
            keywords = [description[:24]]
        return keywords

    @tool
    def inspect_feature_coverage(description: str) -> str:
        """Inspect available executable feature families, mechanism templates, and semantic types."""
        from .feature_registry import feature_capability_manifest

        manifest = feature_capability_manifest()
        return json.dumps(
            {
                "mechanism_capabilities": manifest,
                "available_feature_generators": [
                    "cyclic_signal",
                    "intermittent_event",
                    "daylight_envelope",
                    "smooth_environmental",
                    "count_process",
                    "bounded_utilization",
                ],
                "available_semantic_types": [
                    "instantaneous",
                    "cumulative",
                    "stock_flow",
                    "regime_switching",
                    "random_walk",
                    "decay_recovery",
                    "saturation_growth",
                    "multivariate_lag",
                ],
                "coverage_guidance": {
                    "cyclic_signal": "Use for domains dominated by repeated cycles plus trend/noise.",
                    "intermittent_event": "Use for sparse event domains with many zeros or bursts.",
                    "daylight_envelope": "Use for processes constrained by daylight or availability windows.",
                    "smooth_environmental": "Use for inertia-driven continuous variables.",
                    "count_process": "Use for nonnegative discrete demand/count data.",
                    "bounded_utilization": "Use for percentage/utilization metrics with hard bounds.",
                },
                "semantic_guidance": {
                    "instantaneous": "Each value is an observation at that time step.",
                    "cumulative": "Values integrate per-step increments from an initial value.",
                    "stock_flow": "A stock evolves through inflow minus outflow with conservation.",
                    "regime_switching": "Persistent latent states control level or dynamics.",
                    "random_walk": "The next value depends on the previous value plus a stochastic step.",
                    "decay_recovery": "A state responds to impulses and recursively returns toward equilibrium.",
                    "saturation_growth": "Growth rate decreases as a finite capacity is approached.",
                    "multivariate_lag": "One or more driver variables affect targets after explicit lags.",
                },
                "instruction": (
                    "Choose generator_type and semantic_type independently. If one generator "
                    "can approximate the domain by feature parameters, call "
                    "finalize_feature_plan. If the request reveals a reusable new domain, call "
                    "create_custom_domain_rule with parameters mapped to the closest generator."
                ),
                "description": description,
            },
            ensure_ascii=False,
            indent=2,
        )

    @tool
    def finalize_feature_plan(
        description: str,
        domain: str,
        unit: str,
        generator_type: str,
        baseline: float,
        trend_slope: float = 0.0,
        daily_amplitude: float = 0.0,
        weekly_enabled: bool = False,
        weekly_amplitude: float = 0.0,
        seasonal_amplitude: float = 0.0,
        heat_effect: float = 0.0,
        noise_sigma: float = 1.0,
        lower_bound: float = 0.0,
        anomaly_count: int = 0,
        anomaly_magnitude: float = 3.0,
        anomaly_width: int = 1,
        anomaly_kind: str = "spike",
        event_probability: float = 0.18,
        mean_duration: float = 5.0,
        intensity_shape: float = 1.4,
        intensity_scale: float = 5.0,
        dry_spell_bias: float = 0.5,
        storm_probability: float = 0.08,
        storm_multiplier: float = 3.0,
        sunrise_hour: float = 6.0,
        sunset_hour: float = 19.0,
        cloud_probability: float = 0.18,
        cloud_drop_min: float = 0.35,
        cloud_drop_max: float = 0.8,
        inertia: float = 0.88,
        peak_hour: float = 15.0,
        morning_peak: float = 8.0,
        evening_peak: float = 18.0,
        overdispersion: float = 1.35,
        upper_bound: float = 100.0,
        batch_hour: float = 2.0,
        batch_probability: float = 0.0,
        semantic_type: str = "instantaneous",
        semantic_config_json: str = "{}",
        output_constraints_json: str = "{}",
        variables_json: str = "[]",
        relationships_json: str = "[]",
        components_json: str = "[]",
        composition_json: str = "{}",
        anomaly_enabled: bool = False,
        anomaly_severity: str = "medium",
        anomaly_target: str = "value",
        rationale: str = "",
    ) -> str:
        """Create the final plan by composing feature parameters.

        Use this for most requests. Prefer components_json: a JSON array of executable
        mechanism components using feature_family names from inspect_feature_coverage.
        The legacy scalar fields remain as fallback defaults. Independently select semantic_type
        from instantaneous, cumulative, stock_flow, regime_switching, random_walk,
        decay_recovery, saturation_growth, or multivariate_lag, and provide the matching
        semantic_config and output_constraints. The LLM must explicitly decide
        anomaly_count, anomaly_kind, anomaly_magnitude, and anomaly_width from the
        domain semantics; use anomaly_count=0 only when anomalies are not requested or
        not natural for the scenario. generator_type must be one of cyclic_signal,
        intermittent_event, daylight_envelope, smooth_environmental, count_process,
        bounded_utilization.
        """
        from .feature_composer import build_feature_plan

        plan = build_feature_plan(
            description=description,
            domain=domain,
            unit=unit,
            generator_type=generator_type,
            baseline=baseline,
            trend_slope=trend_slope,
            daily_amplitude=daily_amplitude,
            weekly_enabled=weekly_enabled,
            weekly_amplitude=weekly_amplitude,
            seasonal_amplitude=seasonal_amplitude,
            heat_effect=heat_effect,
            noise_sigma=noise_sigma,
            lower_bound=lower_bound,
            anomaly_count=anomaly_count,
            anomaly_magnitude=anomaly_magnitude,
            anomaly_width=anomaly_width,
            anomaly_kind=anomaly_kind,
            event_probability=event_probability,
            mean_duration=mean_duration,
            intensity_shape=intensity_shape,
            intensity_scale=intensity_scale,
            dry_spell_bias=dry_spell_bias,
            storm_probability=storm_probability,
            storm_multiplier=storm_multiplier,
            sunrise_hour=sunrise_hour,
            sunset_hour=sunset_hour,
            cloud_probability=cloud_probability,
            cloud_drop_min=cloud_drop_min,
            cloud_drop_max=cloud_drop_max,
            inertia=inertia,
            peak_hour=peak_hour,
            morning_peak=morning_peak,
            evening_peak=evening_peak,
            overdispersion=overdispersion,
            upper_bound=upper_bound,
            batch_hour=batch_hour,
            batch_probability=batch_probability,
            semantic_type=semantic_type,
            semantic_config=_parse_params(semantic_config_json),
            output_constraints=_parse_params(output_constraints_json),
            variables=_parse_list(variables_json),
            relationships=_parse_list(relationships_json),
            components=_parse_list(components_json),
            composition=_parse_params(composition_json),
            anomaly_enabled=anomaly_enabled,
            anomaly_severity=anomaly_severity,
            anomaly_target=anomaly_target,
            rationale=rationale,
        )
        return _plan_json(plan, "finalize_feature_plan")

    @tool
    def use_existing_custom_domain_rule(description: str) -> str:
        """Use when a previously registered custom domain rule matches the request."""
        from .domain_rules import match_rule, rule_to_plan

        rule = match_rule(description)
        if rule is None:
            return json.dumps({"error": "No matching custom domain rule found"}, ensure_ascii=False)
        return _plan_json(rule_to_plan(rule, description), "use_existing_custom_domain_rule")

    @tool
    def create_custom_domain_rule(
        description: str,
        domain: str,
        generator_type: str,
        unit: str,
        baseline: float,
        daily_amplitude: float,
        trend_slope: float,
        noise_sigma: float,
        lower_bound: float,
        weekly_enabled: bool,
        keywords_csv: str,
        domain_params_json: str,
        rationale: str = "",
        semantic_type: str = "instantaneous",
        semantic_config_json: str = "{}",
        output_constraints_json: str = "{}",
        anomaly_enabled: bool = False,
        anomaly_severity: str = "medium",
        anomaly_target: str = "value",
    ) -> str:
        """Stage a candidate domain rule when no existing mapping is a good fit.

        generator_type must be one of: cyclic_signal, intermittent_event, daylight_envelope,
        smooth_environmental, count_process, bounded_utilization. domain_params_json must be
        a JSON object with parameters that adapt that generator to the new domain.
        """
        from .domain_rules import plan_to_rule

        if generator_type not in {
            "cyclic_signal",
            "intermittent_event",
            "daylight_envelope",
            "smooth_environmental",
            "count_process",
            "bounded_utilization",
        }:
            generator_type = "cyclic_signal"
        plan = SeriesPlan(
            domain=domain,
            generator_type=generator_type,
            unit=unit,
            baseline=baseline,
            daily_amplitude=daily_amplitude,
            trend_slope=trend_slope,
            noise_sigma=noise_sigma,
            lower_bound=lower_bound,
            weekly_enabled=weekly_enabled,
            domain_params=_parse_params(domain_params_json),
            semantic_type=semantic_type,
            semantic_config=_parse_params(semantic_config_json),
            output_constraints=_parse_params(output_constraints_json),
            anomaly_enabled=anomaly_enabled,
            anomaly_severity=anomaly_severity,
            anomaly_target=anomaly_target,
            metadata={
                "domain_selection_reason": "LangChain agent created a custom domain rule",
                "custom_rule_rationale": rationale,
            },
        )
        keywords = _parse_keywords(keywords_csv, description)
        plan.metadata["candidate_domain_rule"] = plan_to_rule(
            plan, keywords=keywords, rationale=rationale
        )
        plan.metadata["candidate_rule_status"] = "pending_validation"
        return _plan_json(plan, "create_custom_domain_rule")

    return [
        inspect_feature_coverage,
        finalize_feature_plan,
        use_existing_custom_domain_rule,
        create_custom_domain_rule,
    ]


def _extract_json_from_agent_result(result: Any) -> dict[str, Any] | None:
    messages = result.get("messages", []) if isinstance(result, dict) else []
    for message in reversed(messages):
        content = getattr(message, "content", "")
        if isinstance(content, list):
            blocks: list[str] = []
            for item in content:
                if isinstance(item, str):
                    blocks.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        blocks.append(text)
            content = "\n".join(blocks)
        if not isinstance(content, str):
            continue
        cleaned = content.strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


def _reflect_on_plan(
    llm: Any,
    description: str,
    plan: SeriesPlan,
    reference_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage
    from .capability_registry import validate_plan_against_capabilities

    compact_plan = compact_plan_for_llm(plan.to_dict())
    capability_validation = validate_plan_against_capabilities(compact_plan)

    def path_exists(payload: dict[str, Any], path: str) -> bool:
        current: Any = payload
        for part in path.split("."):
            if not part or not isinstance(current, dict) or part not in current:
                return False
            current = current[part]
        return True

    response = llm.invoke(
        [
            SystemMessage(content=REFLECTION_SYSTEM_PROMPT),
            HumanMessage(
                content=build_reflection_input(
                    description,
                    compact_plan,
                    reference_evidence,
                )
            ),
        ]
    )
    payload = _extract_json_from_agent_result({"messages": [response]})
    if payload is None:
        return {
            "status": "REVISE",
            "hard_errors": ["Reflection output could not be parsed or verified."],
            "soft_warnings": [],
            "issues": ["Reflection output could not be parsed or verified."],
            "revision_instruction": "Re-evaluate the complete plan against every reflection check.",
            "parser_fallback": True,
        }
    status = str(payload.get("status", "REVISE")).upper()
    hard_errors = payload.get("hard_errors", payload.get("issues", []))
    soft_warnings = payload.get("soft_warnings", [])
    if not isinstance(hard_errors, list):
        hard_errors = [str(hard_errors)] if hard_errors else []
    if not isinstance(soft_warnings, list):
        soft_warnings = [str(soft_warnings)] if soft_warnings else []
    evidence_items = payload.get("hard_error_evidence", [])
    grounded_indices: set[int] = set()
    if isinstance(evidence_items, list):
        for item in evidence_items:
            if not isinstance(item, dict):
                continue
            index = item.get("failure_index")
            sources = item.get("sources", [])
            if isinstance(sources, str):
                sources = [sources]
            source_set = {str(source) for source in sources} if isinstance(sources, list) else set()
            plan_path = str(item.get("plan_path", "")).strip()
            request_quote = str(item.get("request_quote", "")).strip()
            capability_path = str(item.get("capability_path", "")).strip()
            request_contradiction = (
                {"original_request", "proposed_plan"}.issubset(source_set)
                and bool(request_quote)
                and request_quote in description
                and path_exists(compact_plan, plan_path)
            )
            capability_contradiction = {
                "proposed_plan",
                "pipeline_capabilities",
            }.issubset(source_set) and path_exists(
                compact_plan, plan_path
            ) and capability_path in set(capability_validation["invalid_paths"])
            if (
                isinstance(index, int)
                and 0 <= index < len(hard_errors)
                and str(item.get("evidence", "")).strip()
                and (request_contradiction or capability_contradiction)
            ):
                grounded_indices.add(index)
    ungrounded_errors = [
        str(issue) for index, issue in enumerate(hard_errors) if index not in grounded_indices
    ]
    hard_errors = [
        issue for index, issue in enumerate(hard_errors) if index in grounded_indices
    ]
    soft_warnings.extend(
        f"Ungrounded reflection concern: {issue}" for issue in ungrounded_errors
    )
    if status not in {"PASS", "REVISE"}:
        status = "REVISE"
        hard_errors.append("Reflection returned an invalid status.")
    if hard_errors:
        status = "REVISE"
    else:
        status = "PASS"
    return {
        "status": status,
        "hard_errors": [str(item) for item in hard_errors],
        "soft_warnings": [str(item) for item in soft_warnings],
        "issues": [str(item) for item in hard_errors],
        "revision_instruction": str(payload.get("revision_instruction", "")),
        "confidence": float(payload.get("confidence", 0.0))
        if isinstance(payload.get("confidence"), (int, float))
        else 0.0,
    }


def decide_anomaly_strategy(
    llm: Any,
    description: str,
    plan: SeriesPlan,
    anomaly_control: str = "auto",
    severity_override: str | None = None,
    attempts: int = 1,
) -> Any:
    """Require the LLM to choose mechanism-level anomaly behavior, never data points."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from .capability_registry import build_pipeline_capability_manifest
    from .semantic_types import AnomalyStrategy

    capabilities = compact_capability_manifest_for_agent(build_pipeline_capability_manifest())
    request = {
        "original_request": description,
        "plan": compact_plan_for_llm(plan.to_dict()),
        "anomaly_control": anomaly_control,
        "severity_override": severity_override,
        "execution_capabilities": capabilities["anomaly_injection"],
    }
    last_error = ""
    attempts = max(1, int(attempts))
    for _attempt in range(attempts):
        response = llm.invoke(
            [
                SystemMessage(content=ANOMALY_STRATEGY_PROMPT),
                HumanMessage(content=json.dumps(request, ensure_ascii=False, indent=2)),
            ]
        )
        payload = _extract_json_from_agent_result({"messages": [response]})
        if isinstance(payload, dict):
            try:
                enabled = bool(payload.get("enabled", False))
                if anomaly_control == "off":
                    enabled = False
                elif anomaly_control == "on":
                    enabled = True
                severity = str(severity_override or payload.get("severity", "medium")).lower()
                if severity not in {"low", "medium", "high"}:
                    severity = "medium"
                return AnomalyStrategy(
                    enabled=enabled,
                    reason=str(payload.get("reason", "")),
                    target=str(payload.get("target", plan.anomaly_target or "value")),
                    kind=str(payload.get("kind", plan.anomaly_kind or "spike")),
                    severity=severity,
                    count=max(0, int(payload.get("count", 0))),
                    width=max(1, int(payload.get("width", 1))),
                    magnitude=max(0.0, float(payload.get("magnitude", 3.0))),
                    constraints_after_injection=tuple(
                        str(item) for item in payload.get("constraints_after_injection", [])
                    ),
                    confidence=float(payload.get("confidence", 0.0)),
                )
            except (TypeError, ValueError) as exc:
                last_error = f"{exc.__class__.__name__}: {exc}"
        else:
            last_error = "response was not valid JSON"
        request["previous_output_error"] = last_error
    raise AnomalyStrategyError(
        f"LLM anomaly strategy failed after {attempts} attempts: {last_error}"
    )


def audit_generated_series(
    llm: Any,
    description: str,
    plan: SeriesPlan,
    series_summary: dict[str, Any],
    validation_report: dict[str, Any],
    component_workflow: dict[str, Any] | None = None,
    component_report: dict[str, Any] | None = None,
    anomaly_execution: dict[str, Any] | None = None,
    diversity_report: dict[str, Any] | None = None,
    attempts: int = 1,
    allow_json_repair: bool = True,
) -> dict[str, Any]:
    """Use domain knowledge to audit generated output without sending the full series."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from .capability_registry import build_pipeline_capability_manifest

    executable_plan = compact_plan_for_llm(plan.to_dict())
    executable_plan["metadata"] = {
        key: value
        for key, value in executable_plan.get("metadata", {}).items()
        if key in {"description", "selected_tool"}
    }
    final_anomaly_execution = anomaly_execution or {
        "enabled": bool(plan.anomaly_enabled),
        "requested_count": int(plan.anomaly_count),
        "target": plan.anomaly_target,
        "kind": plan.anomaly_kind,
        "observed_fraction": float(series_summary.get("anomaly_fraction", 0.0)),
    }
    payload = {
        "original_request": description,
        "evidence_authority": {
            "hard_failure_sources": [
                "original_request",
                "deterministic_validation",
                "executable_contract",
            ],
            "llm_assumptions_are_not_explicit_requirements": True,
        },
        "pipeline_capabilities": compact_capability_manifest_for_agent(
            build_pipeline_capability_manifest()
        ),
        "plan": executable_plan,
        "series_summary": compact_series_summary_for_quality(series_summary),
        "deterministic_validation": validation_report,
        "component_workflow": compact_component_workflow_for_agent(component_workflow),
        "component_report": compact_component_report_for_agent(component_report),
        "anomaly_execution": final_anomaly_execution,
        "diversity_report": diversity_report or {},
    }
    request = json.dumps(payload, ensure_ascii=False, indent=2)
    parsed = None
    attempts = max(1, int(attempts))
    for attempt in range(attempts):
        response = llm.invoke(
            [
                SystemMessage(content=QUALITY_EVALUATOR_PROMPT),
                HumanMessage(
                    content=request
                    + (
                        "\n\nThe previous response was not valid JSON. Return exactly one JSON object matching the contract, with no prose."
                        if attempt
                        else ""
                    )
                ),
            ]
        )
        parsed = _extract_json_from_agent_result({"messages": [response]})
        if isinstance(parsed, dict):
            break
    if allow_json_repair and not isinstance(parsed, dict):
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a strict JSON quality evaluator for a time-series generation agent. "
                        "Evaluate whether the locally generated summary satisfies the request, plan, "
                        "component mechanisms, deterministic validation, and business constraints. "
                        "Return only one JSON object. Do not include markdown or prose. "
                        "Use exactly these top-level keys: status, hard_failures, hard_failure_evidence, soft_warnings, "
                        "revision_instruction, confidence. Each hard failure needs a zero-based failure_index, sources, "
                        "and grounded request_quote, plan_path, validation_path, or execution_path as applicable. status must be PASS, PASS_WITH_WARNINGS, "
                        "or REGENERATE. hard_failures and soft_warnings must be arrays of strings. "
                        "Never output or modify data points."
                    )
                ),
                HumanMessage(content=request),
            ]
        )
        parsed = _extract_json_from_agent_result({"messages": [response]})
    if not isinstance(parsed, dict):
        return {
            "status": "UNVERIFIED",
            "hard_failures": ["Quality evaluation output could not be parsed."],
            "soft_warnings": [],
            "issues": ["Quality evaluation output could not be parsed."],
            "revision_instruction": "",
        }
    status = str(parsed.get("status", "UNVERIFIED")).upper()
    def _as_issue_list(value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if value:
            return [value]
        return []

    hard_failures = _as_issue_list(parsed.get("hard_failures"))
    soft_warnings = _as_issue_list(parsed.get("soft_warnings"))
    if not hard_failures and status == "REVISE":
        hard_failures = _as_issue_list(parsed.get("component_issues")) + _as_issue_list(parsed.get("final_series_issues"))
    evidence_items = parsed.get("hard_failure_evidence", [])
    grounded_indices: set[int] = set()

    def resolve_path(root: dict[str, Any], path: str) -> tuple[bool, Any]:
        current: Any = root
        for part in path.split("."):
            if not part or not isinstance(current, dict) or part not in current:
                return False, None
            current = current[part]
        return True, current

    if isinstance(evidence_items, list):
        for item in evidence_items:
            if not isinstance(item, dict):
                continue
            index = item.get("failure_index")
            sources = item.get("sources", item.get("source", []))
            if isinstance(sources, str):
                sources = [sources]
            source_set = {str(source) for source in sources} if isinstance(sources, list) else set()
            request_quote = str(item.get("request_quote", "")).strip()
            request_grounded = "original_request" not in source_set or (
                bool(request_quote) and request_quote in description
            )
            plan_ok, _ = resolve_path(
                executable_plan, str(item.get("plan_path", "")).strip()
            )
            validation_ok, validation_value = resolve_path(
                validation_report, str(item.get("validation_path", "")).strip()
            )
            execution_ok, _ = resolve_path(
                payload, str(item.get("execution_path", "")).strip()
            )
            deterministic_grounded = (
                "deterministic_validation" in source_set
                and validation_ok
                and validation_value is False
            )
            execution_grounded = (
                "executable_contract" in source_set and plan_ok and execution_ok
            )
            if (
                isinstance(index, int)
                and 0 <= index < len(hard_failures)
                and str(item.get("evidence", "")).strip()
                and request_grounded
                and (deterministic_grounded or execution_grounded)
            ):
                grounded_indices.add(index)
    ungrounded = [
        str(issue) for index, issue in enumerate(hard_failures) if index not in grounded_indices
    ]
    hard_failures = [
        issue for index, issue in enumerate(hard_failures) if index in grounded_indices
    ]
    soft_warnings.extend(
        f"Ungrounded quality concern: {issue}" for issue in ungrounded
    )
    if hard_failures:
        status = "REGENERATE"
    elif status in {"PASS", "PASS_WITH_WARNINGS", "REVISE"}:
        status = (
            "PASS_WITH_WARNINGS"
            if soft_warnings or status == "PASS_WITH_WARNINGS"
            else "PASS"
        )
    else:
        status = "UNVERIFIED"
    return {
        "status": status,
        "hard_failures": [str(item) for item in hard_failures],
        "soft_warnings": [str(item) for item in soft_warnings],
        "issues": [str(item) for item in hard_failures],
        "revision_instruction": str(
            parsed.get("revision_instruction")
            or parsed.get("revision_hints")
            or parsed.get("revision_target")
            or ""
        ),
        "confidence": float(parsed.get("confidence", 0.0))
        if isinstance(parsed.get("confidence"), (int, float))
        else 0.0,
        "quality_evaluator": parsed,
    }


def revise_plan_from_series_audit(
    llm: Any,
    description: str,
    plan: SeriesPlan,
    audit: dict[str, Any],
    attempts: int = 1,
    direct_attempts: int = 1,
) -> SeriesPlan | None:
    """Compile one corrected plan after output-level audit requests regeneration."""
    from langchain.agents import create_agent

    evidence = plan.metadata.get("workflow_evidence") or compact_specialist_evidence(
        plan.metadata.get("specialist_evidence", {})
    )
    revision_input = build_revision_input(
        description,
        revision_plan_template(plan.to_dict()),
        audit.get("issues", []),
        audit.get("revision_instruction", ""),
        evidence={"workflow_evidence": evidence, "series_audit": audit},
    )
    agent = create_agent(llm, tools=_build_tools(), system_prompt=PLAN_COMPILER_SYSTEM_PROMPT)
    try:
        payload = _invoke_plan_agent_with_retries(agent, revision_input, attempts=attempts)
    except PlanReviewError as tool_error:
        try:
            payload = _revise_plan_directly(
                llm,
                description,
                plan,
                audit.get("hard_failures", audit.get("issues", [])),
                audit.get("revision_instruction", ""),
                evidence={"workflow_evidence": evidence, "series_audit": audit},
                attempts=direct_attempts,
            )
        except PlanReviewError as direct_error:
            raise PlanReviewError(
                f"series-audit revision failed; tool_error={tool_error}; direct_error={direct_error}"
            ) from direct_error
    revised = normalize_plan_for_execution(
        SeriesPlan.from_dict(payload), description
    )
    preserved = {
        key: value
        for key, value in plan.metadata.items()
        if key not in {"candidate_domain_rule", "candidate_rule_status"}
    }
    revised.metadata = {**preserved, **revised.metadata}
    revised.metadata["replanned_after_series_audit"] = True
    return revised


def _invoke_plan_agent_with_retries(agent: Any, prompt: str, attempts: int = 3) -> dict[str, Any]:
    """Invoke a tool-calling plan agent with bounded JSON/tool recovery."""
    current_prompt = prompt
    last_error = "no response"
    for _attempt in range(attempts):
        result = agent.invoke({"messages": [{"role": "user", "content": current_prompt}]})
        payload = _extract_json_from_agent_result(result)
        if isinstance(payload, dict):
            try:
                SeriesPlan.from_dict(payload)
                return payload
            except (TypeError, ValueError) as exc:
                last_error = f"invalid plan schema: {exc}"
        else:
            last_error = "no valid plan JSON or final tool result"
        current_prompt = (
            prompt
            + "\n\nThe previous response failed validation: "
            + last_error
            + ". Call a final planning tool and return only its complete JSON result."
        )
    raise PlanReviewError(f"LLM planning failed after {attempts} attempts: {last_error}")


def _revise_plan_directly(
    llm: Any,
    description: str,
    plan: SeriesPlan,
    hard_errors: list[str],
    revision_instruction: str,
    evidence: dict[str, Any] | None = None,
    attempts: int = 3,
) -> dict[str, Any]:
    """LLM-only JSON revision fallback for gateways with unreliable tool calls."""
    from langchain_core.messages import HumanMessage, SystemMessage

    system = """You revise an existing executable time-series SeriesPlan. Correct only the supplied hard errors while preserving supported fields. Return one complete JSON object matching the existing SeriesPlan keys. Do not call tools, do not return prose, do not output time-series values, timestamps, indices, or point-level edits. The local numerical engine will generate every value."""
    prompt = build_revision_input(
        description,
        revision_plan_template(plan.to_dict()),
        hard_errors,
        revision_instruction,
        evidence=evidence,
    )
    last_error = "no response"
    for _attempt in range(attempts):
        response = llm.invoke(
            [
                SystemMessage(content=system),
                HumanMessage(
                    content=prompt
                    + "\n\nReturn the complete corrected SeriesPlan as one JSON object."
                ),
            ]
        )
        payload = _extract_json_from_agent_result({"messages": [response]})
        if isinstance(payload, dict):
            try:
                SeriesPlan.from_dict(payload)
                return payload
            except (TypeError, ValueError) as exc:
                last_error = f"invalid plan schema: {exc}"
        else:
            last_error = "response was not a complete JSON object"
        prompt += f"\n\nPrevious output error: {last_error}. Return valid JSON only."
    raise PlanReviewError(
        f"Direct LLM plan revision failed after {attempts} attempts: {last_error}"
    )


def _invoke_json_agent(
    llm: Any,
    system_prompt: str,
    user_payload: str,
    attempts: int = 1,
) -> dict[str, Any]:
    """Invoke a compact JSON-only LLM agent with bounded repair."""
    from langchain_core.messages import HumanMessage, SystemMessage

    current_payload = user_payload
    last_error = "no response"
    for attempt in range(max(1, int(attempts))):
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=current_payload
                    + (
                        "\n\nPrevious output error: "
                        + last_error
                        + ". Return exactly one valid JSON object matching the requested contract, with no prose."
                        if attempt
                        else ""
                    )
                ),
            ]
        )
        payload = _extract_json_from_agent_result({"messages": [response]})
        if isinstance(payload, dict):
            return payload
        last_error = "response was not valid JSON"
    raise PlanReviewError(f"JSON agent failed after {attempts} attempts: {last_error}")


def _build_compact_workflow_evidence(
    *,
    description: str,
    demand_understanding: dict[str, Any],
    mechanism_plan: dict[str, Any],
    reference_profile: dict[str, Any] | None,
) -> dict[str, Any]:
    from .capability_registry import build_pipeline_capability_manifest

    return {
        "workflow": WORKFLOW_NAME,
        "original_request": description,
        "requirement_understanding_agent": demand_understanding,
        "mechanism_planning_agent": mechanism_plan,
        "reference_profile": compact_reference_profile_for_agent(reference_profile),
        "available_capabilities": compact_capability_manifest_for_agent(
            build_pipeline_capability_manifest()
        ),
        "data_policy": {
            "llm_outputs_data_points": False,
            "llm_directly_edits_data_points": False,
            "numeric_values_computed_by": LOCAL_KERNEL_NAME,
        },
        "execution_order": WORKFLOW_STEPS,
    }


def plan_with_langchain_tools(
    description: str,
    model: str = DEFAULT_LLM_MODEL,
    reference_profile: dict[str, Any] | None = None,
    cost_mode: str = "balanced",
) -> SeriesPlan | None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for LLM agent planning")
    from .dependency_check import ensure_llm_dependencies

    ensure_llm_dependencies()

    from langchain.agents import create_agent
    from .llm_client import create_chat_openai

    policy = get_cost_policy(cost_mode)
    llm = create_chat_openai(
        model=model,
        temperature=0,
        max_completion_tokens=3000,
        timeout=60,
        max_retries=1,
    )

    reference_context = compact_reference_profile_for_agent(reference_profile) if reference_profile else None
    requirement_understanding = _invoke_json_agent(
        llm,
        REQUIREMENT_UNDERSTANDING_AGENT_PROMPT,
        build_requirement_understanding_input(
            description,
            reference_profile=reference_context,
            generation_mode="sequence",
        ),
        attempts=max(2, policy.plan_attempts),
    )
    mechanism_plan = _invoke_json_agent(
        llm,
        MECHANISM_PLANNING_AGENT_PROMPT,
        build_mechanism_planning_input(description, requirement_understanding),
        attempts=max(2, policy.plan_attempts),
    )
    workflow_evidence = _build_compact_workflow_evidence(
        description=description,
        demand_understanding=requirement_understanding,
        mechanism_plan=mechanism_plan,
        reference_profile=reference_profile,
    )
    compact_evidence = {
        "requirement_understanding_agent": requirement_understanding,
        "mechanism_planning_agent": mechanism_plan,
        "reference_profile": reference_context or {},
        "available_feature_families": workflow_evidence.get("available_capabilities", {}).get("feature_generation", {}),
        "available_semantics": workflow_evidence.get("available_capabilities", {}).get("semantic_transforms", {}),
    }
    planning_input = build_compilation_input(description, compact_evidence)

    agent = create_agent(llm, tools=_build_tools(), system_prompt=PLAN_COMPILER_SYSTEM_PROMPT)
    payload = _invoke_plan_agent_with_retries(
        agent,
        planning_input,
        attempts=policy.plan_attempts,
    )
    plan = normalize_plan_for_execution(SeriesPlan.from_dict(payload), description)

    payload.setdefault("metadata", {})
    plan.metadata.update(payload["metadata"])
    plan.metadata["planner"] = "compact_agent_workflow"
    plan.metadata["model"] = model
    plan.metadata["base_url"] = os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)
    plan.metadata["description"] = description
    plan.metadata["workflow"] = WORKFLOW_NAME
    plan.metadata["cost_mode"] = policy.mode
    plan.metadata["requirement_understanding_agent"] = requirement_understanding
    plan.metadata["mechanism_planning_agent"] = mechanism_plan
    plan.metadata["workflow_evidence"] = workflow_evidence
    plan.metadata["workflow_steps"] = WORKFLOW_STEPS
    if reference_profile:
        plan.metadata["reference_strength"] = "structure"
        plan.metadata["reference_source"] = reference_profile.get("source")
    plan.metadata["reflection"] = {
        "status": "MERGED_INTO_QUALITY_EVALUATION",
        "rounds": [],
        "soft_warnings": [],
    }
    return plan
