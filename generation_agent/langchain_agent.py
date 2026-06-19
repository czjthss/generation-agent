from __future__ import annotations

import json
import os
from typing import Any

from .env import load_project_env
from .planning_prompts import (
    PLAN_COMPILER_SYSTEM_PROMPT,
    QUALITY_EVALUATOR_PROMPT,
    REFLECTION_SYSTEM_PROMPT,
    SERIES_AUDITOR_PROMPT,
    build_reflection_input,
    build_revision_input,
    build_compilation_input,
)
from .planner import DEFAULT_LLM_MODEL, DEFAULT_OPENAI_BASE_URL, SeriesPlan


load_project_env()


class PlanReviewError(RuntimeError):
    """Raised when a plan cannot pass the mandatory reflection gate."""


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
        """Inspect whether existing feature generators can cover the request and suggest a generator_type."""
        return json.dumps(
            {
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
        anomaly_enabled: bool = False,
        anomaly_severity: str = "medium",
        anomaly_target: str = "value",
        rationale: str = "",
    ) -> str:
        """Create the final plan by composing feature parameters.

        Use this for most requests. The LLM should infer the domain and then set feature
        parameters: cycles, trends, sparse events, daylight envelope, smooth inertia,
        count process, bounds, noise, and anomalies. Independently select semantic_type
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
            content = "\n".join(str(item) for item in content)
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

    response = llm.invoke(
        [
            SystemMessage(content=REFLECTION_SYSTEM_PROMPT),
            HumanMessage(
                content=build_reflection_input(description, plan.to_dict(), reference_evidence)
            ),
        ]
    )
    payload = _extract_json_from_agent_result({"messages": [response]})
    if payload is None:
        return {
            "status": "REVISE",
            "issues": ["Reflection output could not be parsed or verified."],
            "revision_instruction": "Re-evaluate the complete plan against every reflection check.",
            "parser_fallback": True,
        }
    status = str(payload.get("status", "REVISE")).upper()
    issues = payload.get("issues", [])
    if status not in {"PASS", "REVISE"}:
        status = "REVISE"
        issues = list(issues) if isinstance(issues, list) else [str(issues)]
        issues.append("Reflection returned an invalid status.")
    return {
        "status": "REVISE" if status == "REVISE" else "PASS",
        "issues": [str(item) for item in issues] if isinstance(issues, list) else [str(issues)],
        "revision_instruction": str(payload.get("revision_instruction", "")),
        "confidence": float(payload.get("confidence", 0.0))
        if isinstance(payload.get("confidence"), (int, float))
        else 0.0,
    }


def audit_generated_series(
    llm: Any,
    description: str,
    plan: SeriesPlan,
    series_summary: dict[str, Any],
    validation_report: dict[str, Any],
    component_workflow: dict[str, Any] | None = None,
    component_report: dict[str, Any] | None = None,
    diversity_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Use domain knowledge to audit generated output without sending the full series."""
    from langchain_core.messages import HumanMessage, SystemMessage

    payload = {
        "original_request": description,
        "plan": plan.to_dict(),
        "series_summary": series_summary,
        "deterministic_validation": validation_report,
        "component_workflow": component_workflow or {},
        "component_report": component_report or {},
        "diversity_report": diversity_report or {},
    }
    response = llm.invoke(
        [
            SystemMessage(content=QUALITY_EVALUATOR_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False, indent=2)),
        ]
    )
    parsed = _extract_json_from_agent_result({"messages": [response]})
    if not isinstance(parsed, dict):
        return {
            "status": "UNVERIFIED",
            "issues": ["Quality evaluation output could not be parsed."],
            "revision_instruction": "",
        }
    status = str(parsed.get("status", "UNVERIFIED")).upper()
    if status == "REVISE":
        status = "REGENERATE"
    if status not in {"PASS", "REGENERATE"}:
        status = "UNVERIFIED"
    def _as_issue_list(value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if value:
            return [value]
        return []

    issues = parsed.get("issues", [])
    if not issues:
        issues = _as_issue_list(parsed.get("component_issues")) + _as_issue_list(parsed.get("final_series_issues"))
    return {
        "status": status,
        "issues": [str(item) for item in issues] if isinstance(issues, list) else [str(issues)],
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
) -> SeriesPlan | None:
    """Compile one corrected plan after output-level audit requests regeneration."""
    from langchain.agents import create_agent

    evidence = plan.metadata.get("specialist_evidence", {})
    revision_input = build_revision_input(
        description,
        plan.to_dict(),
        audit.get("issues", []),
        audit.get("revision_instruction", ""),
        evidence={"specialist_evidence": evidence, "series_audit": audit},
    )
    agent = create_agent(llm, tools=_build_tools(), system_prompt=PLAN_COMPILER_SYSTEM_PROMPT)
    result = agent.invoke({"messages": [{"role": "user", "content": revision_input}]})
    payload = _extract_json_from_agent_result(result)
    if payload is None:
        return None
    revised = SeriesPlan.from_dict(payload)
    preserved = {
        key: value
        for key, value in plan.metadata.items()
        if key not in {"candidate_domain_rule", "candidate_rule_status"}
    }
    revised.metadata = {**preserved, **revised.metadata}
    revised.metadata["replanned_after_series_audit"] = True
    return revised


def plan_with_langchain_tools(
    description: str,
    model: str = DEFAULT_LLM_MODEL,
    reference_profile: dict[str, Any] | None = None,
) -> SeriesPlan | None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for LLM agent planning")

    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
        temperature=0,
        max_completion_tokens=1200,
        timeout=60,
        max_retries=1,
    )
    from .specialist_workflow import GenerationSpecialistWorkflow

    reference_context = (
        {"reference_strength": "structure", "reference_profile": reference_profile}
        if reference_profile
        else None
    )
    specialist_evidence = GenerationSpecialistWorkflow(llm).analyze(
        description, reference_context=reference_context
    ).to_dict()
    planning_input = build_compilation_input(description, specialist_evidence)

    agent = create_agent(llm, tools=_build_tools(), system_prompt=PLAN_COMPILER_SYSTEM_PROMPT)
    result = agent.invoke({"messages": [{"role": "user", "content": planning_input}]})
    payload = _extract_json_from_agent_result(result)
    if payload is None:
        return None
    plan = SeriesPlan.from_dict(payload)

    try:
        reflection = _reflect_on_plan(llm, description, plan, specialist_evidence)
    except Exception as exc:
        reflection = {
            "status": "REVISE",
            "issues": ["Reflection service failed; the plan is not verified."],
            "revision_instruction": "Re-evaluate the complete plan before generation.",
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        }

    if reflection["status"] == "REVISE":
        revision_input = build_revision_input(
            description,
            plan.to_dict(),
            reflection["issues"],
            reflection["revision_instruction"],
            evidence=specialist_evidence,
        )
        revised_result = agent.invoke({"messages": [{"role": "user", "content": revision_input}]})
        revised_payload = _extract_json_from_agent_result(revised_result)
        if revised_payload is not None:
            payload = revised_payload
            plan = SeriesPlan.from_dict(payload)
            plan.metadata["replanned_after_reflection"] = True
            try:
                second_reflection = _reflect_on_plan(llm, description, plan, specialist_evidence)
            except Exception as exc:
                second_reflection = {
                    "status": "REVISE",
                    "issues": ["Post-revision reflection failed; the plan is not verified."],
                    "revision_instruction": "",
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                }
            reflection["post_revision"] = second_reflection
            if second_reflection["status"] != "PASS":
                raise PlanReviewError("revised plan did not pass reflection")
        else:
            raise PlanReviewError("reflection requested revision but no revised plan was produced")

    payload.setdefault("metadata", {})
    plan.metadata.update(payload["metadata"])
    plan.metadata["planner"] = "langchain_role_workflow"
    plan.metadata["model"] = model
    plan.metadata["base_url"] = os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)
    plan.metadata["description"] = description
    plan.metadata["workflow"] = "role_specialized_multi_agent"
    if specialist_evidence:
        plan.metadata["specialist_evidence"] = specialist_evidence
    if reference_profile:
        plan.metadata["reference_strength"] = "structure"
        plan.metadata["reference_source"] = reference_profile.get("source")
    plan.metadata["reflection"] = reflection
    return plan
