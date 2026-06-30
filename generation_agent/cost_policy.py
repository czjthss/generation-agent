from __future__ import annotations

from dataclasses import dataclass
from typing import Any


VALID_COST_MODES = {"cheap", "balanced", "strict"}


@dataclass(frozen=True)
class CostPolicy:
    mode: str
    plan_attempts: int
    reflection_rounds: int
    direct_revision_attempts: int
    anomaly_attempts: int
    quality_attempts: int
    quality_json_repair: bool
    max_regenerations: int
    dataset_max_diversity_retries: int | None


def get_cost_policy(mode: str | None) -> CostPolicy:
    normalized = (mode or "balanced").strip().lower()
    if normalized not in VALID_COST_MODES:
        raise ValueError(f"cost_mode must be one of {sorted(VALID_COST_MODES)}")
    if normalized == "cheap":
        return CostPolicy(
            mode="cheap",
            plan_attempts=1,
            reflection_rounds=1,
            direct_revision_attempts=1,
            anomaly_attempts=1,
            quality_attempts=1,
            quality_json_repair=False,
            max_regenerations=0,
            dataset_max_diversity_retries=0,
        )
    if normalized == "strict":
        return CostPolicy(
            mode="strict",
            plan_attempts=3,
            reflection_rounds=3,
            direct_revision_attempts=3,
            anomaly_attempts=3,
            quality_attempts=3,
            quality_json_repair=True,
            max_regenerations=2,
            dataset_max_diversity_retries=None,
        )
    return CostPolicy(
        mode="balanced",
        plan_attempts=2,
        reflection_rounds=2,
        direct_revision_attempts=2,
        anomaly_attempts=1,
        quality_attempts=1,
        quality_json_repair=True,
        max_regenerations=1,
        dataset_max_diversity_retries=1,
    )


def compact_plan_for_llm(plan_payload: dict[str, Any]) -> dict[str, Any]:
    metadata = plan_payload.get("metadata", {}) if isinstance(plan_payload.get("metadata"), dict) else {}
    compact_metadata = {
        key: metadata[key]
        for key in (
            "description",
            "workflow",
            "selected_tool",
            "domain_selection_reason",
            "domain_knowledge",
            "specialist_summary",
            "reference_strength",
            "reference_source",
        )
        if key in metadata
    }
    return {
        "domain": plan_payload.get("domain"),
        "unit": plan_payload.get("unit"),
        "generator_type": plan_payload.get("generator_type"),
        "baseline": plan_payload.get("baseline"),
        "trend_slope": plan_payload.get("trend_slope"),
        "daily_amplitude": plan_payload.get("daily_amplitude"),
        "weekly_enabled": plan_payload.get("weekly_enabled"),
        "weekly_amplitude": plan_payload.get("weekly_amplitude"),
        "seasonal_amplitude": plan_payload.get("seasonal_amplitude"),
        "heat_effect": plan_payload.get("heat_effect"),
        "noise_sigma": plan_payload.get("noise_sigma"),
        "anomaly_count": plan_payload.get("anomaly_count"),
        "anomaly_magnitude": plan_payload.get("anomaly_magnitude"),
        "anomaly_width": plan_payload.get("anomaly_width"),
        "anomaly_kind": plan_payload.get("anomaly_kind"),
        "lower_bound": plan_payload.get("lower_bound"),
        "domain_params": plan_payload.get("domain_params", {}),
        "semantic_type": plan_payload.get("semantic_type"),
        "semantic_config": plan_payload.get("semantic_config", {}),
        "output_constraints": plan_payload.get("output_constraints", {}),
        "variables": plan_payload.get("variables", []),
        "relationships": plan_payload.get("relationships", []),
        "anomaly_enabled": plan_payload.get("anomaly_enabled"),
        "anomaly_severity": plan_payload.get("anomaly_severity"),
        "anomaly_target": plan_payload.get("anomaly_target"),
        "metadata": compact_metadata,
    }


def revision_plan_template(plan_payload: dict[str, Any]) -> dict[str, Any]:
    """Keep all executable SeriesPlan fields while pruning verbose metadata."""
    template = dict(plan_payload)
    template["metadata"] = compact_plan_for_llm(plan_payload).get("metadata", {})
    return template


def compact_reference_profile(reference_profile: dict[str, Any] | None) -> dict[str, Any] | None:
    if not reference_profile:
        return None
    keep = (
        "source",
        "time_column",
        "value_column",
        "sample_count",
        "valid_count",
        "missing_ratio",
        "inferred_frequency",
        "statistics",
        "dynamics",
        "sparsity",
        "periodicity",
        "variables",
        "relationships",
        "length",
        "frequency",
        "start",
        "value_summary",
        "shape_summary",
        "missing_fraction",
        "zero_fraction",
        "integer_fraction",
        "lag1_autocorrelation",
        "dominant_periods",
        "bounds",
    )
    return {key: reference_profile[key] for key in keep if key in reference_profile}


def compact_component_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    compact: dict[str, Any] = {}
    for key in ("components", "composition", "final_transform", "anomaly_target", "validator_rules"):
        if key in payload:
            compact[key] = payload[key]
    if "summary" in payload:
        compact["summary"] = payload["summary"]
    return compact


def compact_specialist_evidence(evidence: dict[str, Any] | None) -> dict[str, Any]:
    if not evidence:
        return {}
    specification = evidence.get("specification", {}) if isinstance(evidence.get("specification"), dict) else {}
    process = evidence.get("process_design", {}) if isinstance(evidence.get("process_design"), dict) else {}
    challenge = evidence.get("challenge", {}) if isinstance(evidence.get("challenge"), dict) else {}
    reference = (
        evidence.get("reference_interpretation", {})
        if isinstance(evidence.get("reference_interpretation"), dict)
        else {}
    )
    return {
        "specification": {
            key: specification[key]
            for key in (
                "observable",
                "domain",
                "unit",
                "time_basis",
                "value_support",
                "invariants",
                "conditions",
                "explicit_requirements",
                "assumptions",
            )
            if key in specification
        },
        "process_design": {
            key: process[key]
            for key in (
                "base_process",
                "temporal_dependence",
                "event_mechanism",
                "evolution_semantics",
                "scale",
                "trend",
                "seasonality",
                "noise_model",
                "relationships",
                "anomaly_intervention",
                "constraints",
                "mandatory_properties",
            )
            if key in process
        },
        "challenge": {
            key: challenge[key]
            for key in (
                "verdict",
                "contradictions",
                "missing_constraints",
                "unsupported_mechanisms",
                "required_corrections",
            )
            if key in challenge
        },
        "reference_interpretation": {
            key: reference[key]
            for key in (
                "usable_priors",
                "semantic_warnings",
                "recommended_constraints",
                "summary",
            )
            if key in reference
        },
        "errors": evidence.get("errors", {}),
    }
