from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _keep(payload: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: payload[key] for key in keys if key in payload}


def _limit(items: Any, limit: int) -> list[Any]:
    if not isinstance(items, list):
        return []
    return items[: max(0, limit)]


def compact_reference_profile_for_agent(
    reference_profile: dict[str, Any] | None,
    max_variables: int = 16,
    max_relationships: int = 24,
) -> dict[str, Any] | None:
    if not reference_profile:
        return None
    compact = _keep(
        reference_profile,
        (
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
            "bounds",
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
            "distribution",
            "nonstationarity",
            "event_clusters",
            "covariance",
        ),
    )
    compact["variables"] = [
        _keep(
            _as_dict(variable),
            (
                "name",
                "unit",
                "role",
                "semantic_type",
                "missing_ratio",
                "zero_fraction",
                "integer_fraction",
                "statistics",
                "dynamics",
                "sparsity",
                "periodicity",
                "bounds",
            ),
        )
        for variable in _limit(reference_profile.get("variables"), max_variables)
    ]
    compact["relationships"] = [
        _keep(
            _as_dict(relation),
            (
                "source",
                "target",
                "lag",
                "correlation",
                "direction",
                "strength",
                "relationship_type",
                "evidence",
            ),
        )
        for relation in _limit(reference_profile.get("relationships"), max_relationships)
    ]
    return compact


def compact_capability_manifest_for_agent(manifest: dict[str, Any]) -> dict[str, Any]:
    feature_generation = _as_dict(manifest.get("feature_generation"))
    semantic_transforms = _as_dict(manifest.get("semantic_transforms"))
    anomaly = _as_dict(manifest.get("anomaly_injection"))
    validation = _as_dict(manifest.get("deterministic_validation"))
    return {
        "manifest_source": manifest.get("manifest_source"),
        "planning_contract": _keep(
            _as_dict(manifest.get("plan_compilation")),
            (
                "accepted_top_level_fields",
                "generator_types",
                "semantic_types",
                "domain_param_keys",
                "output_constraint_keys",
                "variable_fields",
                "relationship_fields",
            ),
        ),
        "feature_generation": _keep(
            feature_generation,
            (
                "generator_types",
                "multivariate_generation",
                "component_feature_families",
                "component_parameter_keys",
            ),
        ),
        "semantic_transforms": _keep(
            semantic_transforms,
            (
                "semantic_types",
                "semantic_config_keys",
                "output_constraint_keys",
                "value_role_mapping",
            ),
        ),
        "anomaly_injection": _keep(
            anomaly,
            (
                "accepted_fields",
                "anomaly_kinds",
                "anomaly_targets",
                "severity_levels",
                "execution_contract",
            ),
        ),
        "deterministic_validation": _keep(
            validation,
            (
                "semantic_checks",
                "component_checks",
                "multivariate_checks",
                "repair_policy",
                "hard_warning_policy",
            ),
        ),
        "execution_order": manifest.get("execution_order"),
    }


def compact_component_workflow_for_agent(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    compact = _keep(
        payload,
        ("composition", "final_transform", "anomaly_target", "validator_rules", "summary"),
    )
    components: list[dict[str, Any]] = []
    for component in _limit(payload.get("components"), 32):
        component_payload = _as_dict(component)
        components.append(
            _keep(
                component_payload,
                (
                    "name",
                    "role",
                    "component_semantic",
                    "value_role",
                    "sign_constraint",
                    "time_scale_behavior",
                    "statistical_shape",
                    "feature_family",
                    "params",
                    "constraints",
                    "variable",
                    "relationship_role",
                    "target_variable",
                ),
            )
        )
    if components:
        compact["components"] = components
    return compact


def compact_component_report_for_agent(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    compact = _keep(payload, ("status", "checks", "warnings", "hard_warnings", "summary"))
    reports: list[dict[str, Any]] = []
    for report in _limit(payload.get("components"), 32):
        report_payload = _as_dict(report)
        reports.append(
            _keep(
                report_payload,
                ("name", "feature_family", "status", "checks", "warnings", "stats", "repairs"),
            )
        )
    if reports:
        compact["components"] = reports
    return compact


def compact_series_summary_for_quality(summary: dict[str, Any] | None, sample_size: int = 8) -> dict[str, Any]:
    if not summary:
        return {}
    compact = _keep(
        summary,
        (
            "rows",
            "columns",
            "value",
            "anomaly",
            "variables",
            "multivariate_report",
            "start",
            "frequency",
            "length",
        ),
    )
    samples = summary.get("samples")
    if isinstance(samples, list) and samples:
        if len(samples) <= sample_size * 3:
            compact["samples"] = samples
        else:
            mid = len(samples) // 2
            compact["samples"] = (
                samples[:sample_size]
                + samples[max(0, mid - sample_size // 2) : mid + sample_size // 2]
                + samples[-sample_size:]
            )
    report = _as_dict(compact.get("multivariate_report"))
    audits = report.get("relationship_audits")
    if isinstance(audits, list):
        report["relationship_audits"] = audits[:24]
        compact["multivariate_report"] = report
    return compact


def compact_plan_for_metadata(plan_payload: dict[str, Any]) -> dict[str, Any]:
    metadata = _as_dict(plan_payload.get("metadata"))
    return {
        **_keep(
            plan_payload,
            (
                "domain",
                "unit",
                "generator_type",
                "baseline",
                "trend_slope",
                "daily_amplitude",
                "weekly_enabled",
                "weekly_amplitude",
                "seasonal_amplitude",
                "heat_effect",
                "noise_sigma",
                "anomaly_enabled",
                "anomaly_count",
                "anomaly_severity",
                "anomaly_target",
                "anomaly_kind",
                "anomaly_magnitude",
                "anomaly_width",
                "lower_bound",
                "upper_bound",
                "domain_params",
                "semantic_type",
                "semantic_config",
                "output_constraints",
                "variables",
                "relationships",
            ),
        ),
        "metadata": _keep(
            metadata,
            (
                "description",
                "workflow",
                "selected_tool",
                "model",
                "base_url",
                "cost_mode",
                "reference_strength",
                "reference_source",
                "domain_knowledge",
                "specialist_summary",
            ),
        ),
    }


def compact_validation_for_metadata(validation: dict[str, Any] | None) -> dict[str, Any]:
    if not validation:
        return {}
    return _keep(
        validation,
        (
            "passed",
            "raw_passed",
            "repaired_passed",
            "status",
            "semantic_type",
            "checks",
            "raw_checks",
            "repairs",
            "repairs_applied",
            "repair_warnings",
            "critical_repairs",
            "warnings",
            "hard_warnings",
            "component_warnings",
            "relationship_warnings",
        ),
    )


def compact_quality_for_metadata(quality: dict[str, Any] | None) -> dict[str, Any]:
    if not quality:
        return {}
    return _keep(
        quality,
        (
            "status",
            "verdict",
            "passed",
            "failures",
            "warnings",
            "issues",
            "confidence",
            "needs_regeneration",
            "revision",
            "regeneration_count",
        ),
    )


def build_generation_trace(plan_payload: dict[str, Any], attrs: dict[str, Any]) -> dict[str, Any]:
    metadata = _as_dict(plan_payload.get("metadata"))
    return {
        "plan": plan_payload,
        "requirement_understanding": metadata.get("requirement_understanding", {}),
        "mechanism_planning": metadata.get("mechanism_planning", {}),
        "parameter_compilation": metadata.get("parameter_compilation", {}),
        "reflection": metadata.get("reflection", {}),
        "anomaly_strategy": metadata.get("anomaly_strategy", {}),
        "component_workflow": attrs.get("component_workflow", {}),
        "component_report": attrs.get("component_report", {}),
        "component_stats": attrs.get("component_stats", {}),
        "deterministic_validation": attrs.get("validation_report", {}),
        "quality_evaluation": attrs.get("series_audit", {}),
        "multivariate_report": attrs.get("multivariate_report", {}),
        "workflow_evidence": metadata.get("workflow_evidence", {}),
    }


def write_json_gz(path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    target = Path(path)
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with gzip.open(target, "wb") as handle:
        handle.write(raw)
    return {
        "path": str(target),
        "bytes": target.stat().st_size,
        "uncompressed_bytes": len(raw),
    }
