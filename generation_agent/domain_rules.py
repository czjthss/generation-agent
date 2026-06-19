from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any


RULES_PATH = Path(
    os.getenv(
        "GENERATION_AGENT_RULES_PATH",
        str(Path(__file__).resolve().parents[1] / "domain_rules.json"),
    )
)


def load_rules() -> list[dict[str, Any]]:
    if not RULES_PATH.exists():
        return []
    try:
        payload = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [rule for rule in payload if isinstance(rule, dict)]


def save_rule(rule: dict[str, Any]) -> None:
    rules = load_rules()
    domain = str(rule.get("domain", "")).strip()
    if not domain:
        return
    rules = [item for item in rules if item.get("domain") != domain]
    rules.append(rule)
    RULES_PATH.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")


def commit_candidate_rule(
    plan,
    validation_report: dict[str, Any],
    series_audit: dict[str, Any],
) -> bool:
    """Persist staged strategy memory only after both validation layers pass."""
    candidate = plan.metadata.get("candidate_domain_rule")
    if not isinstance(candidate, dict):
        return False
    if not validation_report.get("passed") or series_audit.get("status") != "PASS":
        plan.metadata["candidate_rule_status"] = "rejected_or_unverified"
        return False
    save_rule(candidate)
    plan.metadata["candidate_rule_status"] = "committed"
    return True


def match_rule(description: str) -> dict[str, Any] | None:
    text = description.lower()
    best_rule = None
    best_score = 0
    for rule in load_rules():
        keywords = rule.get("keywords", [])
        score = 0
        for keyword in keywords:
            keyword_text = str(keyword).strip().lower()
            if keyword_text and keyword_text in text:
                score += len(keyword_text)
        if score > best_score:
            best_rule = rule
            best_score = score
    return best_rule


def rule_to_plan(rule: dict[str, Any], description: str):
    from .planner import SeriesPlan

    plan = SeriesPlan(
        domain=str(rule.get("domain", "custom")),
        generator_type=str(rule.get("generator_type", "cyclic_signal")),
        unit=str(rule.get("unit", "value")),
        baseline=float(rule.get("baseline", 100.0)),
        trend_slope=float(rule.get("trend_slope", 0.0)),
        daily_amplitude=float(rule.get("daily_amplitude", 10.0)),
        weekly_enabled=bool(rule.get("weekly_enabled", False)),
        weekly_amplitude=float(rule.get("weekly_amplitude", 3.0)),
        seasonal_amplitude=float(rule.get("seasonal_amplitude", 0.0)),
        heat_effect=float(rule.get("heat_effect", 0.0)),
        noise_sigma=float(rule.get("noise_sigma", 2.0)),
        anomaly_count=int(rule.get("anomaly_count", 0)),
        anomaly_magnitude=float(rule.get("anomaly_magnitude", 3.0)),
        anomaly_width=int(rule.get("anomaly_width", 1)),
        anomaly_kind=str(rule.get("anomaly_kind", "spike")),
        anomaly_enabled=bool(rule.get("anomaly_enabled", int(rule.get("anomaly_count", 0)) > 0)),
        anomaly_severity=str(rule.get("anomaly_severity", "medium")),
        anomaly_target=str(rule.get("anomaly_target", "value")),
        lower_bound=rule.get("lower_bound", 0.0),
        domain_params=rule.get("domain_params", {}),
        semantic_type=str(rule.get("semantic_type", "instantaneous")),
        semantic_config=rule.get("semantic_config", {}),
        output_constraints=rule.get("output_constraints", {}),
        variables=rule.get("variables", []),
        relationships=rule.get("relationships", []),
        metadata={
            "planner": "custom_domain_rule",
            "description": description,
            "rule_domain": rule.get("domain"),
            "rule_keywords": rule.get("keywords", []),
            "rule_rationale": rule.get("rationale", ""),
        },
    )
    if plan.lower_bound is not None:
        plan.lower_bound = float(plan.lower_bound)
    return plan


def plan_to_rule(plan, keywords: list[str], rationale: str = "") -> dict[str, Any]:
    payload = asdict(plan)
    return {
        "domain": payload["domain"],
        "generator_type": payload["generator_type"],
        "unit": payload["unit"],
        "baseline": payload["baseline"],
        "trend_slope": payload["trend_slope"],
        "daily_amplitude": payload["daily_amplitude"],
        "weekly_enabled": payload["weekly_enabled"],
        "weekly_amplitude": payload["weekly_amplitude"],
        "seasonal_amplitude": payload["seasonal_amplitude"],
        "heat_effect": payload["heat_effect"],
        "noise_sigma": payload["noise_sigma"],
        "anomaly_count": payload["anomaly_count"],
        "anomaly_magnitude": payload["anomaly_magnitude"],
        "anomaly_width": payload["anomaly_width"],
        "anomaly_kind": payload["anomaly_kind"],
        "anomaly_enabled": payload["anomaly_enabled"],
        "anomaly_severity": payload["anomaly_severity"],
        "anomaly_target": payload["anomaly_target"],
        "lower_bound": payload["lower_bound"],
        "domain_params": payload["domain_params"],
        "semantic_type": payload["semantic_type"],
        "semantic_config": payload["semantic_config"],
        "output_constraints": payload["output_constraints"],
        "variables": payload["variables"],
        "relationships": payload["relationships"],
        "keywords": keywords,
        "rationale": rationale,
    }
