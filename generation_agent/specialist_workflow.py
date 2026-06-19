from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any

from .planning_prompts import (
    DOMAIN_CHALLENGER_PROMPT,
    PROCESS_ARCHITECT_PROMPT,
    REFERENCE_INTERPRETER_PROMPT,
    SPECIFICATION_AGENT_PROMPT,
)


FEATURE_CAPABILITY_SUMMARY = {
    "generator_types": [
        "cyclic_signal",
        "intermittent_event",
        "daylight_envelope",
        "smooth_environmental",
        "count_process",
        "bounded_utilization",
    ],
    "semantic_types": [
        "instantaneous",
        "cumulative",
        "stock_flow",
        "regime_switching",
        "random_walk",
        "decay_recovery",
        "saturation_growth",
        "multivariate_lag",
    ],
}

SPECIFICATION_CONTRACT = {
    "observable": str,
    "domain": str,
    "unit": str,
    "time_basis": str,
    "value_support": str,
    "invariants": list,
    "conditions": list,
    "explicit_requirements": list,
    "assumptions": list,
    "ambiguities": list,
    "requires_user_input": bool,
    "confidence": (int, float),
}
PROCESS_CONTRACT = {
    "base_process": str,
    "temporal_dependence": dict,
    "event_mechanism": dict,
    "evolution_semantics": dict,
    "scale": dict,
    "trend": dict,
    "seasonality": list,
    "noise_model": dict,
    "relationships": list,
    "anomaly_intervention": dict,
    "constraints": dict,
    "mandatory_properties": list,
    "optional_properties": list,
    "implementation_risks": list,
    "assumptions": list,
    "confidence": (int, float),
}
CHALLENGE_CONTRACT = {
    "verdict": str,
    "contradictions": list,
    "unrealistic_assumptions": list,
    "missing_constraints": list,
    "unsupported_mechanisms": list,
    "reference_conflicts": list,
    "required_corrections": list,
    "confidence": (int, float),
}
REFERENCE_CONTRACT = {
    "usable_priors": dict,
    "ignored_features": list,
    "semantic_warnings": list,
    "mismatch_checks": dict,
    "recommended_constraints": dict,
    "summary": str,
    "confidence": (int, float),
}


def _extract_json(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("specialist did not return a JSON object")
        payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("specialist JSON must be an object")
    return payload


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return content if isinstance(content, str) else str(content)


class JsonSpecialist:
    def __init__(self, llm: Any, system_prompt: str, contract: dict[str, Any]) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.contract = contract

    def _validate(self, payload: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        for name, expected_type in self.contract.items():
            if name not in payload:
                errors.append(f"missing required field: {name}")
            elif not isinstance(payload[name], expected_type):
                errors.append(f"field {name} has wrong type")
        confidence = payload.get("confidence")
        if isinstance(confidence, (int, float)) and not 0.0 <= float(confidence) <= 1.0:
            errors.append("confidence must be between 0 and 1")
        if "verdict" in payload and payload["verdict"] not in {"ACCEPT", "REVISE"}:
            errors.append("verdict must be ACCEPT or REVISE")
        for name in ("observable", "domain", "unit", "time_basis", "value_support", "base_process"):
            if name in payload and not str(payload[name]).strip():
                errors.append(f"field {name} must not be empty")
        return errors

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False, indent=2)),
        ]
        errors: list[str] = []
        for attempt in range(2):
            response = self.llm.invoke(messages)
            try:
                result = _extract_json(_message_text(response))
                errors = self._validate(result)
            except (ValueError, json.JSONDecodeError) as exc:
                result = {}
                errors = [f"invalid JSON object: {exc}"]
            if not errors:
                return result
            if attempt == 0:
                messages.extend(
                    [
                        response,
                        HumanMessage(
                            content=(
                                "Your JSON violated the required contract: "
                                + json.dumps(errors, ensure_ascii=False)
                                + ". Return a complete corrected JSON object only."
                            )
                        ),
                    ]
                )
        raise ValueError("specialist contract validation failed: " + "; ".join(errors))


class SpecificationAgent(JsonSpecialist):
    def __init__(self, llm: Any) -> None:
        super().__init__(llm, SPECIFICATION_AGENT_PROMPT, SPECIFICATION_CONTRACT)


class ProcessArchitectAgent(JsonSpecialist):
    def __init__(self, llm: Any) -> None:
        super().__init__(llm, PROCESS_ARCHITECT_PROMPT, PROCESS_CONTRACT)


class DomainChallengerAgent(JsonSpecialist):
    def __init__(self, llm: Any) -> None:
        super().__init__(llm, DOMAIN_CHALLENGER_PROMPT, CHALLENGE_CONTRACT)


class ReferenceInterpreterAgent(JsonSpecialist):
    def __init__(self, llm: Any) -> None:
        super().__init__(llm, REFERENCE_INTERPRETER_PROMPT, REFERENCE_CONTRACT)


@dataclass
class SpecialistEvidence:
    specification: dict[str, Any] = field(default_factory=dict)
    process_design: dict[str, Any] = field(default_factory=dict)
    challenge: dict[str, Any] = field(default_factory=dict)
    reference_interpretation: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GenerationSpecialistWorkflow:
    """Build a measurement contract, process design, and adversarial domain review."""

    def __init__(self, llm: Any) -> None:
        self.specification_agent = SpecificationAgent(llm)
        self.process_architect = ProcessArchitectAgent(llm)
        self.domain_challenger = DomainChallengerAgent(llm)
        self.reference_interpreter = ReferenceInterpreterAgent(llm)

    def analyze(
        self,
        description: str,
        reference_context: dict[str, Any] | None = None,
    ) -> SpecialistEvidence:
        evidence = SpecialistEvidence()
        if reference_context:
            try:
                evidence.reference_interpretation = self.reference_interpreter.run(
                    {"request": description, **reference_context}
                )
            except Exception as exc:
                evidence.errors["reference_interpretation"] = f"{exc.__class__.__name__}: {exc}"
        try:
            evidence.specification = self.specification_agent.run(
                {
                    "request": description,
                    "reference_interpretation": evidence.reference_interpretation,
                }
            )
        except Exception as exc:
            evidence.errors["specification"] = f"{exc.__class__.__name__}: {exc}"

        try:
            evidence.process_design = self.process_architect.run(
                {
                    "request": description,
                    "specification": evidence.specification,
                    "reference_interpretation": evidence.reference_interpretation,
                }
            )
        except Exception as exc:
            evidence.errors["process_design"] = f"{exc.__class__.__name__}: {exc}"

        try:
            evidence.challenge = self.domain_challenger.run(
                {
                    "request": description,
                    "specification": evidence.specification,
                    "process_design": evidence.process_design,
                    "reference_interpretation": evidence.reference_interpretation,
                    "available_capabilities": FEATURE_CAPABILITY_SUMMARY,
                }
            )
        except Exception as exc:
            evidence.errors["challenge"] = f"{exc.__class__.__name__}: {exc}"
        return evidence
