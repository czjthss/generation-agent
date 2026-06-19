from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
import re
from typing import Any

from .env import load_project_env
from .planner import DEFAULT_LLM_MODEL, DEFAULT_OPENAI_BASE_URL
from .planning_prompts import DATASET_SCENARIO_PROMPT


load_project_env()


@dataclass
class DatasetScenario:
    description: str
    observable: str = ""
    unit: str = "unknown"
    suggested_frequency: str = "unknown"
    temporal_context: str = ""
    semantic_hint: str = "instantaneous"
    diversity_axis: str = ""
    rationale: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_description(value: str) -> str:
    return re.sub(r"[\s，,。.;；:：]+", "", value).lower()


def _parse_scenarios(content: str) -> list[DatasetScenario]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            return []
        try:
            payload = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return []
    raw = payload.get("scenarios", []) if isinstance(payload, dict) else []
    result: list[DatasetScenario] = []
    for item in raw:
        required = (
            "description",
            "observable",
            "unit",
            "suggested_frequency",
            "temporal_context",
            "semantic_hint",
            "diversity_axis",
            "rationale",
        )
        if not isinstance(item, dict) or any(
            not str(item.get(name, "")).strip() for name in required
        ):
            continue
        tags = item.get("tags", [])
        result.append(
            DatasetScenario(
                description=str(item["description"]).strip(),
                observable=str(item["observable"]).strip(),
                unit=str(item["unit"]).strip(),
                suggested_frequency=str(item["suggested_frequency"]).strip(),
                temporal_context=str(item["temporal_context"]).strip(),
                semantic_hint=str(item["semantic_hint"]).strip(),
                diversity_axis=str(item["diversity_axis"]).strip(),
                rationale=str(item.get("rationale", "")).strip(),
                tags=[str(tag) for tag in tags] if isinstance(tags, list) else [],
            )
        )
    return result


class DatasetScenarioDesigner:
    def __init__(self, model: str | None = DEFAULT_LLM_MODEL) -> None:
        self.model = model

    def design(self, domain: str, count: int) -> list[DatasetScenario]:
        if count <= 0:
            raise ValueError("series_count must be positive")
        if not self.model or not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY and model are required for dataset scenario Agent")

        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=self.model,
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
                temperature=0.2,
                max_completion_tokens=3000,
                timeout=90,
                max_retries=1,
            )
            scenarios: list[DatasetScenario] = []
            seen: set[str] = set()
            while len(scenarios) < count:
                batch_size = min(25, count - len(scenarios))
                payload = {
                    "domain": domain,
                    "requested_count": batch_size,
                    "existing_descriptions": [item.description for item in scenarios],
                    "required_coverage_axes": [
                        "observable",
                        "location_or_context",
                        "season_or_time_basis",
                        "scale",
                        "temporal_behavior",
                        "evolution_semantics",
                    ],
                }
                messages = [
                    SystemMessage(content=DATASET_SCENARIO_PROMPT),
                    HumanMessage(content=json.dumps(payload, ensure_ascii=False, indent=2)),
                ]
                response = llm.invoke(messages)
                parsed = _parse_scenarios(str(getattr(response, "content", "")))
                if len(parsed) != batch_size:
                    messages.extend(
                        [
                            response,
                            HumanMessage(
                                content=(
                                    f"Contract violation: expected exactly {batch_size} complete, "
                                    f"distinct scenarios but parsed {len(parsed)}. Return corrected JSON only."
                                )
                            ),
                        ]
                    )
                    response = llm.invoke(messages)
                    parsed = _parse_scenarios(str(getattr(response, "content", "")))
                added = 0
                for scenario in parsed:
                    key = _normalize_description(scenario.description)
                    if key and key not in seen:
                        scenarios.append(scenario)
                        seen.add(key)
                        added += 1
                    if len(scenarios) >= count:
                        break
                if added == 0:
                    break
            if len(scenarios) < count:
                raise RuntimeError(f"Dataset Scenario Agent produced only {len(scenarios)} of {count} scenarios")
            return scenarios[:count]
        except Exception as exc:
            raise RuntimeError(f"Dataset Scenario Agent failed: {exc}") from exc
