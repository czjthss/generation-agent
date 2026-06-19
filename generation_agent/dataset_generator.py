from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agent import GenerationAgent
from .compact_storage import preview_points, write_series_arrow
from .dataset_scenario_agent import DatasetScenarioDesigner
from .dataset_diversity import (
    diversify_plan,
    max_similarity,
    shape_signature,
    validate_diversity_strength,
)
from .reference_profiler import ReferenceProfile, compare_to_reference
from .semantic_types import AnomalyOverrides, apply_anomaly_overrides


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
    ) -> dict[str, Any]:
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
                json.dumps(profile_payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        entries: list[dict[str, Any]] = []
        accepted_signatures: list[dict[str, Any]] = []
        preview: dict[str, Any] | None = None
        for index in range(series_count):
            scenario = scenarios[index % len(scenarios)]
            variant_index = index // len(scenarios)
            series_id = f"series_{index + 1:04d}"
            base_seed = None if seed is None else seed + index * 17
            base_plan = apply_anomaly_overrides(
                self.agent.plan(scenario.description, reference_profile=profile_payload),
                anomaly_overrides,
            )
            best: tuple[Any, Any, dict[str, Any], dict[str, Any], int, int | None] | None = None
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
                    freq=freq,
                    start=start,
                    seed=attempt_seed,
                )
                final_plan_payload = frame.attrs.get("final_plan")
                if isinstance(final_plan_payload, dict):
                    from .planner import SeriesPlan

                    candidate_plan = SeriesPlan.from_dict(final_plan_payload)
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
                best = (frame, candidate_plan, signature, diversity_report, attempt + 1, attempt_seed)
                if is_unique:
                    break
            assert best is not None
            frame, plan, signature, diversity_report, attempts_used, series_seed = best
            accepted_signatures.append({"series_id": series_id, "signature": signature})
            if preview is None:
                preview = preview_points(frame, max_points=preview_max_points)
                preview["series_id"] = series_id
                preview["description"] = scenario.description
            filename = f"{series_id}.arrow"
            storage = write_series_arrow(
                frame,
                root / filename,
                metadata={
                    "series_id": series_id,
                    "description": scenario.description,
                    "scenario_index": index % len(scenarios),
                    "variant_index": variant_index,
                    "start": start,
                    "frequency": freq,
                    "length": length,
                    "unit": plan.unit,
                    "domain": plan.domain,
                    "generator_type": plan.generator_type,
                    "semantic_type": plan.semantic_type,
                    "plan": plan.to_dict(),
                    "component_workflow": frame.attrs.get("component_workflow", {}),
                    "component_report": frame.attrs.get("component_report", {}),
                },
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
                    "temporal_context": scenario.temporal_context,
                    "semantic_hint": scenario.semantic_hint,
                    "diversity_axis": scenario.diversity_axis,
                    "file": filename,
                    "storage": storage,
                    "seed": series_seed,
                    "diversity": diversity_report,
                    "diversity_retries": attempts_used - 1,
                    "plan": plan.to_dict(),
                    "validation": frame.attrs.get("validation_report", {}),
                    "series_audit": frame.attrs.get("series_audit", {}),
                    "component_workflow": frame.attrs.get("component_workflow", {}),
                    "component_report": frame.attrs.get("component_report", {}),
                    "component_stats": frame.attrs.get("component_stats", {}),
                    "reference_similarity": similarity,
                }
            )

        manifest = {
            "mode": "dataset",
            "storage_format": "arrow_ipc",
            "domain": domain,
            "series_count": len(entries),
            "scenario_count": len(scenarios),
            "length_per_series": length,
            "frequency": freq,
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
            "timestamp_policy": "start/frequency/length stored in manifest and Arrow metadata; no per-row timestamp column",
            "row_metadata_policy": "unit/domain/generator_type/semantic_type stored in manifest and Arrow metadata; not repeated per row",
            "value_dtype": "float32",
            "reference_strength": "structure" if profile_payload else None,
            "reference_source": profile_payload.get("source") if profile_payload else None,
            "dataset_file": None,
            "scenarios_file": "scenarios.json",
            "preview": preview or {"points": [], "count": 0},
            "series": entries,
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest
