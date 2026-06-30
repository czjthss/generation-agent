import json

from generation_agent.capability_registry import (
    build_pipeline_capability_manifest,
    validate_plan_against_capabilities,
)
from generation_agent.feature_composer import GENERATOR_TYPES
from generation_agent.semantic_types import SEMANTIC_TYPES


def test_manifest_is_built_from_installed_pipeline_code():
    manifest = build_pipeline_capability_manifest()

    assert manifest["manifest_source"] == "runtime_code_introspection"
    assert set(manifest["feature_generation"]["generator_types"]) == GENERATOR_TYPES
    assert set(manifest["semantic_transforms"]["semantic_types"]) == SEMANTIC_TYPES
    assert "finalize_feature_plan" in manifest["plan_compilation"]["tools"]
    assert "heat_effect" in manifest["plan_compilation"]["accepted_plan_fields"]
    assert "working_day_shift" in manifest["feature_generation"][
        "component_feature_families"
    ]
    assert "shift_start_hour" in manifest["feature_generation"][
        "component_parameter_keys"
    ]
    heat_contracts = [
        item
        for item in manifest["feature_generation"]["component_execution_contracts"]
        if item["feature_family"] == "heat_index_effect"
    ]
    assert any(
        "plan.heat_effect" in item["parameter_bindings"].get("amplitude", "")
        for item in heat_contracts
    )
    assert "every implemented kind" in manifest["anomaly_injection"][
        "execution_contract"
    ]["windowing"]
    assert manifest["anomaly_injection"]["targets_by_semantic"]["instantaneous"] == [
        "value"
    ]
    assert manifest["anomaly_injection"]["kind_aliases"]["temporary_outage"] == "drop"


def test_manifest_covers_every_execution_stage_without_domain_instructions():
    manifest = build_pipeline_capability_manifest()
    serialized = json.dumps(manifest, ensure_ascii=False).lower()

    assert manifest["execution_order"] == [
        "upstream_analysis",
        "plan_compilation",
        "reflection",
        "component_generation",
        "semantic_transform",
        "anomaly_injection",
        "deterministic_validation",
        "quality_evaluation",
    ]
    assert "industrial load uses" not in serialized
    assert "rainfall" not in serialized


def test_deterministic_capability_validator_accepts_valid_anomaly_window():
    result = validate_plan_against_capabilities(
        {
            "generator_type": "cyclic_signal",
            "semantic_type": "instantaneous",
            "semantic_config": {},
            "domain_params": {},
            "output_constraints": {"nonnegative": True},
            "anomaly_kind": "spike",
            "anomaly_count": 6,
            "anomaly_width": 2,
            "anomaly_magnitude": 0.25,
        }
    )

    assert result["passed"] is True
    assert result["invalid_paths"] == []


def test_deterministic_capability_validator_reports_exact_invalid_paths():
    result = validate_plan_against_capabilities(
        {
            "generator_type": "cyclic_signal",
            "semantic_type": "regime_switching",
            "semantic_config": {"regimes": ["day", "night"]},
            "domain_params": {},
            "output_constraints": {},
            "anomaly_kind": "temporary_outage",
            "anomaly_count": 1,
            "anomaly_width": 1,
            "anomaly_magnitude": 1.0,
        }
    )

    assert result["passed"] is False
    assert result["invalid_paths"] == ["semantic_config.regimes"]
