from generation_agent.planning_prompts import (
    DOMAIN_CHALLENGER_PROMPT,
    DATASET_SCENARIO_PROMPT,
    DEMAND_UNDERSTANDING_AGENT_PROMPT,
    MECHANISM_PLANNING_AGENT_PROMPT,
    PLAN_COMPILER_SYSTEM_PROMPT,
    PROCESS_ARCHITECT_PROMPT,
    REFERENCE_INTERPRETER_PROMPT,
    SPECIFICATION_AGENT_PROMPT,
    REFLECTION_SYSTEM_PROMPT,
    INPUT_PROFILER_PROMPT,
    SCENARIO_VARIABLE_AGENT_PROMPT,
    COMPONENT_MECHANISM_AGENT_PROMPT,
    COMPONENT_FEATURE_PLANNER_PROMPT,
    QUALITY_EVALUATOR_PROMPT,
    ANOMALY_STRATEGY_PROMPT,
    build_reflection_input,
    build_revision_input,
)


def test_reflection_prompt_has_explicit_contract():
    assert "independent Reflection role" in REFLECTION_SYSTEM_PROMPT
    assert '\"status\":\"PASS or REVISE\"' in REFLECTION_SYSTEM_PROMPT
    assert "hard_errors" in REFLECTION_SYSTEM_PROMPT
    assert "soft_warnings" in REFLECTION_SYSTEM_PROMPT
    assert "must not block generation" in REFLECTION_SYSTEM_PROMPT
    assert "pipeline_capabilities" in REFLECTION_SYSTEM_PROMPT
    assert "relevant stage" in REFLECTION_SYSTEM_PROMPT
    assert "hard_error_evidence" in REFLECTION_SYSTEM_PROMPT
    assert "capability_path" in REFLECTION_SYSTEM_PROMPT
    assert "request_quote" in REFLECTION_SYSTEM_PROMPT


def test_reflection_input_documents_runtime_pipeline_capabilities():
    result = build_reflection_input("generate electric load", {"generator_type": "cyclic_signal"})
    assert "pipeline_capabilities" in result
    assert "deterministic_capability_validation" in result
    assert "runtime_code_introspection" in result
    assert "component_feature_families" in result


def test_revision_input_preserves_request_plan_and_feedback():
    result = build_revision_input(
        "generate cumulative sales",
        {"semantic_type": "instantaneous"},
        ["wrong evolution law"],
        "select semantics from the observable definition",
        evidence={"reference_interpretation": {"summary": "scale prior"}},
    )
    assert "generate cumulative sales" in result
    assert '\"semantic_type\": \"instantaneous\"' in result
    assert "wrong evolution law" in result
    assert "reference_interpretation" in result


def test_compact_planning_agents_have_distinct_responsibilities():
    assert "Requirement Understanding Agent" in DEMAND_UNDERSTANDING_AGENT_PROMPT
    assert "reference-series profile" in DEMAND_UNDERSTANDING_AGENT_PROMPT
    assert "Mechanism Planning Agent" in MECHANISM_PLANNING_AGENT_PROMPT
    assert "decompose each target variable" in MECHANISM_PLANNING_AGENT_PROMPT
    assert "Parameter Compilation Agent" in PLAN_COMPILER_SYSTEM_PROMPT
    assert "executable SeriesPlan" in PLAN_COMPILER_SYSTEM_PROMPT
    assert "diverse, concrete" in DATASET_SCENARIO_PROMPT
    assert "exactly requested_count" in DATASET_SCENARIO_PROMPT
    assert "untrusted data" in PLAN_COMPILER_SYSTEM_PROMPT


def test_legacy_specialist_prompts_remain_available_for_reference():
    assert "measurement contract" in SPECIFICATION_AGENT_PROMPT
    assert "coherent stochastic process" in PROCESS_ARCHITECT_PROMPT
    assert "Try to falsify" in DOMAIN_CHALLENGER_PROMPT
    assert "deterministic statistics" in REFERENCE_INTERPRETER_PROMPT


def test_component_workflow_prompts_have_clear_boundaries():
    assert "Input Profiling Agent" in INPUT_PROFILER_PROMPT
    assert "Scenario and Variable Analysis Agent" in SCENARIO_VARIABLE_AGENT_PROMPT
    assert "variable-level semantics" in SCENARIO_VARIABLE_AGENT_PROMPT
    assert "Component Mechanism Modeling Agent" in COMPONENT_MECHANISM_AGENT_PROMPT
    assert "component-level semantic" in COMPONENT_MECHANISM_AGENT_PROMPT
    assert "Component Feature Planning Agent" in COMPONENT_FEATURE_PLANNER_PROMPT
    assert "feature_family" in COMPONENT_FEATURE_PLANNER_PROMPT
    assert "Quality Evaluation Agent" in QUALITY_EVALUATOR_PROMPT
    assert "component-level quality" in QUALITY_EVALUATOR_PROMPT
    assert "Anomaly Strategy Agent" in ANOMALY_STRATEGY_PROMPT
    assert "Never return data values" in ANOMALY_STRATEGY_PROMPT
    assert "never output, identify, or directly edit individual data points" in QUALITY_EVALUATOR_PROMPT
    assert "hard_failure_evidence" in QUALITY_EVALUATOR_PROMPT
    assert "LLM specialist" in QUALITY_EVALUATOR_PROMPT
