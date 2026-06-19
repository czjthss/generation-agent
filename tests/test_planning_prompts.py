from generation_agent.planning_prompts import (
    DOMAIN_CHALLENGER_PROMPT,
    DATASET_SCENARIO_PROMPT,
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
    build_revision_input,
)


def test_reflection_prompt_has_explicit_contract():
    assert "independent Reflection role" in REFLECTION_SYSTEM_PROMPT
    assert '\"status\":\"PASS or REVISE\"' in REFLECTION_SYSTEM_PROMPT
    assert "evidence is missing" in REFLECTION_SYSTEM_PROMPT


def test_revision_input_preserves_request_plan_and_feedback():
    result = build_revision_input(
        "生成累计销售额",
        {"semantic_type": "instantaneous"},
        ["wrong evolution law"],
        "select semantics from the observable definition",
        evidence={"reference_interpretation": {"summary": "scale prior"}},
    )
    assert "生成累计销售额" in result
    assert '\"semantic_type\": \"instantaneous\"' in result
    assert "wrong evolution law" in result
    assert "reference_interpretation" in result


def test_generation_specialists_have_distinct_responsibilities():
    assert "measurement contract" in SPECIFICATION_AGENT_PROMPT
    assert "coherent stochastic process" in PROCESS_ARCHITECT_PROMPT
    assert "Try to falsify" in DOMAIN_CHALLENGER_PROMPT
    assert "deterministic statistics" in REFERENCE_INTERPRETER_PROMPT
    assert "diverse, concrete" in DATASET_SCENARIO_PROMPT
    assert "exactly requested_count" in DATASET_SCENARIO_PROMPT
    assert "untrusted data" in PLAN_COMPILER_SYSTEM_PROMPT


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
