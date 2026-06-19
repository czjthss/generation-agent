from generation_agent.agent import GenerationAgent
from generation_agent.component_workflow import build_component_workflow
from generation_agent.planner import heuristic_plan


def test_component_workflow_decomposes_electric_load():
    plan = heuristic_plan("生成中国南方夏季工业园区的电力负载")
    workflow = build_component_workflow(
        "生成中国南方夏季工业园区的电力负载",
        plan,
        length=48,
        freq="h",
        start="2026-07-01 00:00:00",
    )
    payload = workflow.to_dict()
    names = {item["name"] for item in payload["components"]}
    assert payload["variable_profile"]["variable_semantic"] == plan.semantic_type
    assert "baseline_level" in names
    assert "daily_cycle" in names
    assert "component_feature_planning_agent" in payload["agent_trace"]


def test_generation_records_component_quality_report():
    agent = GenerationAgent(model=None)
    plan = heuristic_plan("生成中国南方夏季工业园区的电力负载")
    frame = agent.generate_from_plan(
        plan,
        length=48,
        freq="h",
        start="2026-07-01 00:00:00",
        seed=7,
    )
    assert "component_workflow" in frame.attrs
    assert "component_report" in frame.attrs
    assert frame.attrs["component_report"]["components"]
    assert frame.attrs["final_plan"]["metadata"]["workflow"] == "component_agent_workflow"
