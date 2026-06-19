import json

import pyarrow.ipc as ipc

from generation_agent.agent import GenerationAgent
from generation_agent.dataset_generator import DatasetGenerator
from generation_agent.dataset_scenario_agent import DatasetScenario
from generation_agent.planner import heuristic_plan


def test_test_scenarios_are_unique():
    scenarios = [
        DatasetScenario(description=f"生成气象测试场景{i + 1}", observable="value", unit="value")
        for i in range(60)
    ]
    assert len(scenarios) == 60
    assert len({item.description for item in scenarios}) == 60


def test_dataset_generator_writes_compact_arrow_series_and_manifest(tmp_path):
    agent = GenerationAgent(model=None)
    generator = DatasetGenerator(agent)
    generator.scenario_designer.design = lambda _domain, count: [
        DatasetScenario(description=f"生成气象场景{i + 1}", observable="value", unit="value")
        for i in range(count)
    ]
    agent.plan = lambda description, reference_profile=None: heuristic_plan(description)
    manifest = generator.generate_to_directory(
        "气象", tmp_path, series_count=3, length=24, seed=10
    )
    assert manifest["series_count"] == 3
    assert manifest["scenario_count"] == 3
    assert manifest["dataset_file"] is None
    assert manifest["storage_format"] == "arrow_ipc"
    assert manifest["diversity"]["strength"] == "medium"
    assert manifest["diversity"]["check_enabled"] is True
    assert not (tmp_path / "dataset.csv").exists()
    assert (tmp_path / "scenarios.json").exists()
    assert (tmp_path / "manifest.json").exists()
    series_file = tmp_path / "series_0001.arrow"
    assert series_file.exists()
    with ipc.open_file(series_file) as reader:
        table = reader.read_all()
    assert table.num_rows == 24
    assert "timestamp" not in table.column_names
    assert "domain" not in table.column_names
    assert table.schema.field("value").type.bit_width == 32
    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert len(saved["series"]) == 3
    assert saved["preview"]["points"]
    assert saved["series"][0]["diversity"]["shape_summary"]
    assert "max_similarity" in saved["series"][0]["diversity"]
    assert saved["series"][0]["component_workflow"]["components"]
    assert saved["series"][0]["component_report"]["components"]
    assert "component_feature_planning_agent" in saved["series"][0]["component_workflow"]["agent_trace"]


def test_dataset_generator_can_use_small_scenario_pool_with_variants(tmp_path):
    agent = GenerationAgent(model=None)
    generator = DatasetGenerator(agent)
    generator.scenario_designer.design = lambda _domain, count: [
        DatasetScenario(description=f"生成电力负载场景{i + 1}", observable="load", unit="kW")
        for i in range(count)
    ]
    agent.plan = lambda description, reference_profile=None: heuristic_plan(description)
    manifest = generator.generate_to_directory(
        "电力负载",
        tmp_path,
        series_count=6,
        scenario_count=2,
        length=72,
        seed=30,
        diversity_strength="high",
    )
    assert manifest["series_count"] == 6
    assert manifest["scenario_count"] == 2
    assert {item["scenario_index"] for item in manifest["series"]} == {0, 1}
    assert max(item["variant_index"] for item in manifest["series"]) >= 2
    assert all(item["file"].endswith(".arrow") for item in manifest["series"])
