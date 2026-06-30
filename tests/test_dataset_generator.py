import gzip
import json

import pandas as pd
import pyarrow.ipc as ipc

from test_helpers import ReviewedFakeAgent
from generation_agent.dataset_generator import DatasetGenerator
from generation_agent.dataset_scenario_agent import DatasetScenario
from generation_agent.planner import heuristic_plan


def test_test_scenarios_are_unique():
    scenarios = [
        DatasetScenario(description=f"generate weather test scenario {i + 1}", observable="value", unit="value")
        for i in range(60)
    ]
    assert len(scenarios) == 60
    assert len({item.description for item in scenarios}) == 60


def test_dataset_generator_writes_compact_arrow_series_and_manifest(tmp_path):
    agent = ReviewedFakeAgent()
    generator = DatasetGenerator(agent)
    generator.scenario_designer.design = lambda _domain, count: [
        DatasetScenario(description=f"generate weather scenario {i + 1}", observable="value", unit="value")
        for i in range(count)
    ]
    manifest = generator.generate_to_directory(
        "weather", tmp_path, series_count=3, length=24, seed=10
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
    assert "agent_trace" not in saved["series"][0]["component_workflow"]
    assert saved["series"][0]["plan_summary"]["domain"]
    assert saved["series"][0]["trace"] is None
    assert saved["series"][0]["variant_audit"]["status"] in {"PASS", "WARN"}


def test_dataset_generator_can_save_full_trace_when_requested(tmp_path):
    agent = ReviewedFakeAgent()
    generator = DatasetGenerator(agent)
    generator.scenario_designer.design = lambda _domain, count: [
        DatasetScenario(description=f"generate weather scenario {i + 1}", observable="value", unit="value")
        for i in range(count)
    ]
    manifest = generator.generate_to_directory(
        "weather", tmp_path, series_count=1, length=24, seed=10, save_trace=True
    )
    trace_info = manifest["series"][0]["trace"]
    assert trace_info["path"].endswith(".trace.json.gz")
    with gzip.open(trace_info["path"], "rt", encoding="utf-8") as handle:
        trace = json.load(handle)
    assert "agent_trace" in trace["component_workflow"]
    assert "parameter_compiler_agent" in trace["component_workflow"]["agent_trace"]


def test_dataset_generator_can_use_small_scenario_pool_with_variants(tmp_path):
    agent = ReviewedFakeAgent()
    generator = DatasetGenerator(agent)
    generator.scenario_designer.design = lambda _domain, count: [
        DatasetScenario(description=f"generate electric load scenario {i + 1}", observable="load", unit="kW")
        for i in range(count)
    ]
    manifest = generator.generate_to_directory(
        "electric load",
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


def test_dataset_generator_records_scenario_frequency_policy(tmp_path):
    agent = ReviewedFakeAgent()
    generator = DatasetGenerator(agent)
    generator.scenario_designer.design = lambda _domain, count: [
        DatasetScenario(
            description=f"generate daily sales scenario {i + 1}",
            observable="sales",
            unit="USD",
            suggested_frequency="D",
        )
        for i in range(count)
    ]
    manifest = generator.generate_to_directory(
        "sales",
        tmp_path,
        series_count=1,
        length=24,
        freq="h",
        seed=11,
    )
    entry = manifest["series"][0]
    assert entry["suggested_frequency"] == "D"
    assert entry["normalized_suggested_frequency"] == "D"
    assert entry["effective_frequency"] == "h"
    assert entry["frequency_warning"]["type"] == "scenario_frequency_ignored"
    assert manifest["frequency_policy"]["conflict_count"] == 1


def test_dataset_generator_can_respect_normalized_scenario_frequency(tmp_path):
    agent = ReviewedFakeAgent()
    generator = DatasetGenerator(agent)
    generator.scenario_designer.design = lambda _domain, count: [
        DatasetScenario(
            description=f"generate daily sales scenario {i + 1}",
            observable="sales",
            unit="USD",
            suggested_frequency="daily",
        )
        for i in range(count)
    ]
    manifest = generator.generate_to_directory(
        "sales",
        tmp_path,
        series_count=1,
        length=24,
        freq="h",
        seed=11,
        respect_scenario_frequency=True,
    )
    entry = manifest["series"][0]
    assert entry["suggested_frequency"] == "daily"
    assert entry["normalized_suggested_frequency"] == "D"
    assert entry["effective_frequency"] == "D"
    assert entry["frequency_warning"] is None
    assert manifest["frequency_policy"]["respect_scenario_frequency"] is True


def test_dataset_generator_invalid_scenario_frequency_falls_back(tmp_path):
    agent = ReviewedFakeAgent()
    generator = DatasetGenerator(agent)
    generator.scenario_designer.design = lambda _domain, count: [
        DatasetScenario(
            description=f"generate odd frequency scenario {i + 1}",
            observable="value",
            unit="value",
            suggested_frequency="every market sneeze",
        )
        for i in range(count)
    ]
    manifest = generator.generate_to_directory(
        "sales",
        tmp_path,
        series_count=1,
        length=24,
        freq="h",
        seed=11,
        respect_scenario_frequency=True,
    )
    entry = manifest["series"][0]
    assert entry["normalized_suggested_frequency"] is None
    assert entry["effective_frequency"] == "h"
    assert entry["frequency_warning"]["type"] == "invalid_scenario_frequency"
    assert entry["frequency_warning"]["effective_frequency"] == "h"


def test_dataset_generator_skips_failed_variant_without_writing_series(tmp_path):
    agent = ReviewedFakeAgent()
    generator = DatasetGenerator(agent)
    generator.scenario_designer.design = lambda _domain, count: [
        DatasetScenario(description="generate failing scenario", observable="value", unit="value")
    ]

    def failing_generate_from_plan(*_args, **_kwargs):
        frame = pd.DataFrame({"value": [1.0, 2.0, 3.0], "anomaly": [0, 0, 0]})
        frame.attrs["validation_report"] = {
            "passed": False,
            "repaired_passed": True,
            "raw_passed": False,
            "critical_repairs": ["recomputed_stock_flow_columns"],
        }
        frame.attrs["series_audit"] = {"status": "FAIL", "llm_reviewed": True}
        frame.attrs["anomaly_execution"] = {"active_points": 0}
        return frame

    agent.generate_from_plan = failing_generate_from_plan
    import pytest

    with pytest.raises(RuntimeError, match="accepted series"):
        generator.generate_to_directory(
            "failing",
            tmp_path,
            series_count=1,
            length=3,
            seed=11,
            max_diversity_retries=0,
        )
    assert not list(tmp_path.glob("series_*.arrow"))
