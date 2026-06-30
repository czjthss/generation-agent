import gzip
import json

import numpy as np

from test_helpers import ReviewedFakeAgent
from generation_agent.dataset_generator import DatasetGenerator
from generation_agent.dataset_scenario_agent import DatasetScenario
from generation_agent.param_pack import materialize_param_pack, read_param_pack
from generation_agent.planner import heuristic_plan


def test_sequence_param_pack_materializes_same_values(tmp_path):
    agent = ReviewedFakeAgent()
    frame, _plan = agent.run_to_files(
        "generate factory electric load",
        tmp_path / "factory",
        length=96,
        seed=123,
        storage_mode="param-pack",
    )
    pack_path = tmp_path / "factory.syn.json.gz"
    assert pack_path.exists()
    pack = read_param_pack(pack_path)
    assert pack["replay_contract"]["llm_required_for_materialize"] is False
    assert "series_values" not in pack
    assert "data_points" not in pack
    restored, storage = materialize_param_pack(pack_path, tmp_path / "factory.arrow")
    assert storage["bytes"] > 0
    np.testing.assert_allclose(frame["value"].to_numpy(), restored["value"].to_numpy(), rtol=0, atol=1e-6)


def test_param_pack_is_compressed_json_without_point_values(tmp_path):
    agent = ReviewedFakeAgent()
    agent.run_to_files(
        "generate summer rainfall",
        tmp_path / "rain.syn.json.gz",
        length=240,
        seed=5,
        storage_mode="param-pack",
    )
    with gzip.open(tmp_path / "rain.syn.json.gz", "rt", encoding="utf-8") as handle:
        pack = json.load(handle)
    assert pack["format"] == "generation-agent-param-pack"
    assert pack["length"] == 240
    assert "points" in pack["preview"]
    assert len(pack["preview"]["points"]) <= 240
    assert "values" not in pack


def test_dataset_generator_writes_param_packs(tmp_path):
    agent = ReviewedFakeAgent()
    generator = DatasetGenerator(agent)
    generator.scenario_designer.design = lambda _domain, count: [
        DatasetScenario(description=f"generate weather scenario {index}", observable="value", unit="value")
        for index in range(count)
    ]
    manifest = generator.generate_to_directory(
        "weather",
        tmp_path,
        series_count=2,
        length=48,
        seed=10,
        storage_mode="param-pack",
    )
    assert manifest["storage_mode"] == "param-pack"
    assert manifest["storage_format"] == "generation_agent_param_pack"
    assert all(item["file"].endswith(".syn.json.gz") for item in manifest["series"])
    assert (tmp_path / "series_0001.syn.json.gz").exists()
