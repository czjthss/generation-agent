import inspect

import numpy as np
import pandas as pd

from generation_agent.mcp_server import generate_time_series_dataset
from generation_agent.mcp_server import generate_time_series_code
from generation_agent.planner import SeriesPlan
from generation_agent.codegen import render_generator_code
from generation_agent.dataset_diversity import diversify_plan
from generation_agent.dataset_scenario_agent import DatasetScenario
from generation_agent.reference_profiler import profile_reference_frame
from generation_agent.synthesizer import _relationship_audit, _relationship_effect, synthesize_series


def test_daily_frequency_does_not_create_intraday_load_peaks():
    plan = SeriesPlan(
        domain="electric_load",
        generator_type="cyclic_signal",
        unit="kW",
        baseline=100.0,
        daily_amplitude=50.0,
        weekly_enabled=False,
        trend_slope=0.0,
        noise_sigma=0.0,
        heat_effect=0.0,
        lower_bound=0.0,
    )
    frame = synthesize_series(plan, length=14, freq="D", start="2026-07-01", seed=7)
    assert np.allclose(frame["value"].to_numpy(), 100.0)


def test_daily_daylight_envelope_generates_aggregated_solar_values():
    plan = SeriesPlan(
        domain="solar",
        generator_type="daylight_envelope",
        unit="kWh",
        baseline=0.0,
        daily_amplitude=50.0,
        noise_sigma=0.0,
        lower_bound=0.0,
        domain_params={"sunrise_hour": 6.0, "sunset_hour": 18.0, "cloud_probability": 0.0},
    )
    hourly = synthesize_series(plan, length=48, freq="h", start="2026-07-01", seed=7)
    daily = synthesize_series(plan, length=7, freq="D", start="2026-07-01", seed=7)
    assert hourly["value"].sum() > 0.0
    assert daily["value"].sum() > 0.0
    assert daily["value"].min() > 0.0


def test_mcp_dataset_generation_exposes_storage_mode_parameter():
    signature = inspect.signature(generate_time_series_dataset)
    assert "storage_mode" in signature.parameters
    assert signature.parameters["storage_mode"].default == "arrow"
    assert "respect_scenario_frequency" in signature.parameters
    assert signature.parameters["respect_scenario_frequency"].default is False


def test_mcp_code_generation_has_no_unused_storage_mode_parameter():
    signature = inspect.signature(generate_time_series_code)
    assert "storage_mode" not in signature.parameters


def test_codegen_replay_script_uses_main_generation_kernel():
    script = render_generator_code(SeriesPlan(), length=8, freq="30min", start="2026-07-01")
    assert "from generation_agent.agent import GenerationAgent" in script
    assert "GenerationAgent(model=" in script
    assert "model=None" not in script
    assert "from generation_agent.synthesizer import synthesize_series" not in script
    assert "np.arange(length, dtype=float) % 24.0" not in script
    assert "Simplified standalone generator" not in script


def test_codegen_replay_uses_agent_component_workflow_path():
    plan = SeriesPlan(
        generator_type="cyclic_signal",
        daily_amplitude=0.0,
        noise_sigma=0.0,
        metadata={
            "mechanism_planning_agent": {
                "target_variables": [
                    {
                        "components": [
                            {
                                "name": "custom_base",
                                "role": "baseline",
                                "component_semantic": "baseline_level",
                                "feature_family": "baseline",
                                "params": {"level": 7.0},
                            },
                            {
                                "name": "custom_trend",
                                "role": "trend",
                                "component_semantic": "trend_component",
                                "feature_family": "linear_trend",
                                "params": {"slope": 1.0},
                            },
                        ]
                    }
                ]
            }
        },
    )
    script = render_generator_code(plan, length=5, freq="D", start="2026-07-01", seed=3)
    assert "GenerationAgent(model=" in script
    assert "generate_from_plan" in script
    assert "synthesize_series" not in script


def test_multivariate_relationship_audit_records_operator_specific_check():
    plan = SeriesPlan(
        domain="electric_load",
        generator_type="cyclic_signal",
        unit="kW",
        baseline=100.0,
        daily_amplitude=20.0,
        noise_sigma=0.0,
        variables=[
            {"name": "load", "role": "target", "unit": "kW"},
            {"name": "temperature", "role": "driver", "unit": "degC", "generator_type": "smooth_environmental"},
        ],
        relationships=[
            {
                "source": "temperature",
                "target": "load",
                "effect": "positive threshold effect",
                "operator": "threshold",
                "threshold": 28.0,
                "coefficient": 15.0,
            }
        ],
    )
    frame = synthesize_series(plan, length=96, freq="h", start="2026-07-01", seed=3)
    audit = frame.attrs["multivariate_report"]["relationship_audit"]["relationships"][0]
    assert audit["operator"] == "threshold"
    assert "operator_check" in audit


def test_threshold_relationship_fails_when_threshold_never_splits_states():
    plan = SeriesPlan(
        domain="electric_load",
        generator_type="cyclic_signal",
        unit="kW",
        baseline=100.0,
        daily_amplitude=20.0,
        noise_sigma=0.0,
        variables=[
            {"name": "load", "role": "target", "unit": "kW"},
            {"name": "temperature", "role": "driver", "unit": "degC", "generator_type": "smooth_environmental"},
        ],
        relationships=[
            {
                "source": "temperature",
                "target": "load",
                "effect": "positive threshold effect",
                "operator": "threshold",
                "threshold": 999.0,
                "coefficient": 15.0,
            }
        ],
    )
    frame = synthesize_series(plan, length=96, freq="h", start="2026-07-01", seed=3)
    audit = frame.attrs["multivariate_report"]["relationship_audit"]["relationships"][0]
    assert audit["operator_check"]["passed"] is False
    assert audit["passed"] is False


def test_cyclic_signal_uses_shift_parameters_without_component_workflow():
    plan = SeriesPlan(
        domain="electric_load",
        generator_type="cyclic_signal",
        unit="kW",
        baseline=100.0,
        daily_amplitude=0.0,
        noise_sigma=0.0,
        weekly_enabled=False,
        domain_params={"shift_start_hour": 8.0, "shift_end_hour": 18.0, "shift_amplitude": 50.0},
    )
    frame = synthesize_series(plan, length=24, freq="h", start="2026-07-01", seed=1)
    assert float(frame.loc[frame.timestamp.dt.hour == 10, "value"].iloc[0]) > 140.0
    assert float(frame.loc[frame.timestamp.dt.hour == 2, "value"].iloc[0]) == 100.0


def test_dataset_diversity_uses_generic_mechanism_variants():
    plan = SeriesPlan(
        domain="electric_load",
        generator_type="cyclic_signal",
        unit="kW",
        baseline=100.0,
        daily_amplitude=20.0,
        weekly_enabled=True,
        weekly_amplitude=5.0,
    )
    scenario = DatasetScenario(description="facility demand with operating schedule", observable="load", unit="kW")
    variant = diversify_plan(plan, scenario, series_index=0, variant_index=0, seed=6, strength="high")
    assert variant.metadata["dataset_diversity"]["mechanism_variant"] in {
        "single_operating_window",
        "double_operating_window",
        "near_continuous_operation",
        "driver_response_dominated",
        "scheduled_event_pulses",
        "strong_calendar_gate",
        "phase_shifted_operation",
        "trend_dominated",
        "seasonal_dominated",
    }
    assert "site_type" not in variant.metadata


def test_reference_profile_can_infer_state_gate_relationship():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=40, freq="h"),
            "target": [10, 11, 10, 11, 40, 42, 41, 43] * 5,
            "machine_state": [0, 0, 0, 0, 1, 1, 1, 1] * 5,
        }
    )
    profile = profile_reference_frame(frame, time_column="timestamp", value_column="target")
    relationship = profile.relationships[0]
    assert relationship["operator"] in {"state_gate", "threshold"}


def test_event_trigger_exact_operator_only_fires_on_matching_low_value():
    driver = np.full(24, 10.0)
    driver[[5, 17]] = 0.0
    target = np.full(24, 100.0)
    effect, operator = _relationship_effect(
        driver,
        target,
        {"operator": "event_trigger", "threshold": 0.0, "trigger_op": "eq", "width": 1},
        sign=-1.0,
        coefficient=20.0,
    )
    assert operator == "event_trigger"
    assert np.flatnonzero(effect != 0.0).tolist() == [5, 17]
    assert effect[5] < 0.0


def test_event_trigger_audit_uses_exact_trigger_operator():
    driver = np.full(48, 10.0)
    driver[[8, 31]] = 0.0
    target = np.full(48, 50.0)
    target[driver == 0.0] = 5.0
    plan = SeriesPlan(
        variables=[
            {"name": "target", "role": "target"},
            {"name": "driver", "role": "driver"},
        ],
        relationships=[
            {
                "source": "driver",
                "target": "target",
                "effect": "negative event-triggered effect",
                "operator": "event_trigger",
                "threshold": 0.0,
                "trigger_op": "eq",
                "width": 1,
            }
        ],
    )
    report = _relationship_audit({"target": target, "driver": driver, "value": target}, plan)
    audit = report["relationships"][0]
    assert audit["operator_check"]["trigger_op"] == "eq"
    assert audit["operator_check"]["passed"] is True
    assert audit["passed"] is True
