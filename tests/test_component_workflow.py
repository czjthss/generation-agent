from generation_agent.component_workflow import build_component_workflow
from generation_agent.planner import SeriesPlan, heuristic_plan
from generation_agent.synthesizer import synthesize_series
from test_helpers import local_generate_from_plan
import pytest


def test_component_workflow_decomposes_electric_load():
    description = "generate electric load for a south China industrial park in summer"
    plan = heuristic_plan(description)
    workflow = build_component_workflow(
        description,
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
    assert "weather_response" in names
    assert "parameter_compiler_agent" in payload["agent_trace"]


def test_llm_mechanism_components_are_executed_as_component_dsl():
    plan = SeriesPlan(
        domain="test",
        generator_type="cyclic_signal",
        baseline=10.0,
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
                                "time_scale_behavior": "persistent",
                                "statistical_shape": "constant",
                                "feature_family": "baseline",
                                "params": {"level": 7.0},
                            },
                            {
                                "name": "custom_trend",
                                "role": "trend",
                                "component_semantic": "trend_component",
                                "time_scale_behavior": "slow increase",
                                "statistical_shape": "linear",
                                "feature_family": "linear_trend",
                                "params": {"slope": 1.0},
                            },
                        ],
                        "composition": {"operator": "add", "final_transform": "identity"},
                    }
                ]
            }
        },
    )
    frame = local_generate_from_plan(plan, length=5, freq="D", seed=3)
    workflow = frame.attrs["component_workflow"]
    assert workflow["agent_trace"]["component_source"] == "llm_mechanism_plan"
    assert list(frame["value"].round(4)) == [7.0, 7.25, 7.5, 7.75, 8.0]


def test_llm_solar_component_dsl_does_not_multiply_daylight_components_explosively():
    plan = SeriesPlan(
        domain="solar",
        generator_type="daylight_envelope",
        unit="kW",
        baseline=0.0,
        daily_amplitude=600.0,
        noise_sigma=0.0,
        lower_bound=0.0,
        output_constraints={"nonnegative": True},
        domain_params={"cloud_probability": 1.0, "cloud_drop_min": 0.5, "cloud_drop_max": 0.5},
        metadata={
            "mechanism_planning_agent": {
                "target_variables": [
                    {
                        "components": [
                            {
                                "name": "daylight envelope",
                                "role": "solar availability envelope",
                                "component_semantic": "availability envelope",
                                "time_scale_behavior": "daylight hours",
                                "statistical_shape": "bounded daily arch",
                            },
                            {
                                "name": "cloud attenuation",
                                "role": "cloud-driven attenuation",
                                "component_semantic": "multiplicative attenuation",
                                "time_scale_behavior": "short cloudy intervals",
                                "statistical_shape": "occasional downward pulses",
                            },
                        ],
                        "composition": {"operator": "multiply_then_add_noise"},
                    }
                ]
            }
        },
    )
    frame = local_generate_from_plan(
        plan,
        length=48,
        freq="h",
        start="2026-07-01 00:00:00",
        seed=3,
    )
    assert frame.attrs["component_workflow"]["agent_trace"]["component_source"] == "llm_mechanism_plan"
    assert frame["value"].max() <= 600.0
    assert frame.attrs["validation_report"]["passed"]
    cloud = next(
        item for item in frame.attrs["component_report"]["components"] if item["name"] == "cloud attenuation"
    )
    assert cloud["checks"]["semantic"] is True


def test_llm_dsl_additive_noise_is_not_dropped_when_multiplicative_base_exists():
    plan = SeriesPlan(
        domain="solar",
        generator_type="daylight_envelope",
        unit="kW",
        baseline=0.0,
        daily_amplitude=300.0,
        noise_sigma=0.0,
        lower_bound=0.0,
        output_constraints={"nonnegative": True},
        domain_params={"cloud_probability": 0.0},
        metadata={
            "mechanism_planning_agent": {
                "target_variables": [
                    {
                        "components": [
                            {
                                "name": "daylight envelope",
                                "role": "solar availability envelope",
                                "component_semantic": "availability envelope",
                                "time_scale_behavior": "daylight hours",
                                "statistical_shape": "bounded daily arch",
                            },
                            {
                                "name": "measurement noise",
                                "role": "meter noise",
                                "component_semantic": "signed noise",
                                "time_scale_behavior": "local perturbation",
                                "statistical_shape": "zero mean noise",
                                "feature_family": "noise",
                                "params": {"sigma": 25.0},
                            },
                        ],
                        "composition": {"operator": "multiply_then_add_noise"},
                    }
                ]
            }
        },
    )
    frame = local_generate_from_plan(
        plan,
        length=48,
        freq="h",
        start="2026-07-01 00:00:00",
        seed=11,
    )
    noise = next(
        item for item in frame.attrs["component_report"]["components"] if item["name"] == "measurement noise"
    )
    assert noise["stats"]["std"] > 0.0
    assert frame["value"].sum() != frame.attrs["component_stats"]["daylight envelope"]["mean"] * len(frame)


def test_unknown_required_llm_component_fails_component_quality_report():
    plan = SeriesPlan(
        domain="custom",
        generator_type="cyclic_signal",
        baseline=10.0,
        daily_amplitude=0.0,
        noise_sigma=0.0,
        metadata={
            "mechanism_planning_agent": {
                "target_variables": [
                    {
                        "components": [
                            {
                                "name": "unimplemented mandatory mechanism",
                                "role": "mandatory domain mechanism",
                                "component_semantic": "unimplemented_physics",
                                "time_scale_behavior": "persistent",
                                "statistical_shape": "nonzero required effect",
                            }
                        ]
                    }
                ]
            }
        },
    )
    frame = local_generate_from_plan(plan, length=12, freq="h", seed=4)
    report = frame.attrs["component_report"]
    validation = frame.attrs["validation_report"]
    assert report["status"] == "REVISE"
    assert "unsupported feature family" in " ".join(report["issues"])
    assert validation["checks"]["component_quality_report"] is False
    assert validation["passed"] is False


def test_llm_storm_multiplier_only_boosts_event_intensity():
    plan = SeriesPlan(
        domain="rain",
        generator_type="intermittent_event",
        baseline=0.0,
        daily_amplitude=0.0,
        noise_sigma=0.0,
        lower_bound=0.0,
        output_constraints={"nonnegative": True},
        domain_params={
            "event_probability": 0.0,
            "storm_probability": 1.0,
            "storm_multiplier": 5.0,
        },
        metadata={
            "mechanism_planning_agent": {
                "target_variables": [
                    {
                        "components": [
                            {
                                "name": "storm multiplier",
                                "role": "rare storm intensity boost",
                                "component_semantic": "sparse extreme event",
                                "time_scale_behavior": "rare bursts",
                                "statistical_shape": "multiplicative boost",
                                "feature_family": "storm_multiplier",
                            }
                        ]
                    }
                ]
            }
        },
    )
    frame = local_generate_from_plan(plan, length=24, freq="h", seed=9)
    assert frame["value"].sum() == 0.0
    assert frame.attrs["component_report"]["status"] == "REVISE"


def test_non_dsl_cloud_attenuation_power_contribution_is_not_factor_checked():
    plan = SeriesPlan(
        domain="solar",
        generator_type="daylight_envelope",
        daily_amplitude=800.0,
        noise_sigma=0.0,
        lower_bound=0.0,
        domain_params={"cloud_probability": 1.0, "cloud_drop_min": 0.5, "cloud_drop_max": 0.5},
    )
    frame = local_generate_from_plan(
        plan,
        length=48,
        freq="h",
        start="2026-07-01 00:00:00",
        seed=7,
    )
    cloud = next(
        item for item in frame.attrs["component_report"]["components"] if item["name"] == "cloud_attenuation"
    )
    assert cloud["stats"]["min"] < -1.0
    assert cloud["checks"]["semantic"] is True


def test_generation_records_component_quality_report():
    plan = heuristic_plan("generate electric load for a south China industrial park in summer")
    plan.output_constraints["upper_bound"] = 110.0
    plan.heat_effect = max(plan.heat_effect, 20.0)
    frame = local_generate_from_plan(
        plan,
        length=48,
        freq="h",
        start="2026-07-01 00:00:00",
        seed=7,
    )
    assert "component_workflow" in frame.attrs
    assert "component_report" in frame.attrs
    assert frame.attrs["component_report"]["components"]
    names = {item["name"] for item in frame.attrs["component_report"]["components"]}
    assert "daily_cycle" in names
    assert "weather_response" in names
    assert frame["value"].max() <= 110.0
    assert "clipped_to_upper_bound" not in frame.attrs["validation_report"]["repairs_applied"]
    assert frame.attrs["final_plan"]["metadata"]["workflow"] == (
        "Requirement Understanding Agent -> Mechanism Planning Agent -> "
        "Parameter Compilation Agent -> Anomaly Strategy Agent -> Local Generation Kernel -> "
        "Quality Evaluation Agent -> Arrow Output"
    )


def test_component_workflow_failure_is_visible_in_validation_report():
    plan = SeriesPlan(
        metadata={"component_workflow": {"invalid": "payload"}},
    )
    frame = synthesize_series(plan, length=12, seed=4)
    validation = frame.attrs["validation_report"]
    assert validation["checks"]["component_workflow_executed"] is False
    assert validation["warnings"][0]["type"] == "component_workflow_fallback"
    assert validation["warnings"][0]["severity"] == "hard_warning"


def test_component_workflow_failure_raises_in_strict_mode():
    plan = SeriesPlan(
        metadata={
            "cost_mode": "strict",
            "component_workflow": {"invalid": "payload"},
        },
    )
    with pytest.raises(RuntimeError, match="Component workflow failed in strict mode"):
        synthesize_series(plan, length=12, seed=4)


def test_intermittent_component_workflow_advances_cursor_on_event_branch():
    plan = SeriesPlan(
        domain="rainfall",
        generator_type="intermittent_event",
        semantic_type="instantaneous",
        domain_params={
            "dry_spell_bias": 0.0,
            "event_probability": 1.0,
            "mean_duration": 6.0,
            "intensity_shape": 1.0,
            "intensity_scale": 2.0,
            "storm_probability": 0.0,
        },
    )
    frame = local_generate_from_plan(
        plan,
        length=24,
        freq="h",
        start="2026-07-01 00:00:00",
        seed=3,
    )
    assert len(frame) == 24
    assert "component_report" in frame.attrs


def test_component_revise_status_fails_deterministic_validation():
    plan = SeriesPlan(
        domain="rainfall",
        generator_type="intermittent_event",
        semantic_type="instantaneous",
        domain_params={
            "dry_spell_bias": 0.0,
            "event_probability": 1.0,
            "mean_duration": 200.0,
            "intensity_shape": 1.0,
            "intensity_scale": 2.0,
            "storm_probability": 0.0,
        },
    )
    frame = local_generate_from_plan(
        plan,
        length=48,
        freq="h",
        start="2026-07-01 00:00:00",
        seed=5,
    )
    assert frame.attrs["component_report"]["status"] == "REVISE"
    validation = frame.attrs["validation_report"]
    assert validation["checks"]["component_quality_report"] is False
    assert validation["passed"] is False
