from generation_agent.plan_normalizer import normalize_plan_for_execution
from generation_agent.planner import SeriesPlan


def test_non_executable_details_are_archived_not_executed():
    plan = SeriesPlan(
        domain="industrial power",
        domain_params={
            "shift_start_hour": 8,
            "shift_end_hour": 20,
            "ramp_rate_kw_per_hour": 150,
        },
        output_constraints={
            "nonnegative": True,
            "maximum_ramp_rate": "150 kW/hour",
        },
    )

    normalized = normalize_plan_for_execution(plan, "generate industrial park electric load")

    assert normalized.domain_params == {"shift_start_hour": 8, "shift_end_hour": 20}
    assert "maximum_ramp_rate" not in normalized.output_constraints
    assert normalized.metadata["non_executable_domain_details"] == {
        "ramp_rate_kw_per_hour": 150
    }
    assert normalized.metadata["non_executable_business_constraints"] == {
        "maximum_ramp_rate": "150 kW/hour"
    }


def test_semantic_config_is_completed_with_kernel_defaults():
    plan = SeriesPlan(
        semantic_type="regime_switching",
        semantic_config={"transition_probability": 0.08},
    )

    normalized = normalize_plan_for_execution(plan, "generate equipment operating-state series")

    assert normalized.semantic_type == "regime_switching"
    assert normalized.semantic_config["states"] == [0.4, 1.0, 1.5]
    assert normalized.semantic_config["transition_probability"] == 0.08
    assert "states" in normalized.metadata["semantic_defaults_applied"]


def test_random_walk_innovation_parameters_remain_executable():
    plan = SeriesPlan(
        semantic_type="random_walk",
        semantic_config={
            "initial_value": 100.0,
            "drift": 0.01,
            "volatility": 2.0,
            "innovation_distribution": "student_t",
            "innovation_df": 4.0,
            "innovation_ar": 0.2,
            "regime_volatility": True,
            "volatility_switch_probability": 0.05,
            "high_volatility_multiplier": 3.0,
        },
    )

    normalized = normalize_plan_for_execution(plan, "generate stock price with heavy-tailed volatility regimes")

    assert normalized.semantic_config["innovation_distribution"] == "student_t"
    assert normalized.semantic_config["innovation_df"] == 4.0
    assert normalized.semantic_config["innovation_ar"] == 0.2
    assert normalized.semantic_config["regime_volatility"] is True
    assert normalized.semantic_config["high_volatility_multiplier"] == 3.0


def test_unsupported_semantic_config_is_archived_before_reflection():
    plan = SeriesPlan(
        semantic_type="regime_switching",
        semantic_config={
            "states": [0.5, 1.0],
            "regimes": ["day", "night"],
            "regime_triggers": {"hour": 8},
        },
    )

    normalized = normalize_plan_for_execution(plan, "generate state-switching series")

    assert normalized.semantic_config["states"] == [0.5, 1.0]
    assert "regimes" not in normalized.semantic_config
    assert normalized.metadata["non_executable_semantic_details"] == {
        "regimes": ["day", "night"],
        "regime_triggers": {"hour": 8},
    }


def test_normalizer_does_not_apply_domain_specific_generator_or_semantic_rules():
    plan = SeriesPlan(
        domain="energy load",
        generator_type="smooth_environmental",
        semantic_type="regime_switching",
        semantic_config={"states": [0.6, 1.0]},
    )

    normalized = normalize_plan_for_execution(plan, "generate hourly industrial park electric load")

    assert normalized.generator_type == "smooth_environmental"
    assert normalized.semantic_type == "regime_switching"
    assert normalized.semantic_config["states"] == [0.6, 1.0]


def test_normalizer_does_not_infer_count_semantics_from_domain_words():
    plan = SeriesPlan(
        domain="restaurant orders",
        generator_type="smooth_environmental",
        lower_bound=None,
    )

    normalized = normalize_plan_for_execution(plan, "generate hourly restaurant orders")

    assert normalized.generator_type == "smooth_environmental"
    assert "integer" not in normalized.output_constraints


def test_invalid_bounds_do_not_reach_the_numerical_kernel():
    plan = SeriesPlan(
        lower_bound=10.0,
        output_constraints={"lower_bound": 10.0, "upper_bound": 2.0},
    )

    normalized = normalize_plan_for_execution(plan, "generate generic continuous observations")

    assert "upper_bound" not in normalized.output_constraints
    assert normalized.metadata["non_executable_business_constraints"][
        "invalid_upper_bound"
    ] == 2.0


def test_instantaneous_output_archives_non_executable_semantic_details():
    plan = SeriesPlan(
        semantic_type="instantaneous",
        semantic_config={"workday_regime": {"shift_start": 8, "shift_end": 20}},
    )

    normalized = normalize_plan_for_execution(plan, "generate hourly load")

    assert normalized.semantic_config == {}
    assert normalized.metadata["non_executable_semantic_details"] == {
        "workday_regime": {"shift_start": 8, "shift_end": 20}
    }


def test_normalizer_coerces_regime_state_labels_to_numeric_levels():
    plan = SeriesPlan(
        semantic_type="regime_switching",
        semantic_config={
            "states": ["weekday_normal", "weekend_low", "fault"],
            "transition_probability": "0.2",
            "initial_state": "1",
        },
    )
    normalized = normalize_plan_for_execution(plan, "generate state-switching load")
    assert all(isinstance(item, float) for item in normalized.semantic_config["states"])
    assert normalized.semantic_config["initial_state"] == 1
    assert normalized.semantic_config["transition_probability"] == 0.2
    assert "coerced_regime_state_labels_to_numeric_levels" in normalized.metadata["execution_normalization"]


def test_normalizer_preserves_llm_feature_family_for_sparse_precipitation():
    plan = SeriesPlan(
        domain="climate and weather",
        unit="mm",
        generator_type="cyclic_signal",
        semantic_type="regime_switching",
        semantic_config={"states": ["dry", "rain"]},
        baseline=1000.0,
        daily_amplitude=300.0,
    )
    normalized = normalize_plan_for_execution(plan, "generate summer precipitation data in south China")
    assert normalized.generator_type == "cyclic_signal"
    assert normalized.domain == "climate and weather"
    assert "applied_strong_domain_prior_feature_family" not in normalized.metadata.get("execution_normalization", [])


def test_normalizer_preserves_llm_feature_family_for_bounded_utilization():
    plan = SeriesPlan(
        domain="IT infrastructure monitoring",
        unit="%",
        generator_type="cyclic_signal",
        semantic_type="regime_switching",
        semantic_config={"states": ["idle", "busy"]},
    )
    normalized = normalize_plan_for_execution(plan, "generate server CPU utilization")
    assert normalized.generator_type == "cyclic_signal"
    assert normalized.domain == "IT infrastructure monitoring"


def test_normalizer_removes_final_upper_bound_from_cumulative_outputs():
    plan = SeriesPlan(
        semantic_type="cumulative",
        output_constraints={"upper_bound": 1000.0},
    )
    normalized = normalize_plan_for_execution(plan, "generate cumulative sales")
    assert "upper_bound" not in normalized.output_constraints
    assert normalized.output_constraints["monotonic"] == "nondecreasing"
    assert "removed_upper_bound_from_cumulative_output" in normalized.metadata["execution_normalization"]
