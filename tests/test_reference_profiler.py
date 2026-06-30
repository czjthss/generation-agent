import numpy as np
import pandas as pd

from generation_agent.planner import SeriesPlan
from generation_agent.reference_profiler import (
    apply_reference_priors,
    compare_to_reference,
    profile_reference_arrow,
    profile_reference_frame,
)
from generation_agent.compact_storage import write_series_arrow
from generation_agent.synthesizer import synthesize_series


def test_reference_profile_extracts_distribution_dynamics_and_periodicity():
    x = np.arange(240, dtype=float)
    values = 20.0 + 0.02 * x + 3.0 * np.sin(2 * np.pi * x / 24)
    frame = pd.DataFrame(
        {"time": pd.date_range("2026-01-01", periods=len(x), freq="h"), "value": values}
    )
    profile = profile_reference_frame(frame)
    assert profile.value_column == "value"
    assert profile.time_column == "time"
    assert profile.statistics["mean"] > 20.0
    assert profile.dynamics["lag1_correlation"] > 0.8
    assert any(item["lag"] == 24 for item in profile.periodicity["autocorrelation_peaks"])


def test_reference_profile_reads_arrow(tmp_path):
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=48, freq="h"),
            "value": np.linspace(10.0, 20.0, 48),
            "anomaly": np.zeros(48, dtype=int),
        }
    )
    path = tmp_path / "reference.arrow"
    write_series_arrow(
        frame,
        path,
        metadata={"start": "2026-01-01 00:00:00", "frequency": "h", "length": 48},
    )
    profile = profile_reference_arrow(path)
    assert profile.value_column == "value"
    assert profile.time_column == "timestamp"
    assert profile.valid_count == 48


def test_reference_comparison_reports_feature_errors():
    frame = pd.DataFrame({"value": np.arange(20, dtype=float)})
    profile = profile_reference_frame(frame)
    report = compare_to_reference(frame, profile)
    assert report["mean_relative_error"] == 0.0
    assert report["std_relative_error"] == 0.0


def test_reference_priors_are_applied_without_llm():
    frame = pd.DataFrame({"value": 100.0 + np.arange(30, dtype=float)})
    profile = profile_reference_frame(frame)
    plan = apply_reference_priors(SeriesPlan(semantic_type="instantaneous"), profile, "structure")
    assert plan.baseline == profile.statistics["quantiles"]["p50"]
    assert plan.trend_slope == profile.dynamics["total_linear_change"]
    assert plan.metadata["reference_priors_applied"]
    assert "reference_executable_targets" in plan.metadata
    assert "distribution_quantiles" in plan.metadata["reference_executable_targets"]


def test_reference_acf_target_is_executed_by_synthesizer():
    base_plan = SeriesPlan(
        semantic_type="instantaneous",
        generator_type="cyclic_signal",
        baseline=0.0,
        daily_amplitude=0.0,
        noise_sigma=5.0,
        lower_bound=None,
    )
    target_plan = SeriesPlan.from_dict(base_plan.to_dict())
    target_plan.metadata["reference_executable_targets"] = {"acf_targets": {"1": 0.85}}
    target_plan.metadata["reference_targets"] = {"lag1_correlation": 0.85}

    base = synthesize_series(base_plan, length=200, seed=17)["value"].to_numpy()
    targeted_frame = synthesize_series(target_plan, length=200, seed=17)
    targeted = targeted_frame["value"].to_numpy()

    base_corr = np.corrcoef(base[:-1], base[1:])[0, 1]
    targeted_corr = np.corrcoef(targeted[:-1], targeted[1:])[0, 1]
    assert targeted_corr > base_corr + 0.1
    assert "acf_lag1" in target_plan.metadata["reference_target_execution"]


def test_positive_acf_target_below_current_is_skipped_not_increased():
    base_plan = SeriesPlan(
        semantic_type="instantaneous",
        generator_type="cyclic_signal",
        baseline=0.0,
        daily_amplitude=5.0,
        noise_sigma=0.2,
        lower_bound=None,
    )
    target_plan = SeriesPlan.from_dict(base_plan.to_dict())
    target_plan.metadata["reference_executable_targets"] = {"acf_targets": {"1": 0.0}}

    base = synthesize_series(base_plan, length=120, seed=13)["value"].to_numpy()
    targeted = synthesize_series(target_plan, length=120, seed=13)["value"].to_numpy()

    assert np.allclose(base, targeted)
    execution = target_plan.metadata["reference_target_execution"]["acf_lag1"]
    assert execution["method"] == "skipped_decrease_not_applied"


def test_sparse_reference_distribution_uses_tail_cap_not_global_affine():
    plan = SeriesPlan(
        semantic_type="instantaneous",
        generator_type="intermittent_event",
        baseline=0.0,
        daily_amplitude=0.0,
        noise_sigma=0.0,
        lower_bound=0.0,
        domain_params={
            "dry_spell_bias": 0.0,
            "event_probability": 1.0,
            "mean_duration": 1.0,
            "intensity_shape": 0.6,
            "intensity_scale": 80.0,
            "storm_probability": 0.6,
            "storm_multiplier": 8.0,
        },
        metadata={
            "reference_executable_targets": {
                "distribution_quantiles": {
                    "p05": 0.0,
                    "p50": 0.0,
                    "p95": 20.0,
                    "p99": 35.0,
                    "p999": 60.0,
                },
                "event_run_length": {"zero_ratio": 0.8},
            }
        },
    )
    frame = synthesize_series(plan, length=240, seed=23)
    assert frame["value"].quantile(0.99) <= 60.0
    assert frame["value"].max() <= 60.0


def test_sparse_event_acf_target_preserves_zero_mask():
    base_plan = SeriesPlan(
        semantic_type="instantaneous",
        generator_type="intermittent_event",
        baseline=0.0,
        lower_bound=0.0,
        domain_params={
            "dry_spell_bias": 0.85,
            "event_probability": 0.25,
            "mean_duration": 1.0,
            "intensity_shape": 1.0,
            "intensity_scale": 5.0,
            "storm_probability": 0.0,
        },
    )
    target_plan = SeriesPlan.from_dict(base_plan.to_dict())
    target_plan.metadata["reference_executable_targets"] = {
        "acf_targets": {"1": 0.9},
        "event_run_length": {"zero_ratio": 0.95},
    }
    targeted = synthesize_series(target_plan, length=240, seed=31)["value"].to_numpy()

    assert float((targeted == 0.0).mean()) >= 0.9
    execution = target_plan.metadata["reference_target_execution"]["acf_lag1"]
    assert execution["method"] == "event_internal_intensity_smoothing"
    assert execution["zero_mask_preserved"] is True
    assert "event_run_length" in target_plan.metadata["reference_target_execution"]


def test_reference_acf_filter_propagates_anomaly_mask():
    plan = SeriesPlan(
        semantic_type="instantaneous",
        generator_type="cyclic_signal",
        baseline=10.0,
        daily_amplitude=0.0,
        noise_sigma=0.0,
        anomaly_enabled=True,
        anomaly_count=1,
        anomaly_width=1,
        anomaly_magnitude=10.0,
        anomaly_kind="spike",
        metadata={"reference_executable_targets": {"acf_targets": {"1": 0.85}}},
    )
    frame = synthesize_series(plan, length=80, seed=5)
    assert frame["anomaly"].sum() > 1
    assert "propagated_anomaly_effect" in plan.metadata["reference_target_execution"]


def test_negative_reference_acf_target_is_executed():
    plan = SeriesPlan(
        semantic_type="instantaneous",
        generator_type="cyclic_signal",
        baseline=0.0,
        daily_amplitude=0.0,
        noise_sigma=5.0,
        lower_bound=None,
        metadata={"reference_executable_targets": {"acf_targets": {"1": -0.6}}},
    )
    frame = synthesize_series(plan, length=200, seed=19)
    values = frame["value"].to_numpy()
    corr = np.corrcoef(values[:-1], values[1:])[0, 1]
    assert corr < -0.1
    assert plan.metadata["reference_target_execution"]["acf_lag1"]["target"] < 0.0


def test_strict_reference_does_not_break_cumulative_semantics():
    frame = pd.DataFrame({"value": np.cumsum(np.arange(1, 31, dtype=float))})
    profile = profile_reference_frame(frame)
    plan = apply_reference_priors(
        SeriesPlan(
            semantic_type="cumulative",
            semantic_config={"initial_value": 0.0, "allow_negative_increment": False},
            output_constraints={"monotonic": "nondecreasing"},
        ),
        profile,
        "strict",
    )
    assert "upper_bound" not in plan.output_constraints
    assert plan.baseline == profile.dynamics["mean_difference"]


def test_reference_profile_prefers_threshold_over_sparse_event_for_broad_split():
    driver = np.tile(np.arange(40, dtype=float), 4)
    target = 10.0 + 25.0 * (driver >= 20.0)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=len(driver), freq="h"),
            "target": target,
            "driver": driver,
        }
    )
    profile = profile_reference_frame(frame, time_column="timestamp", value_column="target")
    relationship = profile.relationships[0]
    assert relationship["operator"] == "threshold"
    assert relationship["mechanism_inference"] == "heuristic_statistical"


def test_reference_profile_marks_sparse_short_response_as_event_trigger():
    driver = np.zeros(96, dtype=float)
    driver[[18, 47, 76]] = 10.0
    target = np.full(96, 5.0)
    target[driver > 0] = 50.0
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=len(driver), freq="h"),
            "target": target,
            "driver": driver,
        }
    )
    profile = profile_reference_frame(frame, time_column="timestamp", value_column="target")
    relationship = profile.relationships[0]
    assert relationship["operator"] == "event_trigger"
    assert relationship["trigger_op"] == "eq"
    assert relationship["mechanism_inference"] == "heuristic_statistical"


def test_reference_profile_marks_sparse_low_value_event_with_exact_trigger():
    driver = np.full(96, 10.0)
    driver[[18, 47, 76]] = 0.0
    target = np.full(96, 50.0)
    target[driver == 0.0] = 5.0
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=len(driver), freq="h"),
            "target": target,
            "driver": driver,
        }
    )
    profile = profile_reference_frame(frame, time_column="timestamp", value_column="target")
    relationship = profile.relationships[0]
    assert relationship["operator"] == "event_trigger"
    assert relationship["threshold"] == 0.0
    assert relationship["trigger_op"] == "eq"
    assert relationship["effect"] == "negative_event_effect"


def test_reference_profile_event_effect_direction_uses_event_mask_not_raw_correlation():
    driver = np.full(96, 10.0)
    driver[[18, 47, 76]] = 0.0
    target = np.full(96, 5.0)
    target[driver == 0.0] = 50.0
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=len(driver), freq="h"),
            "target": target,
            "driver": driver,
        }
    )
    profile = profile_reference_frame(frame, time_column="timestamp", value_column="target")
    relationship = profile.relationships[0]
    assert relationship["operator"] == "event_trigger"
    assert relationship["threshold"] == 0.0
    assert relationship["trigger_op"] == "eq"
    assert relationship["correlation"] < 0.0
    assert relationship["effect"] == "positive_event_effect"
