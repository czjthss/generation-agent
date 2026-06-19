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
