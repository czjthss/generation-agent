import pandas as pd
import pytest


mcp = pytest.importorskip("mcp")

from generation_agent.compact_storage import write_series_arrow
from generation_agent.mcp_server import _reference_payload


def test_reference_payload_accepts_csv(tmp_path):
    path = tmp_path / "reference.csv"
    pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=12, freq="h"), "value": range(12)}).to_csv(
        path,
        index=False,
    )

    profile = _reference_payload(None, str(path))

    assert profile["source"].endswith("reference.csv")
    assert profile["value_column"] == "value"


def test_reference_payload_rejects_ambiguous_reference_inputs(tmp_path):
    csv_path = tmp_path / "reference.csv"
    csv_path.write_text("value\n1\n2\n3\n4\n5\n6\n7\n8\n", encoding="utf-8")
    arrow_path = tmp_path / "reference.arrow"
    frame = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=8, freq="h"), "value": range(8)})
    write_series_arrow(frame, arrow_path, metadata={})

    with pytest.raises(ValueError, match="Pass only one reference input"):
        _reference_payload(str(arrow_path), str(csv_path))
