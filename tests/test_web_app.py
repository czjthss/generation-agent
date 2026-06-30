from pathlib import Path
from io import BytesIO

import pytest

import generation_agent.web_app as web_app
from generation_agent.web_app import _available_models, _normalize_payload, create_app


def test_normalize_sequence_payload(tmp_path: Path) -> None:
    payload = _normalize_payload({"description": "generate summer rainfall", "output_dir": str(tmp_path),
                                  "filename": "rain", "length": "48", "seed": "7"})
    assert payload["generation_mode"] == "sequence"
    assert payload["filename"] == "rain.arrow"
    assert payload["length"] == 48
    assert payload["freq"] == "h"
    assert payload["start"] == "2026-07-01 00:00:00"
    assert payload["output_dir"] == tmp_path.resolve()


def test_normalize_accepts_custom_frequency(tmp_path: Path) -> None:
    payload = _normalize_payload({
        "description": "generate electric load",
        "output_dir": str(tmp_path),
        "freq": "custom",
        "custom_freq": "2h",
        "start": "2026-01-01T03:30",
    })
    assert payload["freq"] == "2h"
    assert payload["start"] == "2026-01-01 03:30:00"


def test_normalize_accepts_dataset_scenario_frequency_toggle(tmp_path: Path) -> None:
    payload = _normalize_payload({
        "generation_mode": "dataset",
        "description": "generate weather dataset",
        "output_dir": str(tmp_path),
        "respect_scenario_frequency": True,
    })
    assert payload["respect_scenario_frequency"] is True


def test_normalize_rejects_invalid_frequency_and_start(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid frequency"):
        _normalize_payload({"description": "generate electric load", "output_dir": str(tmp_path), "freq": "abc"})
    with pytest.raises(ValueError, match="Invalid start time format"):
        _normalize_payload({
            "description": "generate electric load",
            "output_dir": str(tmp_path),
            "start": "not-a-time",
        })


def test_normalize_rejects_unknown_model(tmp_path: Path) -> None:
    payload = _normalize_payload({
        "description": "generate electric load",
        "output_dir": str(tmp_path),
        "model": _available_models()[0],
    })
    assert payload["model"] == _available_models()[0]
    with pytest.raises(ValueError, match="Invalid model parameter"):
        _normalize_payload({
            "description": "generate electric load",
            "output_dir": str(tmp_path),
            "model": "unknown/model",
        })


def test_normalize_accepts_only_arrow_reference(tmp_path: Path) -> None:
    arrow = tmp_path / "reference.arrow"
    arrow.write_bytes(b"placeholder")
    payload = _normalize_payload({
        "description": "generate electric load",
        "output_dir": str(tmp_path),
        "reference": str(arrow),
    })
    assert payload["reference_path"] == arrow.resolve()

    csv = tmp_path / "reference.csv"
    csv.write_text("value\n1\n2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Arrow"):
        _normalize_payload({
            "description": "generate electric load",
            "output_dir": str(tmp_path),
            "reference": str(csv),
        })


def test_normalize_rejects_empty_description() -> None:
    with pytest.raises(ValueError, match="required"):
        _normalize_payload({"description": "  "})


def test_web_routes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_app, "_open_native_dialog",
                        lambda kind: str(tmp_path if kind == "directory" else tmp_path / "reference.arrow"))
    app = create_app()
    client = app.test_client()
    assert client.get("/").status_code == 200
    assert client.post("/api/generate", json={"description": ""}).status_code == 400
    assert client.post("/api/select-output-dir").get_json()["path"] == str(tmp_path)
    assert client.post("/api/select-reference-arrow").get_json()["path"].endswith("reference.arrow")


def test_upload_reference_arrow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_app, "DEFAULT_OUTPUT_DIR", tmp_path)
    app = create_app()
    client = app.test_client()
    response = client.post(
        "/api/upload-reference-arrow",
        data={"file": (BytesIO(b"arrow-bytes"), "reference.arrow")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    path = Path(response.get_json()["path"])
    assert path.suffix == ".arrow"
    assert path.is_file()

    bad = client.post(
        "/api/upload-reference-arrow",
        data={"file": (BytesIO(b"csv"), "reference.csv")},
        content_type="multipart/form-data",
    )
    assert bad.status_code == 400
