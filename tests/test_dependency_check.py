import importlib.util

import pytest

from generation_agent.dependency_check import ensure_llm_dependencies, missing_llm_dependencies


def test_missing_llm_dependencies_reports_requirements(monkeypatch):
    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name):
        if name == "langchain":
            return None
        return original_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    assert "langchain>=1.3" in missing_llm_dependencies()
    with pytest.raises(RuntimeError, match="pip install -r requirements.txt"):
        ensure_llm_dependencies()
