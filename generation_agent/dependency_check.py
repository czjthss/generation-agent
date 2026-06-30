from __future__ import annotations

import importlib.util


LLM_DEPENDENCIES = {
    "langchain": "langchain>=1.3",
    "langchain_openai": "langchain-openai>=1.2",
    "openai": "openai>=1.0",
}


def missing_llm_dependencies() -> list[str]:
    return [
        requirement
        for module_name, requirement in LLM_DEPENDENCIES.items()
        if importlib.util.find_spec(module_name) is None
    ]


def ensure_llm_dependencies() -> None:
    missing = missing_llm_dependencies()
    if not missing:
        return
    raise RuntimeError(
        "LLM planning dependencies are not installed: "
        + ", ".join(missing)
        + ". Install them in the active conda environment with "
        + "`pip install -r requirements.txt`. Materializing .syn.json.gz "
        + "parameter packs and replaying an existing SeriesPlan do not require "
        + "these LLM dependencies."
    )
