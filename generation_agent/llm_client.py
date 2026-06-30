from __future__ import annotations

import os
from typing import Any

import httpx

from .planner import DEFAULT_OPENAI_BASE_URL


def ssl_verify_enabled() -> bool:
    raw = os.getenv("OPENAI_SSL_VERIFY", "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _http_client_kwargs() -> dict[str, Any]:
    if ssl_verify_enabled():
        return {}
    return {"http_client": httpx.Client(verify=False)}


def create_chat_openai(**kwargs: Any):
    from langchain_openai import ChatOpenAI

    kwargs.setdefault("api_key", os.getenv("OPENAI_API_KEY"))
    kwargs.setdefault("base_url", os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL))
    kwargs.update(_http_client_kwargs())
    return ChatOpenAI(**kwargs)


def create_openai_client(**kwargs: Any):
    from openai import OpenAI

    kwargs.setdefault("base_url", os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL))
    if not ssl_verify_enabled():
        kwargs["http_client"] = httpx.Client(verify=False)
    return OpenAI(**kwargs)
