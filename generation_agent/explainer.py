from __future__ import annotations

import json
import os

from .env import load_project_env
from .planner import DEFAULT_LLM_MODEL, DEFAULT_OPENAI_BASE_URL, SeriesPlan


load_project_env()


def _fallback_explanation(description: str, plan: SeriesPlan, length: int, freq: str, start: str) -> str:
    params = json.dumps(plan.domain_params, ensure_ascii=False, indent=2)
    semantic_params = json.dumps(plan.semantic_config, ensure_ascii=False, indent=2)
    constraints = json.dumps(plan.output_constraints, ensure_ascii=False, indent=2)
    return (
        f"Task description: {description}\n\n"
        f"1. Domain understanding\n"
        f"- Domain: {plan.domain}\n"
        f"- Unit: {plan.unit}\n"
        f"- Generator type: {plan.generator_type}\n\n"
        f"2. Time range\n"
        f"- Start time: {start}\n"
        f"- Series length: {length}\n"
        f"- Frequency: {freq}\n\n"
        f"3. Main generation parameters\n"
        f"- baseline：{plan.baseline}\n"
        f"- trend_slope：{plan.trend_slope}\n"
        f"- daily_amplitude：{plan.daily_amplitude}\n"
        f"- weekly_enabled：{plan.weekly_enabled}\n"
        f"- weekly_amplitude：{plan.weekly_amplitude}\n"
        f"- noise_sigma：{plan.noise_sigma}\n"
        f"- lower_bound：{plan.lower_bound}\n"
        f"- domain_params：\n{params}\n\n"
        f"4. Output semantics and mathematical mechanism\n"
        f"- semantic_type：{plan.semantic_type}\n"
        f"- semantic_config：\n{semantic_params}\n"
        f"- output_constraints：\n{constraints}\n\n"
        f"5. Anomaly generation logic\n"
        f"- anomaly_enabled：{plan.anomaly_enabled}\n"
        f"- anomaly_severity：{plan.anomaly_severity}\n"
        f"- anomaly_target：{plan.anomaly_target}\n"
        f"- anomaly_count：{plan.anomaly_count}\n"
        f"- anomaly_kind：{plan.anomaly_kind}\n"
        f"- anomaly_magnitude：{plan.anomaly_magnitude}\n"
        f"- anomaly_width：{plan.anomaly_width}\n"
        f"- The generator injects anomalies according to these parameters and marks them in the anomaly column.\n\n"
        f"6. Output fields\n"
        f"- timestamp: timestamp\n"
        f"- value: generated value\n"
        f"- anomaly: anomaly flag, 0 for normal and 1 for anomalous\n"
        f"- unit: unit\n"
        f"- domain: domain\n"
        f"- generator_type: generation mechanism\n"
    )


def explain_generation(
    description: str,
    plan: SeriesPlan,
    length: int,
    freq: str,
    start: str,
    model: str | None = DEFAULT_LLM_MODEL,
) -> str:
    if not model or not os.getenv("OPENAI_API_KEY"):
        return _fallback_explanation(description, plan, length, freq, start)

    try:
        from .llm_client import create_openai_client

        client = create_openai_client()
        prompt = (
            "Explain the time-series generation logic and intermediate steps in English. Requirements:\n"
            "1. Explain how the user request and domain are understood.\n"
            "2. Explain why the selected generator_type is suitable.\n"
            "3. Explain how the main feature parameters affect the series shape.\n"
            "4. Explain the mathematical mechanism represented by semantic_type.\n"
            "5. Explain anomaly injection logic, target, and severity, and state that anomaly parameters are guided by the LLM from domain semantics.\n"
            "6. Explain output constraint validation and final Arrow fields.\n"
            "Do not output code.\n\n"
            f"User description: {description}\n"
            f"length={length}, freq={freq}, start={start}\n"
            f"SeriesPlan JSON:\n{json.dumps(plan.to_dict(), ensure_ascii=False, indent=2)}"
        )
        response = client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=1600,
        )
        text = getattr(response, "output_text", "") or ""
        return text.strip() or _fallback_explanation(description, plan, length, freq, start)
    except Exception as exc:
        fallback = _fallback_explanation(description, plan, length, freq, start)
        return f"{fallback}\n\nLLM explanation failed; local explanation was used. Error type: {exc.__class__.__name__}\nError message: {exc}\n"
