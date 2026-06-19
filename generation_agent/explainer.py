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
        f"任务描述：{description}\n\n"
        f"1. 领域识别\n"
        f"- 领域：{plan.domain}\n"
        f"- 单位：{plan.unit}\n"
        f"- 生成机制：{plan.generator_type}\n\n"
        f"2. 时间范围\n"
        f"- 起始时间：{start}\n"
        f"- 序列长度：{length}\n"
        f"- 时间粒度：{freq}\n\n"
        f"3. 主要生成参数\n"
        f"- baseline：{plan.baseline}\n"
        f"- trend_slope：{plan.trend_slope}\n"
        f"- daily_amplitude：{plan.daily_amplitude}\n"
        f"- weekly_enabled：{plan.weekly_enabled}\n"
        f"- weekly_amplitude：{plan.weekly_amplitude}\n"
        f"- noise_sigma：{plan.noise_sigma}\n"
        f"- lower_bound：{plan.lower_bound}\n"
        f"- domain_params：\n{params}\n\n"
        f"4. 输出语义和数学机制\n"
        f"- semantic_type：{plan.semantic_type}\n"
        f"- semantic_config：\n{semantic_params}\n"
        f"- output_constraints：\n{constraints}\n\n"
        f"5. 异常值生成逻辑\n"
        f"- anomaly_enabled：{plan.anomaly_enabled}\n"
        f"- anomaly_severity：{plan.anomaly_severity}\n"
        f"- anomaly_target：{plan.anomaly_target}\n"
        f"- anomaly_count：{plan.anomaly_count}\n"
        f"- anomaly_kind：{plan.anomaly_kind}\n"
        f"- anomaly_magnitude：{plan.anomaly_magnitude}\n"
        f"- anomaly_width：{plan.anomaly_width}\n"
        f"- 生成器会根据这些参数在基础序列上注入异常，并在 anomaly 列标记为 1。\n\n"
        f"6. 输出字段\n"
        f"- timestamp：时间戳\n"
        f"- value：生成值\n"
        f"- anomaly：异常标记，0 表示正常，1 表示异常\n"
        f"- unit：单位\n"
        f"- domain：领域\n"
        f"- generator_type：生成机制\n"
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
        from openai import OpenAI

        client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL))
        prompt = (
            "请用中文解释本次时间序列数据的生成逻辑和中间步骤。要求：\n"
            "1. 说明如何理解用户需求和领域。\n"
            "2. 说明选择的 generator_type 为什么适合。\n"
            "3. 说明主要特征参数如何影响序列形态。\n"
            "4. 说明 semantic_type 对应的递推、累计、守恒、状态切换或滞后数学机制。\n"
            "5. 单独说明异常值注入逻辑、注入对象和强度，并指出异常参数由 LLM 根据领域语义给出。\n"
            "6. 说明输出约束校验结果和最终 CSV 字段。\n"
            "不要输出代码。\n\n"
            f"用户描述：{description}\n"
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
        return f"{fallback}\n\nLLM 说明生成失败，已使用本地说明。错误类型：{exc.__class__.__name__}\n错误信息：{exc}\n"
