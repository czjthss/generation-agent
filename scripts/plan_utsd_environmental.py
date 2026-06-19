from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from generation_agent.agent import GenerationAgent


DESCRIPTIONS = [
    *[
        (
            "完全从零合成一条澳大利亚降雨相关的长单变量观测序列，不读取或拟合任何真实数据。"
            f"这是第{i + 1}个独立地点或变量变体。降雨应为大量零值、成簇湿润事件、偏态正降雨强度和少量暴雨事件，"
            "不得使用平滑正弦波模拟降雨。使用intermittent_event和instantaneous语义。"
            "请在semantic_config中给出dry_spell_mean、wet_spell_mean、intensity_shape、"
            "intensity_scale、storm_probability、storm_multiplier。"
        )
        for i in range(7)
    ],
    *[
        (
            "完全从零合成一条北京空气质量与PM2.5相关的长单变量环境监测信号，不读取或拟合任何真实数据。"
            f"这是第{i + 1}个独立监测变量或站点变体。应体现非负、强自相关、缓慢气象背景、污染状态切换、"
            "短时污染积累和少量严重污染过程；周期只表示索引空间潜在结构，不解释为真实采样频率。"
            "使用smooth_environmental基础过程和regime_switching语义。"
            "请在semantic_config中给出correlation_width、regime_width、regime_amplitude、"
            "short_cycle、long_cycle，并明确异常参数。"
        )
        for i in range(7)
    ],
    *[
        (
            "完全从零合成一条空气中苯浓度及相关传感器测量的长单变量序列，不读取或拟合任何真实数据。"
            f"这是第{i + 1}个独立传感器或变量变体。应体现非负背景浓度、相关噪声、活动相关变化、"
            "偶发排放峰和少量传感器短时跌落；周期只表示索引空间潜在结构，不解释为真实采样频率。"
            "使用smooth_environmental基础过程和regime_switching语义。"
            "请在semantic_config中给出correlation_width、activity_cycle、sensor_dropout_count，"
            "并明确异常参数。"
        )
        for i in range(6)
    ],
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    agent = GenerationAgent(model=args.model) if args.model else GenerationAgent()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        payload = existing if isinstance(existing, list) else []
    else:
        payload = []
    if len(payload) > len(DESCRIPTIONS):
        raise ValueError("Checkpoint contains more plans than expected")
    print(f"Resuming from {len(payload)}/{len(DESCRIPTIONS)} plans", flush=True)

    for index, description in enumerate(DESCRIPTIONS[len(payload) :], start=len(payload)):
        plan = None
        planner = ""
        for attempt in range(3):
            print(
                f"[{index + 1:02d}/{len(DESCRIPTIONS)}] LLM planning attempt {attempt + 1}/3...",
                flush=True,
            )
            candidate = agent.plan(description)
            candidate_planner = str(candidate.metadata.get("planner", ""))
            if candidate_planner in {"langchain_agent", "llm"}:
                plan = candidate
                planner = candidate_planner
                break
            planner = candidate_planner
            time.sleep(2.0 * (attempt + 1))
        if plan is None:
            raise RuntimeError(
                f"Sequence {index} was not planned by an LLM after 3 attempts "
                f"(last planner={planner!r}). Generation is stopped."
            )
        plan.metadata["synthetic_only"] = True
        plan.metadata["reference_data_used"] = False
        plan.metadata["sequence_index"] = index
        payload.append(plan.to_dict())
        print(
            f"[{index + 1:02d}/{len(DESCRIPTIONS)}] "
            f"{plan.domain} / {plan.generator_type} / {plan.semantic_type}",
            flush=True,
        )
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(payload)} LLM plans to {output}", flush=True)


if __name__ == "__main__":
    main()
