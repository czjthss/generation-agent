from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from generation_agent.agent import GenerationAgent


DESCRIPTIONS = [
    *[
        (
            "Synthesize a long univariate observational series related to Australian rainfall entirely "
            "from scratch, without reading or fitting any real data. "
            f"This is independent location or variable variant {i + 1}. Rainfall should contain many "
            "zero values, clustered wet events, positively skewed nonzero intensity, and a few storms. "
            "Do not model rainfall with a smooth sine wave. Use the intermittent_event generator and "
            "instantaneous semantics. In semantic_config, specify dry_spell_mean, wet_spell_mean, "
            "intensity_shape, intensity_scale, storm_probability, and storm_multiplier."
        )
        for i in range(7)
    ],
    *[
        (
            "Synthesize a long univariate environmental monitoring signal related to Beijing air "
            "quality and PM2.5 entirely from scratch, without reading or fitting any real data. "
            f"This is independent monitoring variable or station variant {i + 1}. The series should be "
            "nonnegative and strongly autocorrelated, with a slowly varying meteorological background, "
            "pollution regime changes, short accumulation episodes, and a few severe pollution events. "
            "Cycles represent only latent structure in index space and must not be interpreted as a real "
            "sampling frequency. Use a smooth_environmental base process with regime_switching semantics. "
            "In semantic_config, specify correlation_width, regime_width, regime_amplitude, short_cycle, "
            "long_cycle, and explicit anomaly parameters."
        )
        for i in range(7)
    ],
    *[
        (
            "Synthesize a long univariate series of airborne benzene concentration and related sensor "
            "measurements entirely from scratch, without reading or fitting any real data. "
            f"This is independent sensor or variable variant {i + 1}. The series should have a "
            "nonnegative background concentration, correlated noise, activity-driven variation, "
            "occasional emission peaks, and a few brief sensor dropouts. Cycles represent only latent "
            "structure in index space and must not be interpreted as a real sampling frequency. Use a "
            "smooth_environmental base process with regime_switching semantics. In semantic_config, "
            "specify correlation_width, activity_cycle, sensor_dropout_count, and explicit anomaly parameters."
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
