from types import SimpleNamespace

from generation_agent.domain_rules import commit_candidate_rule, load_rules
from generation_agent.langchain_agent import (
    _extract_json_from_agent_result,
    _reflect_on_plan,
    _revise_plan_directly,
    audit_generated_series,
    decide_anomaly_strategy,
)
from generation_agent.planner import SeriesPlan
from generation_agent.semantic_types import AnomalyOverrides, apply_anomaly_strategy


class InvalidReflectionLLM:
    def invoke(self, messages):
        return SimpleNamespace(content="not-json")


class SoftWarningReflectionLLM:
    def invoke(self, messages):
        return SimpleNamespace(
            content='{"status":"REVISE","hard_errors":[],"soft_warnings":["optional realism"],"revision_instruction":"","confidence":0.8}'
        )


class UngroundedHardReflectionLLM:
    def invoke(self, messages):
        return SimpleNamespace(
            content='{"status":"REVISE","hard_errors":["feature is unsupported"],"hard_error_evidence":[],"soft_warnings":[],"revision_instruction":"change it","confidence":0.8}'
        )


class GroundedHardReflectionLLM:
    def invoke(self, messages):
        return SimpleNamespace(
            content='{"status":"REVISE","hard_errors":["requested cumulative total uses instantaneous semantics"],"hard_error_evidence":[{"failure_index":0,"sources":["original_request","proposed_plan"],"request_quote":"cumulative","plan_path":"semantic_type","capability_path":"","evidence":"request says cumulative while proposed_plan says instantaneous"}],"soft_warnings":[],"revision_instruction":"use cumulative semantics","confidence":0.9}'
        )


class AnomalyLLM:
    def invoke(self, messages):
        return SimpleNamespace(
            content='{"enabled":true,"reason":"equipment fault requested","target":"value","kind":"drop","severity":"high","count":2,"width":3,"magnitude":4.0,"constraints_after_injection":["nonnegative"],"confidence":0.9}'
        )


class DirectRevisionLLM:
    def invoke(self, messages):
        return SimpleNamespace(
            content='{"domain":"energy","generator_type":"cyclic_signal","unit":"kW","baseline":1200,"lower_bound":0,"semantic_type":"instantaneous","output_constraints":{"nonnegative":true}}'
        )


class UngroundedQualityLLM:
    def invoke(self, messages):
        return SimpleNamespace(
            content='{"status":"REVISE","hard_failures":["an invented three-shift schedule is missing"],"hard_failure_evidence":[],"soft_warnings":[],"revision_target":"generation","confidence":0.9}'
        )


class GroundedQualityLLM:
    def invoke(self, messages):
        return SimpleNamespace(
            content='{"status":"REVISE","hard_failures":["values violate the executable upper bound"],"hard_failure_evidence":[{"failure_index":0,"sources":["deterministic_validation"],"request_quote":"","plan_path":"","validation_path":"checks.upper_bound","execution_path":"","evidence":"upper_bound=false"}],"soft_warnings":[],"revision_target":"generation","confidence":0.9}'
        )


class PassWithWarningsQualityLLM:
    def invoke(self, messages):
        return SimpleNamespace(
            content='{"status":"PASS_WITH_WARNINGS","hard_failures":[],"hard_failure_evidence":[],"soft_warnings":["short horizon limits weekly assessment"],"revision_target":"none","confidence":0.8}'
        )


def test_unparseable_reflection_fails_closed():
    result = _reflect_on_plan(InvalidReflectionLLM(), "generate load", SeriesPlan())
    assert result["status"] == "REVISE"
    assert result["parser_fallback"] is True


def test_soft_reflection_warning_does_not_block_generation():
    result = _reflect_on_plan(SoftWarningReflectionLLM(), "generate load", SeriesPlan())
    assert result["status"] == "PASS"
    assert result["hard_errors"] == []
    assert result["soft_warnings"] == ["optional realism"]


def test_ungrounded_reflection_error_is_downgraded_to_warning():
    result = _reflect_on_plan(
        UngroundedHardReflectionLLM(), "generate load", SeriesPlan()
    )
    assert result["status"] == "PASS"
    assert result["hard_errors"] == []
    assert "Ungrounded reflection concern" in result["soft_warnings"][0]


def test_grounded_reflection_error_still_requires_revision():
    result = _reflect_on_plan(
        GroundedHardReflectionLLM(), "generate cumulative sales", SeriesPlan()
    )
    assert result["status"] == "REVISE"
    assert result["hard_errors"] == [
        "requested cumulative total uses instantaneous semantics"
    ]


def test_llm_anomaly_strategy_obeys_user_off_override():
    plan = SeriesPlan(anomaly_enabled=True)
    strategy = decide_anomaly_strategy(
        AnomalyLLM(), "generate load with equipment fault", plan, anomaly_control="off"
    )
    updated = apply_anomaly_strategy(
        plan, strategy, AnomalyOverrides(enabled=False)
    )
    assert not updated.anomaly_enabled
    assert updated.anomaly_count == 0
    assert updated.metadata["anomaly_strategy"]["reason"] == "equipment fault requested"


def test_anomaly_strategy_target_and_kind_are_normalized_to_executable_values():
    plan = SeriesPlan(semantic_type="instantaneous")
    strategy = SimpleNamespace(
        enabled=True,
        reason="temporary interruption",
        target="state",
        kind="temporary_outage",
        severity="medium",
        count=2,
        width=2,
        magnitude=2.0,
        constraints_after_injection=(),
        confidence=0.8,
        to_dict=lambda: {
            "enabled": True,
            "reason": "temporary interruption",
            "target": "state",
            "kind": "temporary_outage",
            "severity": "medium",
            "count": 2,
            "width": 2,
            "magnitude": 2.0,
            "constraints_after_injection": (),
            "confidence": 0.8,
        },
    )

    updated = apply_anomaly_strategy(plan, strategy)

    assert updated.anomaly_target == "value"
    assert updated.anomaly_kind == "drop"
    assert updated.metadata["anomaly_strategy"]["normalization"]["target_changed"] is True
    assert updated.metadata["anomaly_strategy"]["normalization"]["kind_changed"] is True


def test_direct_revision_fallback_is_still_llm_generated_plan_json():
    payload = _revise_plan_directly(
        DirectRevisionLLM(),
        "generate park load",
        SeriesPlan(),
        ["load must be nonnegative"],
        "add a nonnegative constraint",
    )
    assert payload["domain"] == "energy"
    assert payload["output_constraints"]["nonnegative"] is True


def test_json_extraction_supports_langchain_content_blocks():
    message = SimpleNamespace(
        content=[{"type": "text", "text": '{"status":"PASS"}'}]
    )
    assert _extract_json_from_agent_result({"messages": [message]}) == {
        "status": "PASS"
    }


def test_ungrounded_quality_failure_is_downgraded_to_warning():
    result = audit_generated_series(
        UngroundedQualityLLM(),
        "generate industrial park load",
        SeriesPlan(),
        {"anomaly_fraction": 0.0},
        {"passed": True},
    )
    assert result["status"] == "PASS_WITH_WARNINGS"
    assert result["hard_failures"] == []
    assert "Ungrounded quality concern" in result["soft_warnings"][0]


def test_grounded_quality_failure_still_requires_regeneration():
    result = audit_generated_series(
        GroundedQualityLLM(),
        "generate a bounded series",
        SeriesPlan(output_constraints={"upper_bound": 100}),
        {"anomaly_fraction": 0.0},
        {"passed": False, "checks": {"upper_bound": False}},
    )
    assert result["status"] == "REGENERATE"
    assert result["hard_failures"] == ["values violate the executable upper bound"]


def test_quality_parser_accepts_pass_with_warnings_status():
    result = audit_generated_series(
        PassWithWarningsQualityLLM(),
        "generate hourly load",
        SeriesPlan(),
        {"anomaly_fraction": 0.0},
        {"passed": True},
    )
    assert result["status"] == "PASS_WITH_WARNINGS"
    assert result["hard_failures"] == []


def test_candidate_rule_commits_only_after_both_reviews_pass(tmp_path, monkeypatch):
    import generation_agent.domain_rules as rules

    monkeypatch.setattr(rules, "RULES_PATH", tmp_path / "rules.json")
    plan = SeriesPlan(
        domain="new-domain",
        metadata={"candidate_domain_rule": {"domain": "new-domain", "keywords": ["new"]}},
    )
    assert not commit_candidate_rule(plan, {"passed": True}, {"status": "UNVERIFIED"})
    assert load_rules() == []

    plan.metadata["candidate_rule_status"] = "pending_validation"
    assert commit_candidate_rule(plan, {"passed": True}, {"status": "PASS"})
    assert load_rules()[0]["domain"] == "new-domain"
