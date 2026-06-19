from types import SimpleNamespace

from generation_agent.domain_rules import commit_candidate_rule, load_rules
from generation_agent.langchain_agent import _reflect_on_plan
from generation_agent.planner import SeriesPlan


class InvalidReflectionLLM:
    def invoke(self, messages):
        return SimpleNamespace(content="not-json")


def test_unparseable_reflection_fails_closed():
    result = _reflect_on_plan(InvalidReflectionLLM(), "生成负载", SeriesPlan())
    assert result["status"] == "REVISE"
    assert result["parser_fallback"] is True


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
