from types import SimpleNamespace

from generation_agent.specialist_workflow import GenerationSpecialistWorkflow


class FakeLLM:
    def __init__(self):
        self.payloads = []

    def invoke(self, messages):
        system = messages[0].content
        self.payloads.append(messages[-1].content)
        if "Specification Agent" in system:
            content = '''{"observable":"load","domain":"energy","unit":"kW","time_basis":"hourly observation","value_support":"nonnegative continuous","invariants":["nonnegative"],"conditions":[],"explicit_requirements":[],"assumptions":[],"ambiguities":[],"requires_user_input":false,"confidence":0.9}'''
        elif "Process Architect" in system:
            content = '''{"base_process":"demand process","temporal_dependence":{},"event_mechanism":{},"evolution_semantics":{},"scale":{},"trend":{},"seasonality":[],"noise_model":{},"relationships":[],"anomaly_intervention":{},"constraints":{"nonnegative":true},"mandatory_properties":["nonnegative"],"optional_properties":[],"implementation_risks":[],"assumptions":[],"confidence":0.8}'''
        else:
            content = '''{"verdict":"ACCEPT","contradictions":[],"unrealistic_assumptions":[],"missing_constraints":[],"unsupported_mechanisms":[],"reference_conflicts":[],"required_corrections":[],"confidence":0.9}'''
        return SimpleNamespace(content=content)


def test_specialists_build_and_challenge_a_process_design():
    llm = FakeLLM()
    evidence = GenerationSpecialistWorkflow(llm).analyze(
        "generate park load",
        reference_context={"reference_strength": "scale", "reference_profile": {}},
    )
    assert evidence.specification["observable"] == "load"
    assert evidence.process_design["base_process"] == "demand process"
    assert evidence.challenge["verdict"] == "ACCEPT"
    assert "specification" not in evidence.errors
    assert "process_design" not in evidence.errors
    assert "challenge" not in evidence.errors
    assert any("available_capabilities" in payload for payload in llm.payloads)
    assert any("reference_interpretation" in payload for payload in llm.payloads)
