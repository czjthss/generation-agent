from generation_agent.cli import build_parser


def test_cli_exposes_only_simplified_reference_and_anomaly_controls():
    parser = build_parser()
    option_names = {option for action in parser._actions for option in action.option_strings}
    assert "--reference" in option_names
    assert "--anomalies" in option_names
    assert "--anomaly-severity" in option_names

    removed = {
        "--no-llm",
        "--agent-mode",
        "--reference-strength",
        "--reference-time-column",
        "--reference-value-column",
        "--anomaly-count",
        "--anomaly-magnitude",
        "--anomaly-width",
        "--anomaly-kind",
        "--anomaly-target",
        "--anomaly-direction",
        "--code-output",
        "--explain-output",
        "--validation-output",
        "--print-plan",
        "--print-validation",
    }
    assert option_names.isdisjoint(removed)
