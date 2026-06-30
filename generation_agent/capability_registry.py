from __future__ import annotations

import ast
import inspect
from typing import Any, Callable


def _tree(callable_obj: Callable[..., Any]) -> ast.AST:
    return ast.parse(inspect.getsource(callable_obj))


def _signature_fields(callable_obj: Callable[..., Any]) -> list[str]:
    return [
        name
        for name in inspect.signature(callable_obj).parameters
        if name not in {"description", "plan", "values", "rng", "length", "columns"}
    ]


def _get_keys(callable_obj: Callable[..., Any], receiver_names: set[str]) -> list[str]:
    keys: set[str] = set()
    for node in ast.walk(_tree(callable_obj)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "get" or not isinstance(node.func.value, ast.Name):
            continue
        if node.func.value.id not in receiver_names or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            keys.add(first.value)
    return sorted(keys)


def _component_feature_families(callable_obj: Callable[..., Any]) -> list[str]:
    families: set[str] = set()
    for node in ast.walk(_tree(callable_obj)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "_component" or len(node.args) < 8:
            continue
        family = node.args[7]
        if isinstance(family, ast.Constant) and isinstance(family.value, str):
            families.add(family.value)
    return sorted(families)


def _constant_text(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _component_execution_contracts(
    callable_obj: Callable[..., Any],
) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    for node in ast.walk(_tree(callable_obj)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "_component" or len(node.args) < 8:
            continue
        family = _constant_text(node.args[7])
        if family is None:
            continue
        bindings = {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in node.keywords
            if keyword.arg is not None
        }
        contracts.append(
            {
                "component": _constant_text(node.args[0]) or "dynamic",
                "component_semantic": _constant_text(node.args[2]) or "dynamic",
                "value_role": _constant_text(node.args[3]) or "dynamic",
                "time_behavior": _constant_text(node.args[5]) or "dynamic",
                "statistical_shape": _constant_text(node.args[6]) or "dynamic",
                "feature_family": family,
                "parameter_bindings": bindings,
            }
        )
    return sorted(contracts, key=lambda item: (item["feature_family"], item["component"]))


def _compared_string_values(
    callable_obj: Callable[..., Any], object_name: str, attribute_name: str
) -> list[str]:
    values: set[str] = set()
    for node in ast.walk(_tree(callable_obj)):
        if not isinstance(node, ast.Compare):
            continue
        left = node.left
        if not (
            isinstance(left, ast.Attribute)
            and left.attr == attribute_name
            and isinstance(left.value, ast.Name)
            and left.value.id == object_name
        ):
            continue
        for comparator in node.comparators:
            if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                values.add(comparator.value)
    return sorted(values)


def _planning_tool_names() -> list[str]:
    from .langchain_agent import _build_tools

    names: set[str] = set()
    for node in ast.walk(_tree(_build_tools)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if any(
            isinstance(decorator, ast.Name) and decorator.id == "tool"
            for decorator in node.decorator_list
        ):
            names.add(node.name)
    return sorted(names)


def build_pipeline_capability_manifest() -> dict[str, Any]:
    """Describe the installed pipeline from code, not domain-specific prompt rules."""
    from .component_workflow import build_component_workflow
    from .feature_composer import GENERATOR_TYPES, build_feature_plan
    from .features import AnomalyConfig, add_anomalies
    from .plan_normalizer import (
        EXECUTABLE_CONSTRAINT_KEYS,
        EXECUTABLE_DOMAIN_PARAM_KEYS,
        VALID_GENERATORS,
        VALID_SEMANTICS,
    )
    from .semantic_transforms import apply_semantic_process
    from .semantic_types import (
        ANOMALY_KIND_ALIASES,
        ANOMALY_TARGETS_BY_SEMANTIC,
        SEMANTIC_TYPES,
        SEVERITY_PRESETS,
        SUPPORTED_ANOMALY_KINDS,
    )
    from .semantic_validators import validate_and_repair
    from .specialist_workflow import (
        CHALLENGE_CONTRACT,
        PROCESS_CONTRACT,
        REFERENCE_CONTRACT,
        SPECIFICATION_CONTRACT,
    )

    return {
        "manifest_source": "runtime_code_introspection",
        "upstream_analysis": {
            "specification_agent_outputs": sorted(SPECIFICATION_CONTRACT),
            "process_architect_outputs": sorted(PROCESS_CONTRACT),
            "domain_challenger_outputs": sorted(CHALLENGE_CONTRACT),
            "reference_interpreter_outputs": sorted(REFERENCE_CONTRACT),
        },
        "plan_compilation": {
            "tools": _planning_tool_names(),
            "accepted_plan_fields": _signature_fields(build_feature_plan),
            "normalizer_generators": sorted(VALID_GENERATORS),
            "normalizer_semantics": sorted(VALID_SEMANTICS),
            "normalizer_domain_parameters": sorted(EXECUTABLE_DOMAIN_PARAM_KEYS),
            "normalizer_constraints": sorted(EXECUTABLE_CONSTRAINT_KEYS),
        },
        "feature_generation": {
            "generator_types": sorted(GENERATOR_TYPES),
            "multivariate_generation": {
                "plan_fields": ["variables", "relationships"],
                "variable_roles": ["target", "driver", "state", "auxiliary"],
                "relationship_operators": [
                    "linear_lag",
                    "threshold",
                    "piecewise",
                    "saturation",
                    "state_gate",
                    "event_trigger",
                ],
                "event_trigger_ops": ["gte", "lte", "eq"],
                "driver_anomaly_policy": "driver anomalies modify a concrete driver column first, then propagate through declared relationships",
                "relationship_audit": "local kernel reports best-lag correlation, sign check, and pass/fail per relationship",
            },
            "component_feature_families": _component_feature_families(
                build_component_workflow
            ),
            "component_execution_contracts": _component_execution_contracts(
                build_component_workflow
            ),
            "component_parameter_keys": _get_keys(
                build_component_workflow, {"params"}
            ),
        },
        "semantic_transforms": {
            "semantic_types": sorted(SEMANTIC_TYPES),
            "semantic_config_keys": _get_keys(
                apply_semantic_process, {"params"}
            ),
        },
        "anomaly_injection": {
            "strategy_fields": sorted(AnomalyConfig.__dataclass_fields__),
            "implemented_kinds": sorted(SUPPORTED_ANOMALY_KINDS),
            "kind_aliases": dict(ANOMALY_KIND_ALIASES),
            "targets_by_semantic": {
                key: list(value) for key, value in ANOMALY_TARGETS_BY_SEMANTIC.items()
            },
            "execution_contract": {
                "position_selection": "random positions without replacement",
                "windowing": "width defines the affected [start, end) window for every implemented kind",
                "magnitude_scale": "magnitude is standardized by the input-series standard deviation",
                "flags": "every affected point is marked in the anomaly flag array",
            },
            "severity_presets": sorted(SEVERITY_PRESETS),
        },
        "deterministic_validation": {
            "constraint_keys": _get_keys(
                validate_and_repair, {"constraints"}
            ),
            "semantic_identity_checks": ["cumulative", "stock_flow"],
        },
        "execution_order": [
            "upstream_analysis",
            "plan_compilation",
            "reflection",
            "component_generation",
            "semantic_transform",
            "anomaly_injection",
            "deterministic_validation",
            "quality_evaluation",
        ],
    }


def validate_plan_against_capabilities(plan: dict[str, Any]) -> dict[str, Any]:
    """Check only machine-verifiable compatibility with the installed pipeline."""
    from .features import AnomalyConfig
    from .plan_normalizer import (
        EXECUTABLE_CONSTRAINT_KEYS,
        EXECUTABLE_DOMAIN_PARAM_KEYS,
        SEMANTIC_CONFIG_KEYS,
        VALID_GENERATORS,
        VALID_SEMANTICS,
    )
    from .semantic_types import ANOMALY_KIND_ALIASES, SUPPORTED_ANOMALY_KINDS

    issues: list[dict[str, str]] = []

    def add(path: str, message: str) -> None:
        issues.append({"path": path, "message": message})

    generator = str(plan.get("generator_type", ""))
    if generator not in VALID_GENERATORS:
        add("generator_type", f"unsupported generator_type: {generator}")

    semantic = str(plan.get("semantic_type", ""))
    if semantic not in VALID_SEMANTICS:
        add("semantic_type", f"unsupported semantic_type: {semantic}")

    domain_params = plan.get("domain_params", {})
    if not isinstance(domain_params, dict):
        add("domain_params", "domain_params must be an object")
    else:
        for key in domain_params:
            if key not in EXECUTABLE_DOMAIN_PARAM_KEYS:
                add(f"domain_params.{key}", "parameter is not consumed by the local kernel")

    semantic_config = plan.get("semantic_config", {})
    if not isinstance(semantic_config, dict):
        add("semantic_config", "semantic_config must be an object")
    else:
        allowed = SEMANTIC_CONFIG_KEYS.get(semantic, set())
        for key in semantic_config:
            if key not in allowed:
                add(
                    f"semantic_config.{key}",
                    f"parameter is not supported for semantic_type {semantic}",
                )

    constraints = plan.get("output_constraints", {})
    if not isinstance(constraints, dict):
        add("output_constraints", "output_constraints must be an object")
    else:
        for key in constraints:
            if key not in EXECUTABLE_CONSTRAINT_KEYS:
                add(f"output_constraints.{key}", "constraint has no deterministic validator")

    anomaly_kinds = set(SUPPORTED_ANOMALY_KINDS) | set(ANOMALY_KIND_ALIASES)
    anomaly_kind = str(plan.get("anomaly_kind", "spike"))
    if anomaly_kind not in anomaly_kinds:
        add("anomaly_kind", f"unsupported anomaly kind: {anomaly_kind}")
    for field, minimum in (("anomaly_count", 0), ("anomaly_width", 1)):
        value = plan.get(field, minimum)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            add(field, f"{field} must be an integer >= {minimum}")
    magnitude = plan.get("anomaly_magnitude", 0.0)
    if isinstance(magnitude, bool) or not isinstance(magnitude, (int, float)) or magnitude < 0:
        add("anomaly_magnitude", "anomaly_magnitude must be a nonnegative number")

    return {
        "passed": not issues,
        "issues": issues,
        "invalid_paths": [issue["path"] for issue in issues],
        "validator": "local_deterministic_capability_validator",
        "anomaly_strategy_fields": sorted(AnomalyConfig.__dataclass_fields__),
    }
