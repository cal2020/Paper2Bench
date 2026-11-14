"""Paper2Bench public API with lazy optional dependencies."""

from importlib import import_module

__version__ = "0.4.0a2"

_EXPORTS = {
    "run": ("api", "run"),
    "BenchmarkResult": ("api", "BenchmarkResult"),
    "parse_paper_to_tree": ("tree_parser", "parse_paper_to_tree"),
    "classify_paper": ("paper_classifier", "classify_paper"),
    "extract_yaml_from_pdf": ("instruction_generator", "extract_yaml_from_pdf"),
    "render_instruction": ("instruction_generator", "render_instruction"),
    "generate_supplementary_plan": ("plan_generator", "generate_supplementary_plan"),
    "build_instruction_gt": ("plan_generator", "build_instruction_gt"),
    "verify_instruction_gt": ("verify", "verify_instruction_gt"),
    "extract_paper_spec": ("paper_spec", "extract_paper_spec"),
    "generate_variants": ("benchmark_generator", "generate_variants"),
    "chat": ("llm", "chat"),
    "RuntimeConfig": ("runtime_config", "RuntimeConfig"),
    "runtime_scope": ("runtime_config", "runtime_scope"),
    "execute": ("workflow", "execute"),
    "validate": ("bundles", "validate_task"),
    "export_bundle": ("bundles", "export_bundle"),
    "resume": ("workflow", "resume"),
    "reproduce": ("workflow", "reproduce"),
    "reconcile": ("workflow", "reconcile"),
    "resume_generation": ("api", "resume_generation"),
    "index_paper": ("evidence", "index_paper"),
    "select_context": ("paper_context", "select_context"),
    "extract_requirements": ("paper_context", "extract_requirements"),
    "freeze_protocol": ("evaluation.science", "freeze_protocol"),
    "evaluate_protocol": ("evaluation.science", "evaluate_protocol"),
    "calibration_report": ("evaluation.calibration", "calibration_report"),
    "certify_runtime": ("certification", "certify_runtime"),
    "ModelBroker": ("model_broker", "ModelBroker"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(f".{module}", __name__), attribute)
    globals()[name] = value
    return value
