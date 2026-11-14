"""Portable runtime and durable task commands, separate from legacy handlers."""
from __future__ import annotations

import json
import os
from contextlib import nullcontext
from pathlib import Path

from .bundles import export_bundle, validate_task
from .journal import RunJournal


def _print(data):
    print(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))


def doctor(args):
    from . import __version__
    from .runtime_config import get_context
    from .providers import get_adapter
    context = get_context()
    result = {"version": __version__, "status": "configuration_valid", "model_calls": 0,
              "model_endpoint_health": "not_probed", "offline_enforcement": "not_attested"}
    if context:
        for role in context.config.roles:
            context.resolve(role)
        result["runtime"] = context.config.to_dict()
        result["adapter_capabilities"] = {n: get_adapter(m.provider).capabilities()
                                          for n, m in context.config.models.items()}
    if args.execution:
        from .executor_factory import build_executor
        result["execution"] = build_executor(image=args.execution_image).preflight()
    if args.probe:
        if context is None:
            raise ValueError("Live probes require an explicit bounded runtime profile")
        from .certification import certify_runtime
        result["certification"] = certify_runtime(context.config, roles=args.role,
            output_path=args.probe_output, probe_max_tokens=args.max_probe_tokens, allow_cloud=args.allow_cloud)
    _print(result)
    if args.probe:
        return 0 if result["certification"]["status"] == "passed" else 1


def validate(args):
    manifest = validate_task(args.task, require_executable=args.executable)
    _print({"schema_valid": True, "task_id": manifest["task_id"],
            "readiness": manifest["readiness"], "pinned_files": len(manifest["files"]),
            "scientific_validity": "requires_curator_assessment"})


def export(args):
    result = export_bundle(args.task, args.output, audience=args.audience)
    _print({"manifest": str(result), "audience": args.audience})


def execute(args):
    from .workflow import execute as run
    report = run(args.task, evaluation_bundle=args.evaluation_bundle,
                 runs_dir=args.runs_dir, image=args.execution_image, evaluation_profile=args.evaluation_profile,
                 seed=args.seed)
    _print(report)
    return 0 if report["pass"] else 1


def status(args):
    _print(RunJournal(args.runs_dir).get(args.run))


def cancel(args):
    state = RunJournal(args.runs_dir).request_cancel(args.run)
    _print({"run_id": args.run, "state": state,
            "note": "Active controller confirms termination; a stopped controller requires reconciliation"})


def resume(args):
    from .workflow import resume as run
    report = run(args.run, runs_dir=args.runs_dir)
    _print(report)
    return 0 if report["pass"] else 1


def reproduce(args):
    from .workflow import reproduce as run
    report = run(args.run, runs_dir=args.runs_dir)
    _print(report)
    return 0 if report["pass"] else 1


def reconcile(args):
    from .workflow import reconcile as run
    _print(run(args.run, runs_dir=args.runs_dir, terminate=args.terminate))


def index_paper(args):
    from .evidence import index_paper as index
    from .bundles import atomic_json
    data = index(args.pdf, max_span_chars=args.span_chars)
    atomic_json(args.output, data)
    _print({"index": args.output, "source_sha256": data["source"]["sha256"],
            "spans": len(data["spans"]), "scientific_readiness": "not_assessed"})


def select_context(args):
    from .evidence import load_index
    from .paper_context import select_context as select
    from .bundles import atomic_json
    result = select(load_index(args.index), args.query, budget_bytes=args.budget_bytes)
    atomic_json(args.output, result)
    _print({"selection": args.output, "coverage": result["coverage"], "readiness": result["readiness"]})


def extract_evidence(args):
    from .evidence import load_index
    from .paper_context import extract_requirements
    from .bundles import atomic_json
    result = extract_requirements(load_index(args.index), model=args.model, checkpoint_path=args.checkpoint,
        chunk_bytes=args.chunk_bytes, max_chunks=args.max_chunks, max_tokens=args.max_tokens,
        retry_failed=args.retry_failed)
    atomic_json(args.output, result)
    _print({"requirements": args.output, "scientific_readiness": "needs_review"})


def validate_protocol(args):
    from .bundles import read_json, validate_task
    from .evaluation.science import freeze_protocol
    manifest = validate_task(args.task, require_executable=True)
    data = read_json(args.protocol)
    frozen = freeze_protocol(data.get("protocol", data), task_manifest=manifest,
                             submission_sha256=manifest["files"][manifest["entrypoint"]])
    _print({"schema_valid": True, "protocol_sha256": frozen.sha256,
            "profile": frozen.as_dict()["profile"], "scope": "contract_validation_not_independent_scientific_review"})


def calibrate(args):
    from .bundles import read_json, atomic_json
    from .evaluation.calibration import calibration_report
    data = read_json(args.records)
    records = data.get("records") if isinstance(data, dict) else data
    report = calibration_report(records, judge_id=args.judge_id, protocol_sha256=args.protocol_sha256,
                                min_labeled=args.min_labeled)
    atomic_json(args.output, report)
    _print(report)


def register(subparsers):
    doctor_p = subparsers.add_parser("doctor", help="Validate runtime settings without paid inference")
    doctor_p.add_argument("--execution", action="store_true", help="Also preflight isolated execution")
    doctor_p.add_argument("--probe", action="store_true", help="Explicitly invoke bounded public-fixture model probes")
    doctor_p.add_argument("--allow-cloud", action="store_true", help="Permit explicitly credential-bound nonlocal probes")
    doctor_p.add_argument("--role", action="append", default=None)
    doctor_p.add_argument("--max-probe-tokens", type=int, default=32)
    doctor_p.add_argument("--probe-output", default=None)
    doctor_p.set_defaults(func=doctor)
    v = subparsers.add_parser("validate", help="Check bounded task schema and pinned files")
    v.add_argument("--task", required=True)
    v.add_argument("--executable", action="store_true")
    v.set_defaults(func=validate)
    ex = subparsers.add_parser("export", help="Export an explicit candidate or curator bundle")
    ex.add_argument("--task", required=True)
    ex.add_argument("--audience", required=True, choices=["candidate", "curator"])
    ex.add_argument("--output", required=True)
    ex.set_defaults(func=export)
    run = subparsers.add_parser("execute", help="Execute a pinned submission in isolation")
    run.add_argument("--task", required=True)
    run.add_argument("--evaluation-bundle")
    run.add_argument("--evaluation-profile", default="execution/v1", choices=["execution/v1", "method_output/v1", "reproduction/v1", "transfer/v1"])
    run.add_argument("--seed", type=int, default=None)
    run.set_defaults(func=execute)
    for name, function in (("status", status), ("cancel", cancel), ("resume", resume), ("reproduce", reproduce)):
        p = subparsers.add_parser(name, help=f"{name.title()} a durable local run")
        p.add_argument("--run", required=True)
        p.set_defaults(func=function)
    reconciliation = subparsers.add_parser("reconcile", help="Inspect or explicitly terminate controller-recorded orphan jobs")
    reconciliation.add_argument("--run", required=True)
    reconciliation.add_argument("--terminate", action="store_true")
    reconciliation.set_defaults(func=reconcile)
    evidence = subparsers.add_parser("index", help="Index PDF pages, sections and source spans")
    evidence.add_argument("--pdf", required=True)
    evidence.add_argument("--span-chars", type=int, default=6000)
    evidence.add_argument("-o", "--output", required=True)
    evidence.set_defaults(func=index_paper)
    selection = subparsers.add_parser("select-context", help="Select bounded source context with explicit coverage")
    selection.add_argument("--index", required=True)
    selection.add_argument("--query", default="")
    selection.add_argument("--budget-bytes", type=int, default=24000)
    selection.add_argument("-o", "--output", required=True)
    selection.set_defaults(func=select_context)
    extraction = subparsers.add_parser("evidence-extract", help="Extract quote-grounded requirements in checkpointed chunks")
    extraction.add_argument("--index", required=True)
    extraction.add_argument("--model", default="gpt-4o")
    extraction.add_argument("--checkpoint", required=True)
    extraction.add_argument("--chunk-bytes", type=int, default=12000)
    extraction.add_argument("--max-chunks", type=int, default=64)
    extraction.add_argument("--max-tokens", type=int, default=2048)
    extraction.add_argument("--retry-failed", action="store_true")
    extraction.add_argument("-o", "--output", required=True)
    extraction.set_defaults(func=extract_evidence)
    protocol = subparsers.add_parser("validate-protocol", help="Validate a source-bound frozen scientific contract")
    protocol.add_argument("--task", required=True)
    protocol.add_argument("--protocol", required=True)
    protocol.set_defaults(func=validate_protocol)
    calibration = subparsers.add_parser("calibrate", help="Compare recorded judge verdicts with expert labels")
    calibration.add_argument("--records", required=True)
    calibration.add_argument("--judge-id", required=True)
    calibration.add_argument("--protocol-sha256", required=True)
    calibration.add_argument("--min-labeled", type=int, default=30)
    calibration.add_argument("-o", "--output", required=True)
    calibration.set_defaults(func=calibrate)
    for name, p in subparsers.choices.items():
        p.add_argument("--runtime-config", default=None, help="Explicit role/provider/policy configuration")
        p.add_argument("--model-runs-dir", default="./runs", help="Durable model budget journal directory")
        p.add_argument("--offline", action="store_true", help="Require local-only policy; no cloud fallback")
        if name in {"doctor", "execute", "verify", "run"}:
            p.add_argument("--execution-image", default=None, help="Staged trusted execution image (never pulled automatically)")
        if name in {"execute", "status", "cancel", "resume", "reproduce", "reconcile"}:
            p.add_argument("--runs-dir", default="./runs")
        if name in {"verify", "run"}:
            p.add_argument("--code-timeout", type=int, default=300)
    subparsers.choices["verify"].add_argument("--data-file", action="append", default=None,
                                              help="Explicit relative input allowlist (repeatable)")


def dispatch(args):
    from .runtime_config import RuntimeConfigError, get_context, load_runtime_config, runtime_scope
    source = args.runtime_config or os.environ.get("PAPER2BENCH_RUNTIME_CONFIG")
    scope = nullcontext(get_context())
    if source:
        config = load_runtime_config(source)
        if args.offline and config.policy["data_egress"] != "local_only":
            raise RuntimeConfigError("--offline requires a local_only runtime profile")
        # Offline describes a restrictive routing policy. Verified egress attestation is
        # unavailable to this CLI and fails closed if required by the supplied profile.
        if args.command in {"parse", "classify", "extract", "plan", "specextract", "generate", "verify", "evidence-extract"} or (args.command == "run" and args.split_rqs) or (args.command == "doctor" and args.probe):
            from .model_runs import durable_runtime_scope
            scope = durable_runtime_scope(config, runs_dir=args.model_runs_dir)
        else:
            scope = runtime_scope(config)
    elif args.offline:
        raise RuntimeConfigError("--offline requires an explicit local runtime configuration")
    with scope:
        if source:
            context = get_context()
            for role in context.config.roles:
                context.resolve(role)
        if getattr(args, "verify", False):
            from .cli import _executor
            _executor(args).preflight()
        return args.func(args)
