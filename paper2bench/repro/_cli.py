"""Small CLI over the same execution core used by programmatic clients."""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

from pydantic import ValidationError

from .authoring import draft_experiment, ingest_source
from .bundles import export_bundle, replay_bundle
from .demo import create_demo, create_optimization_demo
from .engine import Engine
from .models import ExperimentSpec
from .providers import ProviderClient, ProviderProfile
from .runtime import DockerRunner


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="paper2bench-repro",
        description="Source-grounded experiments and trusted numerical evaluation",
    )
    result.add_argument("--state", type=Path, default=Path(".paper2bench-repro"))
    sub = result.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="Normalize a local Markdown, text, or text-bearing PDF")
    ingest.add_argument("path", type=Path)
    ingest.add_argument(
        "--parser-image", help="Pinned isolated parser image; env PAPER2BENCH_PARSER_IMAGE"
    )
    prepare = sub.add_parser(
        "prepare-parser", help="Stage a no-network PDF parser image from installed dependencies"
    )
    prepare.add_argument("--base-image", default="python:3.12-slim")
    dataset = sub.add_parser("dataset", help="Validate and register a train/test JSON dataset")
    dataset.add_argument("path", type=Path)
    profile = sub.add_parser(
        "profile", help="Register a model endpoint profile without credentials"
    )
    profile.add_argument("path", type=Path)
    draft = sub.add_parser("draft", help="Draft source-linked code/spec with one model request")
    draft.add_argument("--source", required=True)
    draft.add_argument("--source-manifest", help="Optional ingestion receipt SHA-256")
    draft.add_argument("--dataset", required=True)
    draft.add_argument("--profile", required=True, help="Registered provider profile SHA-256")
    draft.add_argument("--model", required=True)
    draft.add_argument("--image", default="python:3.12-slim", help="Locally staged Python image")
    draft.add_argument("--allow-cloud", action="store_true")
    draft.add_argument("--max-output-tokens", type=int, default=8192)
    draft.add_argument("--out", type=Path, required=True)
    draft.add_argument("--budget", help="Shared model-call/output-token budget ID")
    budget_create = sub.add_parser("budget-create", help="Create a durable cross-run model quota")
    budget_create.add_argument("budget_id")
    budget_create.add_argument("--calls", type=int, required=True)
    budget_create.add_argument("--output-tokens", type=int, required=True)
    budget_get = sub.add_parser("budget", help="Inspect quota reservations and consumption")
    budget_get.add_argument("budget_id")
    freeze = sub.add_parser("freeze", help="Validate and content-address a reviewed specification")
    freeze.add_argument("path", type=Path)
    run = sub.add_parser("run", help="Submit and execute an immutable experiment")
    run.add_argument("spec_hash")
    run.add_argument("--key", required=True, help="Idempotency key; reuse returns the same run")
    submit = sub.add_parser("submit", help="Create a durable queued run without executing it")
    submit.add_argument("spec_hash")
    submit.add_argument("--key", required=True)
    for name in ("resume", "cancel", "report", "inspect"):
        command = sub.add_parser(name)
        command.add_argument("run_id")
    sub.add_parser("runs", help="List retained runs")
    worker = sub.add_parser("worker", help="Execute queued/expired runs using durable leases")
    worker.add_argument("--once", action="store_true", help="Process the current queue and exit")
    worker.add_argument("--poll", type=float, default=1.0)
    serve = sub.add_parser("serve", help="Run the optional authenticated API on loopback")
    serve.add_argument("--port", type=int, default=8000)
    comparison = sub.add_parser("comparison-freeze", help="Freeze a paired comparison plan")
    comparison.add_argument("path", type=Path)
    benchmark = sub.add_parser("benchmark", help="Run a frozen candidate/baseline comparison")
    benchmark.add_argument("plan_hash")
    benchmark.add_argument("--key", required=True)
    study_freeze = sub.add_parser("study-freeze", help="Freeze paired multi-seed trial plan")
    study_freeze.add_argument("path", type=Path)
    study = sub.add_parser("study", help="Execute a frozen multi-seed plan with failure accounting")
    study.add_argument("plan_hash")
    study.add_argument("--key", required=True)
    study_export = sub.add_parser(
        "study-export", help="Export a complete study and all trial bundles"
    )
    study_export.add_argument("result_hash")
    study_export.add_argument("path", type=Path)
    study_replay = sub.add_parser(
        "study-replay", help="Verify a study and recompute its assessment offline"
    )
    study_replay.add_argument("path", type=Path)
    remote_worker = sub.add_parser(
        "remote-worker", help="Execute assigned tasks through authenticated HTTPS"
    )
    remote_worker.add_argument("--url", required=True)
    remote_worker.add_argument("--worker-id", required=True)
    remote_worker.add_argument("--token-env", required=True)
    remote_worker.add_argument("--spool", type=Path, required=True)
    remote_worker.add_argument("--image", action="append", required=True)
    remote_worker.add_argument("--memory-mb", type=int, default=256)
    remote_worker.add_argument("--once", action="store_true")
    remote_worker.add_argument("--poll", type=float, default=1.0)
    export = sub.add_parser(
        "export", help="Export source/data/code/results; check data rights first"
    )
    export.add_argument("run_id")
    export.add_argument("path", type=Path)
    replay = sub.add_parser(
        "replay", help="Verify artifacts and recompute scores without code execution"
    )
    replay.add_argument("path", type=Path)
    schema = sub.add_parser("schema", help="Print experiment JSON Schema")
    schema.add_argument("--out", type=Path)
    demo = sub.add_parser("demo", help="Run a clearly labeled synthetic fixture")
    demo.add_argument(
        "--python", action="store_true", help="Execute isolated Python instead of builtin"
    )
    demo.add_argument("--image", default="python:3.12-slim")
    demo.add_argument("--key", default=None)
    demo.add_argument("--task", choices=["classification", "knapsack"], default="classification")
    return result


def _read_json(path: Path) -> dict:
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("JSON input exceeds 16 MiB")
    return json.loads(path.read_text(encoding="utf-8"))


def _print(value) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "replay":
            _print(replay_bundle(args.path))
            return 0
        if args.command == "study-replay":
            from .study_bundles import replay_study

            _print(replay_study(args.path))
            return 0
        if args.command == "prepare-parser":
            from .parsers import prepare_parser_image

            _print({"parser_image": prepare_parser_image(args.base_image)})
            return 0
        if args.command == "remote-worker":
            from .remote_client import WorkerClient

            if not 0.1 <= args.poll <= 5:
                raise ValueError("poll interval must be 0.1 to 5 seconds")
            images = tuple(DockerRunner().resolve_image(image) for image in args.image)
            client = WorkerClient(
                args.url, args.worker_id, args.token_env, args.spool, images, args.memory_mb
            )
            while True:
                outcome = client.run_once()
                _print(outcome)
                if args.once:
                    return 2 if outcome["state"] in {"error", "fenced", "pending_submission"} else 0
                time.sleep(args.poll)
        if args.command == "schema":
            schema = ExperimentSpec.model_json_schema()
            if args.out:
                args.out.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
            else:
                _print(schema)
            return 0
        engine = Engine(args.state)
        if args.command == "ingest":
            receipt = ingest_source(args.path, engine.artifacts, parser_image=args.parser_image)
            receipt["ingestion_receipt_sha256"] = engine.artifacts.put_json(receipt)
            _print(receipt)
        elif args.command == "budget-create":
            _print(engine.budgets.create(args.budget_id, args.calls, args.output_tokens))
        elif args.command == "budget":
            _print(engine.budgets.get(args.budget_id))
        elif args.command == "dataset":
            _print({"dataset_sha256": engine.register_dataset(_read_json(args.path))})
        elif args.command == "profile":
            _print({"profile_sha256": engine.register_profile(_read_json(args.path))})
        elif args.command == "draft":
            if args.out.exists():
                raise FileExistsError("draft output already exists")
            profile = ProviderProfile.model_validate(engine.artifacts.get_json(args.profile))
            client = ProviderClient(profile)
            if args.budget:
                from .budgets import BudgetedProviderClient

                client = BudgetedProviderClient(client, engine.budgets, args.budget)
            spec = draft_experiment(
                args.source,
                args.dataset,
                engine.artifacts,
                client,
                args.model,
                not args.allow_cloud,
                DockerRunner().resolve_image(args.image),
                args.max_output_tokens,
            )
            if args.source_manifest:
                spec = ExperimentSpec.model_validate(
                    {**spec.model_dump(mode="json"), "source_manifest_sha256": args.source_manifest}
                )
            # Exclusive creation keeps an existing reviewed draft from being overwritten.
            with args.out.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(spec.model_dump(mode="json"), indent=2) + "\n")
            _print({"draft": str(args.out), "unresolved": list(spec.unresolved), "frozen": False})
        elif args.command == "freeze":
            _print({"spec_hash": engine.freeze(_read_json(args.path))})
        elif args.command in {"run", "submit"}:
            run = engine.submit(args.spec_hash, args.key)
            if args.command == "run":
                run = engine.execute(run["run_id"])
            _print(run)
            if args.command == "run" and run["state"] != "completed":
                return 2
        elif args.command == "resume":
            run = engine.execute(args.run_id)
            _print(run)
            if run["state"] != "completed":
                return 2
        elif args.command == "cancel":
            _print(engine.runs.cancel(args.run_id))
        elif args.command == "inspect":
            _print(engine.result(args.run_id))
        elif args.command == "report":
            print(engine.report(args.run_id))
        elif args.command == "runs":
            _print(engine.runs.list_runs())
        elif args.command == "worker":
            if not 0.1 <= args.poll <= 5:
                raise ValueError("worker polling interval must be between 0.1 and 5 seconds")
            try:
                while True:
                    for run in engine.runs.list_runs():
                        if run["state"] in {"queued", "running"}:
                            _print(engine.execute(run["run_id"]))
                    if args.once:
                        break
                    time.sleep(args.poll)
            except KeyboardInterrupt:
                return 130
        elif args.command == "serve":
            if not 1 <= args.port <= 65535:
                raise ValueError("port must be between 1 and 65535")
            import uvicorn

            from .api import create_app

            uvicorn.run(create_app(args.state), host="127.0.0.1", port=args.port)
        elif args.command == "comparison-freeze":
            from .comparisons import freeze_comparison

            _print({"plan_hash": freeze_comparison(engine, _read_json(args.path))})
        elif args.command == "benchmark":
            from .comparisons import run_comparison

            result = run_comparison(engine, args.plan_hash, args.key)
            _print(result)
            if result["assessment"]["verdict"] == "not_tested":
                return 2
        elif args.command == "study-freeze":
            from .studies import freeze_study

            _print({"plan_hash": freeze_study(engine, _read_json(args.path))})
        elif args.command == "study":
            from .studies import run_study

            result = run_study(engine, args.plan_hash, args.key)
            _print(result)
            if result["assessment"]["verdict"] == "not_tested":
                return 2
        elif args.command == "study-export":
            from .study_bundles import export_study

            _print({"bundle": str(export_study(engine, args.result_hash, args.path))})
        elif args.command == "export":
            _print({"bundle": str(export_bundle(engine, args.run_id, args.path))})
        elif args.command == "demo":
            create = create_demo if args.task == "classification" else create_optimization_demo
            spec_hash = create(engine, python=args.python, image=args.image)
            run = engine.run(spec_hash, args.key or f"synthetic-demo-{uuid.uuid4().hex}")
            print(engine.report(run["run_id"]))
            if run["state"] != "completed":
                return 2
        return 0
    except KeyboardInterrupt:
        return 130
    except ValidationError as exc:
        _print_error(
            "validation_error",
            exc.errors(include_input=False, include_url=False, include_context=False),
        )
    except (ValueError, OSError, KeyError) as exc:
        _print_error(type(exc).__name__, str(exc))
    except Exception as exc:  # noqa: BLE001 - keep remote diagnostics out of CLI output
        # Provider/runtime exceptions are intentionally not echoed with remote text.
        _print_error(
            type(exc).__name__, "operation failed; check configuration and runtime availability"
        )
    return 2


def _print_error(code, detail) -> None:
    print(json.dumps({"error": code, "detail": detail}, ensure_ascii=False), file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
