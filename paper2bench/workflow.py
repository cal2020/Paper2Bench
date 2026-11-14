"""Durable execution of pinned candidate bundles and protected deterministic scoring."""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

from .bundles import BundleError, atomic_json, canonical_json, digest, read_json, safe_file, validate_task
from .journal import JournalError, RunJournal
from .methods import SCORERS, SCORER_PROTOCOLS, score


def _identity(path):
    return digest(canonical_json(read_json(path)))


def _curator(path, task_digest):
    if path is None:
        return None
    data = read_json(path)
    if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
            or data["schema_version"] != 1 or data.get("task_manifest_sha256") != task_digest
            or data.get("scorer") not in SCORERS or not isinstance(data.get("reference", {}), dict)):
        raise BundleError("Protected evaluator is unsupported or does not match the task")
    return data


def _check_scorer_protocol(spec):
    if spec.get("evaluation_bundle") is None:
        return
    curator = read_json(spec["evaluation_bundle"])
    current = SCORER_PROTOCOLS.get(curator.get("scorer")) if isinstance(curator, dict) else None
    if current is None or spec.get("scorer_protocol") != current:
        raise JournalError("Installed scorer protocol changed or was not pinned; start an explicit fresh run")


def execute(task, *, evaluation_bundle=None, runs_dir="./runs", executor=None,
            evaluation_profile="execution/v1", image=None, _run_id=None, parent_run=None,
            seed=None, executor_config=None):
    """Run a sealed Python submission. Never execute a candidate on the host.

    execution/v1 checks job completion only. method_output/v1 uses one of the
    two installed demonstration scorers. Scientific reproduction is not inferred.
    """
    from .executors import DockerExecutor, ExecutionRequest

    if evaluation_profile in {"reproduction/v1", "transfer/v1"}:
        if seed is not None:
            raise BundleError("Scientific seeds must come from the frozen protocol schedule")
        from .scientific_runs import execute_scientific
        return execute_scientific(task, evaluation_bundle=evaluation_bundle, runs_dir=runs_dir,
                                  executor=executor, image=image, evaluation_profile=evaluation_profile,
                                  parent_run=parent_run, _run_id=_run_id, executor_config=executor_config)

    if evaluation_profile not in {"execution/v1", "method_output/v1"}:
        raise BundleError("Unsupported evaluation profile; no calibrated reproduction protocol is installed")
    if Path(task).is_symlink():
        raise BundleError("Task manifest must not be a symlink")
    path = Path(task).resolve()
    if path.is_dir():
        path = path / "task_manifest.json"
    manifest = validate_task(path, require_executable=True)
    task_hash = digest(canonical_json(manifest))
    curator = _curator(evaluation_bundle, task_hash)
    if evaluation_profile == "method_output/v1" and curator is None:
        raise BundleError("method_output/v1 requires an explicit protected evaluation bundle")
    evaluator_hash = digest(canonical_json(curator)) if curator else None
    scorer_protocol = SCORER_PROTOCOLS[curator["scorer"]] if curator else None
    from .execution_policy import admit_limits
    limits = admit_limits(manifest.get("limits", {}))
    from .runtime_config import get_context
    context = get_context()
    image = image or (context.config.execution.get("image") if context else None)
    from .executor_factory import build_executor, executor_selection
    executor = executor or build_executor(image=image, selection=executor_config)
    preflight = executor.preflight() or {}
    image = preflight.get("image_id", image)
    if hasattr(executor, "image") and image:
        executor.image = image  # Subsequent preflight and execution resolve this exact immutable image.
    frozen_executor = executor_selection(executor, image=image)
    max_attempts = context.config.retry.get("max_attempts_per_step", 3) if context else 3
    total_wall = (context.config.budgets.get("experiments", {}).get("wall_seconds", limits["timeout_seconds"] * max_attempts)
                  if context else limits["timeout_seconds"] * max_attempts)
    total_cpu = (context.config.budgets.get("experiments", {}).get("cpu_seconds", total_wall * limits["cpus"])
                 if context else total_wall * limits["cpus"])
    journal = RunJournal(runs_dir)
    spec = {"task": str(path), "task_sha256": task_hash,
            "kind": "execution", "seed": seed, "executor_config": frozen_executor,
            "evaluation_bundle": str(Path(evaluation_bundle).resolve()) if curator else None,
            "evaluation_sha256": evaluator_hash, "evaluation_profile": evaluation_profile,
            "image": image, "limits": limits, "max_attempts": max_attempts}
    if scorer_protocol is not None:
        spec["scorer_protocol"] = scorer_protocol
    if _run_id:
        stored = journal.get(_run_id)
        spec["max_attempts"] = stored["spec"]["max_attempts"]
        if len(stored["attempts"]) >= spec["max_attempts"]:
            raise JournalError("Run attempt budget exhausted")
    run_id = _run_id or journal.create(spec, limits={"wall_seconds": total_wall, "cpu_seconds": total_cpu},
                                       parent_run=parent_run)
    if _run_id and journal.get(run_id)["spec"] != dict(spec, parent_run=journal.get(run_id)["spec"].get("parent_run")):
        raise JournalError("Resume configuration or task identity changed")
    remaining = journal.get(run_id)["budget"]
    remaining_wall = remaining["limits"]["wall_seconds"] - remaining["used"].get("wall_seconds", 0) - remaining["reserved"].get("wall_seconds", 0)
    remaining_cpu = remaining["limits"].get("cpu_seconds", 0) - remaining["used"].get("cpu_seconds", 0) - remaining["reserved"].get("cpu_seconds", 0)
    if remaining_wall <= 0 or remaining_cpu <= 0:
        raise JournalError("Run wall-time budget exhausted")
    attempt_limits = dict(limits, timeout_seconds=min(limits["timeout_seconds"], remaining_wall,
                                                     remaining_cpu / limits["cpus"]))
    attempt = journal.begin_attempt(run_id)
    run_dir = journal.root / run_id
    attempt_dir = run_dir / "attempts" / attempt
    workspace = attempt_dir / "workspace"
    try:
        workspace.mkdir(parents=True, mode=0o700)
        for name in manifest["files"]:
            source = safe_file(path.parent, name)
            destination = workspace / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as stream:
                raw = stream.read(8 * 1024 * 1024 + 1)
            if len(raw) > 8 * 1024 * 1024:
                raise BundleError("Submission changed beyond staging size limit")
            destination.write_bytes(raw)
        atomic_json(run_dir / "run_manifest.json", {"schema_version": 1, "run_id": run_id,
                    "attempt_id": attempt, "submission_sha256": task_hash, "state": "submission_intent", **spec})
        for name, expected in manifest["files"].items():
            if digest((workspace / name).read_bytes()) != expected:
                raise BundleError("Submission changed during staging")
        original_inputs = read_json(safe_file(workspace, manifest["inputs"]))
        code = safe_file(workspace, manifest["entrypoint"]).read_text(encoding="utf-8")
        if len(code.encode()) > 256 * 1024:
            raise BundleError("Entrypoint exceeds code limit")
        # Validate backend-specific constraints before reserving resources or
        # entering the ambiguous-dispatch boundary. An invalid request never
        # created a backend job and must remain a known preparation failure.
        request = ExecutionRequest(code=code, workspace=workspace, seed=seed, **attempt_limits)
    except BaseException as error:
        # Nothing was dispatched and nothing was reserved: this is a known blocked
        # preparation failure, not an ambiguous remote execution.
        journal.finish(run_id, attempt, "blocked", {"reason": "preparation_failed", "error_type": type(error).__name__})
        raise
    reservation = attempt + ":execution"
    try:
        journal.reserve(run_id, reservation, {"wall_seconds": attempt_limits["timeout_seconds"],
                                              "cpu_seconds": attempt_limits["timeout_seconds"] * limits["cpus"]})
    except JournalError as error:
        journal.finish(run_id, attempt, "blocked", {"reason": "budget_exhausted"})
        raise error
    stop = threading.Event()
    owned_by_scientific_parent = bool(parent_run and journal.get(parent_run)["spec"].get("kind") == "scientific")

    def monitor_cancel():
        while not stop.wait(0.2):
            if owned_by_scientific_parent and journal.get(parent_run)["state"] in {"cancellation_requested", "cancelled"}:
                journal.request_cancel(run_id)
            if journal.get(run_id)["state"] == "cancellation_requested":
                executor.cancel()
                # Keep observing until execute returns: an early cancel may occur
                # before the backend has created its container.

    monitor = threading.Thread(target=monitor_cancel, daemon=True)
    if hasattr(executor, "lifecycle_callback"):
        previous_callback = executor.lifecycle_callback

        def record_lifecycle(event, payload):
            journal.event(run_id, event, dict(payload, attempt_id=attempt))
            cancelled = journal.get(run_id)["state"] == "cancellation_requested"
            if owned_by_scientific_parent:
                cancelled = cancelled or journal.get(parent_run)["state"] in {"cancellation_requested", "cancelled"}
            if event == "job_intent" and cancelled:
                raise JournalError("Cancellation requested before backend creation")
            if previous_callback:
                previous_callback(event, payload)

        executor.lifecycle_callback = record_lifecycle
    monitor.start()
    started = time.monotonic()
    try:
        outcome = executor.execute(request)
    except BaseException as error:
        # A crash or ambiguous dispatch never automatically releases its reservation.
        journal.finish(run_id, attempt, "reconciling", {"reason": type(error).__name__})
        raise
    finally:
        stop.set()
        monitor.join(timeout=2)
        if hasattr(executor, "lifecycle_callback"):
            executor.lifecycle_callback = previous_callback
    result = asdict(outcome) if is_dataclass(outcome) else dict(outcome)
    atomic_json(attempt_dir / "execution_result.json", result)
    report = {"schema_version": 1, "evaluation_protocol": evaluation_profile,
              "run_id": run_id, "attempt_id": attempt, "task_sha256": task_hash, "seed": seed,
              "evaluator_sha256": evaluator_hash, "execution": "fail",
              "method_output": "not_run", "scientific_reproduction": "not_assessed",
              "evidence": {"execution_result": f"attempts/{attempt}/execution_result.json"}}
    if scorer_protocol is not None:
        report["scorer_protocol"] = scorer_protocol
    successful = outcome.status == "succeeded" and outcome.exit_code == 0
    state = "succeeded" if successful else "failed"
    if outcome.status in {"cancelled", "blocked"}:
        state = outcome.status
    report["execution"] = "pass" if successful else "fail"
    if successful and curator:
        try:
            prediction_path = safe_file(workspace, manifest["predictions"])
            predicted = read_json(prediction_path)
            metric = score(curator["scorer"], original_inputs, predicted, curator.get("reference", {}))
            report.update(method_output=metric["verdict"], metric=metric,
                          predictions_sha256=digest(prediction_path.read_bytes()))
        except (ValueError, KeyError, OverflowError, OSError) as error:
            report.update(method_output="inconclusive", evaluation_error=type(error).__name__)
    # A failed experiment is a completed evaluation, not a successful scientific result.
    report["evaluation_status"] = "error" if "evaluation_error" in report else "complete"
    report["pass"] = successful and (evaluation_profile == "execution/v1" or report["method_output"] == "pass")
    cleanup_known = outcome.evidence.get("cleanup_confirmed") is True
    if cleanup_known:
        elapsed = time.monotonic() - started
        journal.settle(run_id, reservation, {"wall_seconds": elapsed, "cpu_seconds": elapsed * limits["cpus"]})
        report["resource_accounting"] = {"wall_seconds": elapsed, "cpu_seconds_upper_bound": elapsed * limits["cpus"],
                                         "cpu_measurement": "conservative_lifecycle_bound_not_measured_cpu"}
    else:
        state = "reconciling"
        report["pass"] = False
        report["evaluation_status"] = "inconclusive"
    report["run_state"] = state
    # Final evidence is durable before the terminal database commit. The committed
    # report copy also makes missing convenience files recoverable after a crash.
    atomic_json(attempt_dir / "evaluation_report.json", report)

    def committed_result(final_state):
        if final_state == "cancelled":
            report.update({"pass": False, "execution": "inconclusive", "evaluation_status": "cancelled"})
        report["run_state"] = final_state
        return {"report": str(attempt_dir / "evaluation_report.json"),
                "report_data": report, "report_data_sha256": digest(canonical_json(report)),
                "cleanup_confirmed": cleanup_known}

    state = journal.finish(run_id, attempt, state, committed_result)
    atomic_json(attempt_dir / "evaluation_report.json", report)
    atomic_json(run_dir / "evaluation_report.json", report)
    atomic_json(run_dir / "run_manifest.json", {"schema_version": 1, "run_id": run_id,
                "attempt_id": attempt, "submission_sha256": task_hash, "state": state, **spec})
    atomic_json(run_dir / "reproduce.json", {"schema_version": 1, "task": str(path),
                "task_sha256": task_hash, "evaluation_bundle": spec["evaluation_bundle"],
                "evaluation_sha256": evaluator_hash, "evaluation_profile": evaluation_profile,
                "scorer_protocol": scorer_protocol,
                "limitations": ["External assets must remain available", "No scientific reproduction protocol is implied"]})
    events = journal.events(run_id)
    (run_dir / "evidence.jsonl").write_text("".join(canonical_json(e).decode() + "\n" for e in events), encoding="utf-8")
    return report


def resume(run_id, *, runs_dir="./runs", executor=None):
    journal = RunJournal(runs_dir)
    run = journal.get(run_id)
    spec = run["spec"]
    if spec.get("kind") == "authoring":
        from .api import resume_generation
        from .runtime_config import get_context
        context = get_context()
        result = resume_generation(run_id, model_runs_dir=runs_dir,
                                   runtime_config=context.config if context else None)
        return {"pass": True, "run_id": run_id, "scope": "authoring_artifact_generation",
                "scientific_reproduction": "not_assessed", "task_id": result.task_id,
                "instruction_path": str(result.instruction_path), "extras": result.extras}
    if spec.get("kind") == "scientific":
        from .scientific_runs import resume_scientific
        return resume_scientific(run_id, runs_dir=runs_dir, executor=executor)
    if spec.get("parent_run") and journal.get(spec["parent_run"])["spec"].get("kind") == "scientific":
        raise JournalError("Resume the scientific parent; individual seed retries would change its fixed schedule")
    if _identity(spec["task"]) != spec["task_sha256"]:
        raise JournalError("Task changed since original run")
    try:
        validate_task(spec["task"], require_executable=True)
    except (BundleError, OSError) as error:
        raise JournalError("Pinned task files changed or are unavailable") from error
    if spec["evaluation_bundle"] and _identity(spec["evaluation_bundle"]) != spec["evaluation_sha256"]:
        raise JournalError("Evaluator changed since original run")
    _check_scorer_protocol(spec)
    if run["state"] == "succeeded":
        committed = run["attempts"][-1]["result"]
        if committed and "report_data" in committed:
            report = committed["report_data"]
            if digest(canonical_json(report)) != committed["report_data_sha256"]:
                raise JournalError("Committed report integrity check failed")
            report["run_state"] = run["state"]
            atomic_json(journal.root / run_id / "evaluation_report.json", report)
            return report
        # Compatibility with alpha records created before committed report copies.
        return read_json(journal.root / run_id / "evaluation_report.json")
    if run["state"] not in {"failed", "blocked"}:
        raise JournalError("Run cannot be resumed until execution/termination is reconciled")
    return execute(spec["task"], evaluation_bundle=spec["evaluation_bundle"], runs_dir=runs_dir,
                   executor=executor, evaluation_profile=spec["evaluation_profile"], image=spec["image"], _run_id=run_id,
                   seed=spec.get("seed"), executor_config=spec.get("executor_config"))


def reproduce(run_id, *, runs_dir="./runs", executor=None):
    run = RunJournal(runs_dir).get(run_id)
    spec = run["spec"]
    if spec.get("kind") == "authoring":
        raise JournalError("Use the original authoring command for a new explicit run")
    if spec.get("kind") == "scientific":
        from .scientific_runs import execute_scientific
        root = Path(runs_dir).resolve() / run_id / "frozen"
        return execute_scientific(root / "candidate/task_manifest.json", evaluation_bundle=root / "curator_manifest.json",
                                 runs_dir=runs_dir, executor=executor, image=spec.get("image"),
                                 evaluation_profile=spec["evaluation_profile"], parent_run=run_id,
                                 executor_config=spec.get("executor_config"))
    if _identity(spec["task"]) != spec["task_sha256"] or (spec["evaluation_bundle"] and
            _identity(spec["evaluation_bundle"]) != spec["evaluation_sha256"]):
        raise JournalError("Task or evaluator identity changed")
    _check_scorer_protocol(spec)
    return execute(spec["task"], evaluation_bundle=spec["evaluation_bundle"], runs_dir=runs_dir,
                   executor=executor, evaluation_profile=spec["evaluation_profile"], image=spec["image"], parent_run=run_id,
                   seed=spec.get("seed"), executor_config=spec.get("executor_config"))


def reconcile(run_id, *, runs_dir="./runs", terminate=False, executor=None):
    """Inspect only controller-recorded job names; explicit termination is opt-in."""
    from .executors import DockerExecutor
    journal = RunJournal(runs_dir)
    run = journal.get(run_id)
    if run["spec"].get("kind") == "scientific":
        if terminate:
            journal.request_cancel(run_id)
        children = []
        for child in journal.children(run_id):
            if child["spec"].get("kind") == "execution" and child["state"] in {"pending", "running", "reconciling", "cancellation_requested"}:
                children.append(reconcile(child["id"], runs_dir=runs_dir, terminate=terminate, executor=executor))
        return {"run_id": run_id, "kind": "scientific", "children": children,
                "state": journal.get(run_id)["state"], "budget": journal.get(run_id)["budget"],
                "termination_confirmed": all(c["termination_confirmed"] for c in children)}
    if run["spec"].get("kind") != "execution" or not run["attempts"]:
        raise JournalError("This run has no reconcilable Docker execution")
    attempt = run["attempts"][-1]
    names = {e["payload"].get("container_name") for e in journal.events(run_id)
             if e["kind"] in {"job_intent", "job_created"} and e["payload"].get("attempt_id") == attempt["id"]}
    names.discard(None)
    if not names:
        raise JournalError("No recorded backend job identity; automatic reconciliation is unsafe")
    if terminate:
        journal.request_cancel(run_id)
    from .executor_factory import build_executor
    backend = executor or build_executor(selection=run["spec"].get("executor_config"))
    evidence = [backend.reconcile(name, terminate=terminate) for name in sorted(names)]
    confirmed = all(item.get("cleanup_confirmed") is True for item in evidence)
    if confirmed and attempt["state"] in {"submission_intent", "running", "reconciling"}:
        journal.confirm_orphan_termination(run_id, attempt["id"], {"backend_jobs": evidence,
                    "usage_status": "unknown_reservations_retained"})
    return {"run_id": run_id, "termination_confirmed": confirmed, "backend_jobs": evidence,
            "state": journal.get(run_id)["state"], "budget": journal.get(run_id)["budget"]}
