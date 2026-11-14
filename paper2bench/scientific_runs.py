"""Durable fixed-seed scientific orchestration over isolated child executions.

Protocols and candidate inputs are frozen before dispatch. Child reports come
from controller-committed evidence, never from a candidate's claimed metrics.
Unknown child outcomes stop the loop and retain aggregate reservations.
"""
from __future__ import annotations

import fcntl
import os
import re
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path

from .bundles import BundleError, atomic_json, canonical_json, digest, read_json, safe_file, validate_task
from .evaluation.science import SCIENTIFIC_EVALUATOR_PROTOCOL, evaluate_protocol, freeze_protocol
from .evaluation.statistics import finite_number
from .evaluation.validation import EvaluationError
from .execution_policy import admit_limits
from .journal import JournalError, RunJournal
from .methods import SCORERS, SCORER_PROTOCOLS
from .runtime_config import get_context

_ACTIVE = {"running", "reconciling", "cancellation_requested"}
_METRICS = {"linear_predictions/v1": ("mean_squared_error", "target_units_squared"),
            "knapsack_optimum/v1": ("feasible_optimal_objective", "objective_units")}


def _check_implementation(spec):
    if spec.get("scientific_evaluator_protocol") != SCIENTIFIC_EVALUATOR_PROTOCOL:
        raise JournalError("Scientific evaluator protocol changed or is unpinned; create a fresh scientific run")
    installed = SCORER_PROTOCOLS.get(spec.get("scorer"))
    if installed is None or spec.get("scorer_protocol") != installed:
        raise JournalError("Scientific scorer protocol changed or is unpinned; create a fresh scientific run")


def _validate_frozen_condition(record, runs_root):
    """A cached report cannot conceal changed or missing pinned input files."""
    spec = record["spec"]
    root = runs_root / record["id"] / "frozen"
    try:
        manifest = validate_task(root / "candidate/task_manifest.json", require_executable=True)
        curator = read_json(root / "curator_manifest.json")
        locked_protocol = read_json(root / "scientific_protocol.json")
        if (digest(canonical_json(manifest)) != spec.get("task_sha256")
                or digest(canonical_json(curator)) != spec.get("evaluation_sha256")
                or digest(canonical_json(locked_protocol)) != spec.get("protocol_sha256")):
            raise JournalError("Frozen scientific input or evaluator identity changed")
    except (BundleError, OSError) as error:
        raise JournalError("Frozen scientific inputs changed or are unavailable") from error


@contextmanager
def _controller_lock(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(run_dir / "scientific-controller.lock", os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise JournalError("A scientific controller is already active for this run") from error
        yield
    finally:
        os.close(descriptor)


def _checked_report(record):
    if not record["attempts"]:
        return None
    result = record["attempts"][-1].get("result") or {}
    if not isinstance(result, dict):
        raise JournalError("Committed report record is malformed")
    report = result.get("report_data")
    if report is None:
        return None
    if not isinstance(report, dict):
        raise JournalError("Committed report must be an object")
    if digest(canonical_json(report)) != result.get("report_data_sha256"):
        raise JournalError("Committed child report integrity mismatch")
    if report.get("run_id") != record["id"]:
        raise JournalError("Committed report belongs to another run")
    spec = record["spec"]
    if report.get("attempt_id") != record["attempts"][-1]["id"]:
        raise JournalError("Committed report belongs to another attempt")
    if report.get("run_state") != record["state"]:
        raise JournalError("Committed report lifecycle differs from its execution condition")
    if spec.get("kind") == "execution":
        if (report.get("task_sha256") != spec.get("task_sha256")
                or report.get("evaluator_sha256") != spec.get("evaluation_sha256")
                or type(report.get("seed")) is not int or report["seed"] != spec.get("seed")
                or report.get("evaluation_protocol") != spec.get("evaluation_profile")
                or report.get("scorer_protocol") != spec.get("scorer_protocol")):
            raise JournalError("Committed child report condition identity differs from its execution")
    elif spec.get("kind") == "scientific":
        if (report.get("task_manifest_sha256") != spec.get("task_sha256")
                or report.get("protocol_sha256") != spec.get("protocol_sha256")
                or report.get("submission_sha256") != spec.get("submission_sha256")
                or report.get("evaluation_protocol") != spec.get("evaluation_profile")
                or report.get("scorer_protocol") != spec.get("scorer_protocol")
                or report.get("scientific_evaluator_protocol") != spec.get("scientific_evaluator_protocol")):
            raise JournalError("Committed scientific report differs from the frozen condition")
    return report


def _snapshot(task_path, manifest, curator, run_dir):
    root = run_dir / "frozen"
    candidate = root / "candidate"
    candidate.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name, expected in manifest["files"].items():
        source = safe_file(task_path.parent, name)
        with source.open("rb") as stream:
            content = stream.read(8 * 1024 * 1024 + 1)
        if len(content) > 8 * 1024 * 1024 or digest(content) != expected:
            raise BundleError("Source changed while freezing the scientific submission")
        target = candidate / name
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        target.write_bytes(content)
    atomic_json(candidate / "task_manifest.json", manifest)
    atomic_json(root / "curator_manifest.json", curator)
    validate_task(candidate / "task_manifest.json", require_executable=True)
    return candidate / "task_manifest.json", root / "curator_manifest.json"


def _children_by_seed(journal, run_id, seeds):
    parent_spec = journal.get(run_id)["spec"]
    if parent_spec.get("kind") != "scientific":
        raise JournalError("Scientific child lookup requires a frozen parent condition")
    result = {}
    for child in journal.children(run_id):
        child_spec = child["spec"]
        seed = child_spec.get("seed")
        if child_spec.get("kind") != "execution" or type(seed) is not int or seed not in seeds:
            raise JournalError("Scientific run contains an unrecognized child execution")
        if child_spec.get("parent_run") != run_id or child_spec.get("evaluation_profile") != "method_output/v1":
            raise JournalError("Scientific child is not bound to this parent and installed scoring condition")
        for field in ("task_sha256", "evaluation_sha256", "image", "executor_config", "limits", "scorer_protocol"):
            if (field not in child_spec or field not in parent_spec
                    or canonical_json(child_spec[field]) != canonical_json(parent_spec[field])):
                raise JournalError(f"Scientific child {field} identity differs from the frozen parent condition")
        report = _checked_report(child)
        if child["state"] == "succeeded" and report is None:
            raise JournalError("Completed scientific child has no committed condition evidence")
        if seed in result:
            raise JournalError("Multiple child executions for one fixed seed require manual reconciliation")
        result[seed] = child
    return result


def _intents(journal, run_id):
    by_seed = {}
    for event in journal.events(run_id):
        if event["kind"] == "scientific_seed_intent":
            by_seed[event["payload"]["seed"]] = event["payload"]
    return by_seed


def _settle_parent(journal, parent_id, intent, child):
    if child["state"] in _ACTIVE or child["state"] == "pending" or child["budget"]["reserved"]:
        return False
    used = child["budget"]["used"]
    actual = {"wall_seconds": used.get("wall_seconds", 0), "cpu_seconds": used.get("cpu_seconds", 0)}
    journal.settle(parent_id, intent["reservation_key"], actual)
    return True


def _release_pre_intent_reservations(journal, run_id):
    """A reserve commit preceding seed-intent commit cannot have dispatched work."""
    events = journal.events(run_id)
    intended = {event["payload"]["reservation_key"] for event in events if event["kind"] == "scientific_seed_intent"}
    settled = {event["payload"]["key"] for event in events if event["kind"] == "budget_settled"}
    children = journal.children(run_id)
    for event in events:
        if event["kind"] != "budget_reserved":
            continue
        key = event["payload"]["key"]
        match = re.fullmatch(re.escape(run_id) + r":seed:([0-9]+):[0-9a-f]{32}", key)
        if match and key not in intended and key not in settled:
            if any(child["spec"].get("seed") == int(match.group(1)) for child in children):
                raise JournalError("A child without its scientific dispatch intent requires reconciliation")
            amounts = event["payload"]["amounts"]
            if set(amounts) != {"wall_seconds", "cpu_seconds"}:
                raise JournalError("Unexpected scientific reservation dimensions")
            journal.settle(run_id, key, {"wall_seconds": 0, "cpu_seconds": 0})


def _metric_record(seed, child, report, scorer):
    metric_id, units = _METRICS[scorer]
    base = {"id": metric_id, "seed": seed, "units": units, "origin": "trusted_recomputation",
            "evidence_refs": [f"runs/{child['id']}/evaluation_report.json"]}
    if not report or report.get("execution") != "pass" or not isinstance(report.get("metric"), dict):
        return dict(base, status="failed", value=None)
    metric = report["metric"]
    if metric.get("metric") != metric_id or metric.get("units") != units:
        raise JournalError("Installed scorer returned an unexpected metric contract")
    if scorer == "knapsack_optimum/v1" and metric.get("feasible") is not True:
        # An infeasible subset's numeric objective is not a valid observation of
        # the constrained objective, even when it coincides with the optimum.
        return dict(base, status="failed", value=None, failure_reason="infeasible_solution")
    finite_number(metric.get("value"), "trusted metric")
    n = metric.get("samples") if scorer == "linear_predictions/v1" else 1
    if type(n) is not int or n <= 0:
        raise JournalError("Installed scorer did not report its sample count")
    artifact = report.get("predictions_sha256")
    if not isinstance(artifact, str) or len(artifact) != 64 or any(c not in "0123456789abcdef" for c in artifact):
        raise JournalError("Trusted metric is missing its output artifact digest")
    return dict(base, status="succeeded", value=metric["value"], n=n, artifact_sha256=artifact)


def execute_scientific(task, *, evaluation_bundle, runs_dir="./runs", executor=None,
                       image=None, evaluation_profile="reproduction/v1", parent_run=None,
                       _run_id=None, executor_config=None):
    """Execute one fresh child per fixed seed and evaluate protected evidence.

    Scientific failure does not retry a seed. Resume only recovers already-known
    outcomes or dispatches seeds for which no child submission ever existed.
    """
    from .executor_factory import build_executor, executor_selection
    from .workflow import execute as execute_child

    if evaluation_profile not in {"reproduction/v1", "transfer/v1"}:
        raise BundleError("Scientific runner requires reproduction/v1 or transfer/v1")
    journal = RunJournal(runs_dir)
    if parent_run is not None:
        parent = journal.get(parent_run)
        if parent["spec"].get("kind") == "scientific":
            _check_implementation(parent["spec"])
            _validate_frozen_condition(parent, journal.root)
    if _run_id is not None:
        stored = journal.get(_run_id)
        if stored["spec"].get("kind") != "scientific":
            raise JournalError("Run is not scientific orchestration")
        _check_implementation(stored["spec"])
        if stored["state"] == "succeeded":
            _validate_frozen_condition(stored, journal.root)
            report = _checked_report(stored)
            if report is None:
                raise JournalError("Completed scientific run has no committed report")
            atomic_json(journal.root / _run_id / "evaluation_report.json", report)
            return report
        task = journal.root / _run_id / "frozen/candidate/task_manifest.json"
        evaluation_bundle = journal.root / _run_id / "frozen/curator_manifest.json"
        image = stored["spec"]["image"]
        executor_config = stored["spec"]["executor_config"]
        evaluation_profile = stored["spec"]["evaluation_profile"]
    if evaluation_bundle is None:
        raise BundleError("A protected scientific evaluation bundle is required")
    task_path = Path(task)
    if task_path.is_symlink():
        raise BundleError("Scientific task manifest cannot be a symlink")
    if task_path.is_dir():
        task_path = task_path / "task_manifest.json"
    task_path = task_path.resolve()
    manifest = validate_task(task_path, require_executable=True)
    task_hash = digest(canonical_json(manifest))
    curator = read_json(evaluation_bundle)
    if (not isinstance(curator, dict) or type(curator.get("schema_version")) is not int or curator.get("schema_version") != 1
            or curator.get("task_manifest_sha256") != task_hash or not isinstance(curator.get("scorer"), str)
            or curator.get("scorer") not in SCORERS or curator.get("scorer") not in _METRICS
            or not isinstance(curator.get("protocol"), dict)):
        raise BundleError("Scientific curator bundle requires a matching installed scorer and protocol")
    protocol = freeze_protocol(curator["protocol"], task_manifest=manifest,
                               submission_sha256=manifest["files"][manifest["entrypoint"]])
    protocol_data = protocol.as_dict()
    if protocol_data["profile"] != evaluation_profile:
        raise BundleError("Requested scientific profile differs from the protected protocol")
    reviews = {review["id"]: review for review in protocol_data.get("reviews", [])}
    for criterion in protocol_data["criteria"]:
        if (criterion["critical"] and criterion["kind"] == "review"
                and reviews.get(criterion["review_id"], {}).get("status") != "pass"):
            raise BundleError("Scientific execution requires a passing critical review of the exact submission")
    metric_name = _METRICS[curator["scorer"]][0]
    for criterion in protocol_data["criteria"]:
        if criterion["kind"] == "metric" and (criterion["metric_id"] != metric_name or "baseline_id" in criterion):
            raise BundleError("Scientific runner requires metrics computed by its installed scorer; baseline orchestration is not installed")
    limits = admit_limits(manifest.get("limits", {}))
    context = get_context()
    image = image or (context.config.execution.get("image") if context else None)
    executor = executor or build_executor(image=image, selection=executor_config)
    preflight = executor.preflight() or {}
    image = preflight.get("image_id", image)
    if hasattr(executor, "image") and image:
        executor.image = image
    selection = executor_selection(executor, image=image)
    seeds = protocol_data["plan"]["seed_schedule"]
    aggregate_wall = limits["timeout_seconds"] * len(seeds)
    aggregate_cpu = aggregate_wall * limits["cpus"]
    if context:
        configured = context.config.budgets.get("experiments", {})
        aggregate_wall = configured.get("wall_seconds", aggregate_wall)
        aggregate_cpu = configured.get("cpu_seconds", aggregate_cpu)
    spec = {"kind": "scientific", "source_task": str(task_path),
            "source_evaluation_bundle": str(Path(evaluation_bundle).resolve()), "task_sha256": task_hash,
            "evaluation_sha256": digest(canonical_json(curator)), "protocol_sha256": protocol.sha256,
            "submission_sha256": protocol_data["submission_sha256"], "evaluation_profile": evaluation_profile,
            "scorer": curator["scorer"], "scorer_protocol": SCORER_PROTOCOLS[curator["scorer"]],
            "scientific_evaluator_protocol": SCIENTIFIC_EVALUATOR_PROTOCOL,
            "image": image, "executor_config": selection, "limits": limits, "seed_schedule": seeds}
    condition_fields = ("task_sha256", "evaluation_sha256", "protocol_sha256", "submission_sha256",
                        "evaluation_profile", "image", "executor_config", "limits", "seed_schedule",
                        "scorer", "scorer_protocol", "scientific_evaluator_protocol")
    if _run_id is not None:
        old = journal.get(_run_id)["spec"]
        for key in condition_fields:
            if old.get(key) != spec[key]:
                raise JournalError("Frozen scientific identity or execution configuration changed")
        run_id = _run_id
    else:
        if parent_run is not None and parent["spec"].get("kind") == "scientific":
            if any(parent["spec"].get(key) != spec[key] for key in condition_fields):
                raise JournalError("Scientific reproduction condition differs from its frozen parent")
        run_id = journal.create(spec, limits={"wall_seconds": aggregate_wall, "cpu_seconds": aggregate_cpu}, parent_run=parent_run)
    run_dir = journal.root / run_id
    with _controller_lock(run_dir):
        current = journal.get(run_id)
        if _run_id is not None:
            if current["state"] in {"cancelled", "cancellation_requested"}:
                raise JournalError("Cancelled scientific runs cannot be resumed")
            journal.recover_orchestration(run_id)
            _release_pre_intent_reservations(journal, run_id)
        attempt = journal.begin_attempt(run_id)
        try:
            if _run_id is None:
                frozen_task, frozen_curator = _snapshot(task_path, manifest, curator, run_dir)
                atomic_json(run_dir / "frozen/scientific_protocol.json", protocol_data)
                journal.event(run_id, "scientific_protocol_locked", {"protocol_sha256": protocol.sha256,
                              "task_sha256": task_hash, "submission_sha256": protocol_data["submission_sha256"]})
            else:
                frozen_task = run_dir / "frozen/candidate/task_manifest.json"
                frozen_curator = run_dir / "frozen/curator_manifest.json"
                if digest(canonical_json(read_json(run_dir / "frozen/scientific_protocol.json"))) != protocol.sha256:
                    raise JournalError("Frozen scientific protocol file changed")
        except BaseException as error:
            journal.finish(run_id, attempt, "blocked", {"reason": "scientific_preparation_failed", "error_type": type(error).__name__})
            raise
        stop = threading.Event()

        def forward_cancellation():
            while not stop.wait(0.2):
                if journal.get(run_id)["state"] == "cancellation_requested":
                    for child in journal.children(run_id):
                        if child["spec"].get("kind") == "execution" and child["state"] in {"pending", "running", "reconciling", "cancellation_requested"}:
                            journal.request_cancel(child["id"])

        monitor = threading.Thread(target=forward_cancellation, daemon=True)
        monitor.start()
        blocked_reason = None
        unknown = False
        try:
            for seed in seeds:
                children = _children_by_seed(journal, run_id, seeds)
                intents = _intents(journal, run_id)
                child = children.get(seed)
                intent = intents.get(seed)
                if child is not None and (child["state"] in _ACTIVE or (child["state"] == "pending" and child["attempts"])):
                    unknown = True
                    blocked_reason = "child_outcome_requires_reconciliation"
                    break
                if child is not None and child["state"] != "pending":
                    if intent is None:
                        raise JournalError("Scientific child has no durable parent reservation intent")
                    if not _settle_parent(journal, run_id, intent, child):
                        unknown = True
                        blocked_reason = "child_resource_usage_requires_reconciliation"
                        break
                    continue
                if journal.get(run_id)["state"] == "cancellation_requested":
                    blocked_reason = "cancellation_requested"
                    break
                # Intent without a child row is provably pre-dispatch. Clear only
                # that unused reservation, then make a fresh dispatch identity.
                if child is None and intent is not None:
                    journal.settle(run_id, intent["reservation_key"], {"wall_seconds": 0, "cpu_seconds": 0})
                    intent = None
                if intent is None:
                    budget = journal.get(run_id)["budget"]
                    remaining_wall = budget["limits"]["wall_seconds"] - budget["used"].get("wall_seconds", 0) - budget["reserved"].get("wall_seconds", 0)
                    remaining_cpu = budget["limits"]["cpu_seconds"] - budget["used"].get("cpu_seconds", 0) - budget["reserved"].get("cpu_seconds", 0)
                    # A smaller deadline would change this locked experimental condition.
                    if remaining_wall < limits["timeout_seconds"] or remaining_cpu < limits["timeout_seconds"] * limits["cpus"]:
                        blocked_reason = "aggregate_resource_budget_exhausted"
                        break
                    reservation = f"{run_id}:seed:{seed}:{uuid.uuid4().hex}"
                    amounts = {"wall_seconds": limits["timeout_seconds"], "cpu_seconds": limits["timeout_seconds"] * limits["cpus"]}
                    journal.reserve(run_id, reservation, amounts)
                    intent = {"seed": seed, "reservation_key": reservation, "amounts": amounts}
                    journal.event(run_id, "scientific_seed_intent", intent)
                if journal.get(run_id)["state"] == "cancellation_requested":
                    if child is not None and child["state"] == "pending":
                        journal.request_cancel(child["id"])
                    journal.settle(run_id, intent["reservation_key"], {"wall_seconds": 0, "cpu_seconds": 0})
                    blocked_reason = "cancellation_requested"
                    break
                try:
                    execute_child(frozen_task, evaluation_bundle=frozen_curator, runs_dir=runs_dir,
                                  executor=executor, image=image, evaluation_profile="method_output/v1",
                                  parent_run=run_id, seed=seed, executor_config=selection,
                                  _run_id=child["id"] if child is not None else None)
                except BaseException as error:
                    children = _children_by_seed(journal, run_id, seeds)
                    child = children.get(seed)
                    journal.event(run_id, "scientific_child_call_error", {"seed": seed, "error_type": type(error).__name__,
                                  "child_run_id": child["id"] if child else None})
                    if child is None:
                        journal.settle(run_id, intent["reservation_key"], {"wall_seconds": 0, "cpu_seconds": 0})
                        blocked_reason = "child_preflight_failed_before_submission"
                    elif not _settle_parent(journal, run_id, intent, child):
                        unknown = True
                        blocked_reason = "child_outcome_requires_reconciliation"
                    if isinstance(error, (KeyboardInterrupt, SystemExit)):
                        raise
                    if blocked_reason:
                        break
                child = _children_by_seed(journal, run_id, seeds).get(seed)
                if child is None:
                    raise JournalError("Child execution returned without a durable identity")
                if not _settle_parent(journal, run_id, intent, child):
                    unknown = True
                    blocked_reason = "child_outcome_requires_reconciliation"
                    break
        except BaseException as error:
            journal.finish(run_id, attempt, "reconciling" if journal.get(run_id)["budget"]["reserved"] else "blocked",
                           {"reason": "scientific_orchestration_interrupted", "error_type": type(error).__name__})
            raise
        finally:
            stop.set()
            monitor.join(timeout=2)
        children = _children_by_seed(journal, run_id, seeds)
        observations, seed_reports = [], []
        all_executed = True
        any_failed = False
        for seed in seeds:
            child = children.get(seed)
            if child is None:
                all_executed = False
                seed_reports.append({"seed": seed, "status": "not_run", "child_run_id": None})
                continue
            report = _checked_report(child)
            observation = _metric_record(seed, child, report, curator["scorer"])
            observations.append(observation)
            executed = report is not None and report.get("execution") == "pass" and child["state"] == "succeeded"
            all_executed = all_executed and executed
            any_failed = any_failed or (report is not None and report.get("execution") == "fail")
            seed_reports.append({"seed": seed, "status": child["state"], "child_run_id": child["id"],
                                 "execution": report.get("execution") if report else "not_run",
                                 "metric": observation})
        locked = any(e["kind"] == "scientific_protocol_locked" and e["payload"].get("protocol_sha256") == protocol.sha256
                     for e in journal.events(run_id))
        evidence = {"origin": "controller", "protocol_sha256": protocol.sha256,
                    "protocol_locked_before_execution": locked, "task_manifest_sha256": task_hash,
                    "submission_sha256": protocol_data["submission_sha256"], "datasets": protocol_data["datasets"],
                    "execution": {"status": "succeeded" if all_executed else "failed" if any_failed else "not_run",
                                  "clean_run": all_executed, "run_id": run_id,
                    "evidence_refs": [f"runs/{run_id}/evidence.jsonl"]},
                    "metrics": [{key: value for key, value in observation.items() if key != "failure_reason"}
                                for observation in observations]}
        try:
            scientific = evaluate_protocol(protocol, task_manifest=manifest,
                                           submission_sha256=protocol_data["submission_sha256"], measurement_evidence=evidence)
        except EvaluationError as error:
            # Invalid observed assumptions (for example a metric outside its
            # declared bounded range) are evaluation errors, not a stranded
            # active controller or a fabricated scientific disagreement.
            scientific = evaluate_protocol(protocol, task_manifest=manifest,
                                           submission_sha256=protocol_data["submission_sha256"],
                                           measurement_evidence=dict(evidence, metrics=[]))
            scientific.update(evaluation_status="error", evaluation_error=str(error)[:256])
            blocked_reason = "scientific_measurements_violate_protocol"
        scientific.update(run_id=run_id, attempt_id=attempt, seeds=seed_reports,
                          scorer_protocol=spec["scorer_protocol"],
                          child_run_ids=[children[s]["id"] for s in seeds if s in children],
                          resource_budget=journal.get(run_id)["budget"],
                          execution="pass" if all_executed else "fail" if any_failed else "inconclusive")
        if blocked_reason:
            scientific["orchestration_reason"] = blocked_reason
        state = "reconciling" if unknown else "blocked" if blocked_reason and blocked_reason != "cancellation_requested" else "succeeded"
        if blocked_reason:
            scientific.update({"pass": False, "scientific_reproduction": "inconclusive" if evaluation_profile == "reproduction/v1" else "not_assessed",
                               "transfer_verdict": "inconclusive" if evaluation_profile == "transfer/v1" else "not_assessed"})

        def committed(final_state):
            result = dict(scientific, run_state=final_state)
            if final_state == "cancelled":
                result.update({"pass": False, "verdict": "inconclusive", "execution": "inconclusive"})
                result["scientific_reproduction" if evaluation_profile == "reproduction/v1" else "transfer_verdict"] = "inconclusive"
            atomic_json(run_dir / "attempts" / attempt / "evaluation_report.json", result)
            return {"report_data": result, "report_data_sha256": digest(canonical_json(result))}

        journal.finish(run_id, attempt, state, committed)
        completed = _checked_report(journal.get(run_id))
        atomic_json(run_dir / "evaluation_report.json", completed)
        atomic_json(run_dir / "measurement_evidence.json", evidence)
        atomic_json(run_dir / "run_manifest.json", {"schema_version": 1, "run_id": run_id, "state": journal.get(run_id)["state"], **spec})
        (run_dir / "evidence.jsonl").write_text("".join(canonical_json(event).decode() + "\n" for event in journal.events(run_id)), encoding="utf-8")
        return completed


def resume_scientific(run_id, *, runs_dir="./runs", executor=None):
    """Resume a fixed protocol without re-running an already submitted seed."""
    journal = RunJournal(runs_dir)
    run = journal.get(run_id)
    if run["spec"].get("kind") != "scientific":
        raise JournalError("Run is not scientific orchestration")
    return execute_scientific(journal.root / run_id / "frozen/candidate/task_manifest.json",
                              evaluation_bundle=journal.root / run_id / "frozen/curator_manifest.json",
                              runs_dir=runs_dir, executor=executor, _run_id=run_id,
                              evaluation_profile=run["spec"]["evaluation_profile"])
