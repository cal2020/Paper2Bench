"""End-to-end fixed-seed orchestration with synthetic scientific contracts.

These fixtures prove runtime behavior, not reproduction of a published paper.
"""
from pathlib import Path
import os
import shutil

import pytest

from paper2bench import workflow
from paper2bench.bundles import BundleError, atomic_json, canonical_json, digest, read_json
from paper2bench.executors import ExecutionResult
from paper2bench.journal import JournalError, RunJournal
from paper2bench.scientific_runs import _controller_lock, execute_scientific, resume_scientific
from paper2bench.runtime_config import runtime_scope

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/tasks/linear-fit"


def make_task(tmp_path, *, profile="reproduction/v1", seeds=(0, 1, 2), approved=True, reference=0, tolerance=1e-12):
    root = tmp_path / "task"
    shutil.copytree(EXAMPLE, root)
    manifest = read_json(root / "task_manifest.json")
    manifest["source"] = {"paper_sha256": digest(b"synthetic contract test; not a published paper")}
    atomic_json(root / "task_manifest.json", manifest)
    task_hash = digest(canonical_json(manifest))
    code_hash = manifest["files"][manifest["entrypoint"]]
    paper_hash = manifest["source"]["paper_sha256"]
    protocol = {"schema_version": 1, "profile": profile, "protocol_id": "runtime-contract-fixture",
                "task_manifest_sha256": task_hash, "submission_sha256": code_hash,
                "source": {"paper_sha256": paper_hash, "citations": [
                    {"id": "method", "locator": "synthetic method definition",
                     "claim": "Test-only linear fitting contract, not a paper result"}]},
                "datasets": {manifest["inputs"]: manifest["files"][manifest["inputs"]]},
                "plan": {"seed_schedule": list(seeds), "stopping_rule": "fixed_schedule",
                         "variability_source": "controlled test seed", "independence_assumption": False},
                "criteria": [
                    {"id": "execution", "kind": "execution", "dimension": "execution", "critical": True, "depends_on": []},
                    {"id": "fidelity", "kind": "review", "dimension": "implementation_fidelity", "critical": True,
                     "depends_on": ["execution"], "review_id": "review"},
                    {"id": "metric", "kind": "metric", "dimension": "empirical_agreement" if profile == "reproduction/v1" else "transfer",
                     "critical": True, "depends_on": ["execution", "fidelity"], "metric_id": "mean_squared_error",
                     "units": "target_units_squared", "rule": "absolute_tolerance", "reference": reference, "tolerance": tolerance}],
                "reviews": [{"id": "review", "reviewer": "test-only curator", "status": "pass" if approved else "inconclusive",
                             "submission_sha256": code_hash, "paper_sha256": paper_hash,
                             "source_citations": ["method"], "evidence_refs": ["curator/test-review.json"]}]}
    if profile == "transfer/v1":
        protocol["transformation"] = {"parent_task_sha256": "a" * 64, "changed_factors": ["synthetic noise condition"],
                                      "expected_outcome_source": "variant_protocol"}
    curator = read_json(root / "curator_manifest.json")
    curator.update(task_manifest_sha256=task_hash, protocol=protocol)
    atomic_json(root / "curator_manifest.json", curator)
    return root / "task_manifest.json", root / "curator_manifest.json"


class Worker:
    def __init__(self, *, fail_seed=None, unknown_seed=None, wrong_seed=None):
        self.calls = []
        self.fail_seed, self.unknown_seed, self.wrong_seed = fail_seed, unknown_seed, wrong_seed

    def preflight(self):
        return {"image_id": "sha256:" + "1" * 64}

    def cancel(self):
        return True

    def execute(self, request):
        self.calls.append(request.seed)
        assert not (request.workspace / "curator_manifest.json").exists()
        offset = 1 if request.seed == self.wrong_seed else 0
        atomic_json(request.workspace / "predictions.json", {"predictions": [9+offset, 11+offset, 21+offset], "pass": True})
        return ExecutionResult("failed" if request.seed == self.fail_seed else "succeeded",
                               1 if request.seed == self.fail_seed else 0, "", "", 0.01,
                               evidence={"cleanup_confirmed": request.seed != self.unknown_seed})


def parent_records(journal):
    import sqlite3
    with sqlite3.connect(journal.path) as db:
        ids = [row[0] for row in db.execute("SELECT id FROM runs")]
    return [journal.get(identity) for identity in ids if journal.get(identity)["spec"].get("kind") == "scientific"]


def test_multiseed_jobs_are_fresh_pinned_and_independently_scored(tmp_path):
    task, curator = make_task(tmp_path)
    worker = Worker()
    report = execute_scientific(task, evaluation_bundle=curator, executor=worker, runs_dir=tmp_path / "runs")
    assert report["pass"] and report["scientific_reproduction"] == "pass"
    assert worker.calls == [0, 1, 2]
    assert len(set(report["child_run_ids"])) == 3
    assert report["resource_budget"]["reserved"] == {}
    journal = RunJournal(tmp_path / "runs")
    children = journal.children(report["run_id"])
    assert {child["spec"]["seed"] for child in children} == {0, 1, 2}
    assert all(child["spec"]["image"] == "sha256:" + "1" * 64 for child in children)
    assert resume_scientific(report["run_id"], runs_dir=tmp_path / "runs", executor=worker) == report
    assert worker.calls == [0, 1, 2]


def test_wrong_seed_result_is_not_retried_or_cherry_picked(tmp_path):
    task, curator = make_task(tmp_path)
    worker = Worker(wrong_seed=1)
    report = execute_scientific(task, evaluation_bundle=curator, executor=worker, runs_dir=tmp_path / "runs")
    assert not report["pass"] and report["scientific_reproduction"] == "fail"
    assert report["contradictory_faithful_result"]
    assert worker.calls == [0, 1, 2]
    assert report["criteria"][-1]["summary"]["values"] == [0, 1, 0]


def test_failed_seed_is_retained_and_never_silently_replaced(tmp_path):
    task, curator = make_task(tmp_path)
    worker = Worker(fail_seed=1)
    report = execute_scientific(task, evaluation_bundle=curator, executor=worker, runs_dir=tmp_path / "runs")
    assert not report["pass"]
    assert [seed["seed"] for seed in report["seeds"]] == [0, 1, 2]
    assert report["seeds"][1]["execution"] == "fail"
    assert worker.calls == [0, 1, 2]


def test_unconfirmed_child_stops_dispatch_and_retains_aggregate_reservation(tmp_path):
    task, curator = make_task(tmp_path)
    worker = Worker(unknown_seed=1)
    report = execute_scientific(task, evaluation_bundle=curator, executor=worker, runs_dir=tmp_path / "runs")
    assert not report["pass"] and report["run_state"] == "reconciling"
    assert worker.calls == [0, 1]
    assert report["resource_budget"]["reserved"]["wall_seconds"] > 0
    with pytest.raises(JournalError, match="child outcome"):
        resume_scientific(report["run_id"], runs_dir=tmp_path / "runs", executor=worker)
    assert worker.calls == [0, 1]


def test_resume_after_parent_interrupt_reuses_completed_children(tmp_path, monkeypatch):
    task, curator = make_task(tmp_path)
    worker = Worker()
    original = workflow.execute

    def interrupt_after_child(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get("seed") == 0:
            raise KeyboardInterrupt("simulated parent interrupt after completed child")
        return result

    monkeypatch.setattr(workflow, "execute", interrupt_after_child)
    with pytest.raises(KeyboardInterrupt):
        execute_scientific(task, evaluation_bundle=curator, executor=worker, runs_dir=tmp_path / "runs")
    journal = RunJournal(tmp_path / "runs")
    parent = parent_records(journal)[0]
    monkeypatch.setattr(workflow, "execute", original)
    report = resume_scientific(parent["id"], runs_dir=tmp_path / "runs", executor=worker)
    assert report["pass"] and worker.calls == [0, 1, 2]
    assert len(journal.children(parent["id"])) == 3


def test_pre_dispatch_interrupt_can_resume_without_losing_reservation_integrity(tmp_path, monkeypatch):
    task, curator = make_task(tmp_path)
    worker = Worker()
    original = workflow.execute
    monkeypatch.setattr(workflow, "execute", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt("before child row")))
    with pytest.raises(KeyboardInterrupt):
        execute_scientific(task, evaluation_bundle=curator, executor=worker, runs_dir=tmp_path / "runs")
    journal = RunJournal(tmp_path / "runs")
    parent = parent_records(journal)[0]
    assert journal.get(parent["id"])["budget"]["reserved"] == {}
    monkeypatch.setattr(workflow, "execute", original)
    report = resume_scientific(parent["id"], runs_dir=tmp_path / "runs", executor=worker)
    assert report["pass"] and worker.calls == [0, 1, 2]


def test_unapproved_fidelity_review_blocks_before_any_dispatch(tmp_path):
    task, curator = make_task(tmp_path, approved=False)
    worker = Worker()
    with pytest.raises(BundleError, match="passing critical review"):
        execute_scientific(task, evaluation_bundle=curator, executor=worker, runs_dir=tmp_path / "runs")
    assert not worker.calls


def test_transfer_uses_own_changed_result_rule(tmp_path):
    task, curator = make_task(tmp_path, profile="transfer/v1", reference=1/3, tolerance=1e-12)
    worker = Worker(wrong_seed=1)
    report = execute_scientific(task, evaluation_bundle=curator, executor=worker, runs_dir=tmp_path / "runs", evaluation_profile="transfer/v1")
    assert report["pass"] and report["transfer_verdict"] == "pass"
    assert report["scientific_reproduction"] == "not_assessed"


def test_live_controller_lock_prevents_resume_stealing(tmp_path, monkeypatch):
    task, curator = make_task(tmp_path)
    original = workflow.execute
    monkeypatch.setattr(workflow, "execute", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        execute_scientific(task, evaluation_bundle=curator, executor=Worker(), runs_dir=tmp_path / "runs")
    parent = parent_records(RunJournal(tmp_path / "runs"))[0]
    monkeypatch.setattr(workflow, "execute", original)
    with _controller_lock(tmp_path / "runs" / parent["id"]):
        with pytest.raises(JournalError, match="already active"):
            resume_scientific(parent["id"], runs_dir=tmp_path / "runs", executor=Worker())


def test_changed_frozen_submission_is_rejected_on_resume(tmp_path, monkeypatch):
    task, curator = make_task(tmp_path)
    original = workflow.execute
    monkeypatch.setattr(workflow, "execute", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        execute_scientific(task, evaluation_bundle=curator, executor=Worker(), runs_dir=tmp_path / "runs")
    parent = parent_records(RunJournal(tmp_path / "runs"))[0]
    frozen = tmp_path / "runs" / parent["id"] / "frozen/candidate/solution.py"
    frozen.write_text("print('changed')")
    monkeypatch.setattr(workflow, "execute", original)
    with pytest.raises(BundleError, match="digest"):
        resume_scientific(parent["id"], runs_dir=tmp_path / "runs", executor=Worker())


def test_pending_child_created_before_attempt_can_resume_once(tmp_path, monkeypatch):
    task, curator = make_task(tmp_path)
    worker = Worker()
    original = RunJournal.begin_attempt
    interrupted = []

    def interrupt_child(self, run_id):
        if self.get(run_id)["spec"].get("kind") == "execution" and not interrupted:
            interrupted.append(run_id)
            raise KeyboardInterrupt("after child creation, before attempt")
        return original(self, run_id)

    monkeypatch.setattr(RunJournal, "begin_attempt", interrupt_child)
    with pytest.raises(KeyboardInterrupt):
        execute_scientific(task, evaluation_bundle=curator, executor=worker, runs_dir=tmp_path / "runs")
    journal = RunJournal(tmp_path / "runs")
    parent = parent_records(journal)[0]
    assert journal.children(parent["id"])[0]["state"] == "pending"
    assert journal.get(parent["id"])["budget"]["reserved"]["wall_seconds"] > 0
    monkeypatch.setattr(RunJournal, "begin_attempt", original)
    report = resume_scientific(parent["id"], runs_dir=tmp_path / "runs", executor=worker)
    assert report["pass"] and worker.calls == [0, 1, 2]
    assert len(journal.children(parent["id"])) == 3


def test_crash_between_reservation_and_seed_intent_does_not_leak_budget(tmp_path, monkeypatch):
    task, curator = make_task(tmp_path)
    worker = Worker()
    original = RunJournal.event
    once = []

    def interrupt_intent(self, run_id, kind, payload):
        if kind == "scientific_seed_intent" and not once:
            once.append(True)
            raise KeyboardInterrupt("reservation committed, intent not committed")
        return original(self, run_id, kind, payload)

    monkeypatch.setattr(RunJournal, "event", interrupt_intent)
    with pytest.raises(KeyboardInterrupt):
        execute_scientific(task, evaluation_bundle=curator, executor=worker, runs_dir=tmp_path / "runs")
    journal = RunJournal(tmp_path / "runs")
    parent = parent_records(journal)[0]
    assert journal.get(parent["id"])["budget"]["reserved"]["wall_seconds"] > 0
    monkeypatch.setattr(RunJournal, "event", original)
    report = resume_scientific(parent["id"], runs_dir=tmp_path / "runs", executor=worker)
    assert report["pass"] and worker.calls == [0, 1, 2]
    assert journal.get(parent["id"])["budget"]["reserved"] == {}


def test_completed_scientific_report_recovers_missing_summary_without_execution(tmp_path):
    task, curator = make_task(tmp_path)
    worker = Worker()
    report = execute_scientific(task, evaluation_bundle=curator, executor=worker, runs_dir=tmp_path / "runs")
    summary = tmp_path / "runs" / report["run_id"] / "evaluation_report.json"
    summary.unlink()
    assert resume_scientific(report["run_id"], runs_dir=tmp_path / "runs", executor=worker) == report
    assert summary.is_file() and worker.calls == [0, 1, 2]


def test_aggregate_experiment_budget_stops_later_seeds_without_false_pass(tmp_path):
    task, curator = make_task(tmp_path)
    worker = Worker()
    config = {"models": {"local": {"provider": "ollama", "model": "unused", "hosting": "local_server"}},
              "roles": {"author": "local"}, "budgets": {"experiments": {"wall_seconds": 10, "cpu_seconds": 10}}}
    with runtime_scope(config):
        report = execute_scientific(task, evaluation_bundle=curator, executor=worker, runs_dir=tmp_path / "runs")
    assert worker.calls == [0]
    assert not report["pass"] and report["scientific_reproduction"] == "inconclusive"
    assert report["run_state"] == "blocked" and report["orchestration_reason"] == "aggregate_resource_budget_exhausted"
    assert report["resource_budget"]["limits"] == {"wall_seconds": 10, "cpu_seconds": 10}


def test_invalid_observed_statistical_assumption_is_durable_error_not_active_run(tmp_path):
    task, curator_path = make_task(tmp_path)
    curator = read_json(curator_path)
    curator["protocol"]["criteria"][-1]["bounds"] = [0, 0.5]
    atomic_json(curator_path, curator)
    worker = Worker(wrong_seed=1)
    report = execute_scientific(task, evaluation_bundle=curator_path, executor=worker, runs_dir=tmp_path / "runs")
    assert worker.calls == [0, 1, 2]
    assert not report["pass"] and report["evaluation_status"] == "error"
    assert report["run_state"] == "blocked" and report["scientific_reproduction"] == "inconclusive"
    assert report["coverage"]["unknown_weight"] > 0
    assert RunJournal(tmp_path / "runs").get(report["run_id"])["state"] == "blocked"


@pytest.mark.skipif(os.getenv("P2B_DOCKER_TESTS") != "1", reason="opt-in real isolated scientific contract execution")
def test_real_multiseed_isolated_contract_execution(tmp_path):
    task, curator = make_task(tmp_path, seeds=(0, 1))
    report = execute_scientific(task, evaluation_bundle=curator, runs_dir=tmp_path / "runs")
    assert report["pass"] and report["scientific_reproduction"] == "pass"
    assert len(report["child_run_ids"]) == 2
    assert len(set(report["child_run_ids"])) == 2
    children = RunJournal(tmp_path / "runs").children(report["run_id"])
    assert {child["spec"]["seed"] for child in children} == {0, 1}
    assert all(child["state"] == "succeeded" for child in children)


def test_infeasible_knapsack_objective_cannot_pass_matching_numeric_reference(tmp_path):
    task_path, curator_path = make_task(tmp_path, seeds=(0,), reference=12, tolerance=0)
    task = read_json(task_path)
    for name in ("solution.py", "input.json"):
        source = EXAMPLE.parent / "knapsack" / name
        (task_path.parent / name).write_bytes(source.read_bytes())
        task["files"][name] = digest(source.read_bytes())
    atomic_json(task_path, task)
    curator = read_json(curator_path)
    task_hash = digest(canonical_json(task))
    code_hash = task["files"]["solution.py"]
    curator.update(scorer="knapsack_optimum/v1", reference={}, task_manifest_sha256=task_hash)
    protocol = curator["protocol"]
    protocol.update(task_manifest_sha256=task_hash, submission_sha256=code_hash,
                    datasets={"input.json": task["files"]["input.json"]})
    protocol["reviews"][0]["submission_sha256"] = code_hash
    protocol["criteria"][-1].update(metric_id="feasible_optimal_objective", units="objective_units")
    atomic_json(curator_path, curator)

    class InfeasibleWorker(Worker):
        def execute(self, request):
            self.calls.append(request.seed)
            # weights 2+3+4=9 exceed capacity8, but values3+4+5=12 match optimum12.
            atomic_json(request.workspace / "predictions.json", {"selected": [0, 1, 2]})
            return ExecutionResult("succeeded", 0, "", "", 0.01, evidence={"cleanup_confirmed": True})

    report = execute_scientific(task_path, evaluation_bundle=curator_path, executor=InfeasibleWorker(), runs_dir=tmp_path / "runs")
    assert report["execution"] == "pass" and not report["pass"]
    assert report["scientific_reproduction"] == "inconclusive"
    assert report["seeds"][0]["metric"]["status"] == "failed"
    assert report["seeds"][0]["metric"]["failure_reason"] == "infeasible_solution"


def test_parent_cancellation_cannot_leave_a_passing_report(tmp_path):
    task, curator = make_task(tmp_path)
    runs = tmp_path / "runs"

    class CancellingWorker(Worker):
        def execute(self, request):
            child_id = request.workspace.parents[2].name
            journal = RunJournal(runs)
            parent_id = journal.get(child_id)["spec"]["parent_run"]
            journal.request_cancel(parent_id)
            return super().execute(request)

    worker = CancellingWorker()
    report = execute_scientific(task, evaluation_bundle=curator, executor=worker, runs_dir=runs)
    assert not report["pass"] and report["run_state"] == "cancelled"
    assert report["scientific_reproduction"] == "inconclusive"
    assert report["verdict"] == "inconclusive" and worker.calls == [0]
    stored = RunJournal(runs).get(report["run_id"])
    assert stored["attempts"][-1]["result"]["report_data"]["pass"] is False


@pytest.mark.parametrize("field,bad_value", [
    ("task_sha256", "a" * 64), ("evaluation_sha256", "b" * 64),
    ("image", "sha256:" + "c" * 64),
    ("executor_config", {"backend": "injected", "class": "another.Worker"}),
    ("limits", {"timeout_seconds": 1000}),
    ("scorer_protocol", "unrelated-scorer/v99"),
])
def test_misattached_child_condition_cannot_supply_scientific_measurement(tmp_path, monkeypatch, field, bad_value):
    """Simulate a trusted controller grouping bug, not candidate DB access."""
    import json
    import sqlite3
    task, curator = make_task(tmp_path, seeds=(0,))
    runs = tmp_path / "runs"
    original = workflow.execute

    def wrong_condition(*args, **kwargs):
        report = original(*args, **kwargs)
        journal = RunJournal(runs)
        child = journal.get(report["run_id"])
        spec = dict(child["spec"], **{field: bad_value})
        with sqlite3.connect(journal.path) as db:
            db.execute("UPDATE runs SET spec=? WHERE id=?", (json.dumps(spec), child["id"]))
        return report

    monkeypatch.setattr(workflow, "execute", wrong_condition)
    with pytest.raises(JournalError, match="identity|condition"):
        execute_scientific(task, evaluation_bundle=curator, executor=Worker(), runs_dir=runs)


@pytest.mark.parametrize("field,bad_value", [
    ("task_sha256", "a" * 64), ("evaluator_sha256", "b" * 64),
    ("seed", 100), ("seed", True),
    ("evaluation_protocol", "execution/v1"), ("attempt_id", "unrelated-attempt"),
    ("scorer_protocol", "unrelated-scorer/v99"),
])
def test_validly_hashed_report_from_wrong_condition_is_rejected(tmp_path, monkeypatch, field, bad_value):
    import json
    import sqlite3
    task, curator = make_task(tmp_path, seeds=(0,))
    runs = tmp_path / "runs"
    original = workflow.execute

    def wrong_report(*args, **kwargs):
        report = original(*args, **kwargs)
        journal = RunJournal(runs)
        child = journal.get(report["run_id"])
        result = child["attempts"][-1]["result"]
        result["report_data"][field] = bad_value
        result["report_data_sha256"] = digest(canonical_json(result["report_data"]))
        with sqlite3.connect(journal.path) as db:
            db.execute("UPDATE attempts SET result=? WHERE id=?",
                       (json.dumps(result), child["attempts"][-1]["id"]))
        return report

    monkeypatch.setattr(workflow, "execute", wrong_report)
    with pytest.raises(JournalError, match="condition|attempt"):
        execute_scientific(task, evaluation_bundle=curator, executor=Worker(), runs_dir=runs)


def test_scientific_reports_pin_both_installed_evaluator_protocols(tmp_path):
    from paper2bench.evaluation.science import SCIENTIFIC_EVALUATOR_PROTOCOL
    from paper2bench.methods import SCORER_PROTOCOLS

    task, curator = make_task(tmp_path, seeds=(0,))
    runs = tmp_path / "runs"
    report = execute_scientific(task, evaluation_bundle=curator, executor=Worker(), runs_dir=runs)
    spec = RunJournal(runs).get(report["run_id"])["spec"]
    assert report["scorer_protocol"] == spec["scorer_protocol"] == SCORER_PROTOCOLS["linear_predictions/v1"]
    assert report["scientific_evaluator_protocol"] == spec["scientific_evaluator_protocol"] == SCIENTIFIC_EVALUATOR_PROTOCOL


@pytest.mark.parametrize("changed", ["scorer", "scientific_evaluator"])
def test_completed_scientific_cache_and_reproduction_reject_changed_implementation(tmp_path, monkeypatch, changed):
    import paper2bench.scientific_runs as scientific

    task, curator = make_task(tmp_path, seeds=(0,))
    runs = tmp_path / "runs"
    worker = Worker()
    report = execute_scientific(task, evaluation_bundle=curator, executor=worker, runs_dir=runs)
    if changed == "scorer":
        monkeypatch.setitem(scientific.SCORER_PROTOCOLS, "linear_predictions/v1", "linear_predictions/v99")
    else:
        monkeypatch.setattr(scientific, "SCIENTIFIC_EVALUATOR_PROTOCOL", "scientific_criteria/v99")
    with pytest.raises(JournalError, match="protocol changed"):
        resume_scientific(report["run_id"], runs_dir=runs, executor=worker)
    with pytest.raises(JournalError, match="protocol changed"):
        workflow.reproduce(report["run_id"], runs_dir=runs, executor=worker)
    assert worker.calls == [0]


def test_unpinned_legacy_scientific_cache_requires_new_run(tmp_path):
    import json
    import sqlite3

    task, curator = make_task(tmp_path, seeds=(0,))
    runs = tmp_path / "runs"
    worker = Worker()
    report = execute_scientific(task, evaluation_bundle=curator, executor=worker, runs_dir=runs)
    journal = RunJournal(runs)
    spec = journal.get(report["run_id"])["spec"]
    spec.pop("scientific_evaluator_protocol")
    with sqlite3.connect(journal.path) as db:
        db.execute("UPDATE runs SET spec=? WHERE id=?", (json.dumps(spec), report["run_id"]))
    with pytest.raises(JournalError, match="unpinned"):
        resume_scientific(report["run_id"], runs_dir=runs, executor=worker)
    assert worker.calls == [0]


@pytest.mark.parametrize("changed", ["solution.py", "curator_manifest.json", "scientific_protocol.json"])
def test_completed_scientific_cache_and_reproduction_revalidate_frozen_inputs(tmp_path, changed):
    task, curator = make_task(tmp_path, seeds=(0,))
    runs = tmp_path / "runs"
    worker = Worker()
    report = execute_scientific(task, evaluation_bundle=curator, executor=worker, runs_dir=runs)
    frozen = runs / report["run_id"] / "frozen"
    target = frozen / "candidate" / changed if changed == "solution.py" else frozen / changed
    target.write_text("tampered")
    with pytest.raises(JournalError, match="Frozen scientific"):
        resume_scientific(report["run_id"], runs_dir=runs, executor=worker)
    with pytest.raises(JournalError, match="Frozen scientific"):
        workflow.reproduce(report["run_id"], runs_dir=runs, executor=worker)
    assert worker.calls == [0]


def test_linked_scientific_reproduction_cannot_change_frozen_evaluation_condition(tmp_path):
    task, curator_path = make_task(tmp_path, seeds=(0,))
    runs = tmp_path / "runs"
    worker = Worker()
    report = execute_scientific(task, evaluation_bundle=curator_path, executor=worker, runs_dir=runs)
    curator = read_json(curator_path)
    curator["protocol"]["criteria"][-1]["tolerance"] = 100
    atomic_json(curator_path, curator)
    with pytest.raises(JournalError, match="condition differs"):
        execute_scientific(task, evaluation_bundle=curator_path, executor=worker, runs_dir=runs,
                           parent_run=report["run_id"])
    assert worker.calls == [0]
