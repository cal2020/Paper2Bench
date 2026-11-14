"""Frozen scientific protocols and evidence-based deterministic verdicts.

The controller owns protocol locking, evidence construction, scorer execution and
curator access. This module never imports candidate code or treats a candidate's
JSON provenance/metric assertions as authoritative. Dicts accepted by this API
are a trusted-controller interface, not an untrusted submission format.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from .statistics import compare_summary, finite_number, paired_difference, summarize
from .validation import EvaluationError, weighted_coverage

PROFILES = {"reproduction/v1", "transfer/v1"}
SCIENTIFIC_EVALUATOR_PROTOCOL = "scientific_criteria/v2"
VERDICTS = {"pass", "fail", "inconclusive", "not_run", "not_applicable"}
_SHA = re.compile(r"[0-9a-f]{64}")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}")


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, RecursionError, UnicodeError) as error:
        raise EvaluationError("Protocol must contain bounded finite JSON data") from error


def _sha(value, label):
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise EvaluationError(f"{label} requires a SHA-256 digest")
    return value


def _text(value, label, *, identifier=False):
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise EvaluationError(f"{label} requires bounded nonempty text")
    if identifier and not _ID.fullmatch(value):
        raise EvaluationError(f"{label} requires a stable identifier")
    return value


def _object(value, allowed, label, required=()):
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        raise EvaluationError(f"Invalid {label} fields")
    return value


def _references(value, label):
    if not isinstance(value, list) or not 1 <= len(value) <= 128:
        raise EvaluationError(f"{label} requires bounded evidence references")
    for ref in value:
        _text(ref, label)
    if len(set(value)) != len(value):
        raise EvaluationError(f"{label} cannot contain duplicates")
    return value


@dataclass(frozen=True)
class FrozenProtocol:
    """Canonical bytes prevent caller mutation after pre-dispatch validation."""
    canonical: bytes
    sha256: str

    def as_dict(self):
        return json.loads(self.canonical)


def freeze_protocol(protocol, *, task_manifest, submission_sha256):
    """Validate all static bindings before the controller dispatches a job."""
    if isinstance(protocol, FrozenProtocol):
        if hashlib.sha256(protocol.canonical).hexdigest() != protocol.sha256:
            raise EvaluationError("Frozen protocol integrity mismatch")
        protocol = protocol.as_dict()
    _object(protocol, {"schema_version", "profile", "protocol_id", "task_manifest_sha256",
                       "submission_sha256", "source", "datasets", "plan", "criteria",
                       "reviews", "transformation"}, "scientific protocol",
            {"schema_version", "profile", "protocol_id", "task_manifest_sha256", "submission_sha256",
             "source", "datasets", "plan", "criteria"})
    if type(protocol["schema_version"]) is not int or protocol["schema_version"] != 1 or not isinstance(protocol["profile"], str) or protocol["profile"] not in PROFILES:
        raise EvaluationError("Unsupported scientific protocol version/profile")
    _text(protocol["protocol_id"], "protocol_id", identifier=True)
    if not isinstance(task_manifest, dict) or not isinstance(task_manifest.get("files"), dict):
        raise EvaluationError("Task file identities are required")
    task_digest = hashlib.sha256(_canonical(task_manifest)).hexdigest()
    if _sha(protocol["task_manifest_sha256"], "task binding") != task_digest:
        raise EvaluationError("Scientific protocol task identity mismatch")
    if _sha(protocol["submission_sha256"], "submission binding") != _sha(submission_sha256, "actual submission"):
        raise EvaluationError("Scientific protocol submission identity mismatch")
    entrypoint = task_manifest.get("entrypoint")
    if entrypoint not in task_manifest["files"] or task_manifest["files"][entrypoint] != submission_sha256:
        raise EvaluationError("Submission must be the task's pinned entrypoint")
    source = _object(protocol["source"], {"paper_sha256", "citations"}, "protocol source", {"paper_sha256", "citations"})
    paper_digest = _sha(source["paper_sha256"], "source paper")
    if (not isinstance(task_manifest.get("source"), dict)
            or task_manifest["source"].get("paper_sha256") != paper_digest):
        raise EvaluationError("Scientific source is not bound to the task's paper")
    citations = source["citations"]
    if not isinstance(citations, list) or not 1 <= len(citations) <= 128:
        raise EvaluationError("Scientific source citations are required")
    citation_ids = set()
    for citation in citations:
        _object(citation, {"id", "locator", "claim"}, "source citation", {"id", "locator", "claim"})
        identifier = _text(citation["id"], "citation id", identifier=True)
        if identifier in citation_ids:
            raise EvaluationError("Duplicate source citation")
        citation_ids.add(identifier)
        _text(citation["locator"], "source locator")
        _text(citation["claim"], "source claim")
    datasets = protocol["datasets"]
    if not isinstance(datasets, dict) or not 1 <= len(datasets) <= 128 or task_manifest.get("inputs") not in datasets:
        raise EvaluationError("Input dataset identities must be declared")
    for path, content_digest in datasets.items():
        _sha(content_digest, "dataset")
        if path not in task_manifest["files"] or task_manifest["files"][path] != content_digest:
            raise EvaluationError("Protocol data identity does not match the task")
    plan = _object(protocol["plan"], {"seed_schedule", "stopping_rule", "variability_source", "sampling_unit",
                                       "independence_assumption", "confidence_level"}, "experiment plan",
                   {"seed_schedule", "stopping_rule", "variability_source", "independence_assumption"})
    seeds = plan["seed_schedule"]
    if (not isinstance(seeds, list) or not 1 <= len(seeds) <= 10000
            or any(type(seed) is not int or not 0 <= seed <= 2**32 - 1 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise EvaluationError("Seed schedule must contain unique bounded integer seeds")
    if plan["stopping_rule"] != "fixed_schedule" or type(plan["independence_assumption"]) is not bool:
        raise EvaluationError("Fixed stopping rule and explicit independence assumption are required")
    _text(plan["variability_source"], "variability source")
    if "sampling_unit" in plan:
        _text(plan["sampling_unit"], "sampling unit")
    confidence = finite_number(plan.get("confidence_level", 0.95), "confidence level")
    if not 0 < confidence <= 0.999999:
        raise EvaluationError("Confidence level must be in (0,0.999999]")
    reviews = protocol.get("reviews", [])
    if not isinstance(reviews, list) or len(reviews) > 128:
        raise EvaluationError("Reviews must be a bounded list")
    review_ids = set()
    for review in reviews:
        _object(review, {"id", "reviewer", "status", "submission_sha256", "paper_sha256", "source_citations",
                         "evidence_refs"}, "curator review", {"id", "reviewer", "status", "submission_sha256",
                                                               "paper_sha256", "source_citations", "evidence_refs"})
        identity = _text(review["id"], "review id", identifier=True)
        if identity in review_ids or not isinstance(review["status"], str) or review["status"] not in {"pass", "fail", "inconclusive"}:
            raise EvaluationError("Duplicate review or invalid review status")
        review_ids.add(identity)
        _text(review["reviewer"], "reviewer")
        if review["submission_sha256"] != submission_sha256 or review["paper_sha256"] != paper_digest:
            raise EvaluationError("Curator review does not apply to the frozen submission/paper")
        _references(review["source_citations"], "review source citations")
        if set(review["source_citations"]) - citation_ids:
            raise EvaluationError("Review cites an unknown source passage")
        _references(review["evidence_refs"], "review evidence")
    criteria = protocol["criteria"]
    if not isinstance(criteria, list) or not 3 <= len(criteria) <= 128:
        raise EvaluationError("Scientific rubric requires 3 to 128 criteria")
    by_id = {}
    desired_dimension = "empirical_agreement" if protocol["profile"] == "reproduction/v1" else "transfer"
    for criterion in criteria:
        common = {"id", "kind", "dimension", "critical", "depends_on", "weight", "source_citations"}
        kind = criterion.get("kind") if isinstance(criterion, dict) else None
        extra = ({"review_id"} if kind == "review" else
                 {"metric_id", "baseline_id", "units", "rule", "reference", "tolerance", "bounds", "min_samples_per_run"}
                 if kind == "metric" else set())
        _object(criterion, common | extra, "criterion", {"id", "kind", "dimension", "critical", "depends_on"})
        identity = _text(criterion["id"], "criterion id", identifier=True)
        if identity in by_id or not isinstance(kind, str) or kind not in {"execution", "review", "metric"} or type(criterion["critical"]) is not bool:
            raise EvaluationError("Duplicate criterion or invalid kind/critical flag")
        by_id[identity] = criterion
        weight = finite_number(criterion.get("weight", 1), "criterion weight")
        if weight <= 0:
            raise EvaluationError("Criterion weights must be positive")
        deps = criterion["depends_on"]
        if not isinstance(deps, list) or len(deps) > 128 or any(not isinstance(x, str) for x in deps) or len(set(deps)) != len(deps):
            raise EvaluationError("Criterion dependencies must be unique identifiers")
        if "source_citations" in criterion:
            _references(criterion["source_citations"], "criterion source citations")
            if set(criterion["source_citations"]) - citation_ids:
                raise EvaluationError("Criterion cites an unknown source passage")
        if kind == "execution" and criterion["dimension"] != "execution":
            raise EvaluationError("Execution criterion has the wrong dimension")
        if kind == "review":
            if criterion["dimension"] != "implementation_fidelity":
                raise EvaluationError("Review criterion must assess implementation fidelity")
            _text(criterion.get("review_id"), "review binding", identifier=True)
        if kind == "metric":
            if criterion["dimension"] != desired_dimension:
                raise EvaluationError("Metric criterion is not scoped to this protocol profile")
            _text(criterion.get("metric_id"), "metric id", identifier=True)
            if "baseline_id" in criterion:
                _text(criterion["baseline_id"], "baseline metric id", identifier=True)
            _text(criterion.get("units"), "metric units")
            rule = criterion.get("rule")
            if not isinstance(rule, str) or rule not in {"absolute_tolerance", "relative_tolerance", "at_least", "at_most", "equivalence_interval"}:
                raise EvaluationError("Unsupported metric comparison")
            reference = finite_number(criterion.get("reference"), "reference")
            if rule in {"absolute_tolerance", "relative_tolerance", "equivalence_interval"}:
                if finite_number(criterion.get("tolerance"), "tolerance") < 0:
                    raise EvaluationError("Tolerance cannot be negative")
            elif "tolerance" in criterion:
                raise EvaluationError("Threshold comparisons do not use a tolerance")
            if rule == "relative_tolerance" and reference == 0:
                raise EvaluationError("Zero-reference relative comparison is undefined")
            if "min_samples_per_run" in criterion and (type(criterion["min_samples_per_run"]) is not int or criterion["min_samples_per_run"] < 1):
                raise EvaluationError("Minimum samples must be positive")
            if "bounds" in criterion:
                summarize([criterion["bounds"][0]] if isinstance(criterion["bounds"], list) and criterion["bounds"] else [],
                          bounds=criterion["bounds"])
    order, visiting, visited = [], set(), set()

    def visit(identity):
        if identity in visiting:
            raise EvaluationError("Criterion dependencies contain a cycle")
        if identity in visited:
            return
        if identity not in by_id:
            raise EvaluationError("Criterion dependency is unknown")
        visiting.add(identity)
        for dependency in by_id[identity]["depends_on"]:
            visit(dependency)
        visiting.remove(identity)
        visited.add(identity)
        order.append(identity)

    for identity in by_id:
        visit(identity)
    execution_ids = {c["id"] for c in criteria if c["kind"] == "execution" and c["critical"]}
    fidelity_ids = {c["id"] for c in criteria if c["kind"] == "review" and c["critical"]}
    metric_ids = {c["id"] for c in criteria if c["kind"] == "metric" and c["critical"]}
    if not execution_ids or not fidelity_ids or not metric_ids:
        raise EvaluationError("Critical execution, fidelity and result gates are required")
    weighted_coverage([dict(c, verdict="not_run") for c in criteria])

    # ``order`` is topological. Memoize closures once per node: recursively
    # expanding shared paths makes a small, valid dense DAG exponential work.
    ancestry = {}
    for identity in order:
        dependencies = by_id[identity]["depends_on"]
        result = set(dependencies)
        for dependency in dependencies:
            result.update(ancestry[dependency])
        ancestry[identity] = result

    for identity in (c["id"] for c in criteria if c["kind"] == "metric"):
        if not execution_ids | fidelity_ids <= ancestry[identity]:
            raise EvaluationError("Critical result criteria must depend on execution and fidelity gates")
    if protocol["profile"] == "transfer/v1":
        transformation = _object(protocol.get("transformation"), {"parent_task_sha256", "changed_factors", "expected_outcome_source"},
                                 "transfer transformation", {"parent_task_sha256", "changed_factors", "expected_outcome_source"})
        _sha(transformation["parent_task_sha256"], "parent task")
        _references(transformation["changed_factors"], "changed factors")
        if transformation["expected_outcome_source"] != "variant_protocol":
            raise EvaluationError("Transfer result rules must belong to the changed protocol, not the source answer")
    elif "transformation" in protocol:
        raise EvaluationError("Reproduction protocols cannot hide a transfer transformation")
    encoded = _canonical(protocol)
    if len(encoded) > 2 * 1024 * 1024:
        raise EvaluationError("Scientific protocol exceeds its size bound")
    return FrozenProtocol(encoded, hashlib.sha256(encoded).hexdigest())


validate_protocol = freeze_protocol


def _measurement_envelope(evidence, frozen, task_manifest, submission_sha256):
    if evidence is None or evidence == {}:
        return {"metrics": [], "execution": {}, "lock_established": False}
    _object(evidence, {"origin", "protocol_sha256", "protocol_locked_before_execution", "task_manifest_sha256",
                       "submission_sha256", "datasets", "execution", "metrics"}, "controller evidence",
            {"origin", "protocol_sha256", "protocol_locked_before_execution", "task_manifest_sha256",
             "submission_sha256", "datasets", "execution", "metrics"})
    if evidence["origin"] != "controller":
        raise EvaluationError("Candidate-authored evidence is not an authoritative measurement envelope")
    protocol = frozen.as_dict()
    if (evidence["protocol_sha256"] != frozen.sha256
            or evidence["task_manifest_sha256"] != protocol["task_manifest_sha256"]
            or evidence["submission_sha256"] != submission_sha256 or evidence["datasets"] != protocol["datasets"]):
        raise EvaluationError("Measurement evidence does not match the frozen protocol/task/submission/data")
    if type(evidence["protocol_locked_before_execution"]) is not bool:
        raise EvaluationError("Protocol lock provenance must be an explicit boolean")
    execution = _object(evidence["execution"], {"status", "clean_run", "run_id", "evidence_refs", "image_id", "job_id"},
                        "execution evidence")
    if execution:
        if not isinstance(execution.get("status"), str) or execution.get("status") not in {"succeeded", "failed", "not_run", "error", "cancelled"} or type(execution.get("clean_run")) is not bool:
            raise EvaluationError("Invalid execution lifecycle evidence")
        _text(execution.get("run_id"), "trusted execution run id")
        _references(execution.get("evidence_refs"), "trusted execution evidence")
    metrics = evidence["metrics"]
    if not isinstance(metrics, list) or len(metrics) > 100000:
        raise EvaluationError("Metric evidence must be a bounded record list")
    seen = set()
    for record in metrics:
        _object(record, {"id", "seed", "status", "value", "units", "n", "artifact_sha256", "evidence_refs", "origin", "evaluator_id"},
                "metric evidence", {"id", "seed", "status", "units", "evidence_refs", "origin"})
        identity = _text(record["id"], "measured metric id", identifier=True)
        if type(record["seed"]) is not int or record["seed"] not in protocol["plan"]["seed_schedule"]:
            raise EvaluationError("Metric seed lies outside the locked schedule")
        key = (identity, record["seed"])
        if key in seen:
            raise EvaluationError("Duplicate metric/seed observations cannot be selectively pooled")
        seen.add(key)
        if not isinstance(record["status"], str) or record["status"] not in {"succeeded", "failed", "error", "not_run"}:
            raise EvaluationError("Invalid metric observation status")
        if not isinstance(record["origin"], str) or record["origin"] not in {"trusted_recomputation", "trusted_evaluator"}:
            raise EvaluationError("Candidate-reported metrics are not authoritative")
        _text(record["units"], "observed units")
        _references(record["evidence_refs"], "metric evidence")
        if record["origin"] == "trusted_evaluator":
            _text(record.get("evaluator_id"), "trusted evaluator identity")
        if record["status"] == "succeeded":
            finite_number(record.get("value"), "measured value")
            _sha(record.get("artifact_sha256"), "measured artifact")
            if type(record.get("n")) is not int or not 1 <= record["n"] <= 10**12:
                raise EvaluationError("Metric sample count must be a positive bounded integer")
        elif record.get("value") is not None:
            raise EvaluationError("Failed/unexecuted observations cannot carry a usable value")
    return dict(evidence, lock_established=evidence["protocol_locked_before_execution"])


def evaluate_protocol(protocol, *, task_manifest, submission_sha256, measurement_evidence):
    """Evaluate frozen criteria; return independent scientific dimensions.

    Root controller code must construct evidence from actual execution and
    installed scorers. An ``origin`` string in candidate JSON confers no trust.
    """
    frozen = freeze_protocol(protocol, task_manifest=task_manifest, submission_sha256=submission_sha256)
    protocol = frozen.as_dict()
    evidence = _measurement_envelope(measurement_evidence, frozen, task_manifest, submission_sha256)
    criteria = {criterion["id"]: criterion for criterion in protocol["criteria"]}
    reviews = {review["id"]: review for review in protocol.get("reviews", [])}
    metrics = {(record["id"], record["seed"]): record for record in evidence["metrics"]}
    evaluated = {}
    result_dimension = "empirical_agreement" if protocol["profile"] == "reproduction/v1" else "transfer"
    n_comparisons = sum(c["kind"] == "metric" for c in criteria.values())
    confidence = 1 - (1 - protocol["plan"].get("confidence_level", 0.95)) / n_comparisons

    def evaluate(identity):
        if identity in evaluated:
            return evaluated[identity]
        criterion = criteria[identity]
        deps = [evaluate(dependency) for dependency in criterion["depends_on"]]
        output = {"id": identity, "kind": criterion["kind"], "dimension": criterion["dimension"],
                  "critical": criterion["critical"], "weight": criterion.get("weight", 1),
                  "depends_on": criterion["depends_on"], "evidence_refs": []}
        if not evidence["lock_established"]:
            output.update(verdict="not_run", reason="protocol_not_locked_before_execution")
        elif any(dep["verdict"] != "pass" for dep in deps):
            output.update(verdict="inconclusive", reason="required_dependency_did_not_pass")
        elif criterion["kind"] == "execution":
            record = evidence["execution"]
            if not record or record.get("status") == "not_run":
                output.update(verdict="not_run", reason="execution_evidence_missing")
            elif record["status"] == "succeeded" and record["clean_run"]:
                output.update(verdict="pass", reason="controller_observed_clean_execution", evidence_refs=record["evidence_refs"])
            else:
                output.update(verdict="fail" if record["status"] == "failed" else "inconclusive",
                              reason="clean_execution_not_established", evidence_refs=record["evidence_refs"])
        elif criterion["kind"] == "review":
            review = reviews.get(criterion["review_id"])
            if review is None:
                output.update(verdict="inconclusive", reason="independent_method_review_missing")
            else:
                output.update(verdict=review["status"], reason="curator_review_of_exact_submission",
                              reviewer=review["reviewer"], evidence_refs=review["evidence_refs"],
                              source_citations=review["source_citations"])
        else:
            seeds = protocol["plan"]["seed_schedule"]
            observed, missing, failures, refs, samples = [], [], [], [], []
            needed_ids = [criterion["metric_id"]] + ([criterion["baseline_id"]] if "baseline_id" in criterion else [])
            wrong_units = False
            for seed in seeds:
                records = [metrics.get((metric_id, seed)) for metric_id in needed_ids]
                if any(record is None for record in records):
                    missing.append(seed)
                    continue
                if any(record["status"] != "succeeded" for record in records):
                    failures.append(seed)
                    continue
                wrong_units = wrong_units or any(record["units"] != criterion["units"] for record in records)
                if any(record["n"] < criterion.get("min_samples_per_run", 1) for record in records):
                    failures.append(seed)
                    continue
                if len(records) == 2 and records[0]["n"] != records[1]["n"]:
                    failures.append(seed)
                    continue
                value = paired_difference(records[0]["value"], records[1]["value"]) if len(records) == 2 else records[0]["value"]
                finite_number(value, "paired difference")
                observed.append(value)
                samples.append(records[0]["n"])
                refs.extend(ref for record in records for ref in record["evidence_refs"])
            output.update(metric_id=criterion["metric_id"], units=criterion["units"],
                          expected_seeds=seeds, missing_seeds=missing, failed_seeds=failures,
                          n_runs=len(observed), sample_counts=samples, evidence_refs=list(dict.fromkeys(refs)),
                          variability_source=protocol["plan"]["variability_source"],
                          aggregation="equal_weight_mean_of_prespecified_run_values")
            if wrong_units:
                output.update(verdict="fail", reason="metric_units_mismatch")
            elif missing or failures:
                output.update(verdict="inconclusive", reason="locked_replication_schedule_incomplete")
            else:
                summary = summarize(observed, bounds=criterion.get("bounds"), confidence=confidence,
                                    independence_assumption=protocol["plan"]["independence_assumption"])
                decision = compare_summary(summary, rule=criterion["rule"], reference=criterion["reference"],
                                           tolerance=criterion.get("tolerance"))
                output.update(summary=summary, comparison=decision, verdict=decision["verdict"],
                              reason=decision.get("reason", "prespecified_comparison"))
        evaluated[identity] = output
        return output

    results = [evaluate(identity) for identity in criteria]
    coverage = weighted_coverage(results)

    def dimension_verdict(dimension):
        items = [item["verdict"] for item in results if item["dimension"] == dimension]
        if not items:
            return "not_assessed"
        return "fail" if "fail" in items else "pass" if all(x == "pass" for x in items) else "inconclusive"

    dimensions = {name: dimension_verdict(name) for name in ("execution", "implementation_fidelity", "empirical_agreement", "transfer")}
    final = "pass" if coverage["status"] == "pass" else "fail" if any(item["verdict"] == "fail" for item in results) else "inconclusive"
    report = {"schema_version": 1, "evaluation_protocol": protocol["profile"], "protocol_id": protocol["protocol_id"],
              "scientific_evaluator_protocol": SCIENTIFIC_EVALUATOR_PROTOCOL,
              "protocol_sha256": frozen.sha256, "task_manifest_sha256": protocol["task_manifest_sha256"],
              "submission_sha256": submission_sha256, "paper_sha256": protocol["source"]["paper_sha256"],
              "dimensions": dimensions, "criteria": results, "coverage": coverage,
              "verdict": final, "pass": final == "pass", "scientific_reproduction": final if protocol["profile"] == "reproduction/v1" else "not_assessed",
              "transfer_verdict": final if protocol["profile"] == "transfer/v1" else "not_assessed",
              "contradictory_faithful_result": dimensions["execution"] == "pass" and dimensions["implementation_fidelity"] == "pass" and dimensions[result_dimension] == "fail",
              "uncertainty_multiplicity": {"policy": "bonferroni_for_metric_intervals", "comparisons": n_comparisons,
                                            "family_confidence_level": protocol["plan"].get("confidence_level", 0.95)},
              "assurance": {"evidence_authority": "trusted_controller_and_installed_scorers",
                            "method_fidelity": "curator_attested_for_exact_submission",
                            "source_fidelity": "curator_attested_not_automatically_proven",
                            "scope": "only_the_frozen_question_datasets_and_protocol"}}
    return report
