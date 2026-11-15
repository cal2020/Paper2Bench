"""Frozen experiment orchestration. Candidate output never supplies its own score."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .authoring import validate_evidence
from .budgets import BudgetExceeded, BudgetStore
from .evaluation import evaluate
from .models import (
    Dataset,
    ExperimentSpec,
    KnapsackDataset,
    Predictions,
    canonical_bytes,
    parse_dataset,
)
from .providers import OutcomeUnknown, ProviderClient, ProviderProfile, UnsupportedCapability
from .runtime import (
    BuiltinKnapsackRunner,
    BuiltinRunner,
    DockerRunner,
    ExecutionCancelled,
    ExecutionFailure,
    ExecutionResult,
    ResourceExhausted,
    RuntimeUnavailable,
)
from .storage import ArtifactStore, RunStore


class InvalidExperiment(ValueError):
    """The declared experiment cannot be executed faithfully."""


class Engine:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.artifacts = ArtifactStore(self.root / "artifacts")
        self.runs = RunStore(self.root / "runs.sqlite3")
        self.budgets = BudgetStore(self.root / "budgets.sqlite3")

    def register_dataset(self, value: dict | Dataset | KnapsackDataset) -> str:
        raw = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
        dataset = parse_dataset(raw)
        return self.artifacts.put(canonical_bytes(dataset))

    def register_profile(self, value: dict | ProviderProfile) -> str:
        raw = value.model_dump(mode="json") if isinstance(value, ProviderProfile) else value
        profile = ProviderProfile.model_validate(raw)
        return self.artifacts.put(canonical_bytes(profile))

    def _validate(self, spec: ExperimentSpec) -> Dataset | KnapsackDataset:
        source = self.artifacts.get(spec.source_sha256).decode("utf-8")
        validate_evidence(spec, source)
        dataset = parse_dataset(self.artifacts.get_json(spec.dataset_sha256))
        workload = spec.workload
        if spec.policy.budget_id:
            self.budgets.get(spec.policy.budget_id)
            if workload.kind != "model_classification":
                raise InvalidExperiment(
                    "model token budgets apply to model_classification workloads"
                )
        optimization = isinstance(dataset, KnapsackDataset)
        if optimization != (spec.claim.metric == "optimality_ratio"):
            raise InvalidExperiment("dataset and claim metric describe different tasks")
        if optimization and workload.kind in {"builtin_centroid", "model_classification"}:
            raise InvalidExperiment("classification workload cannot run optimization task")
        if not optimization and workload.kind == "builtin_knapsack":
            raise InvalidExperiment("optimization workload requires knapsack instances")
        if spec.policy.placement == "remote":
            if spec.policy.local_only:
                raise InvalidExperiment("local-only policy forbids remote placement")
            if workload.kind != "python":
                raise InvalidExperiment("remote workers require a pinned Python implementation")
        if workload.kind == "python":
            code = self.artifacts.get(workload.implementation_sha256)
            if len(code) > 256 * 1024:
                raise InvalidExperiment("implementation exceeds 256 KiB")
            try:
                # Parsing/compilation does not execute the candidate.
                compile(code.decode("utf-8"), "candidate.py", "exec")
            except (UnicodeError, SyntaxError) as exc:
                raise InvalidExperiment("implementation is not valid UTF-8 Python") from exc
        if workload.kind == "model_classification":
            try:
                profile = ProviderProfile.model_validate(
                    self.artifacts.get_json(workload.provider_profile)
                )
            except (ValueError, OSError) as exc:
                raise InvalidExperiment(
                    "provider_profile must reference a registered profile"
                ) from exc
            ProviderClient(profile).validate(local_only=spec.policy.local_only)
            if spec.policy.max_provider_calls < len(dataset.test):
                raise InvalidExperiment("provider call budget cannot cover the complete test set")
        receipt = spec.authoring_receipt_sha256
        if receipt:
            provenance = self.artifacts.get_json(receipt)
            for field in ("source_sha256", "dataset_sha256"):
                if provenance.get(field) != getattr(spec, field):
                    raise InvalidExperiment("authoring provenance does not match specification")
            if provenance.get("implementation_sha256") != spec.workload.implementation_sha256:
                raise InvalidExperiment("authoring implementation identity mismatch")
            for field in ("prompt_sha256", "response_sha256"):
                self.artifacts.get(provenance[field])
        if spec.source_manifest_sha256:
            manifest = self.artifacts.get_json(spec.source_manifest_sha256)
            if manifest.get("source_sha256") != spec.source_sha256:
                raise InvalidExperiment("source manifest identity mismatch")
            self.artifacts.get(manifest["original_sha256"])
        if spec.engine_identity_sha256 and spec.engine_identity_sha256 != self._engine_identity():
            raise InvalidExperiment("installed engine differs from frozen execution identity")
        return dataset

    def _engine_identity(self) -> str:
        modules = {
            path.name: self.artifacts.put(path.read_bytes())
            for path in sorted(Path(__file__).parent.glob("*.py"))
        }
        return self.artifacts.put_json(
            {
                "schema_version": "1",
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "os": platform.system(),
                "architecture": platform.machine(),
                "dependencies": {
                    name: importlib.metadata.version(name)
                    for name in ("pydantic", "httpx", "pypdf")
                },
                "modules": modules,
            }
        )

    def freeze(self, value: dict | ExperimentSpec) -> str:
        # model_copy/model_construct can bypass Pydantic validation. Revalidate
        # even already-instantiated objects at this trust boundary.
        raw = value.model_dump(mode="json") if isinstance(value, ExperimentSpec) else value
        spec = ExperimentSpec.model_validate(raw)
        spec = spec.model_copy(update={"engine_identity_sha256": self._engine_identity()})
        self._validate(spec)
        return self.artifacts.put(canonical_bytes(spec))

    def load_spec(self, spec_hash: str) -> ExperimentSpec:
        raw = self.artifacts.get_json(spec_hash)
        spec = ExperimentSpec.model_validate(raw)
        # Check the stored representation, retaining compatibility when a newer
        # schema adds a field with a stable default to an older frozen artifact.
        if canonical_bytes(raw) != self.artifacts.get(spec_hash):
            raise InvalidExperiment("specification is not in canonical frozen form")
        if not spec.engine_identity_sha256:
            raise InvalidExperiment("specification has not passed the freeze boundary")
        return spec

    def submit(self, spec_hash: str, idempotency_key: str) -> dict:
        self._validate(self.load_spec(spec_hash))
        return self.runs.create_run(spec_hash, idempotency_key)

    def run(self, spec_hash: str, idempotency_key: str, **kwargs: Any) -> dict:
        run = self.submit(spec_hash, idempotency_key)
        return self.execute(run["run_id"], **kwargs)

    def execute(self, run_id: str, *, worker: str | None = None, lease_seconds: float = 30) -> dict:
        before = self.runs.get_run(run_id)
        if before["state"] not in {"queued", "running"}:
            return before
        if self.load_spec(before["spec_hash"]).policy.placement == "remote":
            return before
        claim = self.runs.claim(run_id, worker or uuid.uuid4().hex, lease_seconds)
        if claim is None:
            return self.runs.get_run(run_id)
        token = claim["token"]
        stop = threading.Event()
        lost = threading.Event()

        def maintain_lease() -> None:
            while not stop.wait(min(lease_seconds / 3, 1)):
                try:
                    if not self.runs.heartbeat(run_id, token, lease_seconds):
                        lost.set()
                        return
                except Exception:  # noqa: BLE001 - fail closed on lost lease storage
                    lost.set()
                    return

        heartbeat = threading.Thread(target=maintain_lease, daemon=True)
        heartbeat.start()

        def cancelled() -> bool:
            return lost.is_set() or not self.runs.is_active(run_id, token)

        started = time.monotonic()
        receipts: list[dict] = []
        state, error, result_hash = "failed", None, None
        cleanup = {}
        interrupted = False
        spec = None
        try:
            spec = self.load_spec(before["spec_hash"])
            dataset = self._validate(spec)
            if spec.workload.kind == "model_classification" and claim["attempt"] > 1:
                # A crashed process may have sent an unacknowledged billable request.
                # This version deliberately does not repeat uncertain model work.
                raise OutcomeUnknown("previous model attempt ended without a confirmed outcome")
            if spec.workload.kind == "builtin_centroid":
                execution = BuiltinRunner().run(dataset, spec, cancelled=cancelled)
            elif spec.workload.kind == "builtin_knapsack":
                execution = BuiltinKnapsackRunner().run(dataset, spec, cancelled=cancelled)
            elif spec.workload.kind == "python":
                code = self.artifacts.get(spec.workload.implementation_sha256)
                execution = DockerRunner().run(code, dataset, spec, cancelled=cancelled)
            else:
                execution = self._execute_model(
                    dataset,
                    spec,
                    cancelled,
                    receipts,
                    lambda call: self.runs.record_call(run_id, token, call),
                    run_id,
                )
            if cancelled():
                raise ExecutionCancelled("lease lost or cancellation requested")
            assessment = evaluate(dataset, execution.predictions, spec.claim)
            result_hash = self.artifacts.put_json(
                {
                    "schema_version": "1",
                    "run_id": run_id,
                    "spec_hash": before["spec_hash"],
                    "execution_state": "completed",
                    "predictions": execution.predictions.model_dump(mode="json"),
                    "assessment": assessment,
                    "measurements": execution.measurements,
                    "provider_calls": receipts,
                    "authoring_receipt_sha256": spec.authoring_receipt_sha256,
                    "assumptions": list(spec.assumptions),
                    "limitations": [
                        "Numerical agreement does not verify implementation or protocol fidelity.",
                        "Wilson intervals assume independent test cases, conditional on this fit.",
                        "This version does not estimate variation across independent training runs.",
                    ],
                }
            )
            state = "completed"
        except KeyboardInterrupt:
            interrupted = True
            if spec is not None and spec.workload.kind == "model_classification":
                state, error = "outcome_unknown", "provider_request_interrupted"
            else:
                self.runs.cancel_if_active(run_id, token)
                state, error = "cancelled", "execution_interrupted"
        except OutcomeUnknown:
            state, error = "outcome_unknown", "provider_outcome_unknown"
        except (RuntimeUnavailable, UnsupportedCapability) as exc:
            state, error = "unsupported", "required_execution_capability_unavailable"
            cleanup = self._cleanup_details(exc)
        except (ResourceExhausted, BudgetExceeded) as exc:
            state, error = "resource_exhausted", "execution_resource_limit_reached"
            cleanup = self._cleanup_details(exc)
        except ExecutionCancelled as exc:
            state, error = "cancelled", "execution_cancelled_or_lease_lost"
            cleanup = self._cleanup_details(exc)
        except (InvalidExperiment, ValueError, UnicodeError):
            state, error = "invalid", "experiment_or_output_validation_failed"
        except ExecutionFailure as exc:
            state, error = "failed", "candidate_execution_failed"
            cleanup = self._cleanup_details(exc)
        except Exception:  # noqa: BLE001 - persist safe failure, never untrusted diagnostics
            # Persist stable diagnostics, never arbitrary provider/code exception content.
            state, error = "failed", "internal_execution_error"
        finally:
            try:
                if error:
                    result_hash = self.artifacts.put_json(
                        {
                            "schema_version": "1",
                            "run_id": run_id,
                            "spec_hash": before["spec_hash"],
                            "execution_state": state,
                            "assessment": {"verdict": "not_tested", "reason": error},
                            "provider_calls": receipts,
                            "wall_seconds": time.monotonic() - started,
                            "cleanup": cleanup,
                        }
                    )
                    self.runs.attach_diagnostic(run_id, token, result_hash)
                # A cancelled or fenced worker must never publish over its successor.
                if state != "cancelled" and not lost.is_set():
                    self.runs.finish(run_id, token, state, result_hash=result_hash, error=error)
            finally:
                # Storage errors must not leave a thread renewing an abandoned lease.
                stop.set()
                heartbeat.join(timeout=2)
        if interrupted:
            raise KeyboardInterrupt
        return self.runs.get_run(run_id)

    @staticmethod
    def _cleanup_details(exc) -> dict:
        details = {}
        confirmed = getattr(exc, "cleanup_confirmed", None)
        name = getattr(exc, "container_name", None)
        if type(confirmed) is bool:
            details["cleanup_confirmed"] = confirmed
        if isinstance(name, str) and name.startswith("paper2bench-") and len(name) <= 64:
            details["container_name"] = name
        return details

    def _execute_model(
        self, dataset, spec, cancelled, receipts, record_call, run_id
    ) -> ExecutionResult:
        profile = ProviderProfile.model_validate(
            self.artifacts.get_json(spec.workload.provider_profile)
        )
        labels = sorted({example.y for example in dataset.train})
        train = [example.model_dump(mode="json") for example in dataset.train]
        predicted, ids = [], []
        start = time.monotonic()
        for example in dataset.test:
            if cancelled():
                raise ExecutionCancelled("cancelled")
            if time.monotonic() - start >= spec.limits.timeout_seconds:
                raise ResourceExhausted("experiment wall-time limit reached")
            messages = [
                {"role": "system", "content": spec.workload.prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "train": train,
                            "test": {"id": example.id, "x": list(example.x)},
                            "allowed_labels": labels,
                            "response_format": {"label": "one allowed label"},
                        },
                        allow_nan=False,
                    ),
                },
            ]
            call = {
                "phase": "dispatched",
                "call_id": uuid.uuid4().hex,
                "test_id": example.id,
                "requested_model": spec.workload.model,
                "provider_profile": spec.workload.provider_profile,
                "input_sha256": self.artifacts.put_json(messages),
                "ordinal": len(receipts) + 1,
            }
            budget_id = spec.policy.budget_id
            if budget_id:
                self.budgets.reserve(
                    budget_id,
                    call["call_id"],
                    output_tokens=spec.policy.max_output_tokens,
                    run_id=run_id,
                )
            if not record_call(call):
                if budget_id:
                    self.budgets.reconcile(
                        budget_id, call["call_id"], actual_output_tokens=None, outcome="rejected"
                    )
                raise ExecutionCancelled("lease lost before dispatch")
            remaining = spec.limits.timeout_seconds - (time.monotonic() - start)
            if remaining <= 0:
                if budget_id:
                    self.budgets.reconcile(
                        budget_id, call["call_id"], actual_output_tokens=None, outcome="rejected"
                    )
                raise ResourceExhausted("experiment wall-time limit reached")
            bounded_profile = profile.model_copy(
                update={"timeout": min(profile.timeout, remaining)}
            )
            client = ProviderClient(bounded_profile)
            call_started = time.monotonic()
            # No automatic retry: an accepted request may already be billable.
            try:
                generation = client.generate(
                    model=spec.workload.model,
                    messages=messages,
                    max_output_tokens=spec.policy.max_output_tokens,
                    local_only=spec.policy.local_only,
                    total_timeout=remaining,
                )
            except OutcomeUnknown as exc:
                if budget_id:
                    self.budgets.reconcile(
                        budget_id, call["call_id"], actual_output_tokens=None, outcome="unknown"
                    )
                unknown = {
                    **call,
                    "phase": "outcome_unknown",
                    "request_id": exc.request_id,
                    "error": "provider_outcome_unknown",
                }
                receipts.append(unknown)
                record_call(unknown)
                raise
            except Exception as exc:
                if budget_id:
                    self.budgets.reconcile(
                        budget_id, call["call_id"], actual_output_tokens=None, outcome="unknown"
                    )
                record_call({**call, "phase": "failed", "error": type(exc).__name__})
                raise
            if budget_id:
                self.budgets.reconcile(
                    budget_id,
                    call["call_id"],
                    actual_output_tokens=generation.usage.output_tokens,
                    outcome="completed",
                )
            raw_hash = self.artifacts.put(generation.text.encode("utf-8"))
            receipt = generation.model_dump(mode="json")
            receipt.pop("text", None)
            completed = {
                **call,
                **receipt,
                "phase": "completed",
                "output_sha256": raw_hash,
                "wall_seconds": time.monotonic() - call_started,
            }
            receipts.append(completed)
            if not record_call(completed) or cancelled():
                raise ExecutionCancelled("lease lost after provider response")
            if budget_id and self.budgets.get(budget_id)["blocked"]:
                raise ResourceExhausted("provider reported usage beyond reserved token quota")
            if time.monotonic() - start >= spec.limits.timeout_seconds:
                raise ResourceExhausted("experiment wall-time limit reached")
            if len(generation.text.encode("utf-8")) > spec.limits.output_bytes:
                raise ResourceExhausted("model output limit reached")
            if generation.finish_reason not in {None, "stop"}:
                raise ValueError("provider response is incomplete or unsupported")

            def unique(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError("duplicate output field")
                    result[key] = value
                return result

            def invalid_constant(_):
                raise ValueError("nonfinite output")

            output = json.loads(
                generation.text, object_pairs_hook=unique, parse_constant=invalid_constant
            )
            if not isinstance(output, dict) or set(output) != {"label"}:
                raise ValueError("expected exactly one label")
            if not isinstance(output["label"], str) or output["label"] not in labels:
                raise ValueError("invalid classification label")
            ids.append(example.id)
            predicted.append(output["label"])
        return ExecutionResult(
            predictions=Predictions(ids=tuple(ids), labels=tuple(predicted)),
            measurements={
                "wall_seconds": time.monotonic() - start,
                "provider_call_count": len(receipts),
                "cost_usd": None,
                "cost_status": "unknown_no_verified_price_schedule",
                "model_revision_status": "provider_reported_only",
            },
        )

    def result(self, run_id: str) -> dict:
        run = self.runs.get_run(run_id)
        return {
            "run": run,
            "result": self.artifacts.get_json(run["result_hash"]) if run["result_hash"] else None,
            "events": self.runs.events(run_id),
            "provider_call_ledger": self.runs.calls(run_id),
            "attempt_diagnostics": [
                {**item, "details": self.artifacts.get_json(item["artifact_sha256"])}
                for item in self.runs.diagnostics(run_id)
            ],
        }

    def report(self, run_id: str) -> str:
        record = self.result(run_id)
        run, result = record["run"], record["result"]
        lines = [f"Paper2Bench run {run_id}", f"Execution: {run['state']}"]
        if result:
            assessment = result["assessment"]
            lines.append(f"Numerical criterion: {assessment['verdict']}")
            if "accuracy" in assessment:
                lines.append(f"Accuracy: {assessment['accuracy']:.6f}")
                lines.append(f"Confidence interval: {assessment['confidence_interval']}")
            if "mean_score" in assessment:
                lines.append(f"Mean optimality ratio: {assessment['mean_score']:.6f}")
                lines.append(f"Confidence interval: {assessment['confidence_interval']}")
            lines.append("Implementation fidelity: unverified")
            lines.append("Protocol fidelity: unverified")
            lines.append("Paper reproduction: not established by this numerical check")
        lines.append(f"Frozen specification: {run['spec_hash']}")
        return "\n".join(lines)
