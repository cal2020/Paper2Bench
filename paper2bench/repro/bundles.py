"""Portable, size-bounded archives; replay verifies scores rather than executing code."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from .evaluation import evaluate
from .models import ExperimentSpec, canonical_bytes, parse_dataset, parse_predictions

MAX_BUNDLE_BYTES = 64 * 1024 * 1024


def _verify_remote_receipt(result, spec, dataset, blobs):
    if not result or not result.get("remote_completion_sha256"):
        return
    from .remote import WorkerCompletion
    from .runtime import sanitized_input

    completion = WorkerCompletion.model_validate(
        json.loads(blobs[result["remote_completion_sha256"]])
    )
    expected = canonical_bytes(sanitized_input(dataset, spec))
    expected_hash = hashlib.sha256(expected).hexdigest()
    if (
        completion.image != spec.workload.image
        or completion.implementation_sha256 != spec.workload.implementation_sha256
        or result.get("implementation_sha256") != spec.workload.implementation_sha256
        or completion.input_sha256 != expected_hash
        or result.get("input_sha256") != expected_hash
        or blobs[expected_hash] != expected
    ):
        raise ValueError("remote completion does not match assigned code or public input")
    if result.get("execution_state") == "completed":
        if completion.state != "completed":
            raise ValueError("remote completion state differs from accepted result")
        submitted = parse_predictions(completion.predictions, dataset)
        retained = parse_predictions(result["predictions"], dataset)
        if canonical_bytes(submitted) != canonical_bytes(retained):
            raise ValueError("remote completion predictions differ from accepted result")


def _closure(spec, result, ledger, read_json) -> set[str]:
    """All semantically referenced artifacts, not merely the archive's file list."""
    refs = {spec.source_sha256, spec.dataset_sha256}
    for ref in (
        spec.workload.implementation_sha256,
        spec.workload.provider_profile,
        spec.authoring_receipt_sha256,
        spec.source_manifest_sha256,
        spec.engine_identity_sha256,
    ):
        if ref:
            refs.add(ref)
    if spec.authoring_receipt_sha256:
        receipt = read_json(spec.authoring_receipt_sha256)
        for field in ("source_sha256", "dataset_sha256"):
            if receipt.get(field) != getattr(spec, field):
                raise ValueError("authoring receipt identity mismatch")
        if receipt.get("implementation_sha256") != spec.workload.implementation_sha256:
            raise ValueError("authoring implementation identity mismatch")
        refs.update(receipt[field] for field in ("prompt_sha256", "response_sha256"))
    if spec.source_manifest_sha256:
        source = read_json(spec.source_manifest_sha256)
        if source.get("source_sha256") != spec.source_sha256:
            raise ValueError("source manifest identity mismatch")
        refs.add(source["original_sha256"])
    if spec.engine_identity_sha256:
        identity = read_json(spec.engine_identity_sha256)
        refs.update(identity["modules"].values())
    calls = list(ledger)
    if result:
        calls.extend(result.get("provider_calls", []))
        for field in ("remote_completion_sha256", "input_sha256", "implementation_sha256"):
            if result.get(field):
                refs.add(result[field])
    for call in calls:
        profile = call.get("provider_profile")
        if profile:
            if profile != spec.workload.provider_profile:
                raise ValueError("provider-call profile does not match frozen workload")
            refs.add(profile)
        for field in ("input_sha256", "output_sha256"):
            if call.get(field):
                refs.add(call[field])
    if any(
        not isinstance(ref, str) or len(ref) != 64 or any(c not in "0123456789abcdef" for c in ref)
        for ref in refs
    ):
        raise ValueError("invalid referenced artifact identifier")
    return refs


def export_bundle(engine, run_id: str, destination: str | Path) -> Path:
    record = engine.result(run_id)
    run, result = record["run"], record["result"]
    spec = engine.load_spec(run["spec_hash"])
    refs = _closure(
        spec, result, record.get("provider_call_ledger", []), engine.artifacts.get_json
    ) | {run["spec_hash"]}
    for diagnostic in record.get("attempt_diagnostics", []):
        refs.add(diagnostic["artifact_sha256"])
        refs.update(_closure(spec, diagnostic["details"], [], engine.artifacts.get_json))
    if run["result_hash"]:
        refs.add(run["result_hash"])
    manifest = {"schema_version": "1", "record": record, "artifacts": sorted(refs)}
    manifest_bytes = canonical_bytes(manifest)
    blobs = {ref: engine.artifacts.get(ref) for ref in refs}
    if sum(map(len, blobs.values())) + len(manifest_bytes) > MAX_BUNDLE_BYTES:
        raise ValueError("bundle exceeds export size limit")
    if len(blobs) + 1 > 20000:
        raise ValueError("bundle exceeds member count limit")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally overwriting an existing research artifact.
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", manifest_bytes)
        for ref, data in sorted(blobs.items()):
            archive.writestr(f"artifacts/{ref}", data)
    return destination


def replay_bundle(path: str | Path) -> dict:
    try:
        return _replay_bundle(path)
    except (KeyError, TypeError, AttributeError, UnicodeError, RecursionError) as exc:
        raise ValueError("invalid bundle or missing required artifact") from exc


def _replay_bundle(path: str | Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        if len(names) != len(set(names)) or len(names) > 20000:
            raise ValueError("duplicate or excessive archive members")
        if sum(item.file_size for item in infos) > MAX_BUNDLE_BYTES:
            raise ValueError("bundle exceeds decompressed size limit")
        if "manifest.json" not in names:
            raise ValueError("manifest missing")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("schema_version") != "1":
            raise ValueError("unsupported bundle version")
        refs = manifest["artifacts"]
        if not isinstance(refs, list) or any(
            not isinstance(ref, str)
            or len(ref) != 64
            or any(c not in "0123456789abcdef" for c in ref)
            for ref in refs
        ):
            raise ValueError("invalid artifact identifier")
        if len(refs) != len(set(refs)):
            raise ValueError("duplicate artifact references")
        expected = {"manifest.json"} | {f"artifacts/{ref}" for ref in refs}
        if set(names) != expected:
            raise ValueError("archive members do not match manifest")
        blobs = {}
        for ref in refs:
            data = archive.read(f"artifacts/{ref}")
            if hashlib.sha256(data).hexdigest() != ref:
                raise ValueError("artifact integrity verification failed")
            blobs[ref] = data
    run = manifest["record"]["run"]
    raw_spec = json.loads(blobs[run["spec_hash"]])
    spec = ExperimentSpec.model_validate(raw_spec)
    if not spec.engine_identity_sha256:
        raise ValueError("bundle specification has not passed the freeze boundary")
    if canonical_bytes(raw_spec) != blobs[run["spec_hash"]]:
        raise ValueError("noncanonical specification")
    from .authoring import validate_evidence

    validate_evidence(spec, blobs[spec.source_sha256].decode("utf-8"))
    dataset = parse_dataset(json.loads(blobs[spec.dataset_sha256]))
    result = None
    if run["result_hash"]:
        result = json.loads(blobs[run["result_hash"]])
        if result["run_id"] != run["run_id"] or result["spec_hash"] != run["spec_hash"]:
            raise ValueError("result identity mismatch")
        if canonical_bytes(manifest["record"]["result"]) != canonical_bytes(result):
            raise ValueError("manifest result differs from hashed result")
        if result["execution_state"] != run["state"]:
            raise ValueError("execution state mismatch")
    required = _closure(
        spec,
        result,
        manifest["record"].get("provider_call_ledger", []),
        lambda ref: json.loads(blobs[ref]),
    ) | {run["spec_hash"]}
    if run["result_hash"]:
        required.add(run["result_hash"])
    if not required.issubset(blobs):
        raise ValueError("missing required referenced artifact")
    _verify_remote_receipt(result, spec, dataset, blobs)
    for diagnostic in manifest["record"].get("attempt_diagnostics", []):
        diagnostic_hash = diagnostic["artifact_sha256"]
        details = json.loads(blobs[diagnostic_hash])
        if (
            canonical_bytes(details) != canonical_bytes(diagnostic["details"])
            or details["run_id"] != run["run_id"]
        ):
            raise ValueError("attempt diagnostic identity mismatch")
        references = _closure(spec, details, [], lambda ref: json.loads(blobs[ref]))
        if not references.issubset(blobs):
            raise ValueError("missing diagnostic referenced artifact")
        _verify_remote_receipt(details, spec, dataset, blobs)
    if run["state"] == "completed":
        if result is None:
            raise ValueError("completed run has no result")
        predictions = parse_predictions(result["predictions"], dataset)
        assessment = evaluate(dataset, predictions, spec.claim)
        if canonical_bytes(assessment) != canonical_bytes(result["assessment"]):
            raise ValueError("stored assessment differs from trusted recomputation")
        return {
            "integrity": "verified",
            "authentication": "not_verified",
            "scores": "recomputed",
            "execution_rerun": False,
            "assessment": assessment,
            "spec_hash": run["spec_hash"],
        }
    return {
        "integrity": "verified",
        "authentication": "not_verified",
        "scores": "not_tested",
        "execution_rerun": False,
        "execution_state": run["state"],
    }
