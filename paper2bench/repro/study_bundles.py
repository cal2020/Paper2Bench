"""Self-contained study archives with bounded, execution-free numerical replay.

Hashes establish internal integrity, not provenance authentication. Nested run
archives retain their ordinary source, implementation, prediction, and attempt
evidence. No archive member is ever extracted to a member-supplied path.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field, StrictInt, StrictStr

from .bundles import export_bundle, replay_bundle
from .models import SHA256, Dataset, ExperimentSpec, FrozenModel, canonical_bytes
from .studies import _assessment, _bootstrap_seed, _check_budget, _FrozenPlan, _resources

if TYPE_CHECKING:
    from .engine import Engine


MAX_STUDY_BYTES = 64 * 1024 * 1024
MAX_MEMBERS = 20_000
MAX_MANIFEST_BYTES = 1024 * 1024


class _TrialMember(FrozenModel):
    role: Literal["candidate", "baseline"]
    seed: Annotated[StrictInt, Field(ge=0, le=2**63 - 1)]
    spec_hash: SHA256
    run_id: Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{32}$")]
    bundle_sha256: SHA256
    member: StrictStr


class _Manifest(FrozenModel):
    schema_version: Literal["1"] = "1"
    kind: Literal["paired_accuracy_study_bundle"] = "paired_accuracy_study_bundle"
    plan_hash: SHA256
    study_result_hash: SHA256
    artifacts: tuple[SHA256, ...]
    trials: Annotated[tuple[_TrialMember, ...], Field(min_length=4, max_length=100)]


def _json(raw: bytes):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON field in study archive")
            value[key] = item
        return value

    def invalid_constant(_value):
        raise ValueError("non-finite JSON value in study archive")

    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid_constant)


def _canonical_model(raw: bytes, cls):
    value = cls.model_validate(_json(raw))
    if canonical_bytes(value) != raw:
        raise ValueError("noncanonical study archive contract")
    return value


def _scan(archive: zipfile.ZipFile):
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)) or len(names) > MAX_MEMBERS:
        raise ValueError("duplicate or excessive study archive members")
    total = sum(info.file_size for info in infos)
    if total > MAX_STUDY_BYTES:
        raise ValueError("study archive exceeds decompressed size limit")
    for info in infos:
        if (
            info.is_dir()
            or info.flag_bits & 1
            or stat.S_ISLNK(info.external_attr >> 16)
            or info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
        ):
            raise ValueError("unsupported study archive member type")
    return names, total, len(infos)


def _incomplete_assessment(dataset, plan):
    return {
        "verdict": "not_tested",
        "criterion": "paired_multiseed_accuracy_superiority",
        "metric": "accuracy_difference",
        "reason": "all_prespecified_trials_must_complete",
        "effect": None,
        "confidence_interval": None,
        "sample_count": len(dataset.test),
        "seed_count": len(plan.study.seeds),
        "minimum_effect": plan.study.minimum_effect,
        "min_test_examples": plan.study.min_test_examples,
        "minimum_sample_count_met": False,
        "bootstrap_samples": plan.study.bootstrap_samples,
        "bootstrap_seed": plan.bootstrap_seed,
        "confidence": plan.study.confidence,
    }


def _trial_ledger(trial, record, original):
    run, result = record["run"], record["result"]
    return {
        "role": trial.role,
        "seed": trial.seed,
        "spec_hash": trial.spec_hash,
        "run_id": run["run_id"],
        "state": run["state"],
        "result_hash": run["result_hash"],
        "attempts": run["attempts"],
        "measurements": (result or {}).get("measurements"),
        "failure_wall_seconds": (result or {}).get("wall_seconds"),
        "limits": original.limits.model_dump(mode="json"),
        "policy": original.policy.model_dump(mode="json"),
        "workload_kind": original.workload.kind,
        "assumptions": list(original.assumptions),
    }


def _verify_study(manifest, blobs, nested, temporary):
    plan = _canonical_model(blobs[manifest.plan_hash], _FrozenPlan)
    result_raw = blobs[manifest.study_result_hash]
    result = _json(result_raw)
    if canonical_bytes(result) != result_raw:
        raise ValueError("noncanonical study result")
    original_refs = {
        "candidate": plan.study.candidate_spec_hash,
        "baseline": plan.study.baseline_spec_hash,
    }
    required_refs = {manifest.plan_hash, manifest.study_result_hash, *original_refs.values()}
    if set(manifest.artifacts) != required_refs:
        raise ValueError("study archive lacks original contracts or contains unexpected artifacts")
    originals = {
        role: _canonical_model(blobs[ref], ExperimentSpec) for role, ref in original_refs.items()
    }
    for original in originals.values():
        if (
            original.claim.metric != "accuracy"
            or original.dataset_sha256 != plan.dataset_sha256
            or original.engine_identity_sha256 != plan.evaluator_identity_sha256
            or original.workload.kind not in {"builtin_centroid", "python"}
        ):
            raise ValueError("study original experiment contracts are incompatible")
    if plan.bootstrap_seed != _bootstrap_seed(plan.study):
        raise ValueError("study bootstrap identity mismatch")
    schedule = [(role, seed) for seed in plan.study.seeds for role in ("candidate", "baseline")]
    if [(trial.role, trial.seed) for trial in plan.trials] != schedule:
        raise ValueError("study plan omits or changes declared role/seed trials")
    if len(manifest.trials) != len(plan.trials):
        raise ValueError("study archive omits a required trial")
    if len({trial.run_id for trial in manifest.trials}) != len(manifest.trials):
        raise ValueError("study archive reuses a run for multiple trials")
    if (
        result.get("schema_version") != "1"
        or result.get("kind") != "paired_accuracy_study"
        or result.get("plan_hash") != manifest.plan_hash
        or result.get("dataset_sha256") != plan.dataset_sha256
        or result.get("evaluator_identity_sha256") != plan.evaluator_identity_sha256
        or result.get("original_specs") != original_refs
        or canonical_bytes(result.get("seeds")) != canonical_bytes(list(plan.study.seeds))
        or result.get("implementation_fidelity") != "unverified"
        or result.get("protocol_fidelity") != "unverified"
    ):
        raise ValueError("study result differs from its frozen plan")

    records, ledger, dataset = [], [], None
    for index, (declared, member) in enumerate(zip(plan.trials, manifest.trials, strict=True)):
        if (
            member.role != declared.role
            or member.seed != declared.seed
            or member.spec_hash != declared.spec_hash
            or member.member != f"trials/{index:03d}.zip"
        ):
            raise ValueError("study trial archive identity differs from the frozen schedule")
        data = nested[member.member]
        if hashlib.sha256(data).hexdigest() != member.bundle_sha256:
            raise ValueError("nested study run archive integrity verification failed")
        run_path = temporary / f"verified-{index:03d}.zip"
        run_path.write_bytes(data)
        # Existing replay validates source evidence, artifact closure, predictions,
        # and each run's score. It never executes candidate/model code.
        replay_bundle(run_path)
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            inner = _json(archive.read("manifest.json"))
            record = inner["record"]
            run = record["run"]
            if (
                run["run_id"] != member.run_id
                or run["id"] != member.run_id
                or run["spec_hash"] != member.spec_hash
            ):
                raise ValueError("nested run record does not match its study trial")
            child = _canonical_model(archive.read(f"artifacts/{member.spec_hash}"), ExperimentSpec)
            expected = originals[member.role].model_copy(update={"seed": member.seed})
            if canonical_bytes(child) != canonical_bytes(expected):
                raise ValueError("study trial changed fields beyond its declared seed")
            if dataset is None:
                dataset = Dataset.model_validate(
                    _json(archive.read(f"artifacts/{plan.dataset_sha256}"))
                )
        records.append((declared, record))
        ledger.append(_trial_ledger(declared, record, originals[declared.role]))
    _check_budget(len(dataset.test), plan.study)
    if canonical_bytes(ledger) != canonical_bytes(result.get("trials")):
        raise ValueError("study trial ledger differs from its nested run evidence or snapshot")
    completed = all(record["run"]["state"] == "completed" for _, record in records)
    assessment = (
        _assessment(dataset, plan, originals, records)
        if completed
        else _incomplete_assessment(dataset, plan)
    )
    if canonical_bytes(assessment) != canonical_bytes(result.get("assessment")):
        raise ValueError("stored study assessment differs from trusted recomputation")
    resources = {
        role: _resources([trial for trial in ledger if trial["role"] == role])
        for role in ("candidate", "baseline")
    }
    conditions = {
        "same_dataset": True,
        "same_evaluator": True,
        "same_seed_schedule": True,
        "equal_limits": originals["candidate"].limits == originals["baseline"].limits,
        "equal_policies": originals["candidate"].policy == originals["baseline"].policy,
    }
    if canonical_bytes(resources) != canonical_bytes(result.get("resources")) or canonical_bytes(
        conditions
    ) != canonical_bytes(result.get("conditions")):
        raise ValueError("stored study resources or comparison conditions are inconsistent")
    return {
        "integrity": "verified",
        "authentication": "not_verified",
        "scores": "recomputed" if completed else "not_tested",
        "execution_rerun": False,
        "plan_hash": manifest.plan_hash,
        "study_result_hash": manifest.study_result_hash,
        "assessment": assessment,
        "trial_count": len(ledger),
        "resources": resources,
        "implementation_fidelity": "unverified",
        "protocol_fidelity": "unverified",
    }


def export_study(engine: Engine, study_result_hash: str, path: str | Path) -> Path:
    """Export a complete retained snapshot; refuse overwrites and stale run snapshots."""
    result = engine.artifacts.get_json(study_result_hash)
    if not isinstance(result, dict) or result.get("kind") != "paired_accuracy_study":
        raise ValueError("artifact is not a retained study result")
    plan = _canonical_model(engine.artifacts.get(result["plan_hash"]), _FrozenPlan)
    refs = {
        study_result_hash,
        result["plan_hash"],
        plan.study.candidate_spec_hash,
        plan.study.baseline_spec_hash,
    }
    blobs = {ref: engine.artifacts.get(ref) for ref in refs}
    if len(result.get("trials", [])) != len(plan.trials):
        raise ValueError("study result omits a required trial")
    destination = Path(path)
    if destination.exists():
        raise FileExistsError("study archive destination already exists")
    with tempfile.TemporaryDirectory(prefix="paper2bench-study-") as directory:
        temporary = Path(directory)
        members, nested = [], {}
        total_bytes, total_members = sum(map(len, blobs.values())), len(blobs) + 1
        for index, (trial, stored) in enumerate(zip(plan.trials, result["trials"], strict=True)):
            run_id = stored["run_id"]
            if (
                not isinstance(run_id, str)
                or len(run_id) != 32
                or any(char not in "0123456789abcdef" for char in run_id)
            ):
                raise ValueError("invalid study run identity")
            nested_path = export_bundle(engine, run_id, temporary / f"trial-{index:03d}.zip")
            if nested_path.stat().st_size > MAX_STUDY_BYTES:
                raise ValueError("nested study archive exceeds compressed size limit")
            with zipfile.ZipFile(nested_path) as archive:
                _names, expanded, count = _scan(archive)
            total_bytes += nested_path.stat().st_size + expanded
            total_members += count + 1
            if total_bytes > MAX_STUDY_BYTES or total_members > MAX_MEMBERS:
                raise ValueError("study archive exceeds cumulative nested size or member limit")
            data = nested_path.read_bytes()
            name = f"trials/{index:03d}.zip"
            nested[name] = data
            members.append(
                _TrialMember(
                    role=trial.role,
                    seed=trial.seed,
                    spec_hash=trial.spec_hash,
                    run_id=run_id,
                    bundle_sha256=hashlib.sha256(data).hexdigest(),
                    member=name,
                )
            )
        manifest = _Manifest(
            plan_hash=result["plan_hash"],
            study_result_hash=study_result_hash,
            artifacts=tuple(sorted(refs)),
            trials=tuple(members),
        )
        manifest_raw = canonical_bytes(manifest)
        if (
            len(manifest_raw) > MAX_MANIFEST_BYTES
            or total_bytes + len(manifest_raw) > MAX_STUDY_BYTES
        ):
            raise ValueError("study archive exceeds cumulative nested size limit")
        # Verify before creating the user-visible destination. This also catches
        # run-state changes between the stored study snapshot and export.
        _verify_study(manifest, blobs, nested, temporary)
        staged = temporary / "study.zip"
        with zipfile.ZipFile(staged, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", manifest_raw)
            for ref, data in sorted(blobs.items()):
                archive.writestr(f"artifacts/{ref}", data)
            for name, data in sorted(nested.items()):
                archive.writestr(name, data)
        if staged.stat().st_size > MAX_STUDY_BYTES:
            raise ValueError("study archive exceeds compressed size limit")
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Stage on the destination filesystem, then publish with an exclusive
        # link. A failed copy cannot leave a partial scientific artifact at the
        # requested path, and a concurrent writer cannot be overwritten.
        staged_destination = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix=".paper2bench-study-", delete=False
            ) as target:
                staged_destination = Path(target.name)
                with staged.open("rb") as source:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            os.link(staged_destination, destination)
        finally:
            if staged_destination is not None:
                staged_destination.unlink(missing_ok=True)
    return destination


def replay_study(path: str | Path) -> dict:
    """Verify nested evidence and recompute the study without executing workloads."""
    try:
        return _replay_study(Path(path))
    except (
        KeyError,
        TypeError,
        AttributeError,
        UnicodeError,
        RecursionError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
    ) as exc:
        raise ValueError("invalid study archive or missing required evidence") from exc


def _replay_study(path: Path):
    if path.stat().st_size > MAX_STUDY_BYTES:
        raise ValueError("study archive exceeds compressed size limit")
    with zipfile.ZipFile(path) as archive:
        names, total_bytes, total_members = _scan(archive)
        if "manifest.json" not in names:
            raise ValueError("study archive manifest missing")
        if archive.getinfo("manifest.json").file_size > MAX_MANIFEST_BYTES:
            raise ValueError("study archive manifest exceeds size limit")
        manifest = _canonical_model(archive.read("manifest.json"), _Manifest)
        if len(manifest.artifacts) != len(set(manifest.artifacts)):
            raise ValueError("duplicate study artifact references")
        member_names = [trial.member for trial in manifest.trials]
        if member_names != [f"trials/{index:03d}.zip" for index in range(len(manifest.trials))]:
            raise ValueError("invalid or duplicate study trial archive paths")
        expected = {"manifest.json", *member_names} | {
            f"artifacts/{ref}" for ref in manifest.artifacts
        }
        if set(names) != expected:
            raise ValueError("study archive members differ from its manifest")
        blobs = {}
        for ref in manifest.artifacts:
            data = archive.read(f"artifacts/{ref}")
            if hashlib.sha256(data).hexdigest() != ref:
                raise ValueError("study artifact integrity verification failed")
            blobs[ref] = data
        nested = {}
        for name in member_names:
            data = archive.read(name)
            with zipfile.ZipFile(io.BytesIO(data)) as inner:
                _names, expanded, count = _scan(inner)
            total_bytes += expanded
            total_members += count
            if total_bytes > MAX_STUDY_BYTES or total_members > MAX_MEMBERS:
                raise ValueError("study archive exceeds cumulative nested size or member limit")
            nested[name] = data
    with tempfile.TemporaryDirectory(prefix="paper2bench-study-replay-") as directory:
        return _verify_study(manifest, blobs, nested, Path(directory))
