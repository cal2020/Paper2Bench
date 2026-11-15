"""Portable study replay verifies nested evidence without running candidates."""

from __future__ import annotations

import hashlib
import io
import json
import struct
import warnings
import zipfile

import pytest

from paper2bench.repro import study_bundles
from paper2bench.repro.demo import create_demo
from paper2bench.repro.engine import Engine
from paper2bench.repro.models import canonical_bytes
from paper2bench.repro.runtime import ExecutionFailure
from paper2bench.repro.studies import StudySpec, freeze_study, run_study
from paper2bench.repro.study_bundles import export_study, replay_study


def make_study(engine):
    baseline = create_demo(engine)
    candidate = engine.load_spec(baseline).model_dump(mode="json")
    candidate["title"] = "Study archive candidate"
    candidate_hash = engine.freeze(candidate)
    return freeze_study(
        engine,
        StudySpec(
            candidate_spec_hash=candidate_hash,
            baseline_spec_hash=baseline,
            seeds=(2, 7, 11),
            bootstrap_samples=200,
        ),
    )


def read_zip(path):
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def write_zip(path, entries):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return path


def zip_bytes(entries):
    value = io.BytesIO()
    with zipfile.ZipFile(value, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return value.getvalue()


def replace_blob(entries, manifest, old_ref, raw):
    new_ref = hashlib.sha256(raw).hexdigest()
    del entries[f"artifacts/{old_ref}"]
    entries[f"artifacts/{new_ref}"] = raw
    manifest["artifacts"].remove(old_ref)
    manifest["artifacts"].append(new_ref)
    return new_ref


def change_result(entries, mutate):
    manifest = json.loads(entries["manifest.json"])
    old_ref = manifest["study_result_hash"]
    result = json.loads(entries[f"artifacts/{old_ref}"])
    mutate(result)
    manifest["study_result_hash"] = replace_blob(
        entries, manifest, old_ref, canonical_bytes(result)
    )
    entries["manifest.json"] = canonical_bytes(manifest)
    return entries


def change_inner(entries, index, transform):
    manifest = json.loads(entries["manifest.json"])
    trial = manifest["trials"][index]
    member = trial["member"]
    entries[member] = transform(entries[member])
    trial["bundle_sha256"] = hashlib.sha256(entries[member]).hexdigest()
    entries["manifest.json"] = canonical_bytes(manifest)
    return entries


@pytest.fixture(scope="module")
def complete(tmp_path_factory):
    root = tmp_path_factory.mktemp("completed-study-bundle")
    engine = Engine(root / "state")
    plan_hash = make_study(engine)
    result = run_study(engine, plan_hash, "archive-study")
    archive = export_study(engine, result["result_hash"], root / "study.zip")
    return engine, result, archive


def test_study_replay_recomputes_without_execution(complete, monkeypatch):
    engine, result, archive = complete

    def forbidden(*_args, **_kwargs):
        pytest.fail("study artifact replay must not execute candidates or contact models")

    monkeypatch.setattr(Engine, "execute", forbidden)
    monkeypatch.setattr("paper2bench.repro.runtime.BuiltinRunner.run", forbidden)
    monkeypatch.setattr("paper2bench.repro.providers.ProviderClient.generate", forbidden)
    replay = replay_study(archive)
    assert replay["integrity"] == "verified"
    assert replay["authentication"] == "not_verified"
    assert replay["scores"] == "recomputed"
    assert replay["execution_rerun"] is False
    assert replay["assessment"] == result["assessment"]
    assert replay["trial_count"] == 6
    assert replay["plan_hash"] == result["plan_hash"]
    assert replay["study_result_hash"] == result["result_hash"]
    assert replay["resources"] == result["resources"]
    assert engine.artifacts.get(result["plan_hash"])


def test_original_spec_source_and_child_evidence_are_self_contained(complete):
    engine, result, archive = complete
    entries = read_zip(archive)
    for ref in result["original_specs"].values():
        assert entries[f"artifacts/{ref}"] == engine.artifacts.get(ref)
    manifest = json.loads(entries["manifest.json"])
    assert len(manifest["trials"]) == 6
    for trial in manifest["trials"]:
        with zipfile.ZipFile(io.BytesIO(entries[trial["member"]])) as inner:
            spec = json.loads(inner.read(f"artifacts/{trial['spec_hash']}"))
            source = inner.read(f"artifacts/{spec['source_sha256']}")
            assert hashlib.sha256(source).hexdigest() == spec["source_sha256"]
            assert inner.read(f"artifacts/{spec['dataset_sha256']}")
            assert inner.read(f"artifacts/{spec['engine_identity_sha256']}")


def test_export_refuses_to_overwrite_existing_archive(complete):
    engine, result, archive = complete
    original = archive.read_bytes()
    with pytest.raises(FileExistsError):
        export_study(engine, result["result_hash"], archive)
    assert archive.read_bytes() == original


def test_failed_publication_does_not_leave_partial_destination(complete, tmp_path, monkeypatch):
    engine, result, _archive = complete

    def broken_copy(_source, target, **_kwargs):
        target.write(b"partial")
        raise OSError("simulated disk failure")

    monkeypatch.setattr(study_bundles.shutil, "copyfileobj", broken_copy)
    destination = tmp_path / "not-published.zip"
    with pytest.raises(OSError, match="simulated disk failure"):
        export_study(engine, result["result_hash"], destination)
    assert not destination.exists()
    assert list(tmp_path.glob(".paper2bench-study-*")) == []


def test_failed_trial_keeps_entire_primary_study_not_tested(tmp_path, monkeypatch):
    engine = Engine(tmp_path / "state")
    plan_hash = make_study(engine)
    from paper2bench.repro.runtime import BuiltinRunner

    original = BuiltinRunner.run

    def fail_one(self, dataset, spec, **kwargs):
        if spec.seed == 7 and spec.title == "Study archive candidate":
            raise ExecutionFailure("not exposed")
        return original(self, dataset, spec, **kwargs)

    monkeypatch.setattr(BuiltinRunner, "run", fail_one)
    result = run_study(engine, plan_hash, "one-failure")
    assert result["assessment"]["verdict"] == "not_tested"
    archive = export_study(engine, result["result_hash"], tmp_path / "failed-study.zip")
    replay = replay_study(archive)
    assert replay["integrity"] == "verified"
    assert replay["scores"] == "not_tested"
    assert replay["trial_count"] == 6
    assert replay["assessment"] == result["assessment"]
    assert replay["resources"]["candidate"]["states"]["failed"] == 1


def test_coherently_rehashed_assessment_tampering_is_rejected(complete, tmp_path):
    _, _, archive = complete
    entries = change_result(
        read_zip(archive), lambda result: result["assessment"].update(effect=0.75)
    )
    with pytest.raises(ValueError, match="assessment.*recomputation"):
        replay_study(write_zip(tmp_path / "score-tampered.zip", entries))


def test_boolean_score_substitution_is_not_treated_as_numeric_equality(complete, tmp_path):
    _, _, archive = complete
    entries = change_result(
        read_zip(archive), lambda result: result["assessment"].update(effect=False)
    )
    with pytest.raises(ValueError, match="assessment.*recomputation"):
        replay_study(write_zip(tmp_path / "boolean-score.zip", entries))


def test_coherently_rehashed_resource_tampering_is_rejected(complete, tmp_path):
    _, _, archive = complete
    entries = change_result(
        read_zip(archive),
        lambda result: result["resources"]["candidate"].update(cost_usd_unknown_trials=0),
    )
    with pytest.raises(ValueError, match="resources"):
        replay_study(write_zip(tmp_path / "resource-tampered.zip", entries))


def test_omitted_trial_cannot_be_hidden_by_updating_manifest_and_result(complete, tmp_path):
    _, _, archive = complete
    entries = read_zip(archive)
    manifest = json.loads(entries["manifest.json"])
    removed = manifest["trials"].pop()
    del entries[removed["member"]]
    entries["manifest.json"] = canonical_bytes(manifest)
    change_result(entries, lambda result: result["trials"].pop())
    with pytest.raises(ValueError, match="omits"):
        replay_study(write_zip(tmp_path / "missing-trial.zip", entries))


def test_coherent_plan_schedule_tamper_is_rejected(complete, tmp_path):
    _, _, archive = complete
    entries = read_zip(archive)
    manifest = json.loads(entries["manifest.json"])
    old_plan = manifest["plan_hash"]
    plan = json.loads(entries[f"artifacts/{old_plan}"])
    plan["trials"] = list(reversed(plan["trials"]))
    new_plan = replace_blob(entries, manifest, old_plan, canonical_bytes(plan))
    manifest["plan_hash"] = new_plan
    entries["manifest.json"] = canonical_bytes(manifest)
    change_result(entries, lambda result: result.update(plan_hash=new_plan))
    with pytest.raises(ValueError, match="role/seed"):
        replay_study(write_zip(tmp_path / "plan-tampered.zip", entries))


@pytest.mark.parametrize("target", ["plan_hash", "study_result_hash"])
def test_outer_artifact_byte_corruption_is_rejected(complete, tmp_path, target):
    _, _, archive = complete
    entries = read_zip(archive)
    manifest = json.loads(entries["manifest.json"])
    entries[f"artifacts/{manifest[target]}"] += b" "
    with pytest.raises(ValueError, match="integrity"):
        replay_study(write_zip(tmp_path / "hash-corrupted.zip", entries))


def test_nested_archive_byte_corruption_is_rejected(complete, tmp_path):
    _, _, archive = complete
    entries = read_zip(archive)
    entries["trials/000.zip"] += b"unused trailer"
    with pytest.raises(ValueError, match="integrity"):
        replay_study(write_zip(tmp_path / "inner-corrupted.zip", entries))


def test_duplicate_outer_member_is_rejected(complete, tmp_path):
    _, _, archive = complete
    path = write_zip(tmp_path / "duplicate.zip", read_zip(archive))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, "a") as output:
            output.writestr("manifest.json", output.read("manifest.json"))
    with pytest.raises(ValueError, match="duplicate"):
        replay_study(path)


@pytest.mark.parametrize("extra", ["extra.txt", "../escape", "/absolute", "trials/../../escape"])
def test_extra_or_unsafe_members_are_rejected_without_extraction(complete, tmp_path, extra):
    _, _, archive = complete
    entries = read_zip(archive)
    entries[extra] = b"unexpected"
    with pytest.raises(ValueError, match="members"):
        replay_study(write_zip(tmp_path / "unsafe.zip", entries))
    assert not (tmp_path / "escape").exists()


def test_nested_extra_member_is_rejected_even_with_updated_outer_hash(complete, tmp_path):
    _, _, archive = complete

    def altered(data):
        with zipfile.ZipFile(io.BytesIO(data)) as inner:
            entries = {name: inner.read(name) for name in inner.namelist()}
        entries["../unexpected"] = b"not extracted"
        return zip_bytes(entries)

    entries = change_inner(read_zip(archive), 0, altered)
    with pytest.raises(ValueError, match="members"):
        replay_study(write_zip(tmp_path / "inner-extra.zip", entries))


def test_nested_duplicate_member_is_rejected_before_run_replay(complete, tmp_path, monkeypatch):
    _, _, archive = complete

    def duplicate(data):
        value = io.BytesIO(data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(value, "a") as inner:
                inner.writestr("manifest.json", inner.read("manifest.json"))
        return value.getvalue()

    def forbidden(*_args, **_kwargs):
        pytest.fail("nested duplicate must be rejected before replay reads artifacts")

    monkeypatch.setattr(study_bundles, "replay_bundle", forbidden)
    entries = change_inner(read_zip(archive), 0, duplicate)
    with pytest.raises(ValueError, match="duplicate"):
        replay_study(write_zip(tmp_path / "inner-duplicate.zip", entries))


def test_nested_declared_size_is_bounded_before_any_inner_read(complete, tmp_path, monkeypatch):
    _, _, archive = complete
    data = bytearray(zip_bytes({"manifest.json": b"{}"}))
    central = data.index(b"PK\x01\x02")
    struct.pack_into("<I", data, central + 24, 64 * 1024 * 1024 + 1)

    def forbidden(*_args, **_kwargs):
        pytest.fail("oversized nested member must be rejected before per-run replay")

    monkeypatch.setattr(study_bundles, "replay_bundle", forbidden)
    entries = change_inner(read_zip(archive), 0, lambda _old: bytes(data))
    with pytest.raises(ValueError, match="decompressed size"):
        replay_study(write_zip(tmp_path / "inner-oversized.zip", entries))


def test_cumulative_nested_size_is_bounded(complete, monkeypatch):
    _, _, archive = complete
    entries = read_zip(archive)
    outer_size = sum(map(len, entries.values()))
    nested_size = 0
    for name, data in entries.items():
        if name.startswith("trials/"):
            with zipfile.ZipFile(io.BytesIO(data)) as inner:
                nested_size += sum(info.file_size for info in inner.infolist())
    monkeypatch.setattr(study_bundles, "MAX_STUDY_BYTES", outer_size + nested_size - 1)
    with pytest.raises(ValueError, match="cumulative nested"):
        replay_study(archive)


def test_compressed_archive_size_is_bounded(complete, monkeypatch):
    _, _, archive = complete
    monkeypatch.setattr(study_bundles, "MAX_STUDY_BYTES", archive.stat().st_size - 1)
    with pytest.raises(ValueError, match="compressed size"):
        replay_study(archive)


def test_export_refuses_a_stale_queued_snapshot(tmp_path, monkeypatch):
    engine = Engine(tmp_path / "state")
    plan_hash = make_study(engine)
    execute = engine.execute
    monkeypatch.setattr(engine, "execute", lambda run_id: engine.runs.get_run(run_id))
    result = run_study(engine, plan_hash, "queued-snapshot")
    assert result["assessment"]["verdict"] == "not_tested"
    for trial in result["trials"]:
        execute(trial["run_id"])
    destination = tmp_path / "stale.zip"
    with pytest.raises(ValueError, match="snapshot"):
        export_study(engine, result["result_hash"], destination)
    assert not destination.exists()
