"""Portable replay checks are numerical integrity checks, never execution claims."""

from __future__ import annotations

import hashlib
import json
import warnings
import zipfile

import pytest

from paper2bench.repro import bundles
from paper2bench.repro.authoring import draft_experiment
from paper2bench.repro.bundles import export_bundle, replay_bundle
from paper2bench.repro.engine import Engine
from paper2bench.repro.models import (
    ClaimSpec,
    Dataset,
    EvidenceAnchor,
    Example,
    ExperimentSpec,
    Policy,
    WorkloadSpec,
    canonical_bytes,
)
from paper2bench.repro.providers import Generation, OutcomeUnknown, ProviderProfile

SOURCE = "Classify each row by its nearest training centroid. Reported accuracy is 1.0."
IMAGE = "sha256:" + "a" * 64
CODE = b"import json,sys\np=json.load(sys.stdin)\nprint(json.dumps({'ids':[], 'labels':[]}))\n"


def make_run(tmp_path, *, kind="builtin_centroid", authoring=False, execute=True):
    """Only the builtin executes; other workloads use genuine queued records."""
    engine = Engine(tmp_path / "state")
    source_hash = engine.artifacts.put(SOURCE.encode())
    dataset = Dataset(
        train=(
            Example(id="train-neg", x=(-1.0,), y="neg"),
            Example(id="train-pos", x=(1.0,), y="pos"),
        ),
        test=tuple(
            Example(id=f"test-{i}", x=((1.0 if i % 2 else -1.0),), y="pos" if i % 2 else "neg")
            for i in range(80)
        ),
    )
    data_hash = engine.register_dataset(dataset)
    policy = Policy()
    if kind == "python":
        workload = WorkloadSpec(
            kind=kind, implementation_sha256=engine.artifacts.put(CODE), image=IMAGE
        )
    elif kind == "model_classification":
        profile_hash = engine.register_profile(
            ProviderProfile(name="local", base_url="http://127.0.0.1:12345/v1", location="local")
        )
        workload = WorkloadSpec(
            kind=kind, provider_profile=profile_hash, model="fixture", prompt="Classify the row."
        )
        policy = Policy(max_provider_calls=80)
    else:
        workload = WorkloadSpec(kind=kind)
    spec = ExperimentSpec(
        title="bundle fixture",
        source_sha256=source_hash,
        dataset_sha256=data_hash,
        evidence=(EvidenceAnchor(start=0, end=len(SOURCE), quote=SOURCE),),
        claim=ClaimSpec(target=1.0),
        workload=workload,
        policy=policy,
    )
    if authoring:
        assert kind == "python"

        class FixtureAuthor:
            profile = ProviderProfile(
                name="author", base_url="http://127.0.0.1:12345/v1", location="local"
            )

            def generate(self, **kwargs):
                return Generation(
                    text=json.dumps(
                        {
                            "title": "authored bundle fixture",
                            "evidence": [{"start": 0, "end": len(SOURCE), "quote": SOURCE}],
                            "assumptions": [],
                            "unresolved": [],
                            "claim": {"target": 1.0},
                            "implementation": CODE.decode(),
                        }
                    ),
                    model="fixture-author",
                )

        spec = draft_experiment(
            source_hash, data_hash, engine.artifacts, FixtureAuthor(), "fixture-author", True, IMAGE
        )
    spec_hash = engine.freeze(spec)
    run = (
        engine.run(spec_hash, "bundle-test") if execute else engine.submit(spec_hash, "bundle-test")
    )
    return engine, run


def read_bundle(path):
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def write_bundle(path, entries):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return path


def alter_manifest(entries, mutate):
    manifest = json.loads(entries["manifest.json"])
    mutate(manifest)
    entries["manifest.json"] = canonical_bytes(manifest)
    return entries


def remove_ref(entries, ref):
    del entries[f"artifacts/{ref}"]
    return alter_manifest(entries, lambda manifest: manifest["artifacts"].remove(ref))


def test_completed_builtin_bundle_recomputes_scores_without_execution(tmp_path, monkeypatch):
    engine, run = make_run(tmp_path)
    assert run["state"] == "completed"
    path = export_bundle(engine, run["run_id"], tmp_path / "experiment.zip")

    def never_execute(*args, **kwargs):
        pytest.fail("artifact replay must never execute candidate workloads")

    monkeypatch.setattr("paper2bench.repro.runtime.BuiltinRunner.run", never_execute)
    replay = replay_bundle(path)
    assert replay["integrity"] == "verified"
    assert replay["authentication"] == "not_verified"
    assert replay["scores"] == "recomputed"
    assert replay["execution_rerun"] is False
    assert replay["spec_hash"] == run["spec_hash"]
    assert replay["assessment"]["accuracy"] == 1.0
    assert replay["assessment"] == engine.result(run["run_id"])["result"]["assessment"]


def test_export_does_not_overwrite_existing_bundle(tmp_path):
    engine, run = make_run(tmp_path)
    path = export_bundle(engine, run["run_id"], tmp_path / "experiment.zip")
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        export_bundle(engine, run["run_id"], path)
    assert path.read_bytes() == original


def test_hashed_artifact_corruption_is_rejected(tmp_path):
    engine, run = make_run(tmp_path)
    entries = read_bundle(export_bundle(engine, run["run_id"], tmp_path / "original.zip"))
    entries[f"artifacts/{run['spec_hash']}"] += b" "
    with pytest.raises(ValueError, match="integrity"):
        replay_bundle(write_bundle(tmp_path / "corrupt.zip", entries))


def test_score_tampering_with_coherent_hashes_is_rejected_by_recomputation(tmp_path):
    engine, run = make_run(tmp_path)
    entries = read_bundle(export_bundle(engine, run["run_id"], tmp_path / "original.zip"))
    manifest = json.loads(entries["manifest.json"])
    old_hash = run["result_hash"]
    result = json.loads(entries.pop(f"artifacts/{old_hash}"))
    result["assessment"]["accuracy"] = 0.125
    result_bytes = canonical_bytes(result)
    new_hash = hashlib.sha256(result_bytes).hexdigest()
    entries[f"artifacts/{new_hash}"] = result_bytes
    manifest["artifacts"].remove(old_hash)
    manifest["artifacts"].append(new_hash)
    manifest["record"]["run"]["result_hash"] = new_hash
    manifest["record"]["result"] = result
    entries["manifest.json"] = canonical_bytes(manifest)
    with pytest.raises(ValueError, match="(assessment|recomput|score)"):
        replay_bundle(write_bundle(tmp_path / "score-tampered.zip", entries))


@pytest.mark.parametrize("field", ["source_sha256", "dataset_sha256", "spec_hash", "result_hash"])
def test_missing_required_core_artifact_rejected(tmp_path, field):
    engine, run = make_run(tmp_path)
    entries = read_bundle(export_bundle(engine, run["run_id"], tmp_path / "original.zip"))
    spec = engine.load_spec(run["spec_hash"])
    ref = getattr(spec, field) if field in {"source_sha256", "dataset_sha256"} else run[field]
    remove_ref(entries, ref)
    with pytest.raises(ValueError):
        replay_bundle(write_bundle(tmp_path / "incomplete.zip", entries))


@pytest.mark.parametrize(
    "kind,field",
    [("python", "implementation_sha256"), ("model_classification", "provider_profile")],
)
def test_missing_required_workload_artifact_rejected(tmp_path, kind, field):
    engine, run = make_run(tmp_path, kind=kind, execute=False)
    entries = read_bundle(export_bundle(engine, run["run_id"], tmp_path / "original.zip"))
    ref = getattr(engine.load_spec(run["spec_hash"]).workload, field)
    remove_ref(entries, ref)
    with pytest.raises(ValueError):
        replay_bundle(write_bundle(tmp_path / "incomplete.zip", entries))


def test_export_includes_authoring_prompt_response_and_receipt(tmp_path):
    engine, run = make_run(tmp_path, kind="python", authoring=True, execute=False)
    entries = read_bundle(export_bundle(engine, run["run_id"], tmp_path / "authored.zip"))
    spec = engine.load_spec(run["spec_hash"])
    receipt = engine.artifacts.get_json(spec.authoring_receipt_sha256)
    assert f"artifacts/{spec.authoring_receipt_sha256}" in entries
    assert f"artifacts/{receipt['prompt_sha256']}" in entries
    assert f"artifacts/{receipt['response_sha256']}" in entries
    replay = replay_bundle(tmp_path / "authored.zip")
    assert replay["integrity"] == "verified"
    assert replay["authentication"] == "not_verified"
    assert replay["scores"] == "not_tested"


@pytest.mark.parametrize("field", ["authoring_receipt_sha256", "prompt_sha256", "response_sha256"])
def test_missing_required_authoring_artifact_rejected(tmp_path, field):
    engine, run = make_run(tmp_path, kind="python", authoring=True, execute=False)
    entries = read_bundle(export_bundle(engine, run["run_id"], tmp_path / "original.zip"))
    spec = engine.load_spec(run["spec_hash"])
    receipt = engine.artifacts.get_json(spec.authoring_receipt_sha256)
    ref = spec.authoring_receipt_sha256 if field == "authoring_receipt_sha256" else receipt[field]
    assert f"artifacts/{ref}" in entries
    remove_ref(entries, ref)
    with pytest.raises(ValueError):
        replay_bundle(write_bundle(tmp_path / "incomplete.zip", entries))


def test_duplicate_archive_member_rejected(tmp_path):
    engine, run = make_run(tmp_path)
    path = export_bundle(engine, run["run_id"], tmp_path / "duplicate.zip")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("manifest.json", archive.read("manifest.json"))
    with pytest.raises(ValueError, match="duplicate"):
        replay_bundle(path)


def test_duplicate_manifest_artifact_reference_rejected(tmp_path):
    engine, run = make_run(tmp_path)
    entries = read_bundle(export_bundle(engine, run["run_id"], tmp_path / "original.zip"))
    alter_manifest(entries, lambda manifest: manifest["artifacts"].append(manifest["artifacts"][0]))
    with pytest.raises(ValueError):
        replay_bundle(write_bundle(tmp_path / "duplicate-ref.zip", entries))


@pytest.mark.parametrize("path", ["extra.txt", "../escape", "/absolute", "artifacts/not-a-hash"])
def test_extra_and_unsafe_paths_rejected(tmp_path, path):
    engine, run = make_run(tmp_path)
    entries = read_bundle(export_bundle(engine, run["run_id"], tmp_path / "original.zip"))
    entries[path] = b"unexpected"
    with pytest.raises(ValueError):
        replay_bundle(write_bundle(tmp_path / "extra.zip", entries))


def test_unsupported_bundle_version_rejected(tmp_path):
    engine, run = make_run(tmp_path)
    entries = read_bundle(export_bundle(engine, run["run_id"], tmp_path / "original.zip"))
    alter_manifest(entries, lambda manifest: manifest.update(schema_version="999"))
    with pytest.raises(ValueError, match="version"):
        replay_bundle(write_bundle(tmp_path / "future.zip", entries))


def test_replay_checks_decompressed_size_before_reading_members(tmp_path, monkeypatch):
    engine, run = make_run(tmp_path)
    path = export_bundle(engine, run["run_id"], tmp_path / "too-large.zip")
    monkeypatch.setattr(bundles, "MAX_BUNDLE_BYTES", 100)
    with pytest.raises(ValueError, match="size"):
        replay_bundle(path)


def test_export_size_limit_does_not_create_partial_archive(tmp_path, monkeypatch):
    engine, run = make_run(tmp_path)
    path = tmp_path / "too-large.zip"
    monkeypatch.setattr(bundles, "MAX_BUNDLE_BYTES", 100)
    with pytest.raises(ValueError, match="size"):
        export_bundle(engine, run["run_id"], path)
    assert not path.exists()


def test_unknown_provider_outcome_replays_as_not_tested(tmp_path, monkeypatch):
    engine, queued = make_run(tmp_path, kind="model_classification", execute=False)

    def unknown(*args, **kwargs):
        raise OutcomeUnknown("simulated accepted request timeout")

    monkeypatch.setattr(engine, "_execute_model", unknown)
    run = engine.execute(queued["run_id"])
    assert run["state"] == "outcome_unknown"
    replay = replay_bundle(export_bundle(engine, run["run_id"], tmp_path / "unknown.zip"))
    assert replay["integrity"] == "verified"
    assert replay["authentication"] == "not_verified"
    assert replay["scores"] == "not_tested"
    assert replay["execution_state"] == "outcome_unknown"
    assert replay["execution_rerun"] is False
