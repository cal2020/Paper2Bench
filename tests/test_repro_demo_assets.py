"""Source-chain, sanitization, and deterministic public evidence regressions."""

import hashlib
import importlib.util
import io
import json
import math
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("repro_exporter", ROOT / "scripts/export_repro_demo.py")
exporter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exporter)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pack(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
    return buffer.getvalue()


def unpack(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def run_fixture(kind, seed=0, role="candidate"):
    artifacts = {}

    def put(value):
        raw = value if isinstance(value, bytes) else canonical(value)
        artifacts[sha(raw)] = raw
        return sha(raw)

    source = put(b"Synthetic verification fixture; no published result is reproduced.")
    engine = put({"modules": {"core.py": put(b"# test-only engine fixture")}})
    code = put(("# test-only " + role).encode())
    if kind == "knapsack":
        data = {"task": "knapsack", "instances": [
            {"id": f"knapsack-{i:03}", "weights": [2, 3, 3], "values": [3, 4, 4], "capacity": 6}
            for i in range(80)
        ]}
        predictions = {"ids": [x["id"] for x in data["instances"]], "selections": [[1, 2]] * 80}
        radius = math.sqrt(math.log(40) / 160)
        assessment = {
            "mean_score": 1.0, "feasible_count": 80, "confidence": .95,
            "confidence_interval": {"lower": 1 - radius, "upper": 1.0},
            "criterion_band": {"lower": .8, "upper": 1.0}, "verdict": "supported",
            "per_item": [{"id": x["id"], "selection": [1, 2], "feasible": True,
                          "objective": 8, "optimal_objective": 8, "score": 1.0,
                          "total_weight": 6} for x in data["instances"]],
        }
    else:
        data = {
            "train": [{"id": "train-a", "x": [-1.0], "y": "a"},
                      {"id": "train-b", "x": [1.0], "y": "b"}],
            "test": [{"id": f"test-{i:03}", "x": [-1.0 if i % 2 == 0 else 1.0],
                      "y": "a" if i % 2 == 0 else "b"} for i in range(80)],
        }
        labels = ([x["y"] for x in data["test"]] if seed == 22 else ["b"] * 80) if role == "candidate" else ["a"] * 80
        predictions = {"ids": [x["id"] for x in data["test"]], "labels": labels}
        assessment = {"accuracy": sum(x["y"] == label for x, label in zip(data["test"], labels)) / 80}
    dataset_hash = put(data)
    spec = {"seed": seed, "dataset_sha256": dataset_hash, "source_sha256": source,
            "engine_identity_sha256": engine,
            "assumptions": ["Synthetic fixture; no published result is reproduced."],
            "policy": {"local_only": True},
            "workload": {"kind": "python", "implementation_sha256": code}}
    spec_hash = put(spec)
    result = {"spec_hash": spec_hash, "execution_state": "completed", "predictions": predictions,
              "assessment": assessment, "provider_calls": [],
              "measurements": {"runtime": "docker", "network_used": False, "cleanup_confirmed": True}}
    result_hash = put(result)
    manifest = {"artifacts": sorted(artifacts), "record": {
        "run": {"state": "completed", "spec_hash": spec_hash, "result_hash": result_hash,
                "worker": "PRIVATE-WORKER", "run_id": "PRIVATE-RUN"}, "result": result}}
    bundle = pack({"manifest.json": canonical(manifest),
                   **{"artifacts/" + key: value for key, value in artifacts.items()}})
    return bundle, spec, spec_hash, dataset_hash


@pytest.fixture
def source_dist(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    suite = ET.Element("testsuites")
    tests = ET.SubElement(suite, "testsuite", tests="735", failures="0", errors="0", skipped="0",
                          timestamp="2026-11-13T07:01:12+00:00", hostname="PRIVATE-HOST")
    for index in range(642):
        ET.SubElement(tests, "testcase", name=f"case-{index}", file="/workspace/PRIVATE/file.py")
    (source / "verification-v0.2.xml").write_bytes(ET.tostring(suite))
    receipt = {"version": "0.2.0", "pytest_passed": 642, "subtests_passed": 93,
               "failures": 0, "errors": 0, "skipped": 0, "live_paid_model_calls": False,
               "cloud_deployment_created": False, "private_extra": "PRIVATE-RECEIPT-SECRET"}
    (source / "verification-v0.2.json").write_bytes(canonical(receipt))
    (source / "optimization-v0.2.zip").write_bytes(run_fixture("knapsack")[0])
    trials, trial_files, top_artifacts = [], {}, {}
    originals = {}
    for role in ("candidate", "baseline"):
        _raw, spec, identity, dataset_hash = run_fixture("classification", role=role)
        top_artifacts[identity] = canonical(spec)
        originals[role + "_spec_hash"] = identity
    for seed in (11, 22, 33):
        for role in ("candidate", "baseline"):
            raw, _spec, identity, dataset_hash = run_fixture("classification", seed, role)
            member = f"trials/{len(trials):03}.zip"
            trial_files[member] = raw
            trials.append({"role": role, "seed": seed, "spec_hash": identity,
                           "member": member, "bundle_sha256": sha(raw), "run_id": "PRIVATE-RUN"})
    plan = {"dataset_sha256": dataset_hash, "bootstrap_seed": 3681840326672871555,
            "study": {**originals, "seeds": [11, 22, 33], "confidence": .95,
                      "bootstrap_samples": 2000, "minimum_effect": .02},
            "trials": [{key: trial[key] for key in ("role", "seed", "spec_hash")} for trial in trials]}
    plan_hash = sha(canonical(plan))
    top_artifacts[plan_hash] = canonical(plan)
    matrix = [[-1, 1] * 40, [0, 1] * 40, [-1, 1] * 40]
    report = {"plan_hash": plan_hash, "dataset_sha256": dataset_hash,
              "assessment": {"candidate_accuracy": 2 / 3, "baseline_accuracy": .5,
                             "effect": 1 / 6, "verdict": "inconclusive",
                             "confidence_interval": {"lower": -.15, "upper": .475},
                             "paired_correctness_differences": matrix}}
    result_hash = sha(canonical(report))
    top_artifacts[result_hash] = canonical(report)
    manifest = {"artifacts": sorted(top_artifacts), "plan_hash": plan_hash,
                "study_result_hash": result_hash, "trials": trials}
    (source / "seeded-study-v0.2.zip").write_bytes(pack({
        "manifest.json": canonical(manifest),
        **{"artifacts/" + key: value for key, value in top_artifacts.items()}, **trial_files,
    }))
    (source / "seeded-study-v0.2.json").write_bytes(canonical({**report, "result_hash": result_hash}))
    return source


def test_export_is_deterministic_sanitized_and_readback_verified(source_dist, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    result = exporter.export(source_dist, first)
    exporter.export(source_dist, second)
    assert result["verified"] is True
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {p.name: p.read_bytes() for p in second.iterdir()}
    assert exporter.verify_public_assets(first) == result
    for path in first.iterdir():
        if path.suffix != ".zip":
            assert b"PRIVATE-" not in path.read_bytes()
            assert b"/workspace/" not in path.read_bytes()
    verification = json.loads((first / "verification-v0.2.json").read_bytes())
    assert verification["counts"] == {"pytest_passed": 642, "subtests_passed": 93, "junit_tests": 735,
                                       "failures": 0, "errors": 0, "skipped": 0}
    assert verification["canonical_repository_verification"]["test_count"] is None
    optimization = json.loads((first / "optimization-v0.2.json").read_bytes())
    assert optimization["summary"]["mean_score"] == 1.0
    assert optimization["recomputed"]["method"] == "independent_exhaustive_enumeration"
    study = json.loads((first / "seeded-study-v0.2.json").read_bytes())
    assert study["summary"]["verdict"] == "inconclusive"
    assert study["assessment"]["confidence_interval"]["lower"] == -.15
    assert study["assessment"]["confidence_interval"]["upper"] == .475
    assert len(study["trials"]) == 6
    with zipfile.ZipFile(first / "evidence-v0.2.zip") as archive:
        assert set(archive.namelist()) == {"verification-v0.2.json", "optimization-v0.2.json",
                                           "seeded-study-v0.2.json", "README.txt", "sha256-manifest.json"}
        assert all(item.date_time == (1980, 1, 1, 0, 0, 0) for item in archive.infolist())
        assert not any(name.endswith((".whl", ".sqlite", ".pdf")) for name in archive.namelist())


def test_corrupt_source_artifact_fails_before_publication(source_dist, tmp_path):
    path = source_dist / "optimization-v0.2.zip"
    files = unpack(path.read_bytes())
    artifact = next(name for name in files if name.startswith("artifacts/"))
    files[artifact] += b"corruption"
    path.write_bytes(pack(files))
    with pytest.raises(exporter.ReproExportError, match="hash mismatch"):
        exporter.export(source_dist, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_forged_score_with_consistent_hash_chain_is_rejected(source_dist, tmp_path):
    path = source_dist / "optimization-v0.2.zip"
    files = unpack(path.read_bytes())
    manifest = json.loads(files["manifest.json"])
    old = manifest["record"]["run"]["result_hash"]
    result = json.loads(files.pop("artifacts/" + old))
    result["assessment"]["mean_score"] = .5
    new = sha(canonical(result))
    files["artifacts/" + new] = canonical(result)
    manifest["artifacts"] = [new if item == old else item for item in manifest["artifacts"]]
    manifest["record"]["run"]["result_hash"] = new
    manifest["record"]["result"] = result
    files["manifest.json"] = canonical(manifest)
    path.write_bytes(pack(files))
    with pytest.raises(exporter.ReproExportError, match="recomputation"):
        exporter.export(source_dist, tmp_path / "output")


def test_forged_study_interval_is_rejected_after_hash_resealing(source_dist, tmp_path):
    path = source_dist / "seeded-study-v0.2.zip"
    files = unpack(path.read_bytes())
    manifest = json.loads(files["manifest.json"])
    old = manifest["study_result_hash"]
    report = json.loads(files.pop("artifacts/" + old))
    report["assessment"]["confidence_interval"]["lower"] = .1
    new = sha(canonical(report))
    files["artifacts/" + new] = canonical(report)
    manifest["artifacts"] = [new if item == old else item for item in manifest["artifacts"]]
    manifest["study_result_hash"] = new
    files["manifest.json"] = canonical(manifest)
    path.write_bytes(pack(files))
    (source_dist / "seeded-study-v0.2.json").write_bytes(canonical({**report, "result_hash": new}))
    with pytest.raises(exporter.ReproExportError, match="recomputation"):
        exporter.export(source_dist, tmp_path / "output")


@pytest.mark.parametrize("member", ["../private", ".env", "state.sqlite", "trials/999.zip"])
def test_unexpected_archive_members_fail_closed(source_dist, tmp_path, member):
    path = source_dist / "optimization-v0.2.zip"
    files = unpack(path.read_bytes())
    files[member] = b"private state"
    path.write_bytes(pack(files))
    with pytest.raises(exporter.ReproExportError, match="member"):
        exporter.export(source_dist, tmp_path / "output")


def test_private_xml_entities_and_count_drift_rejected(source_dist, tmp_path):
    path = source_dist / "verification-v0.2.xml"
    original = path.read_bytes()
    path.write_bytes(b'<!DOCTYPE x [<!ENTITY secret SYSTEM "file:///private">]>' + original)
    with pytest.raises(exporter.ReproExportError, match="declarations"):
        exporter.export(source_dist, tmp_path / "output")
    path.write_bytes(original.replace(b'tests="735"', b'tests="736"'))
    with pytest.raises(exporter.ReproExportError, match="counts"):
        exporter.export(source_dist, tmp_path / "output")


def test_public_drift_and_symlink_substitution_rejected(source_dist, tmp_path):
    output = tmp_path / "public"
    exporter.export(source_dist, output)
    file = output / "verification-v0.2.json"
    original = file.read_bytes()
    file.write_bytes(original + b"\n")
    with pytest.raises(exporter.ReproExportError, match="manifest"):
        exporter.verify_public_assets(output)
    file.unlink()
    target = tmp_path / "private.json"
    target.write_bytes(original)
    file.symlink_to(target)
    with pytest.raises(exporter.ReproExportError, match="symlink"):
        exporter.verify_public_assets(output)


def test_public_zip_resealing_cannot_add_unreviewed_bytes(source_dist, tmp_path):
    output = tmp_path / "public"
    exporter.export(source_dist, output)
    archive_path = output / "evidence-v0.2.zip"
    files = unpack(archive_path.read_bytes())
    files["extra.txt"] = b"unreviewed"
    archive_path.write_bytes(pack(files))
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"]["evidence-v0.2.zip"] = {"sha256": sha(archive_path.read_bytes()),
                                                "bytes": archive_path.stat().st_size}
    manifest_path.write_bytes(canonical(manifest))
    with pytest.raises(exporter.ReproExportError, match="ZIP"):
        exporter.verify_public_assets(output)


def test_output_refuses_unrelated_files_and_symlink_parents(source_dist, tmp_path):
    output = tmp_path / "public"
    output.mkdir()
    (output / "operator.txt").write_text("preserve")
    with pytest.raises(exporter.ReproExportError, match="unrelated"):
        exporter.export(source_dist, output)
    assert (output / "operator.txt").read_text() == "preserve"
    link = tmp_path / "link"
    link.symlink_to(output, target_is_directory=True)
    with pytest.raises(exporter.ReproExportError, match="symlink"):
        exporter.export(source_dist, link / "nested")


def test_committed_public_assets_verify_without_local_raw_state():
    assert exporter.verify_public_assets(ROOT / "docs/repro")["verified"] is True
    guide = (ROOT / "docs/repro/README.txt").read_text()
    assert "paper2bench-repro --state .paper2bench-repro demo --task knapsack" in guide
    assert "NOT a full runtime replay bundle" in guide
    assert "standalone v0.2" in guide


def test_static_builder_admits_nested_repro_files_without_changing_existing_provenance_rules():
    spec = importlib.util.spec_from_file_location("repro_asset_builder", ROOT / "scripts/build_demo.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    files = builder._sources(release=False)
    for name in exporter.PUBLIC_NAMES:
        assert files["docs/repro/" + name] == (ROOT / "docs/repro" / name).read_bytes()
    assert "docs/results/adam-counterexample.json" in files
