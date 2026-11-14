"""A published archive must be bound to reviewed source and exact payload bytes."""
import importlib.util
import gzip
import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

SPEC = importlib.util.spec_from_file_location("package_demo", Path(__file__).resolve().parents[1] / "scripts/package_demo.py")
packager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packager)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True) + "\n")


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    for name in packager.ESSENTIAL_INPUTS | {"docs/index.html", "docs/example.html", "docs/main.js"}:
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("public fixture\n")
    hosting = {"project_id": "appgprj_test_identity", "static": {"directory": "dist"}}
    write_json(repository / "deployment/demo/.openai/hosting.json", hosting)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repository), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture"], check=True)
    commit = git(repository, "rev-parse", "HEAD")
    monkeypatch.setattr(packager, "ROOT", repository)
    source = tmp_path / "deploy"
    source.mkdir()
    (source / ".openai").mkdir()
    hosting_raw = (repository / "deployment/demo/.openai/hosting.json").read_bytes()
    (source / ".openai/hosting.json").write_bytes(hosting_raw)
    dist = source / "dist"
    dist.mkdir()
    (dist / "index.html").write_text('<script src="assets/main.js"></script>')
    (dist / "example.html").write_text("<main>Illustrative example</main>")
    (dist / "assets").mkdir()
    (dist / "assets/main.js").write_text("console.log('public fixture')")
    inputs = {name: packager.sha256((repository / name).read_bytes()) for name in packager._source_inputs(commit)}
    provenance = {"schema_version": 1, "kind": "paper2bench-static-demo",
                  "source": {"repository": packager.REPOSITORY, "commit": commit, "tree": git(repository, "rev-parse", "HEAD^{tree}"), "dirty": False, "files": inputs},
                  "build": {"mode": "release", "package_lock_sha256": inputs["package-lock.json"]},
                  "hosting": {"project_id": hosting["project_id"], "manifest_sha256": packager.sha256(hosting_raw)}}
    write_json(dist / "build-provenance.json", provenance)
    manifest = {"schema_version": 1, "files": {p.relative_to(dist).as_posix(): packager.sha256(p.read_bytes()) for p in dist.rglob("*") if p.is_file()}}
    write_json(dist / "asset-manifest.json", manifest)
    return source, commit


def reseal(source):
    dist = source / "dist"
    manifest = json.loads((dist / "asset-manifest.json").read_text())
    for name in manifest["files"]:
        manifest["files"][name] = packager.sha256((dist / name).read_bytes())
    write_json(dist / "asset-manifest.json", manifest)


def test_archive_is_deterministic_and_contains_only_verified_payloads(artifact, tmp_path):
    source, commit = artifact
    first, second = tmp_path / "first.tar.gz", tmp_path / "second.tar.gz"
    report = packager.package(source, first, commit)
    packager.package(source, second, commit)
    assert first.read_bytes() == second.read_bytes()
    assert report["source_commit"] == commit
    with tarfile.open(first) as archive:
        assert set(archive.getnames()) == {".openai/hosting.json", "dist/asset-manifest.json", "dist/build-provenance.json", "dist/index.html", "dist/example.html", "dist/assets/main.js"}
        assert all(item.uid == item.gid == item.mtime == 0 for item in archive.getmembers())


@pytest.mark.parametrize("change", ["extra", "bytes", "symlink", "project", "commit", "tree", "dirty", "source_hash", "lock_hash"])
def test_release_artifact_rejects_identity_and_payload_drift(artifact, tmp_path, change):
    source, commit = artifact
    dist = source / "dist"
    if change == "extra":
        (dist / "unlisted.txt").write_text("unreviewed")
    elif change == "bytes":
        (dist / "assets/main.js").write_text("changed")
    elif change == "symlink":
        (dist / "assets/main.js").unlink()
        (dist / "assets/main.js").symlink_to(tmp_path / "private")
    elif change == "project":
        write_json(source / ".openai/hosting.json", {"project_id": "wrong", "static": {"directory": "dist"}})
    else:
        path = dist / "build-provenance.json"
        data = json.loads(path.read_text())
        if change in {"commit", "tree"}:
            data["source"][change] = "0" * 40
        elif change == "dirty":
            data["source"]["dirty"] = True
        elif change == "source_hash":
            data["source"]["files"]["docs/main.js"] = "0" * 64
        else:
            data["build"]["package_lock_sha256"] = "0" * 64
        write_json(path, data)
        reseal(source)
    with pytest.raises(packager.ArtifactError):
        packager.package(source, tmp_path / "bad.tar.gz", commit)
    assert not (tmp_path / "bad.tar.gz").exists()


@pytest.mark.parametrize("name", ["../outside.js", "/outside.js", "nested/.env", "node_modules/token.js", "assets\\escape.js"])
def test_manifest_paths_cannot_escape_or_include_private_runtime_files(artifact, name):
    source, commit = artifact
    path = source / "dist/asset-manifest.json"
    data = json.loads(path.read_text())
    data["files"][name] = "0" * 64
    write_json(path, data)
    with pytest.raises(packager.ArtifactError):
        packager.validate(source, commit)


def test_verify_only_writes_no_archive(artifact, tmp_path):
    source, commit = artifact
    destination = tmp_path / "must-not-exist.tar.gz"
    report = packager.package(source, destination, commit, verify_only=True)
    assert report["verified"] and not destination.exists()


def test_safe_extract_round_trip_admits_expected_commit(artifact, tmp_path):
    source, commit = artifact
    archive = tmp_path / "release.tar.gz"
    packager.package(source, archive, commit)
    destination = tmp_path / "extracted"
    report = packager.extract_archive(archive, destination, commit)
    assert report["source_commit"] == commit
    assert (destination / "dist/assets/main.js").read_bytes() == (source / "dist/assets/main.js").read_bytes()
    assert packager.validate(destination, commit)[1]["verified"]


@pytest.mark.parametrize("kind", ["escape", "symlink", "hardlink", "device", "duplicate", "extra"])
def test_safe_extract_rejects_untrusted_tar_entries(artifact, tmp_path, kind):
    source, commit = artifact
    archive = tmp_path / "malicious.tar.gz"
    payloads, _ = packager.validate(source, commit)
    with tarfile.open(archive, "w:gz") as output:
        for name, raw in payloads.items():
            item = tarfile.TarInfo(name)
            item.size = len(raw)
            output.addfile(item, io.BytesIO(raw))
        item = tarfile.TarInfo("../escaped" if kind == "escape" else "dist/extra.txt")
        raw = b"untrusted"
        if kind in {"symlink", "hardlink", "device"}:
            item.type = {"symlink": tarfile.SYMTYPE, "hardlink": tarfile.LNKTYPE, "device": tarfile.CHRTYPE}[kind]
            item.linkname = "../../private"
            output.addfile(item)
        else:
            if kind == "duplicate":
                item.name = "dist/index.html"
            item.size = len(raw)
            output.addfile(item, io.BytesIO(raw))
    destination = tmp_path / "extracted"
    with pytest.raises(packager.ArtifactError):
        packager.extract_archive(archive, destination, commit)
    assert not destination.exists()
    assert not (tmp_path / "escaped").exists()


def test_safe_extract_bounds_decompression_before_tar_parsing(artifact, tmp_path, monkeypatch):
    source, commit = artifact
    archive = tmp_path / "bomb.tar.gz"
    archive.write_bytes(gzip.compress(b"0" * 200000))
    monkeypatch.setattr(packager, "MAX_TOTAL_BYTES", 1000)
    monkeypatch.setattr(packager, "MAX_FILES", 1)
    with pytest.raises(packager.ArtifactError, match="Decompressed"):
        packager.extract_archive(archive, tmp_path / "extracted", commit)


def test_safe_extract_never_overwrites_existing_destination(artifact, tmp_path):
    source, commit = artifact
    archive = tmp_path / "release.tar.gz"
    packager.package(source, archive, commit)
    destination = tmp_path / "extracted"
    destination.mkdir()
    (destination / "preserved").write_text("keep")
    with pytest.raises(packager.ArtifactError, match="empty"):
        packager.extract_archive(archive, destination, commit)
    assert (destination / "preserved").read_text() == "keep"
