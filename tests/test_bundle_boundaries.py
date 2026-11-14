"""Bundle reads stay bounded when filesystem names or YAML aliases change."""
import os
import shutil
from pathlib import Path

import pytest

from paper2bench import bundles
from paper2bench.bundles import (
    BundleError, atomic_json, digest, export_bundle, public_config,
    read_artifact, read_json, validate_task, write_draft_manifest,
)


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/tasks/linear-fit"


def test_json_open_cannot_follow_a_symlink_inserted_at_dispatch(tmp_path, monkeypatch):
    source = tmp_path / "input.json"
    source.write_text('{"public":true}')
    secret = tmp_path / "secret.json"
    secret.write_text('{"secret":"host credential"}')
    original_open = os.open

    def replace_before_open(path, flags, *args, **kwargs):
        if Path(path) == source:
            source.unlink()
            source.symlink_to(secret)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(bundles.os, "open", replace_before_open)
    with pytest.raises(BundleError, match="nonsymlink"):
        read_json(source)


@pytest.mark.parametrize("reader", ["json", "artifact"])
def test_growth_after_descriptor_stat_does_not_bypass_read_bound(tmp_path, monkeypatch, reader):
    source = tmp_path / "input.json"
    source.write_text("{}")
    inode = source.stat().st_ino
    original_fstat = os.fstat
    grown = False

    def grow_after_stat(fd):
        nonlocal grown
        info = original_fstat(fd)
        if info.st_ino == inode and not grown:
            grown = True
            with source.open("ab") as stream:
                stream.write(b" " * 128)
        return info

    monkeypatch.setattr(bundles.os, "fstat", grow_after_stat)
    with pytest.raises(BundleError, match="size limit"):
        if reader == "json":
            read_json(source, limit=16)
        else:
            read_artifact(tmp_path, source.name, limit=16)
    assert grown


def test_artifact_read_remains_bound_to_opened_parent_directory(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (nested / "payload.txt").write_bytes(b"public artifact")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "payload.txt").write_bytes(b"host secret")
    original_open = os.open

    def replace_parent(path, flags, *args, **kwargs):
        if path == "payload.txt" and "dir_fd" in kwargs:
            nested.rename(root / "original")
            nested.symlink_to(outside, target_is_directory=True)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(bundles.os, "open", replace_parent)
    assert read_artifact(root, "nested/payload.txt") == b"public artifact"


def test_bundle_validator_never_uses_an_unbounded_file_read(tmp_path, monkeypatch):
    shutil.copytree(EXAMPLE, tmp_path / "task")

    def unbounded_read(*args, **kwargs):
        pytest.fail("Bundle validation reopened a checked path for an unbounded read")

    monkeypatch.setattr(Path, "read_bytes", unbounded_read)
    assert validate_task(tmp_path / "task")["task_id"] == "linear-fit"


def test_artifact_size_is_rejected_before_reading_sparse_oversized_file(tmp_path):
    source = tmp_path / "large.json"
    with source.open("wb") as stream:
        stream.truncate(bundles.MAX_FILE_BYTES + 1)
    with pytest.raises(BundleError, match="size limit"):
        read_artifact(tmp_path, source.name)


def test_fifo_json_is_rejected_without_waiting_for_a_writer(tmp_path):
    fifo = tmp_path / "input.json"
    os.mkfifo(fifo)
    with pytest.raises(BundleError, match="regular"):
        read_json(fifo)


@pytest.mark.parametrize("field,value", [
    ("task_mode", []), ("readiness", {}), ("entrypoint", []),
    ("inputs", {}), ("predictions", "result\x00.json"),
    ("predictions", "a" * 241 + ".json"),
    ("limits", {"cpus": 10**1000}),
])
def test_malformed_manifest_fields_raise_bounded_validation_errors(tmp_path, field, value):
    source = tmp_path / "task"
    shutil.copytree(EXAMPLE, source)
    manifest = read_json(source / "task_manifest.json")
    manifest[field] = value
    atomic_json(source / "task_manifest.json", manifest)
    with pytest.raises(BundleError):
        validate_task(source)


def test_yaml_alias_expansion_has_an_independent_node_budget():
    repeated = ["one public value"] * 256
    with pytest.raises(BundleError, match="expanded node"):
        public_config({"datasets": [repeated] * 256})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_public_configuration_is_rejected(value):
    with pytest.raises(BundleError, match="Nonfinite"):
        public_config({"datasets": [{"metric": value}]})


def test_invalid_yaml_export_does_not_create_destination(tmp_path):
    source = tmp_path / "task"
    shutil.copytree(EXAMPLE, source)
    raw = b"datasets: [unterminated"
    (source / "task_config.yaml").write_bytes(raw)
    manifest = read_json(source / "task_manifest.json")
    manifest["files"]["task_config.yaml"] = digest(raw)
    atomic_json(source / "task_manifest.json", manifest)
    destination = tmp_path / "export"
    with pytest.raises(BundleError, match="YAML"):
        export_bundle(source, destination)
    assert not destination.exists()


def test_invalid_draft_identifier_does_not_create_artifacts(tmp_path):
    destination = tmp_path / "task"
    with pytest.raises(BundleError, match="task ID"):
        write_draft_manifest(destination, {"task_id": "../escape", "research_question": "Question"})
    assert not destination.exists()


def test_draft_replaces_file_symlink_without_overwriting_target(tmp_path):
    secret = tmp_path / "host.txt"
    secret.write_text("private host data")
    candidate = tmp_path / "task/candidate"
    candidate.mkdir(parents=True)
    (candidate / "task_config.yaml").symlink_to(secret)
    manifest = write_draft_manifest(candidate.parent, {"task_id": "task", "research_question": "Question"})
    assert secret.read_text() == "private host data"
    assert not (candidate / "task_config.yaml").is_symlink()
    assert validate_task(manifest)["task_id"] == "task"


def test_draft_cannot_follow_candidate_directory_symlink(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "task"
    root.mkdir()
    (root / "candidate").symlink_to(outside, target_is_directory=True)
    with pytest.raises(BundleError, match="directory cannot be a symlink"):
        write_draft_manifest(root, {"task_id": "task", "research_question": "Question"})
    assert not list(outside.iterdir())
