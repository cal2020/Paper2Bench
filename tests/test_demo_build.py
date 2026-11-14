"""Release build admission and immutable artifact regressions."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("demo_builder", SOURCE_ROOT / "scripts/build_demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def test_release_rejects_dirty_source_before_mutating_output(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    (output / "keep.txt").write_text("Existing operator files")
    monkeypatch.setattr(demo, "_identity", lambda: {"dirty": True})
    monkeypatch.setattr(demo, "_sources", lambda **kwargs: pytest.fail("No source dispatch after failed admission"))
    with pytest.raises(demo.DemoBuildError, match="clean Git checkout"):
        demo.build(output, release=True)
    assert list(output.iterdir()) == [output / "keep.txt"]


def test_source_symlink_substitution_and_actual_read_bounds(tmp_path, monkeypatch):
    monkeypatch.setattr(demo, "ROOT", tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "private.txt").write_text("Not a demo asset")
    (tmp_path / "docs/linked.txt").symlink_to(tmp_path / "private.txt")
    with pytest.raises(demo.DemoBuildError, match="symlink"):
        demo._read("docs/linked.txt")
    (tmp_path / "docs/large.txt").write_bytes(b"x" * 17)
    monkeypatch.setattr(demo, "MAX_FILE_BYTES", 16)
    with pytest.raises(demo.DemoBuildError, match="byte limit"):
        demo._read("docs/large.txt")


@pytest.mark.parametrize("path", ["/private.txt", "../private.txt", "docs/../private.txt", "docs//input.json", "docs\\input.json", ".env"])
def test_escaping_or_hidden_source_names_are_not_admitted(path):
    with pytest.raises(demo.DemoBuildError):
        demo._read(path)


def test_fifo_source_is_rejected_without_blocking(tmp_path, monkeypatch):
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO admission is a POSIX profile")
    monkeypatch.setattr(demo, "ROOT", tmp_path)
    (tmp_path / "docs").mkdir()
    os.mkfifo(tmp_path / "docs/pipe.txt")
    with pytest.raises(demo.DemoBuildError, match="regular files"):
        demo._read("docs/pipe.txt")


def test_output_cannot_overwrite_source_or_follow_dist_symlink(tmp_path, monkeypatch):
    monkeypatch.setattr(demo, "ROOT", tmp_path)
    (tmp_path / "docs").mkdir()
    with pytest.raises(demo.DemoBuildError, match="outside docs"):
        demo._admit_output(tmp_path / "docs/build")
    out = tmp_path / "site"
    out.mkdir()
    (out / "dist").symlink_to(tmp_path / "docs")
    with pytest.raises(demo.DemoBuildError, match="symlink"):
        demo._admit_output(out)


def test_different_site_identity_is_preserved(tmp_path, monkeypatch):
    output = tmp_path / "site"
    (output / ".openai").mkdir(parents=True)
    old = output / ".openai/hosting.json"
    old.write_text('{"project_id":"appgprj_other"}')
    monkeypatch.setattr(demo, "_identity", lambda: {"dirty": False})
    monkeypatch.setattr(demo, "_sources", lambda **kwargs: {
        demo.BUILD_INPUTS[-1]: b'{"project_id":"appgprj_expected","static":{"directory":"dist"}}'})
    with pytest.raises(demo.DemoBuildError, match="different Site identity"):
        demo.build(output, release=True)
    assert old.read_text() == '{"project_id":"appgprj_other"}'


@pytest.fixture
def build_repository(tmp_path, monkeypatch):
    try:
        demo._node()
    except (demo.DemoBuildError, OSError, subprocess.CalledProcessError):
        pytest.skip("Integration build needs Node 20+; use PAPER2BENCH_NODE when necessary")
    if not (SOURCE_ROOT / "node_modules/esbuild/package.json").is_file():
        pytest.skip("Integration build needs npm ci")
    root = tmp_path / "repository"
    root.mkdir()
    shutil.copytree(SOURCE_ROOT / "docs", root / "docs")
    for name in demo.BUILD_INPUTS:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE_ROOT / name, target)
    (root / ".gitignore").write_text("node_modules/\ndist/\n")
    (root / "node_modules").symlink_to(SOURCE_ROOT / "node_modules", target_is_directory=True)
    subprocess.run(["git", "init", "--initial-branch=main", str(root)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "-c", "commit.gpgsign=false", "commit", "-m", "Pinned demo fixture"], check=True, capture_output=True)
    monkeypatch.setattr(demo, "ROOT", root)
    return root


def artifact_bytes(path):
    return {p.relative_to(path).as_posix(): p.read_bytes() for p in path.rglob("*") if p.is_file()}


def test_release_build_is_reproducible_and_has_complete_local_script_closure(tmp_path, build_repository):
    a, b = tmp_path / "first", tmp_path / "second"
    first = demo.build(a, release=True)
    second = demo.build(b, release=True)
    assert first["source_commit"] == second["source_commit"]
    assert artifact_bytes(a / "dist") == artifact_bytes(b / "dist")
    files = artifact_bytes(a / "dist")
    manifest = json.loads(files["asset-manifest.json"])
    assert set(manifest["files"]) == set(files) - {"asset-manifest.json"}
    assert all(hashlib.sha256(files[name]).hexdigest() == expected for name, expected in manifest["files"].items())
    provenance = json.loads(files["build-provenance.json"])
    assert provenance["source"]["dirty"] is False and provenance["build"]["mode"] == "release"
    assert provenance["build"]["package_lock_sha256"] == hashlib.sha256((build_repository / "package-lock.json").read_bytes()).hexdigest()
    assert files["scripts.js"]  # Legacy static example tabs still have their local script.
    assert files["examples/cgcnn/instruction.txt"] and files["examples/cgcnn/task_config.yaml"]
    for page in ("index.html", "example.html"):
        html = files[page].decode()
        scripts = re.findall(r'<script\b[^>]*src="([^"]+)"', html)
        assert len(scripts) == 1 and scripts[0] in files
        assert scripts[0].startswith("assets/") and "text/babel" not in html and "unpkg.com" not in html


def test_source_change_during_build_does_not_replace_prior_artifact(tmp_path, build_repository, monkeypatch):
    output = tmp_path / "published"
    demo.build(output, release=True)
    original = artifact_bytes(output / "dist")
    original_run = subprocess.run

    def mutate_after_compile(arguments, *args, **kwargs):
        result = original_run(arguments, *args, **kwargs)
        if len(arguments) > 1 and str(arguments[1]).endswith("build_demo.mjs"):
            target = build_repository / "docs/site-app.jsx"
            target.write_text(target.read_text() + "\n// Concurrent source change\n")
        return result

    monkeypatch.setattr(subprocess, "run", mutate_after_compile)
    with pytest.raises(demo.DemoBuildError, match="changed during build"):
        demo.build(output, release=True)
    assert artifact_bytes(output / "dist") == original


def test_release_output_inside_repository_does_not_dirty_its_source(build_repository):
    output = build_repository / "deployment/demo"
    result = demo.build(output, release=True)
    assert result["source_dirty"] is False
    assert demo._identity()["dirty"] is False
    assert (output / "dist/index.html").is_file()
