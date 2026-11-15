from __future__ import annotations

import json

import pytest

from paper2bench.repro.bundles import export_bundle, replay_bundle
from paper2bench.repro.demo import create_demo
from paper2bench.repro.engine import Engine
from paper2bench.repro.runtime import ExecutionCancelled


def test_cancelled_cleanup_diagnostics_survive_restart_and_bundle(tmp_path, monkeypatch):
    root = tmp_path / "state"
    engine = Engine(root)
    run = engine.submit(create_demo(engine), "cancel-diagnostic")

    def cancel(*_, **__):
        engine.runs.cancel(run["run_id"])
        exc = ExecutionCancelled("safe cancellation")
        exc.cleanup_confirmed = False
        exc.container_name = "paper2bench-" + "a" * 32
        raise exc

    monkeypatch.setattr("paper2bench.repro.runtime.BuiltinRunner.run", cancel)
    cancelled = engine.execute(run["run_id"])
    assert cancelled["state"] == "cancelled"
    assert cancelled["result_hash"] is None
    reopened = Engine(root)
    diagnostics = reopened.result(run["run_id"])["attempt_diagnostics"]
    assert diagnostics[0]["details"]["cleanup"] == {
        "cleanup_confirmed": False,
        "container_name": "paper2bench-" + "a" * 32,
    }
    assert diagnostics[0]["details"]["assessment"]["verdict"] == "not_tested"
    bundle = export_bundle(reopened, run["run_id"], tmp_path / "cancelled.zip")
    assert replay_bundle(bundle)["scores"] == "not_tested"
    import zipfile

    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["record"]["attempt_diagnostics"] == diagnostics


def test_publication_error_does_not_leave_heartbeat_thread(tmp_path, monkeypatch):
    import threading

    engine = Engine(tmp_path)
    run = engine.submit(create_demo(engine), "publication-error")

    def fail(*_, **__):
        raise OSError("simulated database publication failure")

    before = set(threading.enumerate())
    monkeypatch.setattr(engine.runs, "finish", fail)
    with pytest.raises(OSError):
        engine.execute(run["run_id"])
    assert set(threading.enumerate()) == before


def test_keyboard_interrupt_cancels_reference_and_retains_diagnostic(tmp_path, monkeypatch):
    engine = Engine(tmp_path / "state")
    run = engine.submit(create_demo(engine), "keyboard-interrupt")

    def interrupt(*_, **__):
        raise KeyboardInterrupt

    monkeypatch.setattr("paper2bench.repro.runtime.BuiltinRunner.run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        engine.execute(run["run_id"])
    assert engine.runs.get_run(run["run_id"])["state"] == "cancelled"
    diagnostics = engine.result(run["run_id"])["attempt_diagnostics"]
    assert diagnostics[0]["details"]["assessment"]["verdict"] == "not_tested"


def test_export_limit_includes_manifest_and_creates_no_partial_file(tmp_path, monkeypatch):
    from paper2bench.repro import bundles

    engine = Engine(tmp_path / "state")
    run = engine.run(create_demo(engine), "limit")
    path = export_bundle(engine, run["run_id"], tmp_path / "first.zip")
    import zipfile

    with zipfile.ZipFile(path) as archive:
        blob_sum = sum(
            item.file_size for item in archive.infolist() if item.filename != "manifest.json"
        )
        manifest_size = archive.getinfo("manifest.json").file_size
    monkeypatch.setattr(bundles, "MAX_BUNDLE_BYTES", blob_sum + manifest_size // 2)
    with pytest.raises(ValueError, match="size"):
        export_bundle(engine, run["run_id"], tmp_path / "too-big.zip")
    assert not (tmp_path / "too-big.zip").exists()
