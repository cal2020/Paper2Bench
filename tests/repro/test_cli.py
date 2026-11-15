from __future__ import annotations

import json

from paper2bench.repro.cli import main


def test_cli_offline_demo_report_export_replay(tmp_path, capsys):
    state = tmp_path / "state"
    common = ["--state", str(state)]
    assert main(common + ["demo", "--key", "cli-demo"]) == 0
    assert "Execution: completed" in capsys.readouterr().out
    assert main(common + ["runs"]) == 0
    runs = json.loads(capsys.readouterr().out)
    assert len(runs) == 1
    run_id = runs[0]["run_id"]
    assert main(common + ["inspect", run_id]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["result"]["assessment"]["accuracy"] == 1.0
    assert main(common + ["demo", "--key", "cli-demo"]) == 0
    capsys.readouterr()
    bundle = tmp_path / "result.zip"
    assert main(common + ["export", run_id, str(bundle)]) == 0
    capsys.readouterr()
    assert main(["replay", str(bundle)]) == 0
    replay = json.loads(capsys.readouterr().out)
    assert replay["scores"] == "recomputed"
    assert replay["execution_rerun"] is False


def test_schema_does_not_create_state_directory(tmp_path, capsys):
    state = tmp_path / "unused"
    assert main(["--state", str(state), "schema"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["title"] == "ExperimentSpec"
    assert not state.exists()


def test_bad_profile_never_echoes_misplaced_secret(tmp_path, capsys):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"api_key": "secret-that-must-not-leak"}))
    assert main(["--state", str(tmp_path / "state"), "profile", str(path)]) == 2
    captured = capsys.readouterr()
    assert "secret-that-must-not-leak" not in captured.err
    assert "validation_error" in captured.err


def test_invalid_input_has_nonzero_exit(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text("{}")
    assert main(["--state", str(tmp_path / "state"), "freeze", str(path)]) == 2
    assert "validation_error" in capsys.readouterr().err


def test_worker_processes_queued_runs(tmp_path, capsys):
    from paper2bench.repro.demo import create_demo
    from paper2bench.repro.engine import Engine

    state = tmp_path / "state"
    engine = Engine(state)
    run = engine.submit(create_demo(engine), "queue-worker-test")
    assert main(["--state", str(state), "worker", "--once"]) == 0
    assert engine.runs.get_run(run["run_id"])["state"] == "completed"
    assert json.loads(capsys.readouterr().out)["state"] == "completed"
