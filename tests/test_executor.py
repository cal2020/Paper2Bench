import io
import os
import signal
import sys
import tarfile
import threading
import time
from pathlib import Path

import pytest

from paper2bench.executors import (
    DockerExecutor, ExecutionPolicyError, ExecutionRequest, ExecutionUnavailable,
)
from paper2bench.executors.docker import _CommandResult, _command, _restore, _snapshot


def request(tmp_path, **kwargs):
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return ExecutionRequest(code="print('hello')", workspace=workspace, **kwargs)


@pytest.mark.parametrize("values", [
    {"timeout_seconds": float("nan")}, {"timeout_seconds": True},
    {"cpus": 0}, {"memory_bytes": -1}, {"output_limit_bytes": True},
    {"workspace_bytes": 1073741824}, {"pids_limit": 1},
])
def test_request_rejects_unbounded_or_invalid_limits(tmp_path, values):
    with pytest.raises(ExecutionPolicyError):
        request(tmp_path, **values)


def test_container_has_limits_and_no_host_mounts(tmp_path):
    req = request(tmp_path)
    command = DockerExecutor()._create_command("test-name", req, "sha256:" + "a" * 64)
    joined = " ".join(command)
    for option in ("--network none", "--read-only", "--user 65534:65534", "--cap-drop ALL",
                   "no-new-privileges=true", "--pull never", "--pids-limit 64", "--log-driver none"):
        assert option in joined
    assert "size=67108864" in joined
    assert "--mount" not in command and "--volume" not in command and "-v" not in command
    assert str(req.workspace) not in joined


def test_credentials_and_symlinks_never_enter_snapshot(tmp_path):
    req = request(tmp_path)
    (req.workspace / ".env").write_text("OPENAI_API_KEY=private")
    with pytest.raises(ExecutionPolicyError):
        _snapshot(req.workspace, req)
    (req.workspace / ".env").unlink()
    (req.workspace / "escape").symlink_to(tmp_path)
    with pytest.raises(ExecutionPolicyError):
        _snapshot(req.workspace, req)


def test_snapshot_roundtrip_and_controller_hashes(tmp_path):
    req = request(tmp_path)
    (req.workspace / "nested").mkdir()
    (req.workspace / "nested" / "data.json").write_text('[1, 2, 3]')
    archive = _snapshot(req.workspace, req)
    artifacts = _restore(archive, req.workspace, req)
    assert (req.workspace / "nested" / "data.json").read_text() == "[1, 2, 3]"
    assert artifacts["nested/data.json"]["bytes"] == 9
    assert len(artifacts["nested/data.json"]["sha256"]) == 64


def make_archive(name, *, kind=None, contents=b"danger"):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        member = tarfile.TarInfo(name)
        if kind:
            member.type, member.linkname = kind, "../../secret"
            archive.addfile(member)
        else:
            member.size = len(contents)
            archive.addfile(member, io.BytesIO(contents))
    return stream.getvalue()


@pytest.mark.parametrize("name,kind", [
    ("../../escape", None), ("/absolute", None), (".env", None),
    ("nested/.aws/credentials", None), ("link", tarfile.SYMTYPE),
    ("hardlink", tarfile.LNKTYPE), ("fifo", tarfile.FIFOTYPE),
])
def test_invalid_artifacts_do_not_replace_workspace(tmp_path, name, kind):
    req = request(tmp_path)
    (req.workspace / "original").write_text("keep")
    with pytest.raises(ExecutionPolicyError):
        _restore(make_archive(name, kind=kind), req.workspace, req)
    assert (req.workspace / "original").read_text() == "keep"
    assert not (tmp_path.parent / "escape").exists()


def test_snapshot_limits_are_not_just_metadata(tmp_path):
    req = request(tmp_path, max_files=1)
    (req.workspace / "one").write_text("1")
    (req.workspace / "two").write_text("2")
    with pytest.raises(ExecutionPolicyError):
        _snapshot(req.workspace, req)


def test_command_does_not_inherit_provider_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "private-test-key")
    result = _command([sys.executable, "-c", "import os; print(os.environ.get('OPENAI_API_KEY', 'absent'))"])
    assert result.returncode == 0 and result.stdout.strip() == b"absent"


def test_output_is_bounded_and_process_is_stopped():
    result = _command([sys.executable, "-u", "-c", "while True: print('x'*8192)"], limit=512, timeout=5)
    assert result.output_exceeded
    assert len(result.stdout) + len(result.stderr) <= 512


def test_timeout_terminates_command():
    start = time.monotonic()
    result = _command([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.1)
    assert result.timed_out and time.monotonic() - start < 5


def test_command_group_signal_denial_still_stops_direct_client(monkeypatch):
    import paper2bench.executors.docker as module
    def denied(*args):
        raise PermissionError("managed sandbox")
    monkeypatch.setattr(module.os, "killpg", denied)
    result = _command([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.05)
    assert result.timed_out and result.returncode is not None


def test_remote_daemon_is_rejected(monkeypatch):
    import paper2bench.executors.docker as module
    monkeypatch.setattr(module.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(module, "_command", lambda *a, **kw: _CommandResult(0, b"tcp://remote:2376\n"))
    with pytest.raises(ExecutionUnavailable):
        DockerExecutor().preflight()


def test_missing_engine_never_falls_back(monkeypatch, tmp_path):
    import paper2bench.executors.docker as module
    monkeypatch.setattr(module.shutil, "which", lambda _: None)
    with pytest.raises(ExecutionUnavailable):
        DockerExecutor().execute(request(tmp_path))


def mock_job(monkeypatch, tmp_path, callback):
    import paper2bench.executors.docker as module
    executor = DockerExecutor(lifecycle_callback=callback)
    req = request(tmp_path)
    operations = []
    monkeypatch.setattr(executor, "preflight", lambda **kw: {"image_id": "sha256:" + "a" * 64})
    def command(args, **kwargs):
        operations.append(args[1])
        return _CommandResult(0, b"container-id\n")
    monkeypatch.setattr(module, "_command", command)
    archive = _snapshot(req.workspace, req)
    values = iter([_CommandResult(0), _CommandResult(0, b"hello"), _CommandResult(0), _CommandResult(0, archive)])
    monkeypatch.setattr(executor, "_exec", lambda *a, **kw: next(values))
    return executor, req, operations


def test_lifecycle_intent_precedes_create_and_records_cleanup(monkeypatch, tmp_path):
    events = []
    executor, req, operations = mock_job(monkeypatch, tmp_path, lambda event, payload: events.append((event, payload, list(operations))))
    result = executor.execute(req)
    assert result.status == "succeeded", result.to_dict()
    assert [event for event, _, _ in events] == ["job_intent", "job_created", "job_cleanup"]
    assert "create" not in events[0][2]
    assert "create" in events[1][2] and "rm" in events[2][2]
    assert events[0][1]["container_name"] == result.evidence["container_name"]


def test_lifecycle_failure_before_create_never_dispatches(monkeypatch, tmp_path):
    def fail(event, payload):
        if event == "job_intent":
            raise OSError("journal unavailable")
    executor, req, operations = mock_job(monkeypatch, tmp_path, fail)
    result = executor.execute(req)
    assert result.status == "execution_error" and result.evidence["cleanup_confirmed"]
    assert "create" not in operations and "start" not in operations


def test_lifecycle_failure_after_create_still_cleans_up(monkeypatch, tmp_path):
    def fail(event, payload):
        if event == "job_created":
            raise OSError("journal unavailable")
    executor, req, operations = mock_job(monkeypatch, tmp_path, fail)
    result = executor.execute(req)
    assert result.status == "execution_error" and result.evidence["cleanup_confirmed"]
    assert operations == ["create", "rm"]


@pytest.mark.parametrize("name", ["", "--all", "other-container", "p2b-../../escape", "p2b-" + "g" * 32])
def test_reconcile_never_accepts_arbitrary_container_names(monkeypatch, name):
    executor = DockerExecutor()
    monkeypatch.setattr(executor, "preflight", lambda **kw: pytest.fail("must validate before Docker"))
    with pytest.raises(ExecutionPolicyError):
        executor.reconcile(name, terminate=True)


def test_reconcile_inspects_without_implicit_termination(monkeypatch):
    import paper2bench.executors.docker as module
    executor = DockerExecutor()
    monkeypatch.setattr(executor, "preflight", lambda **kw: {"backend": "docker"})
    monkeypatch.setattr(module, "_command", lambda *a, **kw: _CommandResult(0, b'{"Running":true,"Status":"running"}'))
    monkeypatch.setattr(executor, "_remove", lambda *a: pytest.fail("must not terminate"))
    result = executor.reconcile("p2b-" + "a" * 32)
    assert result["running"] and not result["cleanup_confirmed"]


def test_reconcile_terminates_exact_recorded_name_only(monkeypatch):
    import paper2bench.executors.docker as module
    executor = DockerExecutor()
    monkeypatch.setattr(executor, "preflight", lambda **kw: {"backend": "docker"})
    monkeypatch.setattr(module, "_command", lambda *a, **kw: _CommandResult(0, b'{"Running":true}'))
    removed = []
    monkeypatch.setattr(executor, "_remove", lambda name: removed.append(name) or True)
    name = "p2b-" + "b" * 32
    result = executor.reconcile(name, terminate=True)
    assert removed == [name] and result["cleanup_confirmed"] and not result["running"]


def test_reconcile_preserves_unknown_inspection_errors(monkeypatch):
    import paper2bench.executors.docker as module
    executor = DockerExecutor()
    monkeypatch.setattr(executor, "preflight", lambda **kw: {"backend": "docker"})
    monkeypatch.setattr(module, "_command", lambda *a, **kw: _CommandResult(1, stderr=b"daemon unavailable"))
    result = executor.reconcile("p2b-" + "c" * 32)
    assert result["found"] is None and result["running"] is None and not result["cleanup_confirmed"]


def test_reconcile_missing_job_confirms_no_remaining_container(monkeypatch):
    import paper2bench.executors.docker as module
    executor = DockerExecutor()
    monkeypatch.setattr(executor, "preflight", lambda **kw: {"backend": "docker"})
    monkeypatch.setattr(module, "_command", lambda *a, **kw: _CommandResult(1, stderr=b"Error: No such object"))
    result = executor.reconcile("p2b-" + "d" * 32)
    assert result["found"] is False and result["cleanup_confirmed"]


@pytest.mark.skipif(os.getenv("P2B_DOCKER_TESTS") != "1", reason="opt-in real Docker isolation test")
def test_real_container_isolation_persistence_and_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-enter-worker")
    req = request(tmp_path)
    first = DockerExecutor().execute(ExecutionRequest(
        code="import os, pathlib; assert 'OPENAI_API_KEY' not in os.environ; pathlib.Path('value.txt').write_text('42'); print('created')",
        workspace=req.workspace,
    ))
    assert first.status == "succeeded", first.to_dict()
    assert first.evidence["cleanup_confirmed"]
    assert first.artifacts["value.txt"]["bytes"] == 2
    reconciled = DockerExecutor().reconcile(first.evidence["container_name"])
    assert reconciled["found"] is False and reconciled["cleanup_confirmed"], reconciled
    second = DockerExecutor().execute(ExecutionRequest(
        code="from pathlib import Path; print(Path('value.txt').read_text()); assert not Path('/Users/escape').exists()",
        workspace=req.workspace,
    ))
    assert second.status == "succeeded" and "42" in second.stdout, second.to_dict()
    timed = DockerExecutor().execute(ExecutionRequest(
        code="import subprocess, time; subprocess.Popen(['python', '-c', 'import time; time.sleep(60)']); time.sleep(60)",
        workspace=req.workspace, timeout_seconds=0.3,
    ))
    assert timed.status == "timed_out" and timed.evidence["cleanup_confirmed"], timed.to_dict()


@pytest.mark.skipif(os.getenv("P2B_DOCKER_TESTS") != "1", reason="opt-in real Docker isolation test")
def test_real_container_blocks_network_and_root_writes(tmp_path):
    req = request(tmp_path)
    result = DockerExecutor().execute(ExecutionRequest(code="""
import pathlib, socket
try:
    pathlib.Path('/etc/p2b-escape').write_text('bad')
except OSError:
    pass
else:
    raise AssertionError('root filesystem writable')
try:
    socket.create_connection(('1.1.1.1', 443), timeout=0.2)
except OSError:
    pass
else:
    raise AssertionError('outbound networking available')
print('isolated')
""", workspace=req.workspace))
    assert result.status == "succeeded" and result.evidence["cleanup_confirmed"], result.to_dict()
