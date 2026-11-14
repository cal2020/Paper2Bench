import base64
import io
import json
import os
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from paper2bench.executors import DockerExecutor, ExecutionPolicyError, ExecutionRequest, ExecutionUnavailable, GPURequest, SSHExecutor, RemoteProtocolError
from paper2bench.executors.docker import _CommandResult, _snapshot
from paper2bench.executors.gpu import GPUDeviceLease, preflight_gpu
from paper2bench.executors.remote_protocol import PROTOCOL, decode, encode
from paper2bench.executors.worker import WorkerPolicy, serve

GPU = "GPU-11111111-2222-3333-4444-555555555555"
IMAGE = "sha256:" + "a" * 64


@pytest.mark.parametrize("seed", [-1, 4294967296, True, 1.5, "1"])
def test_seed_has_strict_bounded_type(tmp_path, seed):
    with pytest.raises(ExecutionPolicyError):
        ExecutionRequest("pass", tmp_path, seed=seed)


def test_seed_is_only_inside_candidate_python_prelude(tmp_path, monkeypatch):
    import paper2bench.executors.docker as module
    executor = DockerExecutor()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    request = ExecutionRequest("print('candidate')", workspace, seed=123)
    monkeypatch.setattr(executor, "preflight", lambda **kw: {"image_id": IMAGE})
    monkeypatch.setattr(module, "_command", lambda *a, **kw: _CommandResult(0, b"container\n"))
    scripts = []
    values = iter([_CommandResult(0), _CommandResult(0), _CommandResult(0), _CommandResult(0, _snapshot(workspace, request))])
    def command(name, code, **kwargs):
        scripts.append(code)
        return next(values)
    monkeypatch.setattr(executor, "_exec", command)
    result = executor.execute(request)
    assert result.status == "succeeded"
    assert "random.seed(123)" in scripts[1] and "PAPER2BENCH_SEED" in scripts[1]
    assert "PAPER2BENCH_SEED" not in os.environ
    assert result.evidence["seed"] == 123
    assert result.evidence["seed_scope"] == "python_random_and_PAPER2BENCH_SEED_only"


@pytest.mark.parametrize("value", [
    {"device_ids": ["all"]}, {"device_ids": [GPU], "hard_memory_limit": True},
    {"device_ids": [GPU], "dedicated": False}, {"device_ids": [GPU, GPU]},
    {"device_ids": [GPU], "minimum_memory_bytes": True},
])
def test_gpu_cannot_imply_unenforced_limits_or_arbitrary_devices(value):
    with pytest.raises(ExecutionPolicyError):
        GPURequest.from_dict(value)


def gpu_command(*, active=False, runtime=True):
    def command(args, **kwargs):
        if args[1:3] == ["info", "--format"]:
            return _CommandResult(0, json.dumps({"nvidia": {}} if runtime else {}).encode())
        if "--query-gpu=uuid,memory.total,memory.used,driver_version" in args:
            return _CommandResult(0, (GPU + ", 24576, 0, 570.12\n").encode())
        return _CommandResult(0, (GPU + ", 1234\n").encode() if active else b"")
    return command


def test_gpu_inventory_admits_explicit_device_and_records_assurance(monkeypatch, tmp_path):
    import paper2bench.executors.gpu as module
    monkeypatch.setattr(module.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    request = GPURequest.from_dict({"device_ids": [GPU], "minimum_memory_bytes": 8 * 1024**3, "dedicated": True})
    metadata = preflight_gpu(request, "docker", gpu_command())
    assert metadata["devices"][0]["uuid"] == GPU
    assert metadata["hard_vram_quota"] is False
    executor = DockerExecutor(gpu_request=asdict(request))
    executor._gpu_devices = [GPU]
    command = executor._create_command("job", ExecutionRequest("pass", tmp_path), IMAGE)
    assert command[command.index("--gpus") + 1] == '"device=' + GPU + '"'
    assert "--read-only" in command and "--cap-drop" in command
    assert "NVIDIA_DRIVER_CAPABILITIES=compute,utility" in command


def test_gpu_missing_runtime_or_active_device_fails_preflight(monkeypatch):
    import paper2bench.executors.gpu as module
    monkeypatch.setattr(module.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    request = GPURequest.from_dict({"device_ids": [GPU], "dedicated": True})
    with pytest.raises(ExecutionUnavailable, match="runtime"):
        preflight_gpu(request, "docker", gpu_command(runtime=False))
    with pytest.raises(ExecutionUnavailable, match="active"):
        preflight_gpu(request, "docker", gpu_command(active=True))


def test_gpu_advisory_lease_blocks_other_local_controller():
    with GPUDeviceLease([GPU]):
        with pytest.raises(ExecutionUnavailable, match="leased"):
            with GPUDeviceLease([GPU]):
                pass
    with GPUDeviceLease([GPU]):
        pass


def ssh_config(tmp_path):
    known, key = tmp_path / "known_hosts", tmp_path / "identity"
    known.write_text("operator-pinned-host-key")
    key.write_text("operator-selected-key")
    return {"host": "worker.example", "user": "research", "known_hosts": known, "identity_file": key}


@pytest.mark.parametrize("change", [{"host": "-oProxyCommand=bad"}, {"user": "root; command"},
                                     {"remote_python": "python3; command"}, {"port": True}])
def test_ssh_connection_fields_cannot_be_shell_or_ssh_options(tmp_path, change):
    with pytest.raises(ExecutionPolicyError):
        SSHExecutor(**dict(ssh_config(tmp_path), **change))


def test_ssh_command_has_pinned_keys_no_agent_no_proxy_and_no_code(tmp_path, monkeypatch):
    import paper2bench.executors.ssh as module
    monkeypatch.setattr(module.shutil, "which", lambda _: "/usr/bin/ssh")
    executor = SSHExecutor(**ssh_config(tmp_path), remote_policy="/etc/paper2bench/worker.json")
    args = executor._command()
    for setting in ("StrictHostKeyChecking=yes", "BatchMode=yes", "IdentitiesOnly=yes", "IdentityAgent=none",
                    "ProxyCommand=none", "ForwardAgent=no", "ClearAllForwardings=yes"):
        assert setting in args
    assert args[-1] == "python3 -m paper2bench.executors.worker --policy /etc/paper2bench/worker.json"
    assert "-F" in args and "/dev/null" in args


def test_direct_ssh_adapter_cannot_bypass_local_only_policy(tmp_path, monkeypatch):
    from paper2bench.runtime_config import PolicyDenied, runtime_scope
    import paper2bench.executors.ssh as module
    profile = {"models": {"local": {"provider": "ollama", "model": "test", "hosting": "local_server",
                                     "locality_assurance": "operator_trusted"}},
               "roles": {"author": "local"}, "policy": {"data_egress": "local_only"}}
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **kw: pytest.fail("No SSH dispatch under local_only"))
    with runtime_scope(profile), pytest.raises(PolicyDenied):
        SSHExecutor(**ssh_config(tmp_path)).preflight()


FAKE_WORKER = r'''
import json, os, uuid
from pathlib import Path
from paper2bench.executors.worker import serve
from paper2bench.executors.docker import ExecutionResult
class Backend:
    def __init__(self, image, gpu_request, lifecycle_callback):
        self.image=image; self.callback=lifecycle_callback
    def preflight(self):
        return {"backend":"docker", "image_id":"sha256:"+"a"*64, "network":"none"}
    def execute(self, request):
        name="p2b-"+uuid.uuid4().hex
        self.callback("job_intent", {"container_name":name,"image_id":self.image})
        self.callback("job_created", {"container_name":name,"container_id":"id", "image_id":self.image})
        value=json.loads((request.workspace/"input.json").read_text())["value"]
        (request.workspace/"output.json").write_text(json.dumps({"value":value+1,"seed":request.seed,"credentials_absent":"OPENAI_API_KEY" not in os.environ}))
        self.callback("job_cleanup", {"container_name":name,"cleanup_confirmed":True})
        return ExecutionResult("succeeded",0,"worker", "",0.1,evidence={"container_name":name,"cleanup_confirmed":True,"image_id":self.image})
    def reconcile(self, name, terminate=False):
        return {"container_name":name,"cleanup_confirmed":True,"found":False,"running":False}
raise SystemExit(serve(executor_factory=Backend))
'''


class LocalTransport(SSHExecutor):
    def _command(self):
        return [sys.executable, "-u", "-c", FAKE_WORKER]


def test_remote_worker_real_subprocess_transport_acknowledges_before_dispatch(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-enter-transport")
    events = []
    executor = LocalTransport(**ssh_config(tmp_path), lifecycle_callback=lambda event, payload: events.append((event, payload)))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "input.json").write_text('{"value":41}')
    result = executor.execute(ExecutionRequest("not executed by transport fixture", workspace, seed=17))
    assert result.status == "succeeded", result.to_dict()
    assert [event for event, _ in events] == ["job_intent", "job_created", "job_cleanup"]
    data = json.loads((workspace / "output.json").read_text())
    assert data == {"value": 42, "seed": 17, "credentials_absent": True}
    assert result.artifacts["output.json"]["sha256"]
    assert result.backend == "ssh" and result.evidence["cleanup_confirmed"]
    assert executor.image == IMAGE
    assert executor.reconcile(events[0][1]["container_name"])["cleanup_confirmed"]


def test_remote_worker_cannot_proceed_when_controller_persistence_fails(tmp_path):
    def fail(event, payload):
        if event == "job_intent":
            raise OSError("journal unavailable")
    executor = LocalTransport(**ssh_config(tmp_path), lifecycle_callback=fail)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "input.json").write_text('{"value":1}')
    result = executor.execute(ExecutionRequest("unused", workspace))
    assert result.status == "execution_error" and not result.evidence["cleanup_confirmed"]
    assert not (workspace / "output.json").exists()
    assert result.evidence["container_name"].startswith("p2b-")


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"x":1e999}', b'{"x":NaN}'])
def test_remote_protocol_rejects_ambiguous_and_nonfinite_values(raw):
    with pytest.raises(ExecutionPolicyError):
        decode(raw)


def test_remote_worker_policy_cannot_be_weakened_by_request(tmp_path):
    policy = WorkerPolicy.from_dict({"ceilings": {"workspace_bytes": 1024}})
    with pytest.raises(ExecutionPolicyError, match="policy"):
        policy.request({"code": "pass", "snapshot_b64": "", "workspace_bytes": 2048}, tmp_path)
    with pytest.raises(ExecutionPolicyError):
        WorkerPolicy.from_dict({"ceilings": {"cpus": 1000}})


def test_remote_deadline_stops_nonreading_peer_with_full_snapshot_pipe(tmp_path, monkeypatch):
    import paper2bench.executors.ssh as module
    class Nonreader(SSHExecutor):
        def _command(self):
            return [sys.executable, "-u", "-c", "import time; time.sleep(30)"]
    executor = Nonreader(**ssh_config(tmp_path))
    original = module.subprocess.Popen
    children = []
    def capture(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(module.subprocess, "Popen", capture)
    before = {thread.ident for thread in threading.enumerate()}
    started = time.monotonic()
    with pytest.raises(RemoteProtocolError, match="deadline"):
        # Much larger than a pipe buffer. The peer never consumes a byte.
        executor._rpc("execute", timeout_seconds=0.1, request={"snapshot_b64": "x" * 2097152})
    assert time.monotonic() - started < 1
    assert children and all(child.poll() is not None for child in children)
    assert {thread.ident for thread in threading.enumerate()} == before


def test_lifecycle_ack_deadline_is_not_blocked_by_initial_snapshot_writer(tmp_path, monkeypatch):
    import paper2bench.executors.ssh as module
    class FixedIdentity:
        hex = "b" * 32
    monkeypatch.setattr(module.uuid, "uuid4", lambda: FixedIdentity())
    class EarlyEventNonreader(SSHExecutor):
        def _command(self):
            # Emit a matching fixture event without consuming the huge request.
            # Both snapshot and ACK share a full pipe and the same deadline.
            code = r'''
import json, time
print(json.dumps({"protocol":"paper2bench.remote/1","request_id":"b"*32,"type":"event","event":"job_intent","payload":{"container_name":"p2b-"+"a"*32}}),flush=True)
time.sleep(30)
'''
            return [sys.executable, "-u", "-c", code]
    events = []
    executor = EarlyEventNonreader(**ssh_config(tmp_path), lifecycle_callback=lambda *event: events.append(event))
    started = time.monotonic()
    with pytest.raises(RemoteProtocolError):
        executor._rpc("execute", timeout_seconds=0.15, request={"snapshot_b64": "x" * 2097152})
    assert time.monotonic() - started < 1
    assert events and events[0][0] == "job_intent"


@pytest.mark.skipif(os.getenv("P2B_DOCKER_TESTS") != "1", reason="opt-in real Docker via worker subprocess")
def test_remote_worker_protocol_with_real_isolated_docker(tmp_path):
    class RealLocalTransport(SSHExecutor):
        def _command(self):
            return [sys.executable, "-u", "-m", "paper2bench.executors.worker"]
    executor = RealLocalTransport(**ssh_config(tmp_path))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    request = ExecutionRequest("import os, random, pathlib; assert 'OPENAI_API_KEY' not in os.environ; pathlib.Path('seed.txt').write_text(os.environ['PAPER2BENCH_SEED']+':'+str(random.random()))",
                               workspace, seed=5, timeout_seconds=5)
    result = executor.execute(request)
    assert result.status == "succeeded", result.to_dict()
    assert (workspace / "seed.txt").read_text().startswith("5:")
    assert result.evidence["cleanup_confirmed"]
