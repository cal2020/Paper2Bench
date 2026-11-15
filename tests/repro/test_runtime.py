from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time

import pytest

from paper2bench.repro.models import (
    ClaimSpec,
    Dataset,
    EvidenceAnchor,
    Example,
    ExperimentSpec,
    Limits,
    WorkloadSpec,
)
from paper2bench.repro.runtime import (
    BuiltinRunner,
    DockerRunner,
    ExecutionCancelled,
    ExecutionFailure,
    ResourceExhausted,
    RuntimeSpec,
    RuntimeUnavailable,
    _decode_predictions,
    sanitized_input,
)

IMAGE = "sha256:" + "a" * 64
CODE = b"import json,sys\np=json.load(sys.stdin)\nprint(json.dumps({'ids':[r['id'] for r in p['test']],'labels':['yes' if r['x'][0]>0 else 'no' for r in p['test']]}))\n"


def dataset():
    return Dataset(
        train=(Example(id="tr0", x=(-2.0,), y="no"), Example(id="tr1", x=(2.0,), y="yes")),
        test=(Example(id="te0", x=(-1.0,), y="no"), Example(id="te1", x=(1.0,), y="yes")),
    )


def spec(kind="python", code=CODE, **overrides):
    workload = WorkloadSpec(
        kind=kind,
        **(
            {"implementation_sha256": hashlib.sha256(code).hexdigest(), "image": IMAGE}
            if kind == "python"
            else {}
        ),
    )
    return ExperimentSpec(
        title="runtime fixture",
        source_sha256="0" * 64,
        dataset_sha256="1" * 64,
        evidence=(EvidenceAnchor(start=0, end=4, quote="test"),),
        claim=ClaimSpec(target=1.0),
        workload=workload,
        **overrides,
    )


def fake_docker(monkeypatch, runner, *, state=None, output=None, attach_error=None):
    calls = []
    captured = {}

    def control(args, **kwargs):
        calls.append(args)
        if args[0] == "create":
            return ("b" * 64 + "\n").encode()
        if args[0] == "inspect":
            return json.dumps(
                state or {"OOMKilled": False, "ExitCode": 0, "Running": False}
            ).encode()
        return b""

    def attached(name, payload, **kwargs):
        captured.update(json.loads(payload))
        if attach_error:
            raise attach_error
        return output or b'{"ids":["te0","te1"],"labels":["no","yes"]}', b"", 0

    monkeypatch.setattr(runner, "_control", control)
    monkeypatch.setattr(runner, "_attached", attached)
    return calls, captured


def test_builtin_fit_uses_training_and_ties_are_deterministic():
    data = Dataset(
        train=(Example(id="a", x=(-1.0,), y="z"), Example(id="b", x=(1.0,), y="a")),
        test=(Example(id="c", x=(0.0,), y="z"),),
    )
    result = BuiltinRunner().run(data, spec("builtin_centroid"), cancelled=lambda: False)
    assert result.predictions.labels == ("a",)
    assert result.measurements["memory_limit_enforced"] is False


def test_builtin_cancel_and_wrong_runner():
    with pytest.raises(ExecutionCancelled):
        BuiltinRunner().run(dataset(), spec("builtin_centroid"), cancelled=lambda: True)
    with pytest.raises(ExecutionFailure):
        BuiltinRunner().run(dataset(), spec(), cancelled=lambda: False)


def test_docker_security_and_hidden_label_boundary(monkeypatch):
    runner = DockerRunner()
    calls, payload = fake_docker(monkeypatch, runner)
    result = runner.run(CODE, dataset(), spec(), cancelled=lambda: False)
    args = calls[0]
    for flag, value in (
        ("--network", "none"),
        ("--user", "0:0"),
        ("--cap-drop", "ALL"),
        ("--security-opt", "no-new-privileges=true"),
        ("--memory", "256m"),
        ("--memory-swap", "256m"),
        ("--pids-limit", "64"),
        ("--cpus", "1"),
        ("--pull", "never"),
    ):
        assert args[args.index(flag) + 1] == value
    assert "--read-only" in args
    assert [args[index + 1] for index, flag in enumerate(args) if flag == "--cap-add"] == [
        "SETUID",
        "SETGID",
        "KILL",
    ]
    assert "--mount" not in args and "--volume" not in args
    assert "HTTP_PROXY=" in args and "https_proxy=" in args
    assert payload["input"]["test"] == [{"id": "te0", "x": [-1.0]}, {"id": "te1", "x": [1.0]}]
    assert all("y" not in row for row in payload["input"]["test"])
    assert payload["input"]["train"][0]["y"] == "no"
    assert result.predictions.labels == ("no", "yes")
    assert result.measurements["cleanup_confirmed"] is True
    assert calls[-1][0:2] == ["rm", "--force"]


def test_minimal_remote_spec_executes_without_fabricated_labels(monkeypatch):
    frozen = spec()
    minimal = RuntimeSpec(
        workload=frozen.workload, limits=frozen.limits, seed=frozen.seed, task="classification"
    )
    runner = DockerRunner()
    _calls, captured = fake_docker(monkeypatch, runner)
    public = sanitized_input(dataset(), frozen)
    result = runner.run_input(CODE, public, minimal, cancelled=lambda: False)
    assert result.predictions.labels == ("no", "yes")
    assert all("y" not in item for item in captured["input"]["test"])
    assert set(minimal.model_dump()) == {"workload", "limits", "seed", "task"}


@pytest.mark.parametrize(
    "alteration",
    [
        lambda p: p["test"][0].update(y="hidden"),
        lambda p: p["test"][0].update(x=[float("nan")]),
        lambda p: p["test"][0].update(x=[True]),
        lambda p: p["test"][0].update(x=[1.0, 2.0]),
        lambda p: p["test"][0].update(id=p["train"][0]["id"]),
        lambda p: p.update(extra="unexpected"),
        lambda p: p.update(seed=True),
        lambda p: p.update(seed=1),
    ],
)
def test_remote_input_rejects_leakage_and_invalid_values_before_docker(monkeypatch, alteration):
    runner = DockerRunner()
    calls, _captured = fake_docker(monkeypatch, runner)
    public = sanitized_input(dataset(), spec())
    alteration(public)
    with pytest.raises(ExecutionFailure, match="input"):
        runner.run_input(CODE, public, spec(), cancelled=lambda: False)
    assert calls == []


def test_mutable_image_and_source_mismatch_never_create_container(monkeypatch):
    runner = DockerRunner()
    calls, _ = fake_docker(monkeypatch, runner)
    invalid = spec().model_copy(
        update={"workload": spec().workload.model_copy(update={"image": "python:latest"})}
    )
    with pytest.raises(RuntimeUnavailable):
        runner.run(CODE, dataset(), invalid, cancelled=lambda: False)
    with pytest.raises(ExecutionFailure, match="identity"):
        runner.run(CODE + b"# changed", dataset(), spec(), cancelled=lambda: False)
    assert calls == []


@pytest.mark.parametrize("failure", [ResourceExhausted("timeout"), ExecutionCancelled("cancel")])
def test_failed_attachment_always_cleans_container(monkeypatch, failure):
    runner = DockerRunner()
    calls, _ = fake_docker(monkeypatch, runner, attach_error=failure)
    with pytest.raises(type(failure)):
        runner.run(CODE, dataset(), spec(), cancelled=lambda: False)
    assert calls[-1][0:2] == ["rm", "--force"]


def test_oom_is_resource_outcome(monkeypatch):
    runner = DockerRunner()
    calls, _ = fake_docker(monkeypatch, runner, state={"OOMKilled": True, "ExitCode": 137})
    with pytest.raises(ResourceExhausted, match="memory"):
        runner.run(CODE, dataset(), spec(), cancelled=lambda: False)
    assert calls[-1][0] == "rm"


def test_cleanup_failure_is_visible(monkeypatch):
    runner = DockerRunner()
    fake_docker(monkeypatch, runner)
    control = runner._control

    def fail_cleanup(args, **kwargs):
        if args[0] == "rm":
            raise RuntimeUnavailable("disconnected")
        return control(args, **kwargs)

    monkeypatch.setattr(runner, "_control", fail_cleanup)
    result = runner.run(CODE, dataset(), spec(), cancelled=lambda: False)
    assert result.measurements["cleanup_confirmed"] is False


def test_setup_timeout_uses_declared_resource_outcome_and_cleanup(monkeypatch):
    runner = DockerRunner()
    calls = []

    def control(args, **kwargs):
        calls.append(args)
        if args[0] == "create":
            time.sleep(0.02)
            raise RuntimeUnavailable("control request timed out")
        return b""

    monkeypatch.setattr(runner, "_control", control)
    with pytest.raises(ResourceExhausted, match="wall-time"):
        runner.run(
            CODE, dataset(), spec(limits=Limits(timeout_seconds=0.01)), cancelled=lambda: False
        )
    assert calls[-1][0:2] == ["rm", "--force"]


def test_cleanup_retries_when_daemon_finishes_create_after_client_timeout(monkeypatch):
    runner = DockerRunner()
    calls = []

    def control(args, **kwargs):
        calls.append(args)
        if args[0] == "create":
            time.sleep(0.02)
            raise RuntimeUnavailable("create timed out") from subprocess.TimeoutExpired(
                "docker", 0.01
            )
        if len([call for call in calls if call[0] == "rm"]) == 1:
            raise RuntimeUnavailable("container not created yet")
        return b""

    monkeypatch.setattr(runner, "_control", control)
    with pytest.raises(ResourceExhausted) as raised:
        runner.run(
            CODE, dataset(), spec(limits=Limits(timeout_seconds=0.01)), cancelled=lambda: False
        )
    assert len([call for call in calls if call[0] == "rm"]) == 2
    assert raised.value.cleanup_confirmed is True
    assert raised.value.container_name.startswith("paper2bench-")


@pytest.mark.parametrize(
    "raw",
    [
        b'{"ids":["x"],"labels":["a"],"ids":["y"]}',
        b'{"ids":["x"],"labels":["a"],"score":1}',
        b"not json",
        b"\xff",
        b'{"ids":["x"],"labels":[NaN]}',
        b'{"ids":["x"],"labels":["\\ud800"]}',
    ],
)
def test_output_is_strict(raw):
    with pytest.raises(ExecutionFailure):
        _decode_predictions(raw)


def test_docker_environment_cannot_redirect_endpoint(monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", "tcp://attacker:1234")
    monkeypatch.setenv("DOCKER_CONTEXT", "attacker")
    monkeypatch.setenv("DOCKER_CONFIG", "/keep/registry/config")
    environment = DockerRunner._environment()
    assert "DOCKER_HOST" not in environment and "DOCKER_CONTEXT" not in environment
    assert environment["DOCKER_CONFIG"] == "/keep/registry/config"
    assert "unix:///var/run/docker.sock" in DockerRunner().command[1]


def test_attached_stream_is_bounded_and_kills_client(monkeypatch):
    original = subprocess.Popen
    created = []

    def start_fixture(*_args, **kwargs):
        # A fixed trusted test program exercises pipe handling, not candidate execution.
        process = original(
            [
                sys.executable,
                "-c",
                "import sys,time; sys.stdout.write('x'*100000); sys.stdout.flush(); time.sleep(5)",
            ],
            **kwargs,
        )
        created.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", start_fixture)
    with pytest.raises(ResourceExhausted, match="output"):
        DockerRunner()._attached(
            "fixture",
            b"{}",
            deadline=time.monotonic() + 3,
            output_limit=1024,
            cancelled=lambda: False,
        )
    assert created[0].poll() is not None


def test_attached_timeout_kills_client(monkeypatch):
    original = subprocess.Popen
    created = []

    def start_fixture(*_args, **kwargs):
        process = original([sys.executable, "-c", "import time; time.sleep(5)"], **kwargs)
        created.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", start_fixture)
    with pytest.raises(ResourceExhausted, match="wall-time"):
        DockerRunner()._attached(
            "fixture",
            b"{}",
            deadline=time.monotonic() + 0.1,
            output_limit=1024,
            cancelled=lambda: False,
        )
    assert created[0].poll() is not None


@pytest.mark.docker
def test_real_docker_hidden_data_environment_and_readonly(monkeypatch):
    image = os.environ.get("PAPER2BENCH_DOCKER_IMAGE")
    if not image:
        pytest.skip("set PAPER2BENCH_DOCKER_IMAGE to a staged immutable Python image")
    monkeypatch.setenv("P2B_HOST_SECRET", "must-not-be-inherited")
    code = b"""import json,os,signal,socket,sys
assert os.getuid()!=0
with open('/proc/self/status') as status:
    assert int(next(line.split()[1] for line in status if line.startswith('CapEff:')),16)==0
try:
    os.kill(os.getppid(),signal.SIGKILL)
except PermissionError:
    pass
else:
    raise AssertionError('candidate could kill supervisor')
assert os.environ.get('P2B_HOST_SECRET') is None
assert not os.environ.get('HTTPS_PROXY')
p=json.load(sys.stdin)
assert all('y' not in r for r in p['test'])
try:
    open('/paper2bench-write-probe','w').write('bad')
except OSError:
    pass
else:
    raise AssertionError('writable root filesystem')
s=socket.socket(); s.settimeout(.2)
try:
    s.connect(('1.1.1.1',80))
except OSError:
    pass
else:
    raise AssertionError('unexpected network access')
finally:
    s.close()
print(json.dumps({'ids':[r['id'] for r in p['test']],'labels':['yes' if r['x'][0]>0 else 'no' for r in p['test']]}))
"""
    frozen = spec(code=code).model_copy(
        update={"workload": spec(code=code).workload.model_copy(update={"image": image})}
    )
    result = DockerRunner().run(code, dataset(), frozen, cancelled=lambda: False)
    assert result.predictions.labels == ("no", "yes")
    assert result.measurements["cleanup_confirmed"] is True


@pytest.mark.docker
@pytest.mark.parametrize(
    "code, limits",
    [
        (b"while True: pass", Limits(timeout_seconds=5)),
        (b"import sys; sys.stdout.write('x'*1000000)", Limits(output_bytes=1024)),
    ],
)
def test_real_docker_resource_limits(code, limits):
    image = os.environ.get("PAPER2BENCH_DOCKER_IMAGE")
    if not image:
        pytest.skip("set PAPER2BENCH_DOCKER_IMAGE to a staged immutable Python image")
    frozen = spec(code=code, limits=limits)
    frozen = frozen.model_copy(
        update={"workload": frozen.workload.model_copy(update={"image": image})}
    )
    with pytest.raises(ResourceExhausted):
        DockerRunner().run(code, dataset(), frozen, cancelled=lambda: False)


@pytest.mark.docker
def test_real_docker_setup_deadline_cleans_late_create():
    image = os.environ.get("PAPER2BENCH_DOCKER_IMAGE")
    if not image:
        pytest.skip("set PAPER2BENCH_DOCKER_IMAGE to a staged immutable Python image")
    frozen = spec(limits=Limits(timeout_seconds=0.01))
    frozen = frozen.model_copy(
        update={"workload": frozen.workload.model_copy(update={"image": image})}
    )
    with pytest.raises(ResourceExhausted) as raised:
        DockerRunner().run(CODE, dataset(), frozen, cancelled=lambda: False)
    assert raised.value.cleanup_confirmed is True


@pytest.mark.docker
def test_container_deadline_survives_host_worker_sigkill(tmp_path):
    image = os.environ.get("PAPER2BENCH_DOCKER_IMAGE")
    if not image:
        pytest.skip("set PAPER2BENCH_DOCKER_IMAGE to a staged immutable Python image")
    candidate = """import json,os,signal,time
assert os.getuid()==65534
try:
    os.kill(os.getppid(),signal.SIGKILL)
except PermissionError:
    pass
else:
    raise AssertionError('candidate killed supervisor')
escaped=os.fork()
if escaped==0:
    os.setsid()
    while True:
        time.sleep(.1)
with open('/tmp/crash-probe','w') as stream:
    json.dump({'signal_denied':True,'uid':os.getuid(),'escaped_pid':escaped},stream)
while True:
    time.sleep(.1)
"""
    frozen = spec(code=candidate.encode(), limits=Limits(timeout_seconds=6))
    minimal = RuntimeSpec(
        workload=frozen.workload.model_copy(update={"image": image}),
        limits=frozen.limits,
        seed=0,
        task="classification",
    )
    request_file = tmp_path / "request.json"
    request_file.write_text(
        json.dumps(
            {
                "code": candidate,
                "input": sanitized_input(dataset(), frozen),
                "spec": minimal.model_dump(mode="json"),
            }
        )
    )
    name_file = tmp_path / "container-name"
    worker_script = """import json,sys
from pathlib import Path
from paper2bench.repro.runtime import DockerRunner,RuntimeSpec
request=json.loads(Path(sys.argv[1]).read_text())
runner=DockerRunner()
original=runner._create_args
def create(name,image,spec):
    Path(sys.argv[2]).write_text(name)
    return original(name,image,spec)
runner._create_args=create
runner.run_input(request['code'].encode(),request['input'],RuntimeSpec.model_validate(request['spec']),cancelled=lambda:False)
"""
    worker = subprocess.Popen(
        [sys.executable, "-c", worker_script, str(request_file), str(name_file)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    runner = DockerRunner()
    name = None
    try:
        setup_deadline = time.monotonic() + 10
        probe = None
        while time.monotonic() < setup_deadline:
            if name_file.exists():
                name = name_file.read_text()
                try:
                    probe = json.loads(
                        runner._control(
                            ["exec", "--user", "65534:65534", name, "cat", "/tmp/crash-probe"]
                        )
                    )
                    break
                except RuntimeUnavailable:
                    pass
            assert worker.poll() is None, "worker exited before candidate crash probe"
            time.sleep(0.05)
        assert probe and probe["signal_denied"] and probe["uid"] == 65534
        worker.kill()
        worker.wait(timeout=3)
        deadline = time.monotonic() + minimal.limits.timeout_seconds + 3
        state = None
        while time.monotonic() < deadline:
            state = json.loads(runner._control(["inspect", "--format", "{{json .State}}", name]))
            if not state["Running"]:
                break
            time.sleep(0.1)
        assert state and state["Running"] is False
        assert state["ExitCode"] == 124
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=3)
        if name:
            runner._control(["rm", "--force", name], check=False)
        for pipe in (worker.stdout, worker.stderr):
            if pipe:
                pipe.close()
