from pathlib import Path

import pytest

from paper2bench.executor_factory import build_executor, executor_selection
from paper2bench.executors import DockerExecutor, SSHExecutor
from paper2bench.runtime_config import runtime_scope, PolicyDenied


def config(execution=None, *, local=True):
    return {"models": {"local": {"provider": "ollama", "model": "test", "hosting": "local_server",
                                 "locality_assurance": "operator_trusted"}},
            "roles": {"author": "local"}, "policy": {"data_egress": "local_only" if local else "allow"},
            "execution": execution or {}}


def test_frozen_local_selection_keeps_pinned_image():
    with runtime_scope(config({"backend": "oci", "image": "operator-image"})):
        worker = build_executor(selection={"backend": "docker", "image": "sha256:" + "a" * 64})
    assert isinstance(worker, DockerExecutor) and worker.image == "sha256:" + "a" * 64


def test_remote_selection_is_explicit_and_local_only_denies_before_dispatch(tmp_path):
    known = tmp_path / "known_hosts"
    key = tmp_path / "worker_key"
    known.write_text("fixture host key")
    key.write_text("fixture identity; not a real key")
    execution = {"backend": "ssh", "ssh": {"host": "worker.example", "user": "research",
                                             "known_hosts": str(known), "identity_file": str(key)}}
    with runtime_scope(config(execution)), pytest.raises(PolicyDenied):
        build_executor()
    with runtime_scope(config(execution, local=False)):
        worker = build_executor()
        frozen = executor_selection(worker)
        copied = build_executor(selection=frozen)
    assert isinstance(copied, SSHExecutor)
    assert copied.host == "worker.example" and copied.known_hosts == str(known)


def test_candidate_cannot_select_unimplemented_backend_or_host_environment():
    with runtime_scope(config({"backend": "arbitrary", "env": {"SECRET": "bad"}})), pytest.raises(ValueError):
        build_executor()


def test_injected_backend_requires_explicit_caller_on_recovery():
    with pytest.raises(ValueError, match="caller-supplied"):
        build_executor(selection={"backend": "injected", "class": "trusted.test.Worker", "image": None})
