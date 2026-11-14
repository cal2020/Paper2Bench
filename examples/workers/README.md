# Dedicated local GPU and remote SSH workers

These adapters extend the same bounded Docker executor. Candidate code never runs
in the SSH helper process. The remote machine and account are selected by the
operator, never by a task manifest.

## Remote worker

Install this version of Paper2Bench and Docker on a Linux worker. Stage the
operator-selected execution image explicitly. Create an SSH account with a
pinned server key and a dedicated identity; no automatic host-key acceptance,
agent forwarding, SSH configuration/proxy inheritance, or software installation
is performed by Paper2Bench.

The worker command is `python3 -m paper2bench.executors.worker`. A trusted local
policy may be supplied using `--policy /etc/paper2bench/worker.json`. The example
[worker-policy.json](worker-policy.json) tightens the built-in resource ceilings.
The helper accepts one bounded JSON request on stdin and streams typed JSON
events/results on stdout. Keep banners and shell startup output off stdout.

For stronger account scoping, configure the public key's server-side forced
command to this worker module with the operator policy, disable PTY and port,
agent, and X11 forwarding. The helper itself does not enforce SSH account policy;
the operator must configure it. The worker account must be allowed to access its
local Docker engine. Docker access is powerful host access and belongs only to
this trusted worker account.

The Python adapter is:

```python
from paper2bench.executors import SSHExecutor

executor = SSHExecutor(
    host="research-worker.example",
    user="paper2bench",
    known_hosts="/absolute/operator/known_hosts",
    identity_file="/absolute/operator/worker_identity",
    remote_policy="/etc/paper2bench/worker.json",
)
metadata = executor.preflight()
```

The controller persists `job_intent` before acknowledging it. The remote helper
cannot create a container until that acknowledgement arrives. Every job has a
recorded `p2b-<UUID>` name. Lost transport or uncertain cleanup returns an unknown
outcome, retains job identity, and requires exact-job reconciliation. It never
blindly reruns an experiment. Input/output snapshots allow regular files only,
reject credential paths and links, and remain bounded to a 64 MiB workspace.

The transport has been tested through a real local worker subprocess and a real
Docker container. An external SSH server, host-key setup, and remote NVIDIA
hardware require operator-specific conformance checks before being called tested.

## NVIDIA GPU worker

GPU selection is explicit and uses stable NVIDIA UUIDs. It requires a dedicated
Linux host with a reachable local Unix Docker socket, an NVIDIA container runtime,
and working `nvidia-smi`. A macOS Docker Desktop host without these capabilities
fails preflight; there is no CPU fallback.

```python
from paper2bench.executors import DockerExecutor

executor = DockerExecutor(
    image="operator-staged-cuda-image",
    gpu_request={
        "device_ids": ["GPU-11111111-2222-3333-4444-555555555555"],
        "minimum_memory_bytes": 8589934592,
        "dedicated": True,
        "require_idle": True,
        "hard_memory_limit": False,
    },
)
```

Replace the illustrative UUID and image with actual tested worker values.
The same GPU request must be present in an operator-owned remote worker policy
before an SSH client can select it. The worker policy admits one staged image;
the controller freezes its resolved image ID. Changing the policy image causes
an old pinned identity to fail rather than silently substituting a new image.

Docker GPU selection does **not** impose a hard VRAM quota. Requests demanding one
are rejected. Dedicated-device allocation is an operator assertion; per-user
advisory leases prevent concurrent Paper2Bench controllers for that user from
using the same GPU. They do not exclude unrelated processes or other users.
Preflight checks NVIDIA compute-process inventory and memory capacity, but this
is not proof of indefinite exclusive hardware ownership. CUDA/framework/driver
compatibility must be tested using the selected image and task.

## Replication seeds

`ExecutionRequest(seed=...)` accepts an integer from 0 through `2**32 - 1`. Each
fresh candidate Python process sets `random.seed(seed)` and exposes
`PAPER2BENCH_SEED` inside the worker. It does not seed NumPy, PyTorch, JAX, native
libraries, or nondeterministic accelerator kernels automatically. Method plugins
must configure their own RNGs; execution evidence states this seed scope.
