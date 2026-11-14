# Paper2Bench 0.4.0a2 implementation

This workspace extends upstream commit `55a3ca5eeb389c994962a375848bb5e9ca455ad9`.
The original MIT license and generation interfaces remain. The extension adds
bounded model runtimes, source evidence, durable authoring checkpoints, isolated
local/SSH workers, and reviewed scientific protocol execution from the
[implementation brief](paper2bench-implementation-brief.md). These changes are in
this checkout; installing an upstream release does not install this extension.

## Install and run without a model

Python 3.10 or newer is declared; verification used Python 3.11 on macOS and Linux
containers. Install from this checkout and stage the execution image explicitly:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
docker pull python:3.11-slim
.venv/bin/python -m paper2bench doctor --execution
.venv/bin/python -m paper2bench validate --task examples/tasks/linear-fit/task_manifest.json --executable
.venv/bin/python -m paper2bench execute \
  --task examples/tasks/linear-fit/task_manifest.json \
  --evaluation-bundle examples/tasks/linear-fit/curator_manifest.json \
  --evaluation-profile method_output/v1
```

Replace `linear-fit` with `knapsack` for the symbolic optimization example. Both
run without a foundation model or API key. Linear fitting emits predictions
scored against protected labels; knapsack uses a dynamic programming submission
checked by a separate exhaustive oracle. They are synthetic computational
demonstrations, not published-paper reproductions.

Docker must already be running. Execution never pulls an image automatically;
the controller resolves and records its immutable image ID. Candidate code runs
inside the selected worker container.

## Local and hosted model runtimes

`paper2bench.llm.chat` routes through bounded nonstreaming text adapters for
OpenAI, Anthropic, Ollama, and explicit OpenAI-compatible endpoints. Start with
[runtime.local.yaml](examples/runtime.local.yaml) or
[runtime.cloud.yaml](examples/runtime.cloud.yaml). Select installed/available
model IDs and endpoints before using a profile.

```bash
export P2B_LOCAL_MODEL=your-installed-ollama-model
.venv/bin/python -m paper2bench doctor --runtime-config examples/runtime.local.yaml
.venv/bin/python -m paper2bench run --pdf paper.pdf --task-id study_a \
  --research-question 'Does the stated method improve the primary metric?' \
  --runtime-config examples/runtime.local.yaml --output-dir output
```

Author, reference, candidate-agent, and judge roles have explicit bindings and
budget pools. A profile takes precedence over historical model defaults; select
all role models in the profile instead of legacy model flags. Cloud credentials
are environment-variable references, never values embedded in YAML or candidate
bundles. Export them explicitly or load them in trusted caller code; there is no
automatic `.env` loading. Calls without a runtime profile retain the legacy
unbudgeted model path.

The local profile is **operator-trusted local routing**. It denies cloud
providers, known cloud aliases, nonlocal endpoints, external dataset lookups, and
cloud fallback. A loopback model server can itself forward requests, so this is
not an independently enforced offline boundary. `verified_offline_required: true`
fails closed without a trusted egress-attestation callback supplied through the
Python API. `--offline` requires a local-only profile and does not create that
attestation. The network-disabled candidate worker is a separate boundary from
the controller's model server.

Only nonstreaming text inference is implemented. Native tool calling, vision,
provider-enforced structured output, streaming, training, and automatic model
fallback are unsupported. Conservative context preflight rejects oversized
prompts instead of silently truncating them.

## Explicit live runtime probes

Ordinary `doctor` validates configuration and makes zero model calls. `--probe`
explicitly sends a bounded public arithmetic fixture through the selected role:

```bash
.venv/bin/python -m paper2bench doctor \
  --runtime-config examples/runtime.certify-local.yaml --probe --role author \
  --max-probe-tokens 32 --probe-output runs/certification/local-probe.json
```

The supplied profile targets an already installed `llama3.2:latest` on local
Ollama; it does not download a model. Hosted probes additionally require explicit
`--allow-cloud` and provider-specific credential references. They consume the
normal durable role budget and have no automatic retries or fallback.

Reports distinguish `passed`, `failed`, and `blocked`, record dispatch intent,
usage and timing, and check exact JSON-text arithmetic. Ollama inventory/model
digests are checked before and after the call. This is server-reported identity,
not independent weight measurement or egress attestation. Passing the fixture
establishes that bounded text interaction worked; it does not certify general
model quality, native JSON schemas, tools, or vision.

## Source evidence and bounded context

The evidence tools index source/page/span hashes and exact text offsets, select
context under an explicit UTF-8 byte budget, and extract citation-grounded
requirements in checkpointed chunks:

```bash
.venv/bin/python -m paper2bench index --pdf paper.pdf -o paper.index.json
.venv/bin/python -m paper2bench select-context --index paper.index.json \
  --query 'method, primary metric, data split and compute budget' \
  --budget-bytes 24000 -o paper.context.json
.venv/bin/python -m paper2bench evidence-extract --index paper.index.json \
  --checkpoint evidence.checkpoint.json --runtime-config examples/runtime.local.yaml \
  -o paper.requirements.json
```

Selections expose omitted/partial spans, coverage, missing sections, and review
needs. Extracted quotes must match their cited source spans exactly. Requirements
distinguish stated, inferred, assumed, unresolved, and conflicting information;
model prose cannot create source evidence. Checkpoints bind source, index,
prompt/schema, and model identity. An interrupted unknown dispatch is not
automatically replayed. Section/equation/table/figure references are heuristic;
PDF layout, visual content, and extraction loss still require review. These
explicit tools are composable with the generation workflow; indexing alone does
not establish a complete experimental method.

Downloads use bounded public-only HTTP(S), validate redirects and DNS answers,
connect to the validated address with TLS hostname verification, and ignore
environment proxies. Local-only policy covers downloads and dataset validators;
offline resource checks report `not_run`.

PDF parsing uses a stripped-environment child with input/page/text, CPU, stdout,
and wall-time limits. It is not a VM boundary for native parser vulnerabilities.
Hard parser RAM enforcement is unsupported on macOS and reported by
`parser_limits_profile()`; requiring it fails before parsing. Use a dedicated
Linux/VM parser when that assurance is necessary.

## Isolated local, remote, and GPU execution

Workers run candidate code as an unprivileged identity in a network-disabled
Linux container with a read-only root, dropped capabilities, PID/memory/CPU
limits, bounded tmpfs/output, and a deadline. Validated regular-file snapshots
move through bounded streams. No host directory, Docker socket, controller
credential, curator label, or hidden scorer is mounted into the candidate.
Timeout/cancellation reports whether cleanup was confirmed. This is a trusted
single-user Docker/VM profile, not a hostile multitenant isolation claim.

The operator can select a local Docker backend or `SSHExecutor` through runtime
configuration/Python. SSH uses an explicit identity and pinned known-hosts file,
disables forwarding/config/proxy inheritance, and invokes the fixed installed
worker module. The controller persists job intent before acknowledging remote
container creation. Lost transport preserves exact job identity for
reconciliation instead of silently repeating the experiment. Local-only policy
denies remote SSH dispatch. See [worker setup](examples/workers/README.md).

The GPU profile selects explicit stable NVIDIA UUIDs on a dedicated Linux worker,
checks the NVIDIA runtime, capacity and optional idleness, and uses per-user
advisory leases. It has no CPU fallback and no hard VRAM quota. Dedicated-device
ownership remains an operator assertion, and framework/image/driver compatibility
needs a real worker check. External SSH servers and NVIDIA hardware have not been
exercised in this workspace; the transport has been tested with a real local
worker subprocess and Docker container.

`--seed` / `ExecutionRequest(seed=...)` seeds Python `random` and exposes
`PAPER2BENCH_SEED` inside each fresh worker. It does not automatically seed
NumPy/PyTorch/JAX or remove accelerator nondeterminism.

## Candidate access to target models

An opt-in file broker lets the verification agent's isolated code request text
experiments against operator-selected model bindings. Start with
[runtime.broker-local.yaml](examples/runtime.broker-local.yaml):

```bash
export P2B_LOCAL_MODEL=your-installed-ollama-model
.venv/bin/python -m paper2bench doctor --runtime-config examples/runtime.broker-local.yaml
.venv/bin/python -m paper2bench verify --instruction-gt instruction_gt.txt --pdf paper.pdf \
  --runtime-config examples/runtime.broker-local.yaml
```

The candidate writes `model_requests.json` using fixed public target IDs. After a
successful code block, the controller validates the batch, calls only those
configured bindings, and places `model_responses.json` into the next snapshot.
The worker keeps `network: none` and receives no provider keys or HTTP endpoint.
This integration applies to `verify` / `run --verify`; the public `ModelBroker`
SDK also supports trusted controller integrations.

Operator ceilings bound requests, each batch, token reservations, and total
output. Calls share the durable experiments budget and remaining agent deadline.
Private receipts freeze request content and runtime identity before dispatch.
Known results are replayed from receipts without another call; changed content
under an existing request ID is rejected, and unknown outcomes are not
automatically resubmitted. Only text generation is admitted; tasks cannot supply
arbitrary providers, URLs, credentials, or tools. The local-server trust caveat
above still applies.

## Evaluation and reviewed scientific protocols

Public task manifests pin files by SHA-256. Candidate exports exclude curator
files and unknown fields; YAML uses an allowlist. Hidden labels and installed
scorers stay in the controller. Scoring consumes bounded data-only JSON,
recomputes metrics, and ignores printed gold scores. Hashes establish identity
and integrity, not scientific truth.

`execution/v1` checks isolated job completion; `method_output/v1` supports the
installed linear and knapsack demonstration scorers. Legacy `verify` uses strict
version-2 claim-F1 or rubric scoring and reports conclusion similarity separately.
Malformed/truncated responses, string booleans, empty claims, and invalid scores
are rejected. `execution_verified` means a successful code block was observed;
legacy verification does not establish full method fidelity or scientific
reproduction. New model-generated tasks remain `needs_review`, and
`instruction_gt.txt` is a reference draft.

For an explicitly reviewed contract, `reproduction/v1` and `transfer/v1` freeze
the exact source, task, code, data, seed schedule, stopping rule, metric criteria,
dependencies, and curator method-review evidence before dispatch. The controller
executes each prescribed seed in a fresh worker and builds protected measurement
evidence through installed scorers. Missing/failed seeds remain visible; it does
not rerun scientific failures to seek a favorable result. Fidelity, execution,
empirical agreement, and transfer remain distinct: faithful contradictory
results can pass fidelity/execution and fail agreement.

Run the synthetic contract fixture to exercise this orchestration:

```bash
.venv/bin/python examples/protocols/build_contract_fixture.py output/science-fixture
.venv/bin/python -m paper2bench validate-protocol \
  --task output/science-fixture/task_manifest.json \
  --protocol output/science-fixture/curator_manifest.json
.venv/bin/python -m paper2bench execute --task output/science-fixture/task_manifest.json \
  --evaluation-bundle output/science-fixture/curator_manifest.json \
  --evaluation-profile reproduction/v1
```

Use a new destination for the builder. Its source text and author review are
explicit synthetic test evidence, not an independent review or a published-paper
reproduction. Real research needs a curator-supplied source, implementation
review, protocol, and installed trusted scorer. No general training/simulation
plugin or arbitrary-paper scientific guarantee is implemented.

Metric rules include absolute/relative agreement and prespecified interval
equivalence. Hoeffding intervals require declared bounded outcomes and an
independence assumption, with Bonferroni correction for multiple metrics;
inconclusive intervals do not establish equivalence. `calibrate` compares recorded
judge decisions with provenance-linked expert labels while preserving
abstentions and sample size. Meeting supplied sample-level targets does not
certify general judge performance. See [protocol details](examples/protocols/README.md).

## Checkpoints, state, and recovery

Runs use a transactional SQLite journal and private artifact directories.
Submission intent, model budget reservations, and job names commit before
external dispatch. Failed attempts and unknown usage remain visible across
restarts. Reports commit before terminal success, allowing recovery without
re-executing completed work.

```bash
.venv/bin/python -m paper2bench status --run RUN_ID
.venv/bin/python -m paper2bench resume --run RUN_ID
.venv/bin/python -m paper2bench reproduce --run RUN_ID
.venv/bin/python -m paper2bench cancel --run RUN_ID
.venv/bin/python -m paper2bench reconcile --run RUN_ID
.venv/bin/python -m paper2bench reconcile --run RUN_ID --terminate
```

Normal non-`--split-rqs` `run` with a runtime profile checkpoints parse,
classification, extraction, rendering, planning, and optional verification.
It freezes a private source snapshot and binds software/prompts/template/runtime
identity. `resume` reuses completed historical artifacts and retries only known
failed stages within the original remaining budgets and attempt limits. Unknown
outcomes block automatic replay; changed identities are rejected. Task-output
locking prevents concurrent authors from mixing artifacts. This is historical
artifact reuse, not a fresh model replication.

`--split-rqs` retains the legacy orchestration and has no multistage checkpoint
resume. Standalone stage commands have durable model budgets when a profile is
used, but no automatic generation-stage orchestration. Model journals default to
`./runs` and can be selected with `--model-runs-dir` / SDK `model_runs_dir`.

For computational runs, `resume` returns a completed report or retries a known
failed execution within remaining limits. Scientific parents recover their fixed
child schedule, preserving completed observations. `reproduce` creates a linked
fresh run using pinned identities; the command name alone establishes no
scientific conclusion. Cancellation requests are distinct from confirmed
termination. Reconciliation inspects only recorded exact jobs; `--terminate`
explicitly removes them. Unknown work retains reservations. Resource accounting
uses lifecycle wall time and a conservative CPU-time upper bound, not measured
CPU consumption or a provider invoice.

## Verification evidence and remaining limits

The final local gate passed **488 tests** with real Docker tests enabled; the
public verification summary is [docs/verification.json](docs/verification.json).
The full local JUnit report remains at `runs/swarm-test-results.xml`.
Editable installation reported `0.4.0a2`, and compilation passed. The
local `runs/swarm-validation.json` receipt records the checks and artifact
hashes. Full local receipts are intentionally excluded from GitHub and deployment.

```bash
.venv/bin/python -m pytest -q
P2B_DOCKER_TESTS=1 .venv/bin/python -m pytest -q
```

Actual CLI runs completed linear fitting (MSE 0), knapsack (objective 12 checked
by the independent oracle), fixed three-seed synthetic scientific orchestration,
completed-run resume, and linked fresh execution. The local Ollama Llama 3.2 text
probe passed with a stable server-reported digest; its bounded receipt is
summarized in the public verification summary (full local receipt:
`runs/certification/swarm-cli-local.json`). A real
Docker → file broker → local Llama → fresh Docker round trip passed with worker
networking disabled, one durable model call, and zero calls on receipt replay;
the public verification summary records that scope. These checks do not demonstrate that Llama can complete a
paper-authoring or reproduction task.

Tests also cover HTTP protocol fixtures, strict scoring, source/import bounds,
credential/curator denial, SSRF/redirects, cancellation, persistent reservations,
restart behavior, worker deadlines, evidence citations, protocol contradictions,
and output tampering. Fixture tests make no billable provider calls. Hosted model
accounts, external SSH, and physical NVIDIA execution remain untested here.
Remote CI configuration exists; this local result does not establish a remote
CI run. Cluster scheduling, full training/simulation integrations, real expert
calibration studies, paper-family holdouts, authenticated team hosting, and
general scientific reproduction remain further work.
