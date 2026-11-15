# Optional reproducibility core

The optional `paper2bench.repro` package turns reviewed source evidence into a frozen experiment, runs a supported workload, and evaluates its outputs outside the candidate environment. Its command is **`paper2bench-repro`**. The existing `paper2bench` command and main framework documented in [IMPLEMENTATION.md](../IMPLEMENTATION.md) remain available.

This core supports numeric tabular classification and bounded zero-one knapsack optimization. It includes source-linked model authoring, trusted reference implementations, isolated Python execution, local and hosted text-model classification, paired comparisons, classification multi-seed studies, durable model quotas, authenticated remote workers, and offline result replay. A numerical result always leaves implementation fidelity and protocol fidelity unverified.

## Install and run the offline examples

Use Python **3.11 or 3.12** for this optional core. The main framework retains its own Python requirements. From this repository's root:

```bash
python3.11 -m venv .venv-repro
.venv-repro/bin/python -m pip install -e '.[dev,repro,repro-api]'
.venv-repro/bin/paper2bench-repro --state .paper2bench-repro demo --key first-repro-demo
.venv-repro/bin/paper2bench-repro --state .paper2bench-repro demo --task knapsack --key first-repro-knapsack
```

The classification example fits a centroid classifier using training rows and evaluates 80 held-out examples; its synthetic fixture should produce accuracy 1.0. The knapsack example uses a trusted reference algorithm and independently checked objective values. These examples verify software behavior, not a published paper's claims. Neither needs Docker, model credentials, or a downloaded model.

`repro` installs the execution core; `repro-api` also includes its API dependencies. `dev` supplies pytest for verification. These commands resolve dependencies from the optional extra's declared ranges; they are not a locked dependency installation. Frozen experiment identities subsequently record the installed dependency versions.

The default state directory is `.paper2bench-repro`. Keep it on a local filesystem with reliable SQLite locking. The following commands use that default. A different `--state PATH` must appear before the subcommand and must be used consistently for ingestion, freezing, execution, and inspection.

## Run an isolated Python candidate

Stage a Python image explicitly, then run the synthetic Python examples:

```bash
docker --host=unix:///var/run/docker.sock pull python:3.12-slim
.venv-repro/bin/paper2bench-repro demo --python --image python:3.12-slim --key isolated-classification
.venv-repro/bin/paper2bench-repro demo --task knapsack --python --image python:3.12-slim --key isolated-knapsack
```

The CLI resolves the staged tag to an immutable local image ID before freezing. Candidate execution never pulls an image or falls back to host execution. This runtime targets Linux and the managed local Docker socket.

Candidates receive no host project mount, provider credentials, held-out labels, or network access. Containers have a read-only root, bounded memory/processes/output/time, and a nonroot candidate process. A small trusted supervisor retains only the capabilities needed to drop the candidate's privileges and terminate its PID namespace at the deadline. It arms that deadline before reading input, so the bound survives a lost host worker. These controls support single-owner experiments; they are not a claim of isolation suitable for hostile public tenants.

## Register a source and dataset

```bash
.venv-repro/bin/paper2bench-repro ingest examples/repro/paper.md
.venv-repro/bin/paper2bench-repro dataset examples/repro/dataset.json
```

Retain `source_sha256`, `ingestion_receipt_sha256`, and `dataset_sha256` from the responses. Evidence anchors use character offsets into normalized source text. Inspect extracted equations and tables: a matching quotation does not prove correct scientific interpretation.

Classification data uses the following shape:

```json
{
  "train": [{"id": "train-1", "x": [-1.0], "y": "negative"}],
  "test": [{"id": "test-1", "x": [-0.5], "y": "negative"}]
}
```

IDs must be unique across both sets, feature dimensions must agree, numbers must be finite, and test labels must occur in training. Authoring receives training data and feature metadata. Execution receives training labels and test features; the evaluator retains test labels.

Optimization data uses:

```json
{
  "task": "knapsack",
  "instances": [{"id": "case-1", "weights": [2, 3, 4], "values": [3, 4, 5], "capacity": 5}]
}
```

A Python optimization candidate receives `{instances, seed}` and returns `{ids, selections}`, where each selection contains item indices. The evaluator checks feasibility and independently computes the exact dynamic-programming optimum. Invalid, repeated, out-of-range, or overweight selections score zero. Schema limits bound oracle work. Optimization authoring receives problem-shape metadata, not evaluated instances or reference solutions.

### Text-bearing PDFs

PDF parsing requires Docker and an explicitly staged isolated parser image:

```bash
.venv-repro/bin/paper2bench-repro prepare-parser --base-image python:3.12-slim
.venv-repro/bin/paper2bench-repro ingest paper.pdf --parser-image PARSER_IMAGE_ID
```

Replace `PARSER_IMAGE_ID` with the returned immutable image ID. The parser build copies the installed parser dependency without package downloads or document bytes; parsing runs with container resource bounds. `PAPER2BENCH_PARSER_IMAGE` can supply the image instead of the flag. Scanned and encrypted PDFs require separately prepared text. OCR and equation reconstruction are not implemented.

## Draft, review, freeze, and execute

Start a compatible model server separately and register its profile:

```bash
.venv-repro/bin/paper2bench-repro profile examples/repro/local-provider.json
```

Local endpoints must use a numeric loopback host such as `127.0.0.1`. To use a hosted endpoint, adapt `examples/repro/cloud-provider.json` and populate its named credential through your environment or secret manager. Profiles store the credential environment-variable name, not its value. This adapter supports text chat; tools, vision, and embeddings require separate adapters.

Replace the uppercase placeholders with returned hashes and a model actually available to your endpoint:

```bash
.venv-repro/bin/paper2bench-repro draft \
  --source SOURCE_SHA256 \
  --source-manifest INGESTION_RECEIPT_SHA256 \
  --dataset DATASET_SHA256 \
  --profile PROFILE_SHA256 \
  --model YOUR_AVAILABLE_MODEL \
  --image python:3.12-slim \
  --out draft.json
```

The Python image must already be staged. Local-only is the default; hosted authoring needs `--allow-cloud`. Model calls have output bounds, a whole-request deadline, and no automatic retry. Uncertain acceptance or billing is reported as an unknown outcome.

Inspect `draft.json` and its referenced implementation artifact before freezing. Resolve material uncertainties explicitly and retain assumptions and deviations. Authoring does not execute generated code. Unresolved settings or incorrect source quotations block freezing.

```bash
.venv-repro/bin/paper2bench-repro freeze draft.json
.venv-repro/bin/paper2bench-repro run SPEC_HASH --key reviewed-experiment-1
.venv-repro/bin/paper2bench-repro inspect RUN_ID
.venv-repro/bin/paper2bench-repro report RUN_ID
```

Frozen identities capture source, data, code, image, settings, engine source snapshots, interpreter, dependency versions, and host architecture. Changing the installed engine requires a new freeze. Hosted-model revision information remains limited to what the provider reports.

Direct model classification uses a `model_classification` workload with a registered profile hash, model ID, and system prompt. Its policy needs enough provider calls for the test set and a per-call output-token limit. Each completion must contain exactly `{"label":"one training label"}`. Inspect the authoritative schema with:

```bash
.venv-repro/bin/paper2bench-repro schema --out repro-schema.json
```

## Interpret scores and compare systems

A single classification run checks whether its Wilson confidence interval lies inside the declared target-accuracy band and whether the minimum sample count is met. Knapsack uses the mean optimality ratio and a bounded-mean Hoeffding interval. Results may be `supported`, `not_supported`, or `inconclusive` **for that numerical criterion**; operational failures are `not_tested`. Passing does not establish paper reproduction, a causal improvement, or faithful implementation.

For a paired comparison, freeze both experiment specifications, then create `comparison.json`:

```json
{
  "candidate_spec_hash": "CANDIDATE_SPEC_HASH",
  "baseline_spec_hash": "BASELINE_SPEC_HASH",
  "minimum_effect": 0.02,
  "confidence": 0.95,
  "bootstrap_samples": 2000,
  "min_test_examples": 20,
  "seed": 0
}
```

```bash
.venv-repro/bin/paper2bench-repro comparison-freeze comparison.json
.venv-repro/bin/paper2bench-repro benchmark PLAN_HASH --key comparison-1
```

Both systems must use the same dataset and evaluator identity. Both role runs are registered before execution. The assessment reports candidate-minus-baseline accuracy, or mean optimality ratio, with a paired percentile-bootstrap interval. Resource and seed differences are disclosed. The comparison conditions on the fitted systems and assumes independent test cases; it does not account for seed variation, clustering, or selection across many comparisons.

For a classification multi-seed study, add a unique `seeds` array such as `[11,22,33]` to the candidate/baseline study plan, along with the effect/confidence/bootstrap/minimum-case fields:

```bash
.venv-repro/bin/paper2bench-repro study-freeze study.json
.venv-repro/bin/paper2bench-repro study STUDY_PLAN_HASH --key study-1
.venv-repro/bin/paper2bench-repro study-export STUDY_RESULT_HASH study.zip
.venv-repro/bin/paper2bench-repro study-replay study.zip
```

All seed-specific trials are registered before execution. Failed or incomplete trials block the primary verdict rather than disappearing from the analysis. The crossed bootstrap resamples paired seeds and shared cases. Python implementations are responsible for consuming the supplied seed; the platform does not attest that they do. Hosted-model seed studies, clustered inference, and optimization multi-seed studies remain outside this contract.

## Quotas, queues, and recovery

```bash
.venv-repro/bin/paper2bench-repro budget-create shared-model-work --calls 100 --output-tokens 200000
.venv-repro/bin/paper2bench-repro budget shared-model-work
.venv-repro/bin/paper2bench-repro submit SPEC_HASH --key queued-experiment-1
.venv-repro/bin/paper2bench-repro worker --once
.venv-repro/bin/paper2bench-repro runs
.venv-repro/bin/paper2bench-repro cancel RUN_ID
.venv-repro/bin/paper2bench-repro resume RUN_ID
```

Use `draft --budget shared-model-work`, or a model experiment's `policy.budget_id`, to share reservations across processes and runs. Reservations cover calls and maximum requested output tokens. Verified completion usage releases unused capacity; missing usage and uncertain outcomes retain conservative reservations. These are not monetary-billing guarantees and do not cover input-token charges or provider misreporting.

SQLite transactions, leases, heartbeats, fencing tokens, and immutable terminal results prevent stale publication. Run IDs, attempts, provider dispatch records, and diagnostics survive restart. Reusing an idempotency key returns the same run; using that key for a different specification fails. A new trial requires a new key and counts toward its study budget. Interrupted external-model batches become `outcome_unknown` rather than automatically repeating potentially billable requests. Cancellation diagnostics retain cleanup uncertainty without replacing an authoritative result.

## Replay and share results

```bash
.venv-repro/bin/paper2bench-repro export RUN_ID experiment.zip
.venv-repro/bin/paper2bench-repro replay experiment.zip
```

Exported bundles include relevant source, data, code, software identity, authoring/provider provenance, results, and diagnostics. **They include held-out labels and may include source or training data captured in prompts.** Review data rights and access requirements before sharing.

Replay verifies bounded archive structure, required referenced artifacts, hashes, evidence quotations, and recomputed scores. It executes no candidate code. Hashes establish internal consistency, not authenticated authorship: replay explicitly reports authentication as unverified. Replay is distinct from rerunning an experiment, which needs the retained state and appropriate runtime.

## API and remote workers

Install `repro-api`, populate `PAPER2BENCH_SERVER_TOKEN` through your environment or secret manager, and run:

```bash
.venv-repro/bin/paper2bench-repro serve --port 8000
```

The API binds to loopback and requires an administrator bearer token for data routes. `/health` contains no experiment data. Submissions queue work; a separate `worker` against the same state executes it. This API is single-owner and does not provide public multi-tenant isolation.

Remote workers need separately provisioned per-worker credentials. On the control server, `PAPER2BENCH_WORKERS` is a JSON map from worker IDs to credential **environment-variable names**, for example `{"cloud-worker-a":"CLOUD_WORKER_A_TOKEN"}`. Populate each named secret separately. Administrator and worker tokens must differ. On the worker host, populate its token environment variable, stage the assigned immutable Python image, and run:

```bash
.venv-repro/bin/paper2bench-repro remote-worker \
  --url https://YOUR_CONTROL_SERVICE \
  --worker-id cloud-worker-a \
  --token-env PAPER2BENCH_WORKER_TOKEN \
  --spool /PRIVATE/WORKER/SPOOL \
  --image IMMUTABLE_PYTHON_IMAGE_ID
```

A submitted Python experiment must explicitly set `policy.placement="remote"` and `policy.local_only=false`. Workers receive assigned code, public inputs, image identity, seed, and limits. They cannot list experiments, read arbitrary artifacts or hidden labels, or retrieve final scores through the worker API. The control server computes scores from returned outputs.

The client requires HTTPS except for numeric-loopback HTTP protocol tests. Production remote access needs an authenticated TLS reverse proxy or an explicitly configured ASGI deployment; the stock loopback server does not provision cloud infrastructure. Private fsynced worker spools preserve completions across lost acknowledgements. Leases prevent stale results; worker measurements and execution claims remain worker-reported, without cryptographic hardware attestation. Remote dispatch does not send model credentials.

## Python imports

The optional core has its own namespace:

```python
from paper2bench.repro.engine import Engine
from paper2bench.repro.bundles import export_bundle, replay_bundle

engine = Engine(".paper2bench-repro")
```

Use models and APIs from `paper2bench.repro.*` when working with these experiment artifacts. The main framework's existing imports remain documented separately.

## Verify this integration

```bash
.venv-repro/bin/python -m pytest -q tests/repro -m 'not docker'
```

For the actual container tests, stage the images first. Replace the parser placeholder with the immutable ID returned by `prepare-parser`:

```bash
docker --host=unix:///var/run/docker.sock pull python:3.12-slim
.venv-repro/bin/paper2bench-repro prepare-parser --base-image python:3.12-slim
PAPER2BENCH_DOCKER_IMAGE="$(docker --host=unix:///var/run/docker.sock image inspect --format '{{.Id}}' python:3.12-slim)" \
PAPER2BENCH_PARSER_IMAGE=PARSER_IMAGE_ID \
  .venv-repro/bin/python -m pytest -q tests/repro -m docker
```

The [optional-core CI workflow](../.github/workflows/repro.yml) installs the extras and runs the non-Docker suite on Python 3.11 and 3.12. A separate Ubuntu job stages the images and runs actual Docker tests, including isolation, resource bounds, isolated PDF parsing, HTTP remote-worker protocol, and generated-code-to-score-to-replay fixtures. These fixtures use local mock model protocols; CI does not prove inference through a real foundation model or a paid hosted account. The staged Python tag is resolved to an immutable image for each run; it is not pinned across separate CI runs.

The implemented task contracts do not cover arbitrary scientific domains, multimodal models, GPU training, agent tools, general dependency builds, or automatic scientific sign-off. Local-only constrains this software's outbound use but cannot attest that a separately configured local model server never forwards requests. PostgreSQL, cloud object stores, monetary billing reconciliation, and automatic cloud provisioning are outside this optional core.
