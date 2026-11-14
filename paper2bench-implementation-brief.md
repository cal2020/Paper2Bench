# Paper2Bench implementation brief

Draft extension specification • 12 November 2026

Upstream: [AbhayAnandUCSD/Paper2Bench](https://github.com/AbhayAnandUCSD/Paper2Bench), inspected at commit [`55a3ca5eeb389c994962a375848bb5e9ca455ad9`](https://github.com/AbhayAnandUCSD/Paper2Bench/tree/55a3ca5eeb389c994962a375848bb5e9ca455ad9), package version 0.3.1. Audience: maintainers and engineers implementing the extension. Requirements below describe proposed behavior unless explicitly identified as the upstream baseline.

## 1 Executive decision

Extend Paper2Bench in place into a portable, evidence-driven paper-to-benchmark toolkit. Preserve its paper-to-task and paper-to-variant workflows, CLI, Python SDK, templates, and readable artifacts. Add explicit provider configuration, local and cloud inference, isolated experiment execution, non-LLM method plugins, and versioned scientific evaluation.

The central product promise is: a researcher can supply a paper, state the question and resource limits, choose where authoring and candidate models run, inspect the generated task, execute permitted experiments, and receive an export containing the implementation, measurements, provenance, and a qualified verdict. Generation alone remains useful; execution is optional and never implied by a successful task-generation command.

Make three architectural decisions first:

1. Separate **benchmark authoring**, **candidate execution**, and **evaluation**. Each gets its own model bindings, permissions, budgets, and artifacts.
2. Separate **model family**, **hosting location**, and **computational method**. A foundation model may run locally or in a cloud. A task may evaluate a classical algorithm, solver, simulation, or hybrid system without using a language model as the subject.
3. Separate **conclusion similarity** from **experimental evidence**. Preserve existing scores as legacy profiles; add explicit execution, methodology, empirical-agreement, and transfer verdicts.

Ship a dependable local CLI and SDK first. A hosted dashboard, team service, cluster scheduler, and distributed training are subsequent adapters, each gated by demonstrated need. The initial implementation should not require a hosted orchestrator, Kubernetes, a vector database, or an agent framework.

## 2 Verified upstream baseline

The current package has two complementary paths: paper to a benchmark task and paper specification to research variants. It recognizes `llm_evaluation`, `novel_architecture`, and `empirical_study`. Existing deliverables include `task_config.yaml`, `instruction.txt`, `instruction_gt.txt`, tree JSON, and variant metadata. Preserve these names and meanings through a migration layer. [Pinned README](https://github.com/AbhayAnandUCSD/Paper2Bench/blob/55a3ca5eeb389c994962a375848bb5e9ca455ad9/README.md)

### 2.1 Model and configuration limitations

`paper2bench/llm.py` routes Claude-prefixed names to Anthropic and other names to OpenAI. Its `chat()` interface accepts text messages and returns content plus token counts. It has no explicit local endpoint, capability contract, residency policy, or resolved model-revision record. Retain this public interface as a compatibility facade while moving implementation into adapters. [Pinned model wrapper](https://github.com/AbhayAnandUCSD/Paper2Bench/blob/55a3ca5eeb389c994962a375848bb5e9ca455ad9/paper2bench/llm.py)

`api.run()` supplies one model across generation stages and uses that model for both verification roles. Its verification call does not forward the SDK's explicit `api_key`. Dataset-resolution helpers also omit explicit key propagation. These are concrete reasons to pass a shared, typed execution context through every nested operation. [Pinned SDK](https://github.com/AbhayAnandUCSD/Paper2Bench/blob/55a3ca5eeb389c994962a375848bb5e9ca455ad9/paper2bench/api.py), [pinned extraction helpers](https://github.com/AbhayAnandUCSD/Paper2Bench/blob/55a3ca5eeb389c994962a375848bb5e9ca455ad9/paper2bench/instruction_generator.py)

### 2.2 Verification and execution limitations

In `verify.py`, generated Python runs through host `subprocess.run`. The working directory receives provider keys in `.env`, and execution inherits the host environment. A final conclusion can end the agent loop before any code runs. The judge parser accepts loosely typed JSON; `bool("false")` becomes true. Claim evaluation removes specific numbers from the candidate conclusion and reverses the conventional precision and recall labels. Swapping those labels alone leaves their harmonic-mean F1 unchanged. The implemented pass threshold is at least 80 on a 0–100 scale. These findings establish implementation risks and scoring limitations; they do not establish an observed compromise. [Pinned verifier](https://github.com/AbhayAnandUCSD/Paper2Bench/blob/55a3ca5eeb389c994962a375848bb5e9ca455ad9/paper2bench/verify.py)

Variant generation writes curator information into `metadata.json` and the combined `instances.json`. Those files must not automatically enter a candidate-visible export. Classification falls back to `llm_evaluation` when output is invalid; strict mode should instead require review. PDF extraction preserves page markers but supplies plain text rather than a structured evidence index. [Pinned CLI](https://github.com/AbhayAnandUCSD/Paper2Bench/blob/55a3ca5eeb389c994962a375848bb5e9ca455ad9/paper2bench/cli.py), [classifier](https://github.com/AbhayAnandUCSD/Paper2Bench/blob/55a3ca5eeb389c994962a375848bb5e9ca455ad9/paper2bench/paper_classifier.py), [PDF utility](https://github.com/AbhayAnandUCSD/Paper2Bench/blob/55a3ca5eeb389c994962a375848bb5e9ca455ad9/paper2bench/pdf_utils.py)

The repository is MIT licensed. Retain upstream notices in distributions and distinguish the software license from paper, data, model-weight, and generated-artifact rights. Package metadata declares Python 3.9 or newer while the README says 3.10 or newer; reconcile the supported floor and exercise it in CI. [Pinned package metadata](https://github.com/AbhayAnandUCSD/Paper2Bench/blob/55a3ca5eeb389c994962a375848bb5e9ca455ad9/pyproject.toml), [license](https://github.com/AbhayAnandUCSD/Paper2Bench/blob/55a3ca5eeb389c994962a375848bb5e9ca455ad9/LICENSE)

## 3 Scope and success conditions

The extension has six required outcomes:

- **Portable authoring:** the entire generation pipeline works with existing hosted providers, a genuinely local model endpoint, and a tested compatible serving endpoint. Per-role configuration reaches every helper.
- **Independent subjects:** the benchmark can evaluate a different model, algorithm, agent, or executable from the model that authored it.
- **Safe execution:** generated workloads run within an explicit isolation boundary with bounded resources and scoped access to inputs and APIs.
- **Auditable science:** important task requirements trace to source evidence or declared assumptions; result claims trace to measured artifacts.
- **Recoverable operations:** interrupted work resumes from compatible completed stages, preserving attempts, costs, errors, and cancellation state.
- **Practical extensibility:** at least one classical/numerical method and one symbolic/optimization method execute through the plugin contract before the release advertises support for those paradigms.

The first release covers computational experiments with obtainable data and supported CPU or single-accelerator execution profiles. Wet-lab procedures, physical robotics, proprietary instruments, unavailable historical APIs, and experiments beyond the configured resources remain representable but explicitly non-executable or manually assessed. An adapter interface by itself is an extension point, not delivered backend support.

## 4 Model roles and computational paradigms

### 4.1 Independent configuration axes

Represent these axes independently in schemas and reports:

- **Authoring role:** parse/classify, extract, resolve resources, plan, generate variants, assess leakage, or propose a rubric.
- **Candidate role:** the system whose research or engineering ability is being evaluated. For an LLM evaluation paper, distinguish this research agent from the target model it calls during the experiment.
- **Evaluation role:** deterministic scorer, semantic judge, or human reviewer. Reference extraction is a separate role and does not establish truth.
- **Model family:** foundation model, specialized neural model, classical estimator, or no learned model. Modality and supported operations are additional properties.
- **Hosting:** on-device, local server, private cluster, managed cloud API, or self-managed cloud deployment.
- **Method:** inference, retrieval, agent loop, multi-agent coordination, training/fine-tuning, statistical analysis, numerical computation, constraint solving, theorem proving, simulation, or a hybrid composition.

Continue using the three existing paper archetypes to choose extraction/rendering templates. Add orthogonal `method_kind` and execution requirements. A method outside those archetypes can use an expert-authored specification until a versioned archetype extension exists; do not misclassify a theorem or simulation merely to fit an LLM schema.

### 4.2 Paradigm contracts and delivery scope

The first portable execution milestone delivers inference/agent execution, one classical or numerical plugin, and one symbolic or optimization plugin. The contracts below define how additional training, simulation, probabilistic, and hybrid plugins fit; their support is delivered only when their named conformance suites pass. Section 18 assigns these capabilities to release gates.

**Foundation-model inference and agents:** support text generation first, followed by negotiated tool and image capabilities. A paper requiring vision must retain figures or become blocked; extracted text is not an equivalent input. Record agent framework, tool definitions, memory policy, retrieval corpus, prompts, context strategy, and iteration limits.

**Specialized neural and classical ML:** support prediction, fitting, tuning, checkpointing, and batch evaluation through executable jobs. Record feature preprocessing, data partitions, estimator parameters, weight hashes, and training/search budgets. ONNX, scikit-learn, PyTorch, JAX, and other libraries belong in optional method/environment integrations rather than the core dependencies.

**Symbolic and optimization methods:** support solver commands with typed problem instances, objective direction, feasibility checks, optimality gaps, timeouts, and verifiable certificates where available. A timeout may return a feasible incumbent plus a gap; it must not be converted to a successful optimum claim.

**Scientific and probabilistic computation:** support simulator seeds, units, mesh/discretization, convergence tolerances, posterior diagnostics, and numerical residuals. Domain plugins select appropriate evidence rather than forcing every task into accuracy or claim F1.

**Hybrid systems:** represent retrieval plus generation, learned heuristics plus a solver, and simulation plus optimization as a dependency graph. Record each component and its data boundary. Run ablations against the same locked protocol.

**Human and manual modes:** permit a researcher to author/repair a task specification, submit an external implementation, or enter a reviewed assessment. Mark human interventions and their timing. A manually supplied valid specification can be rendered, executed, scored, and exported without any foundation-model call.

## 5 User workflows

### 5.1 Generate a task

The researcher supplies a local PDF or selects a precise paper version, enters the research question, selects an authoring profile, and sets limits. Paper2Bench extracts evidence, produces the familiar artifacts, and shows unresolved requirements. Generation succeeds with a useful draft even when execution is unavailable, but the task's readiness state makes that limitation visible.

### 5.2 Re-execute or implement a paper

The researcher selects one of three modes: `from_scratch`, `author_artifact_reexecution`, or `adaptation`. The curator freezes the task, evaluation protocol, allowed resources, and search budget before candidate work. The candidate may design experiments and perform permitted exploration, then records its confirmatory plan before the corresponding held-out runs. A separate evaluator executes the sealed submission in a clean environment, exposing test inputs to candidate code while retaining hidden labels and scoring assets elsewhere. These modes measure different abilities and must not share an undifferentiated leaderboard.

### 5.3 Generate and evaluate transfer variants

Generate bounded perturbations, ablations, or future-work tasks using the existing workflow. Each variant records its parent paper, changed factors, inherited assumptions, feasibility evidence, and curator hypothesis. Evaluate the changed experiment on its own terms. A reversed or absent effect can be a correct transfer result even when it differs from the source paper.

### 5.4 Compare models and methods

A comparison command creates separate runs from the same locked task and evaluator. Authoring is held fixed when comparing candidates. Compare both common-budget conditions and disclosed resource/frontier conditions, with costs, failures, and human intervention visible. A cloud model, local quantized model, and symbolic method may appear in one study only when they solve the same defined task under a defensible protocol.

### 5.5 Operate offline or across a cluster

Offline operation uses staged papers, data, models, dependencies, and evaluator assets with outbound networking disabled. Hybrid operation explicitly permits named cloud stages while retaining restricted data locally. Remote workers later use the same execution request contract; the submission and evaluation semantics do not change with scheduler choice.

## 6 Architecture and package boundaries

Use a modular Python library with a thin CLI. Keep existing modules callable. Introduce three trust domains even when they share one installation:

1. **Controller and curator:** configuration, credentials, source processing, task generation, policy, budget accounting, immutable manifests, and run state.
2. **Candidate worker:** approved task bundle, candidate implementation, permitted data/tools, disposable writable workspace, and resource limits. Paper text and model output are untrusted inputs.
3. **Evaluator:** protected scoring code, hidden labels/references, a sealed submission, controlled execution inputs, and authoritative metric/verdict production. Candidate execution gets only the permitted test inputs in a separate sandbox/process identity. Hidden labels, references, and protected scoring assets are neither readable nor writable by candidate code; trusted scoring consumes collected predictions and artifacts afterward.

The controller invokes model-provider adapters for authoring and agent decisions. It dispatches executable workloads through execution backends. Method plugins translate typed scientific tasks into those jobs. Evaluation plugins compute deterministic criteria or request bounded semantic review. This separation permits a local authoring model to prepare a cloud GPU experiment or a hosted model to author an offline solver task.

### 6.1 Proposed module additions

- `runtime_config.py`: `RuntimeConfig`, `ModelRef`, `RoleBinding`, policy, budgets, and a shared `RunContext`.
- `providers/`: typed generation contract, existing provider adapters, Ollama adapter, compatible-endpoint adapter, capability tests, and normalized errors.
- `executors/`: execution requests/results, isolated OCI execution, capability/preflight profiles, cancellation, and later remote/cluster adapters.
- `methods/`: entry-point plugins for prediction, training, agent loops, solvers, and simulation. Controller/evaluator plugins are explicitly installed, trusted code with pinned versions. Never auto-load a plugin named by a paper or task. Untrusted preparation scripts run inside workers.
- `schemas/`: published JSON Schemas and typed internal models for tasks, experiments, runs, evidence, and reports.
- `evidence/`: paper spans, table/figure references, extraction transformations, and claim-to-artifact links.
- `evaluation/`: deterministic metrics, rubric dependencies, semantic judging, calibration, and legacy compatibility profiles.
- `runs/`: SQLite journal, attempt/event records, atomic artifact store, cache policy, reconciliation, and resume.
- `bundles/`: candidate/curator exports, import validation, checksums, reproducibility export, and migration.

Refactor `api.run()` and CLI handlers onto a shared orchestration function to prevent their behavior diverging. Keep `llm.chat()` and `ChatResponse` as public facades. Use `BenchmarkResult.extras` for additive manifest/report paths initially. Avoid a simultaneous rewrite of extraction prompts, package layout, and all report formats.

### 6.2 Core implementation choices

Use typed validation with JSON Schema for interchange; a small validation dependency such as Pydantic may implement internal models. Use safe YAML loading with explicit bounds. Use SQLite transactions and local files for one-controller operation. Store large artifacts by SHA-256 content identity with atomic finalization. Local file locking and a single-writer policy prevent competing controllers from corrupting a run.

Keep model SDKs, accelerator libraries, solvers, OCR, and serving runtimes optional. Publish tested dependency constraints and locked execution environments. Model servers run as separately managed processes/services; importing Paper2Bench should not download weights or start a server. SDK/CLI logs and outbound telemetry must not contain document content or secrets by default.

## 7 Provider and runtime contract

### 7.1 Explicit provider identity

A `ModelRef` records provider adapter, endpoint identity, hosting location, requested model ID, resolved identity/revision when exposed, weight digest when obtainable, quantization, adapters, tokenizer/template revision, modalities, context/output limits, and supported operations. Use `unknown` for unavailable metadata. A mutable hosted alias cannot promise identical future weights.

For legacy calls only, preserve the existing name-based provider inference. In the new runtime configuration, provider and endpoint are explicit. Do not infer privacy from an endpoint being `localhost`: a local gateway may forward to a cloud. Ollama documents both local and cloud API use; deployment policy must therefore apply to resolved execution, not just URL shape. [Ollama API documentation](https://docs.ollama.com/api/introduction)

### 7.2 Capabilities and normalized responses

Adapters expose `capabilities()`, `generate()`, and usage estimation where supported. Capabilities include text/image input, structured output, tool calling, streaming, context limits, batching, cancellation semantics, and revision/usage visibility. Include `supported`, `unsupported`, and `unverified` states with probe version/time.

The normalized response records content, finish reason, refusals, tool calls, usage and its completeness, latency components, provider request ID, resolved identity, and a redacted raw-response reference. A truncated response is not a valid complete specification. Unknown token usage is not measured zero; preserve legacy numeric facade fields while keeping the accounting record's missingness explicit.

Treat compatible HTTP protocols as transport compatibility. vLLM documents supported interfaces and feature-specific constraints; test the exact endpoint/model combination. Ollama currently documents cloud structured-output limitations. Native schema enforcement, JSON mode, and prompt-plus-validation fallback are distinct strategies and must be recorded. [vLLM compatible serving](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/), [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)

### 7.3 Routing and fallback

Resolve a role in this order: data/egress policy, required operation and modality, context and hardware feasibility, budget and rate availability, then explicit preference. Validate endpoint URLs, TLS policy, and allowed destinations. Keep secrets in credential references rather than serializing values into configuration or bundles.

An authoring role may use an explicitly permitted fallback and must log the change. A candidate or target-model binding is frozen for a benchmark condition. Changing its model, quantization, retrieval configuration, or toolset creates a new condition; it is never an invisible retry. A judge change creates a new evaluation identity or requires calibrated comparability.

Normalize errors into authentication, rate limit, timeout, unavailable model, unsupported capability, context overflow, refusal, malformed output, policy denial, and unknown outcome. Retry only retryable failures within both a per-call and total-run limit. An ambiguous billed timeout enters reconciliation or records possible duplicate cost; do not promise exactly-once external API execution.

### 7.4 Small and local model support

Whole-paper prompts are insufficient for smaller context windows. Build a reusable page/section evidence index; perform schema-guided extraction over bounded sections, then reconcile claims against cited spans. Budget tokens using the endpoint's tokenizer where available, with conservative estimates otherwise. Reserve space for instructions, responses, tools, and accumulated conversation. Never silently truncate required methods or results.

Cache immutable extraction inputs and intermediate results by paper hash, parser/prompt/schema version, model identity, generation parameters, and policy. Keep caches private by default. Cached stochastic calls are reused artifacts, not fresh independent trials. For mutable aliases or unknown revisions, a cache hit describes the historical response and cannot establish the model's current behavior. Record cache hits in run evidence.

Preflight local inference for model availability, RAM/VRAM, disk, context settings, OS/architecture, and acceleration support. Distinguish load time, cold-start latency, warm inference, queueing, and experiment compute. Avoid co-locating authoring inference with candidate latency measurements unless resource sharing is part of the protocol.

## 8 Execution and method plugin contract

Every execution request contains an immutable task/plan identity, run and attempt IDs, code/image identity, typed input mounts, operation/entrypoint arguments, permitted outputs, resource limits, network/API policy, deadline, seed, and required environment capabilities. Commands are argument arrays rather than interpolated shell text. The trusted controller validates plugin-prepared jobs against policy; untrusted preparation cannot enlarge permissions or obtain controller credentials. Installing a controller plugin is a code-trust decision, not a sandboxed task action.

An execution result contains backend job ID, actual start/end time, exit code or signal, lifecycle status, stdout/stderr references, resource measurements, checkpoints, artifact hashes, and structured errors. Agent-authored result JSON is an input to evaluation, not an authoritative controller record.

Backends implement `preflight`, `submit`, `status`, `cancel`, `collect`, and `reconcile`. Method plugins expose metadata, schema validation, job preparation, and expected evidence. Metrics plugins run in the protected evaluation domain. Avoid a universal `chat()` interface for a solver, estimator, or simulator.

Begin with isolated Linux CPU execution and a separately tested single-GPU profile. On macOS, use a managed Linux VM/container-engine environment for generated code, or a remote isolated worker. Apple Silicon inference can run through a local server while candidate code executes in Linux. Native Metal/MPS experiments need a dedicated, explicitly documented worker profile; a Linux image is not evidence of Metal compatibility.

Do not advertise an ordinary container as a sufficient hostile multitenant boundary. Select a hardened sandbox or dedicated VM according to the threat profile. Test runtime, accelerator, driver, framework, and isolation together. gVisor documents both a security model and accelerator support; neither all GPU work being unsupported nor every workload being safely compatible should be assumed. [gVisor security model](https://gvisor.dev/docs/architecture_guide/security/)

## 9 Source ingestion and task construction

### 9.1 Controlled ingestion

Accept local PDFs and existing arXiv download flows first. Pin the selected paper version and hash all source bytes. DOI/URL, HTML/LaTeX, supplemental archives, and repositories can follow through controlled fetch/import adapters. Preserve bibliographic identity, source URL, retrieval time, license/access status, and attachment relationships.

Run parsers in bounded workers. Limit input bytes, pages, expanded archive bytes, file count, nesting, processing time, memory, and extracted text. Reject path traversal and escaping symlinks. Validate the initial destination and each redirect before following it, and enforce permitted resolved addresses at connection time without an unchecked second DNS lookup. Reject internal/metadata destinations unless an explicit private-data integration permits them. Package builds, dataset loaders, repository scripts, and model remote-code loading are executable imports and belong inside an isolated preparation stage.

### 9.2 Evidence and uncertainty

Keep a structured document representation with page, section, paragraph, equation, table, and figure identities where extraction supports them. Store source spans for each consequential dataset, method component, metric, hyperparameter, resource requirement, and result. Mark OCR/table reconstruction confidence and preserve the source crop/reference for review. Do not turn a model's extraction confidence into a calibrated probability of truth.

Represent facts as `stated`, `inferred`, `assumed`, `unresolved`, or `conflicting`, with evidence and a resolution record. Missing learning rates, inaccessible data, conflicting units, or ambiguous splits must be visible. A task cannot become executable while unresolved critical inputs remain; a curator may resolve them explicitly and create a new task revision.

Dataset name existence is only the first check. Validate revision, content/schema, split membership, label availability, license/access, preprocessing, subject overlap, and loadability in the intended environment. Pin model artifacts and repository commits similarly. HTTP success or Hub repository existence does not establish scientific suitability.

### 9.3 Task readiness gates

A task progresses through `draft`, `needs_review`, `ready`, `blocked`, or `retired`. Readiness requires source fidelity, sufficient instructions, resource feasibility, a usable scorer/evidence contract, and a candidate-bundle leakage check. Human review is required for new scientific rubrics and unresolved methodological choices; routine schema-valid generation can remain automatic.

Keep `instruction_gt.txt` for compatibility but describe it as a generated reference plan with a validation status. It is not inherently ground truth. New execution profiles build candidate instructions from an explicit allowlist of permitted content; the historical reference plan is not automatically included. Legacy verification may continue to use its original inputs under an explicit legacy profile.

## 10 Experiment and artifact schemas

### 10.1 Two plans with different owners

The **curator protocol** defines the research question, permissible resources and search, scoring rules, hidden evaluation, critical methodology, and reproduction tolerances. Freeze and hash it before candidate work. The **candidate experiment plan** records the candidate's methodological decisions and planned confirmatory analyses before those analyses run. This preserves experiment design as a research-agent skill while preventing retrospective selection of successful outcomes.

Allow exploratory trials within a declared search budget, preserving every attempt and its purpose. Any confirmatory change creates a plan revision and states whether prior held-out feedback influenced it. Never silently substitute a smaller model, synthetic data, alternate dataset, easier metric, or approximate method when the specified experiment cannot run.

### 10.2 Required artifact contracts

- **`task_manifest.json`:** schema version, stable task ID and revision, source-paper family/hash, archetype, method kind, research question, task mode, allowed candidate resources, declared outputs, resource limits, candidate-file allowlist, protocol/evaluator identity, readiness state, and unresolved items.
- **`curator_manifest.json`:** restricted reference plans, original result claims, hidden labels, expected-outcome hypotheses, rubric/scorer assets, validation/reviewer records, and their hashes. Export separately from candidate materials.
- **`experiment_plan.json`:** curator-protocol identity, candidate plan revision, dataset and split identities, methods and baselines, preprocessing, metric definitions/units/direction, estimands, comparison/tolerance/decision rules, replication schedule, stopping rule, search budget, resource profile, assumptions, and deviations.
- **`run_manifest.json`:** task/plan/submission identities, requested and resolved models, provider/worker metadata, package/repository/image/dependency identities, policy, authoring and execution budgets, timestamps, parent run, attempts, cache use, and final lifecycle state. Secrets are excluded.
- **`evidence.jsonl`:** controller-issued events linking activities, input/output entities, actor identity, trust origin, timestamps, hashes, and artifact references. Candidate-authored observations remain identified as untrusted until checked.
- **`evaluation_report.json`:** evaluation protocol, criterion verdicts/reasons, evidence references, metric values and uncertainty, critical gates, coverage, legacy comparison scores when requested, execution/fidelity/agreement/transfer states, and evaluation errors.
- **`reproduce.json`:** entrypoint arguments, required assets and access instructions, environment identity, expected output contracts, supported hardware, and explicit reproducibility limitations.

Publish schema files, examples, migrations, and validation commands together. Reject unknown major schema versions; preserve unknown additive fields in private archival round trips, but exclude them from candidate exports until explicitly reviewed. Canonicalize the exact serialization used for hashing, including numeric and path normalization. Restrict artifact paths to the bundle root; use content digests for identity and controller validation for authority.

### 10.3 Bundle layout and visibility

Retain the familiar generated files in the curator's working output. A versioned export creates two independent roots:

```text
candidate/
  task_manifest.json
  instruction.txt
  task_config.yaml         field-allowlisted public projection
  inputs/                 permitted files or pinned fetch manifests
  interfaces/             public input and output schemas
curator/
  curator_manifest.json
  instruction_gt.txt      reviewed reference plan when available
  metadata.json           hypotheses and curation records
  evaluator/              protected scoring and hidden assets
runs/<run-id>/
  run_manifest.json
  experiment_plan.json
  evidence.jsonl
  attempts/<attempt-id>/  logs, code, outputs and resource evidence
  evaluation_report.json
  reproduce.json
```

The layout is logical; filesystem placement alone does not provide access control. Materialize only `candidate/` into the candidate worker. Export both files and their fields by allowlist: the public task configuration excludes curator outcomes, disallowed author-code/reference-implementation hints, and unknown fields. Neither `instances.json` nor the curator root is mounted. New candidate exports exclude the paper itself when its results would reveal a transfer task's intended answer; each task declares exactly which method/source material is intentionally provided.

Use provenance concepts compatible with W3C PROV's entities, activities, agents, and derivations. Add an optional export pinned to a named RO-Crate version after validating its profile; do not claim conformance merely for emitting JSON-LD. Hashes establish integrity and identity, not scientific correctness. [W3C PROV-DM](https://www.w3.org/TR/prov-dm/), [RO-Crate 1.2](https://www.researchobject.org/ro-crate/specification/1.2/)

## 11 Evaluation and scientific validity

### 11.1 Evaluate task quality separately from candidate performance

A benchmark task can be invalid even when an agent completes it. Evaluate source fidelity, instruction sufficiency, resource feasibility, evaluator adequacy, and leakage before interpreting candidate results. A task without accessible data remains a useful draft but is excluded from an executable leaderboard with its reason disclosed.

For candidate performance, record independent verdicts for implementation fidelity, required execution, empirical agreement, clean re-execution, and transfer. Criterion verdicts are `pass`, `fail`, `inconclusive`, `not_run`, or `not_applicable`. Use structured reason codes such as `missing_data`, `unsupported_hardware`, `invalid_submission`, `timeout`, `judge_error`, or `insufficient_replication`. Lifecycle state and scientific verdict are different fields.

A faithful experiment that contradicts a paper can pass implementation and execution while failing empirical agreement. A copied conclusion without computation can score highly on text similarity while earning no execution or reproduction pass. An infrastructure outage is not evidence that a scientific method failed.

### 11.2 Versioned evaluation profiles

- **Historical `legacy_claim_f1/v1` and `legacy_llm_judge/v1`:** identify imported upstream results without changing their values. Their `pass` is the upstream-recorded conclusion acceptance flag, not an independently validated finding; retain the original parsing and scoring limitations in provenance.
- **Live conclusion profiles in version 2:** retain `--eval-mode claim_f1` and `--eval-mode llm_judge` as documented aliases, with explicit version-2 protocol identities. Apply strict parsing, correct precision/recall terminology, preserve quantitative details, and distinguish evaluation errors. These changes require fresh calibration and are not directly comparable with historical scores.
- **`execution/v1`:** required operations ran under the declared environment, produced valid artifacts, and satisfied execution evidence requirements. This alone does not establish method fidelity or result agreement.
- **`reproduction/v1`:** required methodology, clean execution, and prespecified empirical comparisons pass. Dependent result criteria cannot pass when critical data/method gates fail.
- **`transfer/v1`:** the changed task's valid investigation and supported conclusions pass its own protocol. The original paper's conclusion is not the gold answer.

Preserve the legacy report keys for the legacy CLI aliases, including the conclusion-score `pass`, and add `evaluation_protocol`, `evaluation_status`, and explicit scope. For malformed/unavailable evaluation, `pass` is false and `evaluation_status` is `error`; aggregators exclude it from scientific-failure counts while reporting its coverage. New reports use named verdicts rather than an overloaded top-level scientific `pass`. Never relabel an old `pass` as paper reproduction.

### 11.3 Trusted evaluation

Seal and hash the candidate's source and environment specification before final evaluation. Rebuild or restore the pinned environment in a fresh worker. Candidate code receives controlled test inputs and emits predictions, traces, or certificates; a separate trusted process computes metrics using hidden labels and protected scoring assets. Do not accept a printed metric as authoritative.

For interactive tasks, place private environment state behind a constrained evaluator API. For simulation and optimization, recompute invariants, feasibility, objectives, residuals, or certificates using the domain plugin. Re-execution records a controller nonce, actual inputs, process exits, resources, and output hashes. This proves which job the controller observed, while the rubric determines whether the job addressed the science.

Treat collected artifacts as hostile inputs. Accept bounded, non-executable typed formats such as validated JSON/CSV or arrays with object loading disabled. Do not deserialize pickle/joblib or execute candidate-provided loader code inside trusted scoring. Validate shapes, row counts, units, identifiers, finite values, and file sizes; isolate complex parsers before transferring validated data to the scorer.

PaperBench provides a relevant precedent for separating code development from fresh execution and result evaluation. It also illustrates why a generated rubric and an LLM judge need their own validation. Use these design lessons without assuming that Paper2Bench tasks automatically satisfy that benchmark's standards. [PaperBench research paper](https://arxiv.org/abs/2504.01848)

### 11.4 Quantitative comparisons and statistics

Prespecify the primary metric, direction, units, population, baseline, estimand, tolerance, and stopping rule. Declare whether agreement means an absolute/relative tolerance, directional effect, confidence-bound criterion, formal equivalence test, or exact property. Do not equate a nonsignificant difference with equivalence.

Select replication counts using expected variability and desired precision or power. A small fixed seed count may support an exploratory report, not a general reproduction claim. Separate variability from training initialization, dataset partition, subject sampling, model sampling, endpoint drift, and whole-agent runs. Seeds are not independent people or independent papers.

Report per-run values, failures, actual sample sizes, effect differences, and uncertainty methods/assumptions. Pair observations when the design permits; account for repeated subjects and clustering. Prespecify treatment of failed runs, missing observations, multiple comparisons, hyperparameter search, and optional stopping. If source uncertainty is unavailable, report agreement with a published point estimate under a declared tolerance.

Keep reported paper numbers, re-executed author-artifact numbers, and new implementation measurements visibly distinct. For symbolic or deterministic tasks, use exact properties, checked proofs/certificates, feasibility, or numerical error bounds where appropriate. NeurIPS guidance supports recording settings, variability, uncertainty methods, and compute; the concrete decision rule remains task specific. [NeurIPS paper checklist](https://neurips.cc/public/guides/PaperChecklist)

### 11.5 Semantic judging and calibration

Use deterministic checks for schemas, execution, predictions, metrics, numerical comparisons, and invariants. Use models only for semantic criteria requiring interpretation. Each judged claim retains polarity, magnitude, units, conditions, comparison, scope, evidence, and `supported`, `contradicted`, or `insufficient_evidence` status. Copied source assertions must be distinguishable from candidate-demonstrated findings.

Validate judge responses against strict schemas: actual booleans, bounded scores, required evidence, and enumerated statuses. Never coerce arbitrary strings to booleans or quietly clamp invalid scores into validity. A bounded repair retry may correct formatting; repeated failure is an evaluation error. Empty claim sets cannot earn a successful default score.

Build an expert-labeled calibration set containing correct, partial, misleading, contradictory, numerically wrong, and adversarial submissions for each supported archetype. Measure false acceptance, false rejection, agreement, and uncertainty by judge/prompt/protocol version. Blind candidate/provider identity where feasible. Cross-model judging is an additional check, not independent ground truth. Automate a scientific semantic gate only after its agreed error bounds are met; otherwise retain human review.

### 11.6 Aggregation and reporting

Report task readiness, completion, evaluation coverage, failure stages, critical gate outcomes, and uncertainty alongside scores. Aggregate criteria within papers and papers within the corpus so papers with many rubric leaves do not dominate. State weighting rules and all exclusion reasons.

For an applicable rubric weight `W`, passed weight `P`, and unknown/inconclusive weight `U`, report the interval `[P/W, (P+U)/W]`. Include applicable `not_run` criteria and evaluation errors in `U`; remove only justified `not_applicable` criteria from `W`, recording reasons. A zero applicable denominator is ungradable, never a perfect score. Keep infrastructure errors, scientific disagreement, candidate defects, and invalid tasks in separate counts.

Report authoring cost, agent inference, target-model calls, experiment compute, failed attempts, wall time, and human intervention separately. Resource-matched comparisons require the same task and evaluation identities plus a declared budget allocation. Different quantizations or hosted revisions are separate conditions, even when marketed under the same model family.

## 12 Leakage and benchmark contamination

Store private hypotheses, source conclusions, hidden labels, and scoring assets outside candidate-readable bundles. A prompt telling an agent not to open a neighboring file is not a boundary. Export by file allowlist and inspect paths, nested archives, metadata, logs, and embedded artifacts for prohibited content.

Define permitted source exposure per task: method implementation may intentionally include methods, while answer discovery may withhold results. Automated leakage classification is advisory; review new templates and transformations against explicit allowed/forbidden material. A method description is not automatically a leak when implementing that method is the intended task.

Assign every research question and variant derived from one paper to the same corpus partition. Group near-duplicate papers, related author repositories, shared subjects, and reused templates where needed. Maintain development tasks, private or chronological holdouts where feasible, and a release history. Repeated hidden-test feedback counts toward a query budget and may retire a task from blind evaluation.

Record allowed web/search access and actual resource traces. Public-paper pretraining contamination is often unobservable; offline execution does not erase it. Report this limitation and use new variants or private materials to test transfer without claiming contamination has been eliminated. MLE-bench's documented split and answer-exposure issues are a useful operational warning about benchmark construction. [MLE-bench repository](https://github.com/openai/mle-bench)

## 13 Security and privacy requirements

### 13.1 Candidate execution boundary

The default verification path must not run model-generated code on the host. Enforce a disposable workspace, a read-only root/input filesystem where practical, unprivileged identity, restricted system calls/capabilities, process limits, CPU/RAM/disk limits, bounded output, a deadline, and termination of descendants. Declare accelerator limits honestly: where hard VRAM quotas are unavailable, use dedicated-device admission and monitoring/termination; reject tasks requiring an unsupported hard guarantee. Prohibit privileged execution, host credential mounts, container-engine sockets, and unapproved host paths.

Separate dependency preparation from experiments. Resolve and lock dependencies in an isolated builder with controlled registry/network access, then run experiments without arbitrary installation privileges. Treat custom wheels, source builds, dataset scripts, pickled objects, and remote model code as executable content. Record supply-chain inputs and scan known vulnerabilities as part of release policy; scanning does not replace isolation.

If a host runner remains for trusted developer debugging, require an explicit `--unsafe-host-execution` option, record it prominently, and exclude its results from assurance claims. When isolation is unavailable, the default command stops before candidate code executes. This is an intentional, documented safety change to historical behavior.

### 13.2 Credentials and network access

Remove generated `.env` files and inherited host environments from candidate execution. Pass a small environment allowlist. Keep provider secrets in the controller or an operating-system/secret-store reference. Avoid global dotenv mutation as a hidden dependency of nested stages.

When a candidate legitimately evaluates a model API, expose a run-scoped broker enforcing allowed model IDs/endpoints, request shapes, rates, token limits, and budgets. Distinguish the research agent's authoring/control credentials from target-model permissions. The worker receives no reusable provider key. The broker's tool capabilities cannot expose arbitrary filesystem reads or external commands.

Deny outbound network access by default. Permit explicit destinations through policy-aware fetch/API gateways. Revalidate redirects and destination addresses; prevent metadata-service access and DNS rebinding bypasses. Treat a URL suggested by a model as untrusted input. Offline mode includes downloaders, dataset validation, telemetry, model gateways, and update checks, not merely inference. [OWASP SSRF guidance](https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html)

### 13.3 Source injection and data governance

Instructions embedded in PDFs, repositories, examples, or model responses are source content. They cannot modify runtime policy, authorize tools, reveal secrets, or change the trusted evaluator. Separate task data from controller instructions, validate generated specifications, and enforce permissions outside prompts.

Classify artifacts as public, workspace-private, restricted, or secret. Policy checks precede every model call and upload. `local_only` prohibits cloud forwarding and content-bearing telemetry; enforce deny-egress controls over model-server/gateway processes as well as the controller and worker. Where a server is operator-managed outside that boundary, record the residency assumption and block a verified-offline claim unless its egress is independently controlled. A permitted cloud stage receives only its approved evidence subset. Do not represent prompt redaction as a guarantee that derived content is free of sensitive information.

Attach access/license/retention metadata to papers, data, weights, code, logs, and bundles. Default to private local storage and explicit export. Provide configurable retention, deletion of indexed/cache copies, and a deletion receipt; immutable audit retention must use redacted identifiers if full-content deletion is required. Later services must enforce workspace authorization on every run, artifact, log, and cancellation action. Opaque IDs alone confer no permission. [OWASP authorization guidance](https://cheatsheetseries.owasp.org/cheatsheets/Authorization_Cheat_Sheet.html)

## 14 Reliability and resource accounting

### 14.1 Durable state and retries

Persist run IDs, attempt IDs, stage inputs, state transitions, budgets, remote job IDs, and output finalization in a transactional journal. Stages move through `pending`, `preflight`, `running`, and a terminal `succeeded`, `failed`, `blocked`, or `cancelled` state. Retries create new attempts. `reconciling` represents an external outcome that is not yet known.

Use immutable artifact paths and atomic writes. Before external submission, persist submission intent and a client idempotency key; then atomically persist the returned job/request identity. If a crash occurs between submission and identity persistence, reconcile by that key or retain an unknown outcome rather than blindly resubmitting. Resume only when input, configuration, schema, model, policy, and evaluator compatibility checks pass. A changed critical identity invalidates dependent stages rather than reusing stale evidence. Preserve failed logs and artifacts with retention controls. Never overwrite a previous run because its task ID matches.

Classify failures before retry: transient transport errors may retry; invalid methods, insufficient data, permission conflicts, and exhausted budgets need a changed plan or user intervention. Repeatedly asking a model until it emits a desired scientific answer is not an infrastructure retry. Persist remote IDs before polling and reconcile orphaned jobs after restarts.

### 14.2 Cancellation and deadlines

Represent `cancellation_requested`, `cancelling`, and confirmed `cancelled` separately. Stop scheduling descendants, ask the backend to terminate, escalate after its configured grace period, and collect partial evidence. Keep unsettled reservations until termination and usage reconciliation establish what can be released. Reject late artifacts from superseded attempts. The controller must confirm process-group/container/remote-job termination before reporting it stopped.

A provider may lack cancellation or may have accepted a request before a connection failed. Preserve `reconciling` or cancellation-pending details and possible continuing charges. Cancellation of a caller does not prove a cloud job or API request ended.

### 14.3 Typed budgets

Replace ambiguous `budget_per_model` interpretation with explicit units: requests, input/output tokens, agent turns, tool calls, search trials, CPU/GPU seconds, wall seconds, RAM/VRAM bytes, storage/download bytes, and optional currency ceiling. Preserve the old field for reading/rendering; migration requires a declared unit or marks it unresolved rather than guessing dollars or tokens.

Reserve budget transactionally before dispatch, then reconcile actual usage and refunds. Enforce worker limits outside candidate code and reject work whose maximum permitted output would exceed the remaining token budget. Separate authoring and evaluation pools from experiment pools so generating a benchmark cannot silently consume its execution budget. Charge authoring/resource-resolution/variant calls to authoring, reference extraction and judges to evaluation, and candidate-agent control plus target-model calls to experiments; no role dispatches without an assigned pool.

Record measured, estimated, and unknown usage/cost independently. Version price assumptions when cost estimation is enabled. Local hardware time is measurable even when its monetary cost is unspecified. Controller-side limits can bound dispatched work, but missing provider usage and asynchronous billing prevent an unconditional exact-invoice guarantee.

### 14.4 Observability and integrity

Structured events include run/task/stage/attempt IDs, provider request or backend job IDs, latency, retries, quotas, errors, and artifact references. Redact secrets and keep full prompts/raw responses behind an explicit private-debug policy. Reports should explain a failure without requiring sensitive content in ordinary console output.

Provide human-readable progress and machine-readable `--json` output. The final summary distinguishes draft generation, task readiness, execution status, and scientific verdict. Set nonzero exit codes for invalid configuration, policy denial, blocked required execution, runtime failure, and evaluation error; document a separate outcome for scientifically valid disagreement so automation does not confuse it with infrastructure failure.

Support backup/export of the journal plus referenced artifacts, integrity checks on restore, and controlled garbage collection of unreferenced content. A later remote service adds authentication, tenant isolation, encrypted storage, quotas, rate limiting, and disaster recovery without changing artifact semantics.

## 15 CLI and SDK evolution

### 15.1 Compatibility rules

Retain `download`, `parse`, `classify`, `extract`, `render`, `plan`, `verify`, `run`, `specextract`, and `generate`. Preserve public imports and `BenchmarkResult` fields. Keep existing generation defaults when no new configuration is supplied, while documenting default execution hardening and evaluation-protocol changes as intentional behavior changes.

Add `--runtime-config` to every network/model-using command and propagate its `RunContext` through classification, nested dataset resolution, plans, variants, leakage checks, reference extraction, candidate control, and judging. Reconcile CLI and SDK behavior through the same controller. Missing credentials for one role must never fall through to another provider's key.

Resolve each role from an explicit role-specific CLI/SDK override, then the selected runtime profile, then an explicitly supplied legacy model argument, then the historical default. An argparse default value must not silently override a selected local profile. Reject conflicting evaluation flags and ambiguous generic keys with mixed providers. A restrictive `--offline` flag tightens the profile; it cannot be weakened by fallback routing.

Read the explicit configuration path or a documented `PAPER2BENCH_RUNTIME_CONFIG` environment reference. Validate and show the resolved, redacted configuration before dispatch. Avoid undisclosed global profile selection. Resolve credential references at call time and omit values from printed/saved configuration.

### 15.2 Proposed commands

The following commands are interface targets for the extension, not commands available in upstream 0.3.1:

```bash
paper2bench doctor --runtime-config runtime.local.yaml
paper2bench run --pdf paper.pdf --task-id study_a \
  --research-question "Does the stated method improve the primary metric?" \
  --runtime-config runtime.local.yaml --output-dir ./output
paper2bench validate --task ./output/tasks/study_a/task_manifest.json
paper2bench export --task ./output/tasks/study_a --audience candidate \
  --output ./bundles/study_a
paper2bench export --task ./output/tasks/study_a --audience curator \
  --output ./protected/study_a
paper2bench execute --task ./bundles/study_a/task_manifest.json \
  --evaluation-bundle ./protected/study_a/curator_manifest.json \
  --runtime-config runtime.local.yaml --evaluation-profile reproduction/v1
paper2bench status --run RUN_ID --json
paper2bench resume --run RUN_ID
paper2bench cancel --run RUN_ID
paper2bench reproduce --run RUN_ID --runtime-config runtime.local.yaml
```

`doctor` checks adapter availability, credentials without printing them, model reachability/identity, capabilities, context, policy, local hardware, and execution isolation. It performs no paid inference probe unless explicitly requested. `validate` distinguishes schema validity, resource checks, leakage checks, and scientific readiness. `execute` is explicit; ordinary generation remains usable without an execution backend.

`export --audience candidate` is field-allowlisted; curator export requires an explicit audience and retains its sensitivity labels. The controller resolves protected evaluator assets from an explicit evaluation bundle or an operator-trusted registry pinned by digest; missing assets fail preflight. A candidate-supplied scorer identity cannot authorize a download, controller import, or plugin installation. Import validates schema and hashes without auto-executing scripts. `reproduce` creates a linked fresh run and reports compatibility/deviations; it does not pretend a remote alias or unavailable historical device is identical.

### 15.3 SDK target

Add optional keyword arguments `runtime_config`, `agent_model`, `eval_model`, and `evaluation_profile` to `run()` or a typed equivalent, retaining legacy arguments. Accept a typed configuration object or explicit file path. New `execute`, `validate`, `export_bundle`, `resume`, and `reproduce` functions share controller implementations with their CLI commands.

Continue returning `BenchmarkResult`; add manifest, run, evidence, and report paths through `extras` initially. Raise typed exceptions with stable reason codes. Permit callers to receive progress events and cancel through a token without requiring an asynchronous framework. Internally asynchronous provider/executor operations may use a synchronous wrapper for the existing SDK.

## 16 Runtime configuration example

This is a proposed minimal local profile. The schema should require the named environment variable to contain an operator-selected installed model ID; it is not a secret. The task supplies its pinned execution image and target-model bindings. The limits below are illustrative operator settings, not measured requirements for all papers.

```yaml
schema_version: 1
profile: local_research
policy:
  data_egress: local_only
  verified_offline_required: true
  telemetry_content: false
models:
  local_text:
    provider: ollama
    endpoint: http://127.0.0.1:11434
    hosting: local_server
    model_from_env: P2B_LOCAL_MODEL
    required_capabilities: [text_generation]
roles:
  author: local_text
  reference: local_text
  agent: local_text
  judge: local_text
routing:
  fallback: disabled
execution:
  backend: oci
  profile: linux_cpu_single_user_vm
  network: none
  max_parallel_jobs: 1
budgets:
  authoring:
    input_tokens: 200000
    output_tokens: 30000
    requests: 40
  evaluation:
    input_tokens: 100000
    output_tokens: 15000
    requests: 50
  experiments:
    input_tokens: 400000
    output_tokens: 60000
    requests: 100
    wall_seconds: 3600
    cpu_seconds: 7200
    memory_bytes: 8589934592
    disk_bytes: 10737418240
    agent_turns: 30
    search_trials: 10
retry:
  max_attempts_per_step: 3
  max_total_attempts: 50
```

Separate `judge` and `reference` bindings do not make their outputs independent or correct when they use the same model. A cloud profile may change those bindings only after an explicit egress policy permits their input classifications. Bind a cloud provider with a provider-specific credential reference and resolved model identity; do not reuse a generic API key across all roles.

Preflight must verify control over the local model server's egress to satisfy this profile's offline requirement. If the operator's existing server cannot meet that condition, return a policy/preflight error or select a controlled offline server. Staged model/dependency downloads are a separate preparation operation.

## 17 Verification strategy and release gates

All criteria in this section are engineering acceptance targets. They are not measured capabilities of the current project.

### 17.1 Compatibility and provider tests

- **G01:** Existing command parsing, public SDK imports, result fields, artifact filenames, and template rendering pass offline characterization tests. Document intentional safety/protocol migrations.
- **G02:** Every nested model call uses the selected role/provider/credential reference. Test mixed providers, explicit SDK keys, missing keys, dataset helpers, variants, and verification.
- **G03:** Hosted existing-provider adapters, a local Ollama endpoint, and one compatible server pass the same response/error conformance suite. Include refusal, truncation, invalid JSON, tools if advertised, usage missingness, authentication, rate limits, and streaming if advertised.
- **G04:** A staged local-only generation/execution workflow produces no outbound connection across controller, worker, and controlled model server. An attempted cloud fallback fails visibly.
- **G05:** Unsupported modality/context/hardware, unavailable model, unresolved critical data, and insufficient budget produce explicit states without silent substitution.

Use deterministic fake providers for ordinary CI and opt-in, budget-capped live conformance jobs before advertising a real endpoint. Label backend support as experimental, tested, or unsupported with the exact model/runtime/hardware profile. Installation or discovery alone is not a passing test.

### 17.2 Isolation and adversarial tests

- **G06:** Candidate code cannot read host credentials, curator metadata, hidden labels, scoring assets, container-engine sockets, or unauthorized host paths.
- **G07:** Malicious paper instructions cannot obtain secrets or change effective policy/tool authorization. Plugin names in source content cannot cause controller imports.
- **G08:** Archive traversal, escaping symlinks, oversized sources, parser hangs, malicious result serialization, and invalid output shapes are rejected within bounds.
- **G09:** Fork/process exhaustion, excessive stdout, disk exhaustion, deadlines, and descendant termination remain bounded by backend controls. Accelerator-limit guarantees match the advertised profile.
- **G10:** Fetch redirects, rebinding attempts, metadata addresses, and unapproved target-model requests are denied. Secrets never enter candidate environment or bundles.

### 17.3 Scientific validity tests

- **G11:** A copied conclusion without experiments receives no execution/reproduction pass. Hardcoded metrics fail recomputation on controlled inputs.
- **G12:** String `"false"`, missing fields, malformed/truncated judge output, nonfinite metrics, empty claims, and unavailable judges never become a scientific pass.
- **G13:** Wrong units, numeric magnitude, polarity, population, or comparison prevent false entailment. Critical method failure blocks dependent result criteria.
- **G14:** A well-executed variant reversing the source effect can pass its own transfer criteria. A faithful contradictory reproduction retains separate fidelity and empirical-agreement verdicts.
- **G15:** Wrong dataset revisions/splits, overlapping subjects, and source-paper variants crossing partitions fail validation. Candidate exports contain only allowed fields and source materials.
- **G16:** Unknown/not-run/error criteria remain in coverage accounting; invalid zero denominators are ungradable. Changed judge/rubric/data/model configurations create distinct evaluation/run identities.

### 17.4 Reliability and paradigm tests

- **G17:** Kill/restart the controller around dispatch, returned-job persistence, artifact writes, and finalization. Completed compatible stages are not repeated; ambiguous external submissions reconcile or remain visibly unknown.
- **G18:** Cancellation stops local descendants within the configured grace/force deadlines. Remote unconfirmed termination remains pending, reservations stay unsettled, and stale outputs cannot overwrite current attempts.
- **G19:** Concurrent reservations cannot exceed configured allocatable budgets. Failed/retried work and unknown usage remain visible in totals.
- **G20:** One task completes through hosted authoring, local authoring, and an explicitly permitted hybrid profile. A manually authored task executes with no generative-model calls.
- **G21:** A classical/numerical plugin and a symbolic/optimization plugin produce schema-valid outputs and independently checked metrics/certificates. Training, simulation, or multi-agent support requires its own additional passing suite.
- **G22:** Export/import and fresh re-execution preserve hashes, visibility, protocol identities, and result contracts. Cache or resume cannot attach evidence from a different task/configuration.

Maintain a curated pilot corpus covering all three current archetypes plus the delivered non-LLM methods, varied document quality, unavailable resources, and contradictory/negative results. Use it to test infrastructure and evaluation behavior; its size alone cannot validate scientific judge accuracy. Every advertised scientific task needs source review and an independently checked reference procedure or appropriate expert assessment.

CI should run formatting/static typing, schema/migration tests, characterization tests, provider contract tests, isolation tests on the supported runner, and integration tests for delivered plugins. Real-provider/GPU suites run separately with pinned environments and budgets. Scientific regression tests check evidence and verdict semantics, not whether a stochastic agent reproduces identical prose.

## 18 Implementation sequence

### Gate A Safe compatible core

**PR 1 — characterization and manifest foundation.** Capture CLI/SDK/artifact behavior at the pinned upstream revision. Add schema/version identities and migration readers. Resolve Python support metadata and add CI for the declared matrix. Preserve original report samples without legitimizing their scientific meaning.

**PR 2 — context and existing-provider adapters.** Introduce `RunContext`, explicit role bindings, normalized responses/errors, and redacted configuration. Refactor `llm.chat()` behind the facade; fix nested credential propagation. No scientific prompt redesign is needed to verify this seam.

**PR 3 — isolated execution and secret removal.** Replace `_run_code` behind the executor contract, remove key copying/environment inheritance, implement limits/cancellation and the model-API broker, and protect the evaluator boundary. Fail closed without isolation. Deliver the parser/fetch/import limits required for supported inputs.

**PR 4 — durable runs and budgets.** Add submission intent, attempt journal, atomic artifacts, reservations, reconciliation, resume, and honest status/error outputs. Gate A passes G01–G02 and the applicable G05–G10/G17–G19 tests before expanding execution reach.

### Gate B Portable runtimes and other paradigms

**PR 5 — local and compatible serving.** Add Ollama and a compatible-server adapter, capability probes, local resource preflight, controlled offline serving, and evidence-aware context budgeting. Preserve the existing cloud adapters. Pass G03–G05/G20 for named tested profiles.

**PR 6 — method plugins and task bundles.** Add field-allowlisted candidate export, manual-spec operation, one deterministic classical/numerical example, and one symbolic/optimization example with an independent checker. Freeze plugin trust/installation rules. Pass G06/G15/G21/G22.

Gate B constitutes delivered local/cloud portability and delivered non-LLM coverage. Do not defer these capabilities to an unspecified future plugin ecosystem while advertising them as supported.

### Gate C Evidence based scientific evaluation

**PR 7 — plans, evidence and scoring.** Add task readiness, curator/candidate plan separation, deterministic metrics, critical rubric dependencies, clean re-execution, versioned conclusion profiles, and reproduction/transfer reports. Preserve legacy reader/field compatibility and document non-comparability of corrected scores.

**PR 8 — calibration and pilot release.** Curate reference procedures, run adversarial and failure-injection suites, calibrate semantic criteria, and publish backend support profiles, migration notes, and reproducibility exports. Pass G11–G16 and the full applicable release suite. Human-review requirements remain where semantic automation lacks sufficient calibration.

The first robust portable release requires Gates A, B, and C. Maintainers can release intermediate previews with narrower, explicit capability claims. Installation profiles should keep generation-only use lightweight and clearly state extra execution requirements.

### Gate D Optional advanced execution and service

Add training/checkpoint and simulation plugins with domain-specific acceptance suites, retrieval/multimodal/hybrid extensions, external agent harnesses, and multi-agent comparisons. Each records its own orchestration overhead, intervention, and resource accounting. Local serving integrations such as llama.cpp, MLX, Transformers, or ONNX may use native or compatible adapters only after conformance tests.

Remote workers, Kubernetes Jobs, Slurm, and distributed training implement executor adapters. Adopt durable distributed orchestration only when multi-worker recovery and scheduling requirements justify it; maintain the CLI/SDK contract. A team service then adds authenticated project access, artifact/log authorization, shared quotas, retention, audit trails, and operations. A browser UI is a client of these APIs, not a second controller.

## 19 Technical decisions and remaining choices

Adopt these defaults for implementation planning:

- Extend the existing package and its MIT distribution; keep user-owned outputs separate from upstream changes.
- Start with one local controller, SQLite, and content-addressed files. Use Linux isolated CPU workers first, a separately tested GPU profile, and macOS as a controller/local-model host.
- Keep all provider choices explicit in new configuration; preserve historical generation defaults only for legacy calls. Add local and compatible adapters before more hosted providers.
- Require curator review for new scientific rubrics and unresolved critical methods. Allow fully automatic processing of already validated templates/tasks within their policy.
- Keep search/exploration budgets explicit and distinguish from-scratch implementation, author-code re-execution, and adaptation.
- Deliver concrete non-LLM examples before claiming paradigm coverage. Use manual specifications for papers outside validated extraction archetypes.

Maintainers must select the supported Python/OS matrix, named local models and hardware profiles, the first pilot papers, judge-calibration error bounds, default resource/retention limits, and which data classifications may leave a machine. These choices depend on deployment capacity and task owners; they do not require redesigning the contracts. Record them as versioned release policy with owners and rationale.

Avoid promising universal paper coverage, identical outcomes across hardware or mutable APIs, complete contamination detection, exact external invoices, or automated proof of scientific correctness. The product's reliable claim is a traceable, controlled procedure with explicit evidence and limits.

## 20 Definition of done

The extension is ready when a maintainer can take an existing Paper2Bench example, generate its compatible artifacts through a hosted provider or an entirely local profile, inspect a source-linked task specification, execute it within the advertised isolation profile, and export a result whose claims point to trusted measurements. A manual non-LLM task must use the same lifecycle and evidence contracts without requiring a foundation model.

The release includes code and schemas, compatibility migrations, example profiles, method examples, a tested support matrix, an adversarial/failure-injection suite, scientific evaluation procedures, and a reproducibility guide. Interrupted, blocked, inconclusive, contradictory, and cancelled runs remain understandable and recoverable. Every advertised capability corresponds to an exercised path; every scientific pass identifies the protocol and evidence that support it.
