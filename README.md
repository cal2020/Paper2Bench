# Paper2Bench

This checkout contains the **0.4.0a2 implementation** extending upstream
Paper2Bench: bounded local/hosted text runtimes, citation-grounded source evidence,
durable generation checkpoints, isolated Docker/SSH workers with explicit NVIDIA
GPU selection, an opt-in target-model file broker, and reviewed fixed-protocol
scientific execution. Start with [IMPLEMENTATION.md](IMPLEMENTATION.md) for usable
commands, verification evidence, and capability limits. The updated
[interactive demo](https://paper2bench-portable-demo.escape4242.chatgpt.site) is
publicly accessible without signing in; its build is documented in
[deployment/demo](deployment/demo/README.md). Actual local Llama
inference and a Docker → model broker → Llama → Docker round trip passed;
external SSH, NVIDIA hardware, and hosted model accounts remain untested here.

The public demo now includes a [recorded published-paper experiment](https://paper2bench-portable-demo.escape4242.chatgpt.site/#real-paper):
the Appendix A counterexample from *On the Convergence of Adam and Beyond*
(ICLR 2018). Four actual local Qwen authoring calls produced reviewable drafts.
Source review found incorrect generated steps and metrics; a separately curated,
digest-bound implementation ran 30,000 steps in each of three fresh isolated
containers. Adam average regret was **0.667180**, versus **0.052132** for AMSGrad.
This is an assisted finite check, not autonomous full-paper reproduction. The
[task and correction audit](examples/papers/adam-counterexample/README.md),
[public receipt](docs/results/adam-counterexample.json), and
[raw trajectories](docs/results/adam-predictions.json) are committed together.

The extension is implemented in this checkout, not in the upstream release tags
linked below. The original generation workflows remain available. Legacy
conclusion scoring remains separate from method fidelity and scientific
agreement; a generated reference draft or matching conclusion is not a
paper-reproduction result.

The optional [reproducibility runtime](docs/REPRODUCIBILITY.md) adds frozen
experiment identities, independently scored classification and knapsack tasks,
scoped remote workers with durable completion replay, container-enforced
deadlines and PDF parsing, shared model quotas, and multi-seed study bundles.
It is isolated under `paper2bench.repro` and leaves the original CLI unchanged.
Use Python 3.11+ and install `.[repro]` (or `.[repro-api]` for its control API),
then run `paper2bench-repro demo --task knapsack`.

The [demo workbench source](docs/site-app.jsx)
explains these boundaries and links to [selected recorded software evidence](docs/repro/README.txt).
Those synthetic fixtures are separate from the Adam paper experiment above.
[Control Center assets](docs/CONTROL_CENTER.md) include an importable protected
verification workflow and capsule; saving them does not execute verification.
The [integrated release checks](docs/integration-verification.json) record the
fresh local framework/runtime and desktop/mobile verification separately from
the historical receipts displayed in the demo.

Convert academic papers into benchmark tasks for evaluating AI coding agents.

Paper2Bench offers two complementary workflows:

1. **Core pipeline** — a reproducible task per paper: extract what the paper used (models, datasets, budget), hand it to an agent, and score the agent's conclusion against the paper's.
2. **Benchmark-variant generator** — turn a paper into a *family* of new research problems (perturbations, ablations, future-work tasks) that test whether an agent can transfer the paper's logic to new settings.

Both workflows auto-detect the paper's archetype (`llm_evaluation` / `novel_architecture` / `empirical_study`) and route extraction through a matching prompt and template — so papers like CGCNN or SchNet don't get shoehorned into an LLM-evaluation schema.

---

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Pipelines at a glance](#pipelines-at-a-glance)
- [Commands](#commands)
- [Usage guide](#usage-guide)
  - [Core pipeline, one command](#core-pipeline-one-command)
  - [Core pipeline, step by step](#core-pipeline-step-by-step)
  - [Paper types and auto-classification](#paper-types-and-auto-classification)
  - [Verification (F1 gate)](#verification-f1-gate)
  - [Splitting papers with multiple research questions](#splitting-papers-with-multiple-research-questions)
  - [Benchmark-variant generator](#benchmark-variant-generator)
  - [Custom templates](#custom-templates)
- [Worked examples](#worked-examples)
- [Output files](#output-files)
- [How it works](#how-it-works)

---

## Install

Install this implementation from the public GitHub repository:

```bash
gh repo clone cal2020/Paper2Bench
cd Paper2Bench
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m paper2bench doctor
```

The remaining upstream installation examples describe the original project and
do **not** install this checkout's 0.4.0a2 changes:

```bash
git clone https://github.com/AbhayAnandUCSD/Paper2Bench.git
cd Paper2Bench
pip install -e .
```

Or install directly from GitHub:

```bash
pip install git+https://github.com/AbhayAnandUCSD/Paper2Bench.git@v0.3.0
```

If you want to use Claude models, install the optional `anthropic` extra:

```bash
pip install "paper2bench[anthropic] @ git+https://github.com/AbhayAnandUCSD/Paper2Bench.git@v0.3.0"
```

Requires Python 3.10+. Hosted models require an appropriate exported credential;
local runtime profiles need no hosted-provider key. Automatic `.env` loading is
disabled in this extension. Export variables explicitly, or load them in trusted
caller code before invocation:

```bash
export OPENAI_API_KEY=sk-...           # required for gpt-* / o-* models
export ANTHROPIC_API_KEY=sk-ant-...    # required for claude-* models
```

Without a runtime profile, legacy **provider selection** uses the model name:

- `gpt-*`, `o1-*`, `o3-*`, `o4-*` → OpenAI
- `claude-*`, `anthropic/claude-*` → Anthropic

The legacy default model is `gpt-4o`; choose a provider/model available to your
account. Each stage accepting `--model` (or `--agent-model` / `--eval-model`) can
select a legacy provider independently. Runtime profiles instead specify every
role's provider, model, endpoint, credential reference, and durable budget; their
bindings take precedence over legacy model defaults. The extension's HTTP text
adapters do not require the optional Anthropic SDK extra.

> **Note on Claude + `--split-rqs`**: a single paper's `run` makes ~6 LLM calls per RQ, each with ~45k input tokens. Lower Anthropic tiers (30k input-tokens/min) will hit rate limits on papers with many RQs. Workarounds: run one RQ at a time via `--research-question "..."` instead of `--split-rqs`, or use OpenAI for batch runs.

## Python SDK

The pipeline is also importable as a library — useful when you want to feed Paper2Bench programmatically rather than via the CLI:

```python
import paper2bench

result = paper2bench.run(
    pdf="paper.pdf",
    research_question="Does X improve Y?",
    api_key="sk-...",                  # or rely on OPENAI_API_KEY / ANTHROPIC_API_KEY env vars
    output_dir="./output",
)

print(result.paper_type)               # auto-classified, or pass paper_type=... to override
print(result.instruction_path)         # ./output/tasks/<task_id>/instruction.txt
print(result.instruction_gt_path)      # ground-truth experimental plan
print(result.config)                   # parsed task_config.yaml as dict
```

Stage functions (`parse_paper_to_tree`, `classify_paper`, `extract_yaml_from_pdf`, `render_instruction`, `generate_supplementary_plan`, `build_instruction_gt`, `verify_instruction_gt`, `extract_paper_spec`, `generate_variants`, `chat`) are also importable from `paper2bench` for power users who want to compose the pipeline manually.

---

## Quick start

One-shot benchmark from an arXiv query:

```bash
paper2bench run \
  --task-id reversal_curse \
  --query "The Reversal Curse Berglund" \
  --research-question "If an LLM is fine-tuned on 'A is B,' does it learn 'B is A'?" \
  --output-dir ./output/ \
  --auto
```

Generate a family of variant benchmarks from an existing PDF:

```bash
paper2bench specextract --pdf paper.pdf -o spec.json
paper2bench generate    --spec spec.json --pdf paper.pdf -o ./variants/
```

---

## Pipelines at a glance

**Core pipeline** — paper → single benchmark task:

```
Paper title/author
  → paper2bench download   → PDF
  → paper2bench parse      → problem tree JSON
  → paper2bench classify   → paper_type (auto)
  → paper2bench extract    → task_config.yaml
  → paper2bench render     → instruction.txt
  → paper2bench plan       → instruction_gt.txt   (ground-truth plan)
  → paper2bench verify     → F1 check             (optional)
```

**Benchmark-variant generator** — paper → family of variant tasks:

```
PDF
  → paper2bench specextract → paper_spec.json  (7-component spec)
  → paper2bench generate    → variants/<id>/{instruction.txt, task_config.yaml, metadata.json}
```

The two workflows are independent — you can use either on its own.

---

## Commands

| Command | Purpose |
|---------|---------|
| `download` | Search arXiv and download a paper PDF |
| `parse` | Parse a PDF into a 3-level problem tree |
| `classify` | Classify a paper as `llm_evaluation` / `novel_architecture` / `empirical_study` |
| `extract` | Extract task config YAML (paper-type aware) |
| `render` | Render YAML into `instruction.txt` |
| `plan` | Build `instruction_gt.txt` from instruction + PDF |
| `verify` | Run an agent with `instruction_gt` and check F1 ≥ 80 |
| `run` | Core pipeline end-to-end |
| `specextract` | Extract 7-component paper specification |
| `generate` | Generate benchmark variant instances from a paper spec |

---

## Usage guide

### Core pipeline, one command

```bash
paper2bench run \
  --task-id my_task \
  --query "Paper title and authors" \
  --research-question "The question the agent will try to answer" \
  --output-dir ./output/ \
  --auto
```

Useful `run` flags:

| Flag | Effect |
|------|--------|
| `--pdf PATH` | Use a local PDF instead of searching arXiv |
| `--paper-type TYPE` | Skip auto-classification and force `llm_evaluation` / `novel_architecture` / `empirical_study` |
| `--skip-hf-validation` | Skip the HuggingFace-Hub existence check on dataset loaders |
| `--split-rqs` | Emit a separate benchmark task per research question in the parsed tree (`--research-question` becomes optional) |
| `--verify` | After generating `instruction_gt.txt`, run an agent with it and check F1 ≥ 80 |
| `--template PATH` | Use a custom instruction template |
| `--model MODEL` | Change the LLM (default `gpt-4o`) |

### Core pipeline, step by step

Run any stage on its own — each writes a self-contained artifact.

```bash
# 1. Download from arXiv
paper2bench download "Paper title and authors" --task-id my_task -o ./papers/

# 2. Parse into problem tree
paper2bench parse ./papers/my_task.pdf -o ./trees/my_task_tree.json

# 3. (optional) Classify standalone
paper2bench classify --pdf ./papers/my_task.pdf -o ./tasks/my_task/classification.json

# 4. Extract task config (paper-type aware)
paper2bench extract \
  --pdf ./papers/my_task.pdf \
  --research-question "The question" \
  --task-id my_task \
  --tree ./trees/my_task_tree.json \
  --paper-type llm_evaluation \
  -o ./tasks/my_task/task_config.yaml

# 5. Render instruction.txt
paper2bench render \
  --config ./tasks/my_task/task_config.yaml \
  -o ./tasks/my_task/instruction.txt

# 6. Generate ground-truth plan
paper2bench plan \
  --instruction ./tasks/my_task/instruction.txt \
  --pdf ./papers/my_task.pdf \
  --tree ./trees/my_task_tree.json \
  -o ./tasks/my_task/instruction_gt.txt

# 7. (optional) Verify clarity
paper2bench verify \
  --instruction-gt ./tasks/my_task/instruction_gt.txt \
  --pdf ./papers/my_task.pdf \
  -o ./tasks/my_task/verify_results.json
```

### Paper types and auto-classification

Every paper is auto-classified into one of three archetypes; the type drives both the **extraction prompt** and the **instruction template**. Pass `--paper-type` anywhere to override.

| Type | When it fires | Schema highlights | Template |
|------|---------------|-------------------|----------|
| `llm_evaluation` | Paper evaluates existing models on a task (e.g. Lost in the Middle, Chain-of-Thought) | `models.api`, `models.huggingface`, `datasets` | `default.txt` |
| `novel_architecture` | Paper introduces a new model / method (e.g. CGCNN, SchNet, Transformer) | `proposed_method`, `baselines`, `reference_implementation` | `novel_architecture.txt` |
| `empirical_study` | Observational / meta-study, no new model | `study_type`, `data_sources`, `analytical_tools` | `empirical_study.txt` |

All three schemas also include:

- **`references`** — author-year citations get their own field so they aren't mis-extracted as fake synthetic datasets.
- **HuggingFace loader validation** — any `source: huggingface` loader is checked against the Hub; missing repos are demoted to `source: unknown` (bypass with `--skip-hf-validation`).

### Verification (F1 gate)

`paper2bench verify` runs a coding agent with `instruction_gt.txt`, extracts its final conclusion, decomposes both the conclusion and a paper-derived reference answer into atomic claims, and computes claim-level **precision / recall / F1**. The task passes if **F1 ≥ 80**.

Verification is opt-in and not suitable for papers whose experiments can't be executed in a Python sandbox.

```bash
paper2bench verify \
  --instruction-gt ./tasks/my_task/instruction_gt.txt \
  --pdf ./papers/my_paper.pdf \
  --data-dir ./tasks/my_task/data \
  --agent-model gpt-4o \
  --eval-model gpt-4o
```

### Splitting papers with multiple research questions

Pass `--split-rqs` on `run` to emit a separate benchmark task per research question found in the parsed problem tree. Each sub-task lands in `tasks/<task_id>_rqN/` with its own config, instruction, and instruction_gt.

```bash
paper2bench run --task-id my_task --pdf paper.pdf --split-rqs
```

### Benchmark-variant generator

For evaluating *transfer* rather than reproduction, `specextract` + `generate` turn a paper into a family of new research problems:

- **Perturbations** — new dataset, tighter budget, different metric, shifted domain
- **Ablation-derived** — does the key component still matter under a shift? Which fails first?
- **Future-work-derived** — bounded instantiations of the paper's stated limitations

Each variant includes an automated leakage check (direct-answer / method / statistical leak) so instructions that accidentally reveal the paper's answer get flagged.

```bash
# Extract the 7-component paper specification
paper2bench specextract --pdf paper.pdf -o spec.json

# Generate a family of variants (auto-classifies paper type; override with --paper-type)
paper2bench generate --spec spec.json --pdf paper.pdf -o ./variants/
```

Each variant directory contains:

- `instruction.txt` — a detailed research instruction, **same format as the core pipeline's instruction.txt** (research question + models/method + datasets with loaders + budget + constraints), rendered through the paper-type-aware template.
- `task_config.yaml` — the resource spec that was rendered into the instruction.
- `metadata.json` — transformation `type`, `difficulty`, rationale, curator-only expected outcome hint, and leakage-check verdict.

Because the generator reuses the paper-type branching, variants of a materials-science paper tell the agent "implement the method yourself" while variants of an LLM-evaluation paper list concrete API model IDs and HuggingFace loaders.

### Custom templates

Pass `--template` to `render` (or `run`) to use your own template:

```bash
paper2bench render --config task.yaml --template my_template.txt -o instruction.txt
```

Template placeholders: `{research_question}`, `{models_section}`, `{datasets_section}`, `{budget_per_model}`, `{constraints_section}`.

---

## Worked examples

Three papers run end-to-end through the core pipeline, showing how the classifier routes each one to a different schema. Each example shows the full rendered `instruction.txt` (what the agent actually sees) and the extracted `task_config.yaml` that produced it.

---

### Lost in the Middle (NLP / LLM)

Classified `llm_evaluation` (high confidence). The LLM-evaluation template lists concrete API + HuggingFace models, dataset loaders, and budget — the agent is expected to evaluate these models, not implement a new one.

<details>
<summary><b>Rendered <code>instruction.txt</code></b> (what the agent sees)</summary>

```
You are a research agent. Conduct research and experiment about the question: "How does model performance vary based on relevant information position in context?"

You have access to the following resources:

Models:
- gpt-3.5-turbo-0613 and gpt-3.5-turbo-16k-0613 and claude-1.3 and claude-1.3-100k via API
- Load with HuggingFace: mpt-30b-instruct
- Load with HuggingFace: longchat-13b-16k
- Load with HuggingFace: flan-t5-xxl
- Load with HuggingFace: flan-ul2
- Load with HuggingFace: Llama-2-7b-chat-hf
- Load with HuggingFace: Llama-2-13b-chat-hf
- Load with HuggingFace: Llama-2-70b-chat-hf
- Load with HuggingFace: Llama-2-7b-hf
- Load with HuggingFace: Llama-2-13b-hf
- Load with HuggingFace: Llama-2-70b-hf
- Computational budget: 1000 API calls per model

Datasets:
- NaturalQuestions-Open: A dataset containing historical queries issued to the Google search engine, coupled with human-annotated answers extracted from Wikipedia.  [loader unverified -- locate the data yourself]
- Synthetic JSON-formatted key-value pairs: A synthetic dataset of JSON-formatted key-value pairs with unique, randomly-generated UUIDs as keys and values.  [generate programmatically]
  Generation code:
    import uuid
    def generate_synthetic_kv_pairs(num_pairs):
        return {str(uuid.uuid4()): str(uuid.uuid4()) for _ in range(num_pairs)}
- Referenced but not loaded: Lee et al. 2019
- Referenced but not loaded: Kwiatkowski et al. 2019

Experimental constraints:
- Do NOT use web search
- Run FULL end-to-end experiments

Please design and execute experiments to investigate this research question. Document your experimental plan, run end-to-end experiments, and provide conclusions at different levels of detail.
```

</details>

<details>
<summary><b>Extracted <code>task_config.yaml</code></b></summary>

```yaml
paper_type: llm_evaluation
models:
  api:         [gpt-3.5-turbo-0613, gpt-3.5-turbo-16k-0613, claude-1.3, claude-1.3-100k]
  huggingface: [mpt-30b-instruct, longchat-13b-16k, flan-t5-xxl, flan-ul2,
                Llama-2-{7b,13b,70b}-{hf,chat-hf}]
datasets:
  - name: NaturalQuestions-Open
    source: unknown                          # demoted — no canonical HF path
  - name: Synthetic JSON key-value pairs
    source: synthetic
    loader: |
      import uuid
      def generate_synthetic_kv_pairs(n):
          return {str(uuid.uuid4()): str(uuid.uuid4()) for _ in range(n)}
references: [Lee et al. 2019, Kwiatkowski et al. 2019]
```

</details>

---

### CGCNN (materials science)

Classified `novel_architecture` (high confidence). The template tells the agent to **implement the method from scratch** — not call a library that bundles the paper's code — and hands it the method's key components, baselines, and a reference repo hint.

<details>
<summary><b>Rendered <code>instruction.txt</code></b></summary>

```
You are a research agent. Conduct research about the question: "Can graph convolutional neural networks predict material properties directly from crystal structure?"

Your central task is to **implement a proposed method from scratch** and evaluate whether it answers the research question. Do NOT import a library that bundles the paper's exact code; the benchmark value comes from your own implementation.

Proposed method to implement:
- Crystal Graph Convolutional Neural Networks (CGCNN): CGCNN is a framework that represents crystal structures as graphs, where nodes represent atoms and edges represent bonds. A convolutional neural network is applied to these graphs to predict material properties directly from the crystal structure, achieving accuracy comparable to DFT calculations while providing interpretability by extracting contributions from local chemical environments.
  - Component: Crystal graph representation with nodes as atoms and edges as bonds
  - Component: Graph convolutional layers to update atom feature vectors
  - Component: Pooling layers to aggregate features into a crystal-level representation
  - Component: Fully-connected layers for property prediction
- Baseline for comparison: DFT calculations
- Reference hint: https://github.com/txie-93/cgcnn

Datasets for training and evaluation:
- Materials Project: A database of inorganic crystal structures and their properties, used for training and evaluating the CGCNN model.  [loader unverified -- locate the data yourself]
- Perovskite database: A dataset containing energy above hull data for perovskite crystals, used to demonstrate the interpretability of CGCNN.  [loader unverified -- locate the data yourself]
- Referenced but not loaded: Jain et al. 2013
- Referenced but not loaded: Kirklin et al. 2015
- Referenced but not loaded: De Jong et al. 2015

Experimental constraints:
- Do NOT use web search
- Run FULL end-to-end experiments
- Implement the proposed method yourself -- do not import it from a library that bundles the paper's exact code.
- Computational budget: 1000 training runs / evaluations total

Deliverables:
1. A working implementation of the proposed method in Python.
2. Training and evaluation runs on the specified datasets.
3. Quantitative results (metrics of your choice, justified against the research question).
4. A written conclusion tying the results back to the research question.

Design your own experimental protocol, metrics, and baselines. Run end-to-end experiments and report results honestly.
```

</details>

<details>
<summary><b>Extracted <code>task_config.yaml</code></b></summary>

```yaml
paper_type: novel_architecture
proposed_method:
  name: Crystal Graph Convolutional Neural Networks (CGCNN)
  summary: Represents crystal structures as graphs (atoms=nodes, bonds=edges);
    a graph CNN predicts material properties directly from structure.
  key_components:
    - Crystal graph representation
    - Graph convolutional layers
    - Pooling layer
    - Fully-connected prediction head
baselines:                 [DFT calculations]
reference_implementation:  [https://github.com/txie-93/cgcnn]
datasets:
  - {name: Materials Project,   source: unknown}
  - {name: Perovskite database, source: unknown}
references: [Jain et al. 2013, Kirklin et al. 2015, De Jong et al. 2015]
```

</details>

---

### SchNet (quantum chemistry)

Classified `novel_architecture` (high confidence). The HF-loader validator rejected all three claimed dataset paths (`qm9`, `md17`, `iso17`) as non-existent on the Hub — they were demoted to `source: unknown` so the agent knows to locate the data itself rather than waste time on hallucinated loaders.

<details>
<summary><b>Rendered <code>instruction.txt</code></b></summary>

```
You are a research agent. Conduct research about the question: "Can continuous-filter convolutional neural networks accurately model quantum-mechanical interactions in molecules?"

Your central task is to **implement a proposed method from scratch** and evaluate whether it answers the research question. Do NOT import a library that bundles the paper's exact code; the benchmark value comes from your own implementation.

Proposed method to implement:
- SchNet: SchNet is a deep learning architecture that uses continuous-filter convolutional layers to model quantum interactions in molecules. It respects essential quantum-chemical constraints, providing rotationally invariant energy predictions and rotationally equivariant force predictions. SchNet is designed to handle molecules with arbitrary atomic positions, ensuring a smooth potential energy surface and energy-conserving force fields.
  - Component: Continuous-filter convolutional layers
  - Component: Rotationally invariant energy prediction
  - Component: Rotationally equivariant force prediction
  - Component: Atom-wise layers and interaction blocks
- Baseline for comparison: Gradient-domain machine learning (GDML)
- Baseline for comparison: Deep tensor neural networks (DTNN)
- Baseline for comparison: enn-s2s

Datasets for training and evaluation:
- QM9: A benchmark dataset for predicting various molecular properties in equilibrium, consisting of approximately 130k organic molecules with up to 9 heavy atoms.  [loader unverified -- locate the data yourself]
- MD17: A collection of molecular dynamics simulations for small organic molecules, used for predicting energy-conserving force fields.  [loader unverified -- locate the data yourself]
- ISO17: A dataset consisting of short molecular dynamics trajectories of 129 isomers, used to evaluate the model's ability to represent complex potential energy surfaces with chemical and conformational changes.  [loader unverified -- locate the data yourself]
- Referenced but not loaded: Ramakrishnan et al. 2014

Experimental constraints:
- Do NOT use web search
- Run FULL end-to-end experiments
- Implement the proposed method yourself -- do not import it from a library that bundles the paper's exact code.
- Computational budget: 1000 training runs / evaluations total

Deliverables:
1. A working implementation of the proposed method in Python.
2. Training and evaluation runs on the specified datasets.
3. Quantitative results (metrics of your choice, justified against the research question).
4. A written conclusion tying the results back to the research question.

Design your own experimental protocol, metrics, and baselines. Run end-to-end experiments and report results honestly.
```

</details>

<details>
<summary><b>Extracted <code>task_config.yaml</code></b></summary>

```yaml
paper_type: novel_architecture
proposed_method:
  name: SchNet
  summary: Continuous-filter convolutional network modeling quantum interactions,
    with rotationally invariant energies and equivariant forces.
  key_components:
    - Continuous-filter convolutional layers
    - Rotationally invariant energy prediction
    - Rotationally equivariant force prediction
    - Atom-wise interaction blocks
baselines: [GDML, DTNN, enn-s2s]
datasets:
  - {name: QM9,   source: unknown}           # HF validator rejected bogus loader
  - {name: MD17,  source: unknown}
  - {name: ISO17, source: unknown}
references: [Ramakrishnan et al. 2014]
```

</details>

Before paper-type branching, CGCNN produced empty model lists and SchNet hallucinated `meta-llama/Llama-3.1-8B-Instruct` as a "model for the task" — neither paper has anything to do with LLMs.

---

### Example variant (from the benchmark generator)

Running `specextract` + `generate` on CGCNN produced six variants across perturbation / ablation / future-work. Here is one — a perturbation that changes the empirical setting while preserving the paper's core hypothesis. Note the detailed `instruction.txt` uses the same `novel_architecture` template as the core pipeline, so the agent is still told to "implement the method yourself":

<details>
<summary><b>Variant: <code>cgcnn_benchmark_elastic_properties</code></b> — perturbation, medium difficulty, leakage-clean</summary>

**`instruction.txt`**

```
You are a research agent. Conduct research about the question: "Can CGCNN improve prediction accuracy for elastic properties with an increased amount of training data?"

Your central task is to **implement a proposed method from scratch** and evaluate whether it answers the research question. Do NOT import a library that bundles the paper's exact code; the benchmark value comes from your own implementation.

Proposed method to implement:
- Crystal Graph Convolutional Neural Network (CGCNN): The CGCNN framework represents crystal structures as graphs where nodes are atoms and edges are bonds. A convolutional neural network is applied to these graphs to learn features that predict material properties. The model is trained using DFT-calculated data and can extract contributions from local chemical environments to global properties.
  - Component: Crystal Graph
  - Component: Convolutional Layers
  - Component: Pooling Layer
  - Component: Fully-Connected Layers
- Baseline for comparison: DFT calculations compared to experimental data
- Reference hint: https://github.com/txie-93/cgcnn

Datasets for training and evaluation:
- Extended Materials Project Database: A larger set of inorganic crystals with DFT-calculated elastic properties, including bulk and shear moduli.  [loader unverified -- locate the data yourself]
- Referenced but not loaded: Xie-Grossman-2018

Experimental constraints:
- Do NOT use web search
- Run FULL end-to-end experiments
- Implement the proposed method yourself -- do not import a library that bundles the paper's exact code.
- Computational budget: 1000 training runs / evaluations total

Deliverables:
1. A working implementation of the proposed method in Python.
2. Training and evaluation runs on the specified datasets.
3. Quantitative results (metrics of your choice, justified against the research question).
4. A written conclusion tying the results back to the research question.

Design your own experimental protocol, metrics, and baselines. Run end-to-end experiments and report results honestly.
```

**`metadata.json`** (summary)

```json
{
  "id": "cgcnn_benchmark_elastic_properties",
  "title": "Benchmarking CGCNN on Elastic Properties with Extended Dataset",
  "type": "perturbation",
  "difficulty": "medium",
  "scientific_question": "Can CGCNN improve prediction accuracy for elastic properties with an increased amount of training data?",
  "rationale": "The original paper noted higher errors for elastic properties due to limited data. Increasing the dataset size should test if the model's accuracy improves as hypothesized.",
  "expected_outcome_hint": "The CGCNN should show improved accuracy for elastic properties with the increased dataset size.",
  "leakage_check": { "leaked": false, "leakage_type": "none", "evidence": "" }
}
```

</details>

---

## Output files

**Core pipeline (`tasks/<task_id>/`):**

| File | Purpose |
|------|---------|
| `<task_id>.pdf` | Downloaded paper |
| `<task_id>_tree.json` | Structured problem tree (Root → Research Questions → Experiments) |
| `task_config.yaml` | Extracted resources — paper-type-aware schema |
| `instruction.txt` | What the AI agent sees — research question + resources |
| `instruction_gt.txt` | Ground truth — detailed experimental procedures |
| `verify_results.json` | Precision / recall / F1 / pass verdict (if `verify` ran) |

**Benchmark-variant generator (`variants/<paper>/`):**

| File | Purpose |
|------|---------|
| `<paper>.spec.json` | 7-component paper specification |
| `instances.json` | Full manifest of all generated variants |
| `<instance>/instruction.txt` | Detailed research instruction for the variant |
| `<instance>/task_config.yaml` | Resource spec rendered into the variant's instruction |
| `<instance>/metadata.json` | Variant metadata: type, difficulty, rationale, leakage check |

---

## How it works

### Core pipeline

1. **Download** — arXiv search by title/author with exponential backoff on HTTP 429.
2. **Parse** — LLM decomposes the paper into a 3-level problem tree (root → research questions → experiments).
3. **Classify** — a small LLM call over the paper's head tags it as `llm_evaluation`, `novel_architecture`, or `empirical_study`. Drives everything downstream.
4. **Extract** — one of three type-specific prompts pulls a structured resource spec (models / method / datasets / budget / constraints) into YAML. HuggingFace loaders are checked against the Hub; hallucinated IDs are demoted to `source: unknown`. Author-year citations go into a separate `references` field.
5. **Render** — a type-aware template converts the YAML into a standardized `instruction.txt`. Deliberately withholds the paper's methodology so the agent must design experiments.
6. **Plan** — LLM reads the paper again and writes detailed experimental procedures as a *supplementary* plan, combined with the base instruction to produce `instruction_gt.txt` (for evaluation, not for the agent).
7. **Verify** *(optional)* — runs a coding agent against `instruction_gt.txt`, then scores the agent's conclusion with claim-level precision / recall / F1 (adapted from FIRE-Bench's RAGChecker evaluator).

### Benchmark-variant generator

1. **specextract** — distills the paper into a 7-component specification: scientific question, method (summary + components), claims, evaluation protocol, assumptions, ablation structure, future work.
2. **generate** — given the spec, emits a JSON family of 6–12 instances across three transformation strategies (perturbation / ablation / future-work), each with a paper-type-matching `task_config`. Each config is rendered into a detailed `instruction.txt` via the same `render_instruction` function used by the core pipeline, then audited by an LLM judge for three leakage types.
