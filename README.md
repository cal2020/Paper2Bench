# Paper2Bench

Convert academic papers into benchmark tasks for evaluating AI coding agents.

## Pipeline

**Core pipeline** — paper → benchmark task:

```
Paper title/author
  → paper2bench download  → PDF
  → paper2bench parse     → problem tree JSON
  → paper2bench classify  → paper_type (llm_evaluation | novel_architecture | empirical_study)
  → paper2bench extract   → task config YAML
  → paper2bench render    → instruction.txt
  → paper2bench plan      → instruction_gt.txt (ground truth)
  → paper2bench verify    → F1 check (optional)
```

**Benchmark-generator components** (independent, research-grade):

```
PDF
  → paper2bench specextract → paper_spec.json (7-component spec)
  → paper2bench generate    → variant benchmark instances
```

## Installation

```bash
pip install -e .
```

Requires an OpenAI API key:
```bash
export OPENAI_API_KEY=sk-...
```

## Usage

### Full pipeline (one command)

```bash
paper2bench run \
  --task-id reversal_curse \
  --query "The Reversal Curse Berglund" \
  --research-question "If an LLM is fine-tuned on 'A is B,' does it learn 'B is A'?" \
  --output-dir ./output/ \
  --auto
```

Additional `run` flags:

| Flag | Effect |
|------|--------|
| `--split-rqs` | Generate a separate task per research question found in the tree (research-question flag becomes optional) |
| `--verify` | After generating `instruction_gt.txt`, run an agent with it and check that F1 ≥ 80 against a paper-derived reference answer |

### Step by step

```bash
# 1. Download paper from arXiv
paper2bench download "The Reversal Curse Berglund" --task-id reversal_curse -o ./papers/

# 2. Parse into problem tree
paper2bench parse ./papers/reversal_curse.pdf -o ./trees/reversal_curse_tree.json

# 3. Extract task config (LLM reads paper, outputs YAML)
paper2bench extract \
  --pdf ./papers/reversal_curse.pdf \
  --research-question "If an LLM is fine-tuned on 'A is B,' does it learn 'B is A'?" \
  --task-id reversal_curse \
  --tree ./trees/reversal_curse_tree.json \
  -o ./tasks/reversal_curse/task_config.yaml

# 4. Review and edit the YAML, then render instruction.txt
paper2bench render \
  --config ./tasks/reversal_curse/task_config.yaml \
  -o ./tasks/reversal_curse/instruction.txt

# 5. Generate ground truth (detailed experimental plan from paper)
paper2bench plan \
  --instruction ./tasks/reversal_curse/instruction.txt \
  --pdf ./papers/reversal_curse.pdf \
  --tree ./trees/reversal_curse_tree.json \
  -o ./tasks/reversal_curse/instruction_gt.txt

# 6. (optional) Verify the instruction_gt is clear enough for an agent
paper2bench verify \
  --instruction-gt ./tasks/reversal_curse/instruction_gt.txt \
  --pdf ./papers/reversal_curse.pdf \
  -o ./tasks/reversal_curse/verify_results.json
```

### Verification component

`paper2bench verify` runs a coding agent with `instruction_gt.txt`, extracts
its final conclusion, decomposes both the conclusion and a paper-derived
reference answer into atomic claims, and computes claim-level
**precision / recall / F1**. The instruction passes if F1 ≥ 80.

Skipped by default — opt in with `--verify` in the `run` pipeline, or invoke
the standalone command. Not suitable for papers whose experiments cannot be
executed in a Python sandbox.

```bash
paper2bench verify \
  --instruction-gt ./tasks/my_task/instruction_gt.txt \
  --pdf ./papers/my_paper.pdf \
  --data-dir ./tasks/my_task/data \
  --agent-model gpt-4o \
  --eval-model gpt-4o
```

### Research-question splitting

When a paper contains multiple research questions, pass `--split-rqs` to
generate a separate benchmark task per RQ. Each sub-task gets its own config,
instruction, and instruction_gt under `tasks/<task_id>_rqN/`.

```bash
paper2bench run --task-id my_task --pdf paper.pdf --split-rqs
```

### Benchmark-generator components (independent)

For research beyond straight reproduction, the `specextract` and `generate`
commands turn a paper into a **family of benchmark variants** — perturbations,
ablation-derived tasks, and future-work instances — with automated leakage
checks.

```bash
# Extract the 7-component paper specification
paper2bench specextract --pdf ./papers/my_paper.pdf -o ./specs/my_paper.json

# Generate a family of variant benchmark instances
paper2bench generate \
  --spec ./specs/my_paper.json \
  --pdf ./papers/my_paper.pdf \
  -o ./variants/my_paper/
```

Each generated instance includes:

- `instruction.txt` — a **full, detailed research instruction** in the same
  format as the core pipeline (research question + models/method + datasets
  with loaders + budget + constraints), rendered via the paper-type-aware
  template
- `task_config.yaml` — the resource spec that was rendered into the
  instruction (schema matches the detected paper type: `llm_evaluation` uses
  `models`/`datasets`; `novel_architecture` uses `proposed_method`/
  `baselines`/`reference_implementation`; `empirical_study` uses
  `data_sources`/`analytical_tools`)
- `metadata.json` — transformation type (perturbation / ablation /
  future_work), difficulty (easy / medium / hard), rationale, expected
  outcome hint (for curators), and the automated leakage-check verdict

The generator auto-classifies the paper unless `--paper-type` is passed, so
variants of a materials-science paper tell the agent "implement the method
yourself" while variants of an LLM-evaluation paper list concrete API model
IDs and HuggingFace loaders.

### Paper types

`run` auto-classifies each paper into one of three archetypes and routes it
through a type-specific extraction prompt and instruction template. Pass
`--paper-type <type>` to skip auto-classification, or run `paper2bench
classify` standalone.

| Type | When it fires | YAML schema highlights | Template |
|------|---------------|------------------------|----------|
| `llm_evaluation` | Paper evaluates existing models on a task | `models.api`, `models.huggingface`, `datasets` | `default.txt` |
| `novel_architecture` | Paper proposes a new model / method | `proposed_method`, `baselines`, `reference_implementation` | `novel_architecture.txt` |
| `empirical_study` | Observational / analytical / meta-study | `study_type`, `data_sources`, `analytical_tools` | `empirical_study.txt` |

All three schemas include a `references` field for author-year citations
(to prevent them being mis-extracted as fake synthetic datasets) and run
any claimed HuggingFace dataset loader through a Hub-existence check —
unverified paths are demoted to `source: unknown` so agents don't waste
time on hallucinated IDs. Pass `--skip-hf-validation` to skip the network
check.

### Worked examples

Three papers exercised end-to-end on the current pipeline (see
`test_output_v2/tasks/` for full artifacts):

**Lost in the Middle (AI / NLP)** — classified `llm_evaluation` (high confidence)
```yaml
paper_type: llm_evaluation
models:
  api: [gpt-3.5-turbo-0613, gpt-3.5-turbo-16k-0613, claude-1.3, claude-1.3-100k]
  huggingface: [mpt-30b-instruct, longchat-13b-16k, flan-t5-xxl, flan-ul2,
                Llama-2-7b-chat-hf, Llama-2-13b-chat-hf, Llama-2-70b-chat-hf, ...]
datasets:
  - name: NaturalQuestions-Open
    source: unknown                # demoted — no canonical HF path
  - name: Synthetic JSON key-value pairs
    source: synthetic
    loader: |
      import uuid
      def generate_synthetic_kv_pairs(num_pairs):
          return {str(uuid.uuid4()): str(uuid.uuid4()) for _ in range(num_pairs)}
references: [Lee et al. 2019, Kwiatkowski et al. 2019]
```

**CGCNN (materials science)** — classified `novel_architecture` (high confidence)
```yaml
paper_type: novel_architecture
proposed_method:
  name: Crystal Graph Convolutional Neural Networks (CGCNN)
  summary: Represents crystal structures as graphs with atoms as nodes and bonds
    as edges; a graph CNN predicts material properties directly from structure.
  key_components:
    - Crystal graph representation with nodes as atoms and edges as bonds
    - Graph convolutional layers to update atom feature vectors
    - Pooling layers to aggregate features into a crystal-level representation
    - Fully-connected layers for property prediction
baselines: [DFT calculations]
reference_implementation: [https://github.com/txie-93/cgcnn]
datasets:
  - name: Materials Project
    source: unknown
  - name: Perovskite database
    source: unknown
references: [Jain et al. 2013, Kirklin et al. 2015, De Jong et al. 2015]
```

**SchNet (quantum chemistry)** — classified `novel_architecture` (high confidence)
```yaml
paper_type: novel_architecture
proposed_method:
  name: SchNet
  summary: Deep learning architecture using continuous-filter convolutional layers
    to model quantum interactions, producing rotationally invariant energies and
    equivariant force predictions.
  key_components:
    - Continuous-filter convolutional layers
    - Rotationally invariant energy prediction
    - Rotationally equivariant force prediction
    - Atom-wise layers and interaction blocks
baselines: [Gradient-domain machine learning (GDML), Deep tensor neural networks (DTNN), enn-s2s]
datasets:
  - {name: QM9,   source: unknown}   # HF validator rejected bogus loader
  - {name: MD17,  source: unknown}
  - {name: ISO17, source: unknown}
references: [Ramakrishnan et al. 2014]
```

Before the classifier and branched prompts, CGCNN produced empty model lists
and SchNet hallucinated `meta-llama/Llama-3.1-8B-Instruct` — neither paper
has anything to do with LLMs.

### Custom templates

The default instruction template is generic. Pass `--template` to use your own:

```bash
paper2bench render --config task.yaml --template my_framework_template.txt -o instruction.txt
```

## Commands

| Command | Purpose |
|---------|---------|
| `download` | Search arXiv and download a paper PDF |
| `parse` | Parse a PDF into a 3-level problem tree |
| `classify` | Classify a paper as llm_evaluation / novel_architecture / empirical_study |
| `extract` | Extract task config YAML from a PDF (paper-type aware) |
| `render` | Render YAML config into instruction.txt |
| `plan` | Build instruction_gt.txt from instruction + PDF |
| `verify` | Run an agent with instruction_gt and check F1 ≥ 80 |
| `run` | Full pipeline in one step |
| `specextract` | Extract 7-component paper specification |
| `generate` | Generate benchmark variant instances from a paper spec |

## Output Files

| File | Purpose |
|------|---------|
| `<task_id>.pdf` | Downloaded paper |
| `<task_id>_tree.json` | Structured problem tree (Root → Research Questions → Experiments) |
| `task_config.yaml` | Extracted models, datasets, budget, constraints |
| `instruction.txt` | What the AI agent sees — research question + resources |
| `instruction_gt.txt` | Ground truth — detailed experimental procedures for evaluation |
| `verify_results.json` | Precision / recall / F1 / pass verdict from agent verification |
| `<paper>.spec.json` | 7-component paper specification |
| `variants/<id>/instruction.txt` | Detailed research instruction for the variant (same format as core pipeline) |
| `variants/<id>/task_config.yaml` | Resource spec rendered into the variant's instruction |
| `variants/<id>/metadata.json` | Variant metadata: type, difficulty, rationale, leakage check |
| `variants/instances.json` | Full manifest of all generated variants |

## How It Works

### Core pipeline
1. **Download**: Searches arXiv API by title/author, downloads PDF
2. **Parse**: LLM reads the PDF and decomposes it into a 3-level problem tree
3. **Extract**: LLM extracts models, datasets, and budget into structured YAML — no experimental procedures (the agent must design those independently)
4. **Render**: Template engine converts YAML into standardized instruction.txt
5. **Plan**: LLM reads the paper again and writes detailed experimental procedures as a supplementary plan, creating instruction_gt.txt for evaluation
6. **Verify** *(optional)*: Runs a coding agent against `instruction_gt.txt`, then scores the agent's conclusion with claim-level precision / recall / F1 (adapted from FIRE-Bench's RAGChecker evaluator)

### Benchmark-generator components
1. **specextract**: Distills the paper into a 7-component specification — scientific question, method, claims, evaluation protocol, assumptions, ablation structure, future work — separating transferable logic from original packaging
2. **generate**: Applies three logic-preserving transformations (task/constraint perturbations, ablation-derived tasks, future-work instantiations) to produce a family of benchmark variants. Each instance runs through an automated leakage check that flags instructions accidentally revealing the paper's conclusions
