# Paper2Bench

Convert academic papers into benchmark tasks for evaluating AI coding agents.

## Pipeline

```
Paper title/author
  → paper2bench download  → PDF
  → paper2bench parse     → problem tree JSON
  → paper2bench extract   → task config YAML
  → paper2bench render    → instruction.txt
  → paper2bench plan      → instruction_gt.txt (ground truth)
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
```

### Custom templates

The default instruction template is generic. Pass `--template` to use your own:

```bash
paper2bench render --config task.yaml --template my_framework_template.txt -o instruction.txt
```

## Output Files

| File | Purpose |
|------|---------|
| `<task_id>.pdf` | Downloaded paper |
| `<task_id>_tree.json` | Structured problem tree (Root → Research Questions → Experiments) |
| `task_config.yaml` | Extracted models, datasets, budget, constraints |
| `instruction.txt` | What the AI agent sees — research question + resources |
| `instruction_gt.txt` | Ground truth — detailed experimental procedures for evaluation |

## How It Works

1. **Download**: Searches arXiv API by title/author, downloads PDF
2. **Parse**: LLM reads the PDF and decomposes it into a 3-level problem tree
3. **Extract**: LLM extracts models, datasets, and budget into structured YAML — no experimental procedures (the agent must design those independently)
4. **Render**: Template engine converts YAML into standardized instruction.txt
5. **Plan**: LLM reads the paper again and writes detailed experimental procedures as a supplementary plan, creating instruction_gt.txt for evaluation
