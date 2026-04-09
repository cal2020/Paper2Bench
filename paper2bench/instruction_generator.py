"""Generate instruction.txt from a paper PDF via YAML extraction + template rendering."""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional

import yaml
import openai
from dotenv import load_dotenv

from .pdf_utils import extract_text_from_pdf

load_dotenv()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EXTRACT_YAML_PROMPT = """\
You are an expert research engineer who specializes in designing benchmark
experiments for evaluating large language models.

You will be given:
1. A research question (provided separately).
2. The full text of an academic paper that investigates this question.
3. Optionally, a structured research-problem tree derived from the paper.

Your task is to extract the key experimental parameters from the paper and
output them as a YAML document that can be used to generate a standardized
benchmark instruction file.

IMPORTANT: The instruction file is given to an AI agent that must DESIGN ITS
OWN experiments. You must extract ONLY the resources available to the agent
(models, datasets, budget). Do NOT extract experimental procedures, evaluation
metrics, prompt templates, or any methodological details -- the agent must
figure those out independently. This is critical for benchmark integrity.

Output ONLY valid YAML (no preamble, no markdown fences, no commentary)
with exactly this schema:

```
task_id: <snake_case identifier>
research_question: <the provided research question, verbatim>
models:
  api:          # models available via API (use exact API model IDs)
    - gpt-4o
  huggingface:  # models to load via HuggingFace (use exact repo IDs)
    - meta-llama/Llama-3.1-8B-Instruct
api_key_var: OPENAI_API_KEY   # or ANTHROPIC_API_KEY, GOOGLE_API_KEY
budget_per_model: 1000        # realistic number based on paper's scale
datasets:
  - name: "<dataset name>"
    source: huggingface        # huggingface | local | synthetic
    loader: |
      <Python code snippet to load the dataset>
    description: "<brief factual description -- NO hints about experimental use>"
constraints:
  - "Do NOT use web search"
  - "Run FULL end-to-end experiments"
```

Rules:
1. Use EXACT model API IDs (e.g., "gpt-4o" not "GPT-4o").
2. For datasets, prefer HuggingFace dataset IDs when publicly available.
   Use "synthetic" for datasets generated programmatically.
3. Set budget_per_model based on the paper's experiment scale (500-5000).
4. Do NOT include conclusions, results, or findings from the paper.
5. Do NOT invent models or datasets not in the paper.
6. Do NOT include evaluation metrics, prompt templates, or experimental procedures.
7. Dataset descriptions should only say WHAT the data is, not HOW to use it.
8. Always include "Do NOT use web search" and "Run FULL end-to-end experiments"
   in constraints.
"""


# ---------------------------------------------------------------------------
# Default template path
# ---------------------------------------------------------------------------

_DEFAULT_TEMPLATE = Path(__file__).parent / "templates" / "default.txt"


# ---------------------------------------------------------------------------
# YAML extraction
# ---------------------------------------------------------------------------

def extract_yaml_from_pdf(
    pdf_path: str,
    research_question: str,
    task_id: str,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    tree_json_path: Optional[str] = None,
) -> str:
    """Read a paper PDF and extract structured YAML config via LLM."""
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be set in the environment or passed directly")

    client = openai.OpenAI(api_key=api_key)
    paper_text = extract_text_from_pdf(pdf_path)
    if not paper_text.strip():
        raise ValueError(f"No text could be extracted from {pdf_path}")

    user_message = (
        f"TASK ID: {task_id}\n\n"
        f"RESEARCH QUESTION: {research_question}\n\n"
        "---\n\n"
        "FULL PAPER TEXT:\n\n"
        f"{paper_text}"
    )

    if tree_json_path:
        tree_path = Path(tree_json_path)
        if tree_path.exists():
            with open(tree_path, encoding="utf-8") as f:
                tree = json.load(f)
            user_message += (
                "\n\n---\n\n"
                "RESEARCH-PROBLEM TREE:\n\n"
                f"```json\n{json.dumps(tree, indent=2)}\n```"
            )
        else:
            print(f"[warn] Tree JSON not found at {tree_json_path}, skipping.")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": EXTRACT_YAML_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    raw_output = response.choices[0].message.content

    usage = response.usage
    print(
        f"Tokens  --  input: {usage.prompt_tokens}  |  "
        f"output: {usage.completion_tokens}  |  "
        f"total: {usage.total_tokens}"
    )

    # Strip markdown fences if present
    cleaned = re.sub(r"^```(?:ya?ml)?\s*\n", "", raw_output, flags=re.MULTILINE)
    cleaned = re.sub(r"\n```\s*$", "", cleaned, flags=re.MULTILINE)

    return cleaned.strip()


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

def render_instruction(config: dict, template_path: Optional[str] = None) -> str:
    """Render a task_config dict into an instruction.txt string using a template."""

    # Load template
    tpl_path = Path(template_path) if template_path else _DEFAULT_TEMPLATE
    template = tpl_path.read_text(encoding="utf-8")

    # Models section
    models_lines = []
    api_models = config.get("models", {}).get("api", [])
    hf_models = config.get("models", {}).get("huggingface", [])
    if api_models:
        models_lines.append("- " + " and ".join(api_models) + " via API")
    for hf in hf_models:
        models_lines.append(f"- Load with HuggingFace: {hf}")
    models_section = "\n".join(models_lines)

    # Datasets section
    datasets_lines = []
    for ds in config.get("datasets", []):
        desc = ds.get("description", ds.get("name", ""))
        loader = ds.get("loader", "").strip()
        if ds.get("source") == "synthetic":
            datasets_lines.append(f"- {desc}")
        elif loader:
            datasets_lines.append(f"- {ds.get('name', 'Dataset')}: {desc}")
            datasets_lines.append("  Loading code:")
            for line in loader.split("\n"):
                datasets_lines.append(f"    {line}")
        else:
            datasets_lines.append(f"- {desc}")
    datasets_section = "\n".join(datasets_lines)

    # Constraints section
    constraints = config.get("constraints", [
        "You should **NOT** use any web search tool or try to answer the question "
        "based on your prior knowledge, but run **FULL** end to end experiment and "
        "draw conclusion from your experiment results."
    ])
    constraints_section = "\n".join(f"- {c}" for c in constraints)

    return template.format(
        research_question=config["research_question"],
        models_section=models_section,
        budget_per_model=config.get("budget_per_model", 1000),
        datasets_section=datasets_section,
        constraints_section=constraints_section,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate instruction.txt: 'extract' (PDF -> YAML) or 'render' (YAML -> txt)."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # --- extract mode ---
    ext = subparsers.add_parser("extract", help="Extract task config YAML from a paper PDF")
    ext.add_argument("--pdf", required=True, help="Path to the paper PDF")
    ext.add_argument("--research-question", required=True, help="The research question")
    ext.add_argument("--task-id", required=True, help="Task ID (snake_case)")
    ext.add_argument("--tree", default=None, help="Optional problem tree JSON")
    ext.add_argument("--model", default="gpt-4o", help="OpenAI model (default: gpt-4o)")
    ext.add_argument("--max-tokens", type=int, default=4096)
    ext.add_argument("--temperature", type=float, default=0.0)
    ext.add_argument("-o", "--output", required=True, help="Output YAML path")

    # --- render mode ---
    rnd = subparsers.add_parser("render", help="Render YAML config into instruction.txt")
    rnd.add_argument("--config", required=True, help="Path to task_config.yaml")
    rnd.add_argument("--template", default=None, help="Custom template file (default: built-in)")
    rnd.add_argument("-o", "--output", required=True, help="Output instruction.txt path")

    args = parser.parse_args()

    if args.mode == "extract":
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        yaml_content = extract_yaml_from_pdf(
            str(pdf_path),
            research_question=args.research_question,
            task_id=args.task_id,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            tree_json_path=args.tree,
        )

        try:
            parsed = yaml.safe_load(yaml_content)
            if not isinstance(parsed, dict):
                raise ValueError("YAML did not parse to a dict")
        except Exception as e:
            print(f"[warn] LLM output may not be valid YAML: {e}")
            print("[warn] Saving raw output for manual editing.")

        output_path.write_text(yaml_content, encoding="utf-8")
        print(f"\nConfig saved to {output_path}")

    elif args.mode == "render":
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        instruction = render_instruction(config, template_path=getattr(args, "template", None))
        output_path.write_text(instruction, encoding="utf-8")
        print(f"instruction.txt saved to {output_path}")


if __name__ == "__main__":
    main()
