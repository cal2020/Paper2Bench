"""Generate instruction.txt from a paper PDF via YAML extraction + template rendering."""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

from .llm import chat
from .pdf_utils import extract_text_from_pdf

load_dotenv()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Shared rules (injected into every extraction prompt)
# ---------------------------------------------------------------------------

_SHARED_RULES = """\
Shared rules for dataset / reference extraction:

- A **dataset** must have a proper-noun or acronym name (e.g., "QM9", "GSM8K",
  "MMLU", "NaturalQuestions", "Materials Project").  If the only mention of a
  data source is an author-year citation (e.g., "Riesselman et al. 2018"),
  record it under the `references:` list, NOT under `datasets`.
- For HuggingFace loaders, only emit a loader if you are confident the exact
  repo ID exists.  If unsure, set `source: unknown` and leave a textual
  description -- downstream validation will flag hallucinations.
- For programmatically-generated data, use `source: synthetic` and write a
  real generation snippet (not a placeholder like `pass` or `np.random.rand`).
- Never fabricate model IDs.  If the paper does not name a HuggingFace or API
  model, leave the list empty rather than inventing one.
- Do NOT include conclusions, results, metrics, prompt templates, or
  experimental procedures -- the agent must design those itself.
- Always include "Do NOT use web search" and "Run FULL end-to-end experiments"
  in `constraints`.
- Output ONLY valid YAML (no preamble, no markdown fences, no commentary).
"""


# ---------------------------------------------------------------------------
# Paper-type-specific prompts
# ---------------------------------------------------------------------------

EXTRACT_YAML_PROMPT_LLM_EVAL = """\
You are an expert benchmark engineer.  The paper you are given is an
**evaluation of existing language models or AI systems on a task**.  Your
job is to extract the resources the paper used to evaluate models, so that
an AI agent can design its own experiments answering the same research
question.

Output ONLY valid YAML with this schema:

```
task_id: <snake_case identifier>
paper_type: llm_evaluation
research_question: <the provided research question, verbatim>
models:
  api:          # models available via API (use exact API model IDs)
    - gpt-4o
  huggingface:  # models to load via HuggingFace (use exact repo IDs)
    - meta-llama/Llama-3.1-8B-Instruct
api_key_var: OPENAI_API_KEY
budget_per_model: 1000
datasets:
  - name: "<dataset name>"
    source: huggingface           # huggingface | local | synthetic | unknown
    loader: |
      <Python code snippet -- omit if source is 'unknown'>
    description: "<what the data is, not how to use it>"
references:
  - "<author-year citation for any mentioned but unverified data source>"
constraints:
  - "Do NOT use web search"
  - "Run FULL end-to-end experiments"
```

""" + _SHARED_RULES


EXTRACT_YAML_PROMPT_NOVEL_ARCH = """\
You are an expert benchmark engineer.  The paper you are given introduces a
**new model, method, or architecture** that must be implemented and trained.
Your job is to extract the resources needed to reproduce (or re-derive) the
paper's central contribution, so that an AI agent can design and run its own
experiments testing the same research question.

Output ONLY valid YAML with this schema:

```
task_id: <snake_case identifier>
paper_type: novel_architecture
research_question: <the provided research question, verbatim>
proposed_method:
  name: "<short name of the new architecture / method>"
  summary: "<one-paragraph description of what it is>"
  key_components:
    - "<component 1 -- the essential pieces an implementation must include>"
baselines:                       # existing methods the paper compares against
  - "<baseline name>"
reference_implementation:        # optional hints, empty if none
  - "<framework / library / GitHub repo mentioned by the paper, if any>"
datasets:
  - name: "<dataset name>"
    source: huggingface          # huggingface | local | synthetic | unknown
    loader: |
      <Python code snippet -- omit if source is 'unknown'>
    description: "<what the data is, not how to use it>"
references:
  - "<author-year citation for any mentioned but unverified data source>"
compute_hint: "<rough compute scale from the paper, e.g. 'single GPU, <24h'>"
constraints:
  - "Do NOT use web search"
  - "Run FULL end-to-end experiments"
  - "Implement the proposed method yourself -- do not import it from a library
    that bundles the paper's exact code."
```

""" + _SHARED_RULES


EXTRACT_YAML_PROMPT_EMPIRICAL = """\
You are an expert benchmark engineer.  The paper you are given is an
**observational or analytical study** -- it neither evaluates existing models
on a task nor proposes a new architecture.  Your job is to extract the
empirical resources the paper used, so that an AI agent can design and run
its own study answering the same research question.

Output ONLY valid YAML with this schema:

```
task_id: <snake_case identifier>
paper_type: empirical_study
research_question: <the provided research question, verbatim>
study_type: "<e.g. dataset audit, meta-analysis, training-data study>"
data_sources:                    # primary corpora / logs / measurement sources
  - name: "<source name>"
    source: huggingface          # huggingface | local | synthetic | unknown
    loader: |
      <Python code snippet -- omit if source is 'unknown'>
    description: "<what the data is>"
analytical_tools:                # libraries / techniques used, if any
  - "<tool name>"
references:
  - "<author-year citation for any mentioned but unverified data source>"
constraints:
  - "Do NOT use web search"
  - "Run FULL end-to-end experiments"
```

""" + _SHARED_RULES


# Backwards-compatible alias -- legacy callers get the LLM-eval prompt.
EXTRACT_YAML_PROMPT = EXTRACT_YAML_PROMPT_LLM_EVAL


_PROMPT_BY_TYPE = {
    "llm_evaluation": EXTRACT_YAML_PROMPT_LLM_EVAL,
    "novel_architecture": EXTRACT_YAML_PROMPT_NOVEL_ARCH,
    "empirical_study": EXTRACT_YAML_PROMPT_EMPIRICAL,
}


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
    paper_type: str = "llm_evaluation",
    validate_hf_loaders: bool = True,
) -> str:
    """Read a paper PDF and extract structured YAML config via LLM.

    ``paper_type`` selects the extraction prompt
    (``llm_evaluation`` | ``novel_architecture`` | ``empirical_study``).
    When ``validate_hf_loaders`` is True, the returned YAML is post-processed:
    any dataset whose claimed HuggingFace ID does not resolve against the
    Hub is demoted to ``source: unknown`` (and its loader stripped), so
    downstream agents know not to trust a hallucinated path.
    """
    prompt = _PROMPT_BY_TYPE.get(paper_type, EXTRACT_YAML_PROMPT_LLM_EVAL)

    paper_text = extract_text_from_pdf(pdf_path)
    if not paper_text.strip():
        raise ValueError(f"No text could be extracted from {pdf_path}")

    user_message = (
        f"TASK ID: {task_id}\n\n"
        f"PAPER TYPE: {paper_type}\n\n"
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

    response = chat(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        api_key=api_key,
    )
    raw_output = response.content
    print(
        f"Tokens  --  input: {response.input_tokens}  |  "
        f"output: {response.output_tokens}  |  "
        f"total: {response.total_tokens}  |  provider: {response.provider}"
    )

    # Strip markdown fences if present
    cleaned = re.sub(r"^```(?:ya?ml)?\s*\n", "", raw_output, flags=re.MULTILINE)
    cleaned = re.sub(r"\n```\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    if validate_hf_loaders:
        cleaned = _validate_hf_loaders(cleaned)

    return cleaned


# ---------------------------------------------------------------------------
# HF loader validation
# ---------------------------------------------------------------------------

def _validate_hf_loaders(yaml_text: str) -> str:
    """Demote any dataset claiming a HuggingFace source whose ID does not
    resolve on the Hub to ``source: unknown`` (and strip its loader).

    This catches hallucinations like ``salesforce/uniref90_bfd30`` that an
    agent would waste time trying to load.  Silently no-ops if the
    ``huggingface_hub`` package or network is unavailable.
    """
    try:
        import yaml as _yaml
        from huggingface_hub import HfApi
        from huggingface_hub.utils import RepositoryNotFoundError
    except ImportError:
        return yaml_text

    try:
        config = _yaml.safe_load(yaml_text)
    except Exception:
        return yaml_text
    if not isinstance(config, dict):
        return yaml_text

    # Both schemas use the same dataset shape under different keys.
    for key in ("datasets", "data_sources"):
        items = config.get(key, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("source") != "huggingface":
                continue
            loader = item.get("loader", "")
            m = re.search(r'load_dataset\(\s*["\']([^"\']+)["\']', loader)
            ds_id = m.group(1) if m else None
            if not ds_id:
                continue
            try:
                HfApi().dataset_info(ds_id)
            except RepositoryNotFoundError:
                print(f"[validate] HF dataset '{ds_id}' not found, "
                      "demoting to source: unknown")
                item["source"] = "unknown"
                item.pop("loader", None)
            except Exception as e:
                # Network errors etc. -- don't mutate the config on failure.
                print(f"[validate] skipped HF check for '{ds_id}': {e}")

    return _yaml.safe_dump(config, sort_keys=False)


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_TEMPLATE_BY_TYPE = {
    "llm_evaluation": _TEMPLATE_DIR / "default.txt",
    "novel_architecture": _TEMPLATE_DIR / "novel_architecture.txt",
    "empirical_study": _TEMPLATE_DIR / "empirical_study.txt",
}


def render_instruction(config: dict, template_path: Optional[str] = None) -> str:
    """Render a task_config dict into an instruction.txt string using a template.

    If ``template_path`` is not provided, the template is auto-selected based
    on ``config['paper_type']`` (falling back to the default LLM-eval template).
    """

    # --- Template selection ---
    if template_path:
        tpl_path = Path(template_path)
    else:
        paper_type = config.get("paper_type", "llm_evaluation")
        tpl_path = _TEMPLATE_BY_TYPE.get(paper_type, _DEFAULT_TEMPLATE)
    template = tpl_path.read_text(encoding="utf-8")

    # --- Models / method section (schema depends on paper type) ---
    models_lines = []
    if "proposed_method" in config:
        # novel_architecture schema
        method = config["proposed_method"] or {}
        if method.get("name"):
            models_lines.append(f"- {method['name']}: {method.get('summary', '').strip()}")
        for comp in method.get("key_components", []) or []:
            models_lines.append(f"  - Component: {comp}")
        for base in config.get("baselines", []) or []:
            models_lines.append(f"- Baseline for comparison: {base}")
        for ref in config.get("reference_implementation", []) or []:
            models_lines.append(f"- Reference hint: {ref}")
    elif "analytical_tools" in config:
        # empirical_study schema
        for tool in config.get("analytical_tools", []) or []:
            models_lines.append(f"- {tool}")
    else:
        # llm_evaluation schema
        api_models = config.get("models", {}).get("api", []) or []
        hf_models = config.get("models", {}).get("huggingface", []) or []
        if api_models:
            models_lines.append("- " + " and ".join(api_models) + " via API")
        for hf in hf_models:
            models_lines.append(f"- Load with HuggingFace: {hf}")
    models_section = "\n".join(models_lines) if models_lines else "- (none specified -- the agent must choose appropriate tools)"

    # --- Datasets / data-sources section ---
    dataset_list = config.get("datasets") or config.get("data_sources") or []
    datasets_lines = []
    for ds in dataset_list:
        name = ds.get("name", "Dataset")
        desc = ds.get("description", name)
        source = ds.get("source", "")
        loader = (ds.get("loader") or "").strip()

        if source == "unknown":
            datasets_lines.append(f"- {name}: {desc}  [loader unverified -- locate the data yourself]")
        elif source == "synthetic":
            datasets_lines.append(f"- {name}: {desc}  [generate programmatically]")
            if loader:
                datasets_lines.append("  Generation code:")
                for line in loader.split("\n"):
                    datasets_lines.append(f"    {line}")
        elif loader:
            datasets_lines.append(f"- {name}: {desc}")
            datasets_lines.append("  Loading code:")
            for line in loader.split("\n"):
                datasets_lines.append(f"    {line}")
        else:
            datasets_lines.append(f"- {name}: {desc}")

    # Surface unverified citations so the agent knows about them.
    for ref in config.get("references", []) or []:
        datasets_lines.append(f"- Referenced but not loaded: {ref}")

    datasets_section = "\n".join(datasets_lines) if datasets_lines else "- (no datasets specified)"

    # --- Constraints ---
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
