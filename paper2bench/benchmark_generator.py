"""Generate benchmark variant instances from a paper specification.

Given a 7-component paper spec (produced by paper_spec.py), this module
generates a family of benchmark tasks via three transformation strategies:

  1. Task / constraint perturbations  (Section 3.3.1 + 3.3.3)
  2. Ablation-derived instances        (Section 3.3.4)
  3. Future-work-derived instances      (Section 3.3.5)

Each generated instance includes:
  - A full task_config block (paper-type-aware schema with models, datasets,
    loaders, budget, constraints) -- the same format as the core pipeline's
    task_config.yaml
  - A rendered instruction.txt produced via render_instruction
  - Difficulty tag (easy / medium / hard)
  - Transformation type metadata
  - A leakage-check verdict

The module is independent of the core Paper2Bench pipeline and can be run
on any paper spec JSON produced by ``paper2bench specextract``.
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .llm import chat
from .pdf_utils import extract_text_from_pdf
from .instruction_generator import render_instruction
from .paper_classifier import classify_paper, PAPER_TYPES

load_dotenv()

# ---------------------------------------------------------------------------
# Per-type resource schema snippets (injected into the generation prompt)
# ---------------------------------------------------------------------------

_RESOURCE_SCHEMA_LLM_EVAL = """\
  "task_config": {
    "task_id": "<same as id>",
    "paper_type": "llm_evaluation",
    "research_question": "<same as scientific_question>",
    "models": {
      "api": ["gpt-4o", ...],
      "huggingface": ["meta-llama/Llama-3.1-8B-Instruct", ...]
    },
    "api_key_var": "OPENAI_API_KEY",
    "budget_per_model": 1000,
    "datasets": [
      {
        "name": "<dataset name>",
        "source": "huggingface | local | synthetic | unknown",
        "loader": "<Python code to load the dataset -- omit if source is 'unknown'>",
        "description": "<what the data is>"
      }
    ],
    "references": ["<author-year citations, NOT datasets>"],
    "constraints": ["Do NOT use web search", "Run FULL end-to-end experiments"]
  }"""

_RESOURCE_SCHEMA_NOVEL_ARCH = """\
  "task_config": {
    "task_id": "<same as id>",
    "paper_type": "novel_architecture",
    "research_question": "<same as scientific_question>",
    "proposed_method": {
      "name": "<short name>",
      "summary": "<one-paragraph description>",
      "key_components": ["<component 1>", "<component 2>", ...]
    },
    "baselines": ["<existing method to compare against>", ...],
    "reference_implementation": ["<library / repo hint, if any>"],
    "datasets": [
      {
        "name": "<dataset name>",
        "source": "huggingface | local | synthetic | unknown",
        "loader": "<Python code -- omit if source is 'unknown'>",
        "description": "<what the data is>"
      }
    ],
    "references": ["<author-year citations>"],
    "compute_hint": "<e.g. 'single GPU, <24h'>",
    "budget_per_model": 1000,
    "constraints": [
      "Do NOT use web search",
      "Run FULL end-to-end experiments",
      "Implement the proposed method yourself -- do not import a library that bundles the paper's exact code."
    ]
  }"""

_RESOURCE_SCHEMA_EMPIRICAL = """\
  "task_config": {
    "task_id": "<same as id>",
    "paper_type": "empirical_study",
    "research_question": "<same as scientific_question>",
    "study_type": "<e.g. dataset audit, meta-analysis>",
    "data_sources": [
      {
        "name": "<source name>",
        "source": "huggingface | local | synthetic | unknown",
        "loader": "<Python code -- omit if source is 'unknown'>",
        "description": "<what the data is>"
      }
    ],
    "analytical_tools": ["<library / technique>", ...],
    "references": ["<author-year citations>"],
    "budget_per_model": 1000,
    "constraints": ["Do NOT use web search", "Run FULL end-to-end experiments"]
  }"""

_RESOURCE_SCHEMA_BY_TYPE = {
    "llm_evaluation": _RESOURCE_SCHEMA_LLM_EVAL,
    "novel_architecture": _RESOURCE_SCHEMA_NOVEL_ARCH,
    "empirical_study": _RESOURCE_SCHEMA_EMPIRICAL,
}


# ---------------------------------------------------------------------------
# Generation prompt template (paper_type is interpolated in)
# ---------------------------------------------------------------------------

VARIANT_GENERATION_PROMPT_TEMPLATE = """\
You are an expert benchmark designer for autonomous research agents.

You are given a **structured paper specification** that captures the
transferable scientific content of an academic paper, plus the full paper text
for reference.

The source paper's benchmark archetype is: **{paper_type}**.

Your task is to generate a set of **new benchmark instances** derived from
this paper.  Each instance must preserve the paper's core scientific logic
while changing the task surface so that the resulting problem is genuinely new
-- it cannot be solved by recalling the original paper's results.

Generate instances from THREE sources:

### 1. Task / Constraint Perturbations
Alter the empirical setting while preserving the scientific question.  Examples:
- Replace the original dataset with a related but different dataset
- Tighten or relax the computational budget
- Change the target metric (e.g. accuracy → robustness, average → worst-case)
- Shift the domain while keeping the hypothesis intact
- Change the model family under test

### 2. Ablation-Derived Instances
Use the paper's ablation structure to create tasks that test *why* the method
works.  Examples:
- Does the key component remain important under a dataset shift?
- Which component fails first under a tighter budget?
- Do two components interact differently in a new setting?
- Reconstruct a missing or incomplete ablation

### 3. Future-Work-Derived Instances
Instantiate the paper's stated limitations or open questions as bounded tasks.
Examples:
- Extend the method to a domain the authors suggested
- Test a robustness concern the authors raised
- Address a scaling question left open

---

For each instance, output the following JSON schema.  The ``task_config`` block
MUST match the source paper's archetype (``{paper_type}``).  It is a full
resource specification -- models or method, datasets with loaders, budget,
constraints -- structured so that it can be rendered into a detailed,
self-contained instruction.txt for an autonomous agent.

{{
  "instances": [
    {{
      "id": "<short snake_case id>",
      "title": "<one-line title>",
      "type": "perturbation | ablation | future_work",
      "difficulty": "easy | medium | hard",
      "scientific_question": "<the specific question this instance answers>",
{resource_schema},
      "expected_outcome_hint": "<for curator use only -- what the correct conclusion should roughly look like, based on the paper's logic>",
      "rationale": "<why this is a valid and meaningful benchmark instance>"
    }}
  ]
}}

Rules:
1. Generate 2-4 instances per source (6-12 total).  Prefer quality over quantity.
2. The ``task_config`` resource spec must be concrete and complete -- the
   same format and detail level as the core pipeline's task_config.yaml.
   List real models, real datasets (with HF paths when you are confident,
   else ``source: unknown``), realistic budgets, and explicit constraints.
3. The perturbed task should use a *different* dataset / metric / component
   than the original paper when relevant -- that is the whole point of the
   transformation.
4. A dataset must have a proper-noun or acronym name.  Author-year citations
   belong in ``references``, never in ``datasets`` / ``data_sources``.
5. Do NOT leak the paper's conclusions, exact results, or method details
   that would let an agent skip experimentation.  Do NOT tell the agent to
   "use the same architecture as the original paper"; if the paper is a
   novel_architecture, the agent must re-derive or adapt the method.
6. Mark difficulty as:
   - **easy**: one-step adaptation (e.g. same method, closely related dataset)
   - **medium**: moderate shift requiring planning (new domain, tighter
     budget, partial ablation)
   - **hard**: multi-step investigation requiring significant adaptation
7. Output ONLY valid JSON (no markdown fences, no preamble).
"""

# ---------------------------------------------------------------------------
# Leakage check prompt
# ---------------------------------------------------------------------------

LEAKAGE_CHECK_PROMPT = """\
You are a benchmark quality auditor.  Your job is to check whether a
research instruction **accidentally leaks** the expected answer, making it
possible for an agent to solve the task by reading the instruction alone
instead of running experiments.

Check for these leakage types:
1. **Direct answer leak**: the instruction states or strongly implies the
   conclusion.
2. **Method leak**: the instruction gives so much methodological detail that
   the experiment is trivially reduced to running a provided recipe.
3. **Statistical leak**: exact numbers, effect sizes, or rankings from the
   source paper appear in the instruction.

Respond with ONLY a JSON object:
{
  "leaked": true/false,
  "leakage_type": "<direct_answer | method | statistical | none>",
  "evidence": "<quote from the instruction that constitutes leakage, or empty string>"
}
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_variants(
    spec_path: str,
    pdf_path: str,
    *,
    paper_type: Optional[str] = None,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    max_tokens: int = 16384,
    temperature: float = 0.3,
) -> list[dict]:
    """Generate benchmark variant instances from a paper spec + PDF.

    If ``paper_type`` is not provided, the paper is auto-classified via
    ``paper_classifier.classify_paper`` so the right resource schema is used
    in both generation and rendering.

    Returns a list of instance dicts.  Each instance carries:
      - ``task_config`` -- the paper-type-aware resource spec
      - ``instruction`` -- the rendered, detailed instruction.txt string
      - ``leakage_check`` -- automated audit verdict
      - metadata: id, title, type, difficulty, rationale, expected_outcome_hint
    """
    # ---- Paper type (classify if not provided) ----
    if paper_type is None:
        print("[generate] Classifying paper type ...")
        classification = classify_paper(pdf_path, model=model, api_key=api_key)
        paper_type = classification["type"]
        print(f"[generate] paper_type={paper_type}  "
              f"(confidence={classification.get('confidence')})")
    if paper_type not in PAPER_TYPES:
        raise ValueError(f"Unknown paper_type: {paper_type}")

    # Load paper spec + text
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    paper_text = extract_text_from_pdf(pdf_path)
    if not paper_text.strip():
        raise ValueError(f"No text extracted from {pdf_path}")

    # ---- Step 1: Generate instances (with task_config per variant) ----
    print("[generate] Generating benchmark variants ...")
    prompt = VARIANT_GENERATION_PROMPT_TEMPLATE.format(
        paper_type=paper_type,
        resource_schema=_RESOURCE_SCHEMA_BY_TYPE[paper_type],
    )
    user_message = (
        f"PAPER ARCHETYPE: {paper_type}\n\n"
        "PAPER SPECIFICATION:\n\n"
        f"```json\n{json.dumps(spec, indent=2)}\n```\n\n"
        "---\n\n"
        "FULL PAPER TEXT (for reference):\n\n"
        f"{paper_text}"
    )

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
    print(
        f"  Tokens  --  input: {response.input_tokens}  |  "
        f"output: {response.output_tokens}  |  "
        f"total: {response.total_tokens}  |  provider: {response.provider}"
    )
    result = _parse_json(response.content)
    instances = result.get("instances", [])
    print(f"[generate] Got {len(instances)} instance(s)")

    # ---- Step 2: Render each task_config into a detailed instruction.txt ----
    print("[generate] Rendering detailed instructions ...")
    for inst in instances:
        cfg = inst.get("task_config") or {}
        if not cfg:
            inst["instruction"] = ""
            continue
        # Ensure required fields exist before rendering
        cfg.setdefault("research_question",
                       inst.get("scientific_question", ""))
        cfg.setdefault("paper_type", paper_type)
        try:
            inst["instruction"] = render_instruction(cfg)
        except Exception as e:
            print(f"  [{inst.get('id', '?')}] render failed: {e}")
            inst["instruction"] = ""

    # ---- Step 3: Leakage check each rendered instruction ----
    print("[generate] Running leakage checks ...")
    for inst in instances:
        inst["leakage_check"] = _check_leakage(
            inst.get("instruction", ""), model, api_key,
        )
        status = "LEAKED" if inst["leakage_check"].get("leaked") else "clean"
        print(f"  [{inst.get('id', '?')}] {status}")

    return instances


def check_leakage(
    instruction: str,
    *,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
) -> dict:
    """Standalone leakage check for any instruction text."""
    return _check_leakage(instruction, model, api_key)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_leakage(instruction: str, model: str, api_key: Optional[str]) -> dict:
    """Run the leakage-check LLM judge on a single instruction."""
    response = chat(
        model=model,
        messages=[
            {"role": "system", "content": LEAKAGE_CHECK_PROMPT},
            {"role": "user", "content": instruction},
        ],
        max_tokens=1024,
        temperature=0.0,
        api_key=api_key,
    )
    return _parse_json(response.content)


def _parse_json(text: str) -> dict:
    """Best-effort JSON extraction from LLM output."""
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        # Try finding a JSON array wrapper
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            try:
                arr = json.loads(text[start : end + 1])
                return {"instances": arr}
            except json.JSONDecodeError:
                pass
    raise ValueError("No valid JSON found in the model response")
