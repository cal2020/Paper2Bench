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
    "reference_implementation": ["<framework hint only, e.g. 'PyTorch Geometric', 'JAX'. NEVER a URL, NEVER a GitHub/HuggingFace repo, NEVER the source paper's own codebase.>"],
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
1. Generate **at least 2 instances of each transformation type** (perturbation,
   ablation, future_work) for a minimum of 6 instances total.  Aim for
   6-9 in total; prefer quality over quantity but do NOT skip a category.
2. The ``task_config`` resource spec must be concrete and complete -- the
   same format and detail level as the core pipeline's task_config.yaml.
   List real models, real datasets (with HF paths when you are confident,
   else ``source: unknown``), realistic budgets, and explicit constraints.
3. The perturbed task should use a *different* dataset / metric / component
   than the original paper when relevant -- that is the whole point of the
   transformation.  At most ONE perturbation may be a pure dataset swap;
   the others MUST change something else (metric family, compute budget,
   model family under test, evaluation regime, distribution shift, etc.).
4. A dataset must have a proper-noun or acronym name.  Author-year citations
   belong in ``references``, never in ``datasets`` / ``data_sources``.
5. **Ablation instances MUST come from the paper's own ablation_structure
   when the spec lists any** -- pick a `component_varied` from the spec and
   build the ablation around it.  Only invent a new ablation axis if the
   spec's ablation_structure is empty.
6. **Future_work instances MUST come from the spec's `future_work` list**
   when it has entries -- map each one to a concrete task.  Do NOT
   instantiate a future_work item that asks the agent to re-derive a
   conclusion already explicitly stated in the source paper.
7. Do NOT leak the paper's conclusions, exact results, or method details
   that would let an agent skip experimentation.  Specifically:
   - Do NOT cite the source paper itself in ``references``.
   - Do NOT include URLs to the source paper's official code repository,
     the first author's GitHub/HuggingFace account, or any "official
     implementation" hint.  ``reference_implementation`` is for generic
     framework names (e.g. "PyTorch Geometric"), never URLs.
   - Do NOT recreate the paper's exact study (same dataset + same models +
     same already-answered question).
   - Do NOT tell the agent to "use the same architecture as the original
     paper"; if the paper is a novel_architecture, the agent must re-derive
     or adapt the method.
8. Mark difficulty as:
   - **easy**: one-step adaptation (e.g. same method, closely related dataset)
   - **medium**: moderate shift requiring planning (new domain, tighter
     budget, partial ablation)
   - **hard**: multi-step investigation requiring significant adaptation
   Bump up one notch if the only available dataset is ``source: unknown``.
9. Output ONLY valid JSON (no markdown fences, no preamble).
"""

# ---------------------------------------------------------------------------
# Leakage check prompt
# ---------------------------------------------------------------------------

LEAKAGE_CHECK_PROMPT = """\
You are a benchmark quality auditor.  Your job is to check whether a
research benchmark variant **accidentally leaks** the expected answer or
trivializes the task, so an agent could "solve" it without genuine
experimentation.

You will be shown:
  SOURCE PAPER DIGEST -- title, authors, and the paper's main claims.
  VARIANT TASK_CONFIG  -- the JSON resource spec (models, datasets, refs).
  RENDERED INSTRUCTION -- the prompt the agent will read.

Check for these leakage types (any one is sufficient to flag):
1. **direct_answer**: the instruction states or strongly implies the conclusion.
2. **method**: the instruction (or task_config) gives so much methodological
   detail that the experiment is reduced to running a provided recipe.
3. **statistical**: exact numbers, effect sizes, or rankings from the source
   paper appear.
4. **repo_link**: the task_config's `reference_implementation` (or any field
   rendered into the instruction) contains a URL to the SOURCE paper's
   official code repository, the first author's personal GitHub/HuggingFace
   account, or anything that would let the agent download the paper's code.
5. **self_citation**: `references` cites the source paper itself (by author
   surname + year, or by paper title).
6. **setup_reproduction**: the variant essentially recreates the source
   paper's exact study -- same dataset, same models, asking a question the
   source paper already explicitly answered.  Flag this even if wording is
   reworded.

Respond with ONLY a JSON object:
{
  "leaked": true/false,
  "leakage_type": "direct_answer | method | statistical | repo_link | self_citation | setup_reproduction | none",
  "evidence": "<quote or field path showing the leakage, or empty string>"
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

    # ---- Step 2: Sanitize task_configs (strip self-references / repo URLs) ----
    print("[generate] Sanitizing task_configs ...")
    _strip_self_references(spec, instances)

    # ---- Step 3: Validate datasets + models against HuggingFace; resolve unknowns ----
    print("[generate] Validating datasets and models ...")
    for inst in instances:
        cfg = inst.get("task_config") or {}
        if not cfg:
            continue
        cfg = _revalidate_task_config(cfg, model, api_key)
        _validate_hf_models(cfg)
        _bump_difficulty_for_unresolved(inst, cfg)
        inst["task_config"] = cfg

    # ---- Step 4: Render each task_config into a detailed instruction.txt ----
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

    # ---- Step 5: Leakage check each rendered instruction (with paper context) ----
    print("[generate] Running leakage checks ...")
    paper_digest = _build_paper_digest(spec)
    for inst in instances:
        inst["leakage_check"] = _check_leakage(
            inst.get("instruction", ""),
            inst.get("task_config") or {},
            paper_digest,
            model, api_key,
        )
        status = "LEAKED" if inst["leakage_check"].get("leaked") else "clean"
        kind = inst["leakage_check"].get("leakage_type", "")
        print(f"  [{inst.get('id', '?')}] {status}"
              f"{f' ({kind})' if status == 'LEAKED' else ''}")

    # ---- Step 6: Coverage report ----
    _report_coverage(instances)

    return instances


def check_leakage(
    instruction: str,
    *,
    task_config: Optional[dict] = None,
    paper_digest: str = "",
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
) -> dict:
    """Standalone leakage check for any instruction text."""
    return _check_leakage(instruction, task_config or {}, paper_digest, model, api_key)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_leakage(
    instruction: str,
    task_config: dict,
    paper_digest: str,
    model: str,
    api_key: Optional[str],
) -> dict:
    """Run the leakage-check LLM judge on a single instruction."""
    user_msg = (
        "SOURCE PAPER DIGEST:\n"
        f"{paper_digest or '(unavailable)'}\n\n"
        "VARIANT TASK_CONFIG:\n"
        f"```json\n{json.dumps(task_config, indent=2, ensure_ascii=False)}\n```\n\n"
        "RENDERED INSTRUCTION:\n"
        f"{instruction}"
    )
    response = chat(
        model=model,
        messages=[
            {"role": "system", "content": LEAKAGE_CHECK_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=1024,
        temperature=0.0,
        api_key=api_key,
    )
    return _parse_json(response.content)


def _build_paper_digest(spec: dict) -> str:
    """Compact summary of the source paper for the leakage judge."""
    paper = (spec or {}).get("paper") or {}
    spec_body = (spec or {}).get("spec") or {}
    title = paper.get("title") or "(unknown title)"
    authors = ", ".join(paper.get("authors") or []) or "(unknown)"
    year = paper.get("year") or "(unknown)"
    claims = spec_body.get("claim") or []
    claim_lines = "\n".join(f"  - {c}" for c in claims[:5]) if claims else "  (none)"
    return (
        f"Title: {title}\n"
        f"Authors: {authors}\n"
        f"Year: {year}\n"
        f"Main claims:\n{claim_lines}"
    )


def _author_last_name(name: str) -> str:
    parts = re.split(r"\s+", str(name or "").strip())
    return parts[-1].lower() if parts and parts[-1] else ""


def _strip_self_references(spec: dict, instances: list) -> None:
    """Remove the source paper from each variant's `references`, and drop
    any URL-flavoured `reference_implementation` entry that points at the
    source paper's repo / first-author account.

    The source paper is never a valid reference for a benchmark *derived*
    from it, and its canonical repo is the worst possible leak (the agent
    is told to re-derive the method from scratch).
    """
    paper = (spec or {}).get("paper") or {}
    authors = [a for a in (paper.get("authors") or []) if a]
    last_names = [_author_last_name(a) for a in authors if a]
    last_names = [ln for ln in last_names if ln]
    year = str(paper.get("year") or "").strip()
    title_lc = (paper.get("title") or "").lower()
    title_tokens = {t for t in re.findall(r"[a-z0-9]{4,}", title_lc)}

    for inst in instances:
        cfg = inst.get("task_config") or {}

        # 1. Strip self-citations from references
        refs = cfg.get("references") or []
        if isinstance(refs, list):
            kept = []
            for ref in refs:
                r = str(ref).lower()
                # self-cite if any author surname AND year both appear,
                # or the title appears verbatim
                self_cite = (
                    (year and any(ln in r for ln in last_names) and year in r)
                    or (title_lc and title_lc[:40] in r)
                )
                if self_cite:
                    print(f"  [{inst.get('id','?')}] dropped self-citation: {ref}")
                    continue
                kept.append(ref)
            if kept != refs:
                cfg["references"] = kept

        # 2. Strip URL-bearing reference_implementation entries that look
        #    like the source paper's repo or the first author's account.
        ri = cfg.get("reference_implementation") or []
        if isinstance(ri, list):
            kept_ri = []
            for hint in ri:
                h = str(hint).lower()
                is_url = "://" in h or h.startswith(("github.com", "huggingface.co"))
                if is_url:
                    # Drop if any author surname appears in the URL,
                    # or any 4+ char title token appears.
                    if any(ln and ln in h for ln in last_names) or any(
                        tok in h for tok in title_tokens
                    ):
                        print(f"  [{inst.get('id','?')}] dropped repo URL: {hint}")
                        continue
                    # Also drop generic GitHub/HF links from variants -- the
                    # benchmark value comes from the agent re-deriving.
                    if "github.com" in h or "huggingface.co" in h or "gitlab.com" in h:
                        print(f"  [{inst.get('id','?')}] dropped code URL: {hint}")
                        continue
                kept_ri.append(hint)
            if kept_ri != ri:
                cfg["reference_implementation"] = kept_ri


def _revalidate_task_config(cfg: dict, model: str, api_key: Optional[str]) -> dict:
    """Round-trip the variant task_config through the same dataset
    validators the main pipeline uses (HF loader check, unknown-dataset
    resolution, external-link discovery).  Silently no-op on any failure
    so a missing dependency or network issue can't break generation.
    """
    try:
        import yaml as _yaml
        from .instruction_generator import (
            _validate_hf_loaders,
            resolve_unknown_datasets,
            find_external_dataset_links,
        )
    except ImportError:
        return cfg
    try:
        yaml_text = _yaml.safe_dump(cfg, sort_keys=False)
        yaml_text = _validate_hf_loaders(yaml_text)
        yaml_text = resolve_unknown_datasets(yaml_text, model=model)
        yaml_text = find_external_dataset_links(yaml_text, model=model)
        result = _yaml.safe_load(yaml_text)
        if isinstance(result, dict):
            return result
    except Exception as e:
        print(f"[validate] task_config revalidation skipped: {e}")
    return cfg


def _validate_hf_models(cfg: dict) -> None:
    """Drop any HuggingFace model ID that does not resolve on the Hub.
    Catches fabrications like ``meta-llama/Llama-2-3B`` (does not exist)
    that would waste agent time."""
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import RepositoryNotFoundError
    except ImportError:
        return
    models = cfg.get("models") or {}
    if not isinstance(models, dict):
        return
    hf = models.get("huggingface") or []
    if not isinstance(hf, list) or not hf:
        return
    api = HfApi()
    kept = []
    for mid in hf:
        if not isinstance(mid, str) or not mid:
            continue
        try:
            api.model_info(mid)
            kept.append(mid)
        except RepositoryNotFoundError:
            print(f"[validate] HF model '{mid}' not found, dropping")
        except Exception as e:
            # network / auth failure -- keep the entry rather than guess
            print(f"[validate] skipped HF model check for '{mid}': {e}")
            kept.append(mid)
    if kept != hf:
        models["huggingface"] = kept


def _bump_difficulty_for_unresolved(inst: dict, cfg: dict) -> None:
    """If the variant's only data source is still ``unknown`` after
    validation, bump difficulty up one notch -- the agent has to locate
    the data themselves, which is materially harder."""
    items = cfg.get("datasets") or cfg.get("data_sources") or []
    if not isinstance(items, list) or not items:
        return
    if not all(
        isinstance(d, dict) and (d.get("source") or "").lower() == "unknown"
        for d in items
    ):
        return
    cur = (inst.get("difficulty") or "").lower()
    bumped = {"easy": "medium", "medium": "hard"}.get(cur)
    if bumped:
        inst["difficulty"] = bumped
        prev_note = inst.get("difficulty_note", "")
        suffix = "auto-bumped: data source unresolved"
        inst["difficulty_note"] = (
            f"{prev_note} ({suffix})" if prev_note else suffix
        )


def _report_coverage(instances: list) -> None:
    """Warn when any of the three transformation sources has fewer than
    two instances -- callers will likely want to re-run."""
    from collections import Counter
    counts = Counter((i.get("type") or "?") for i in instances)
    line = " | ".join(f"{k}: {counts.get(k, 0)}" for k in
                      ("perturbation", "ablation", "future_work"))
    print(f"[generate] Coverage -- {line}")
    sparse = [k for k in ("perturbation", "ablation", "future_work")
              if counts.get(k, 0) < 2]
    if sparse:
        print(f"[generate] WARN: under-covered transformation type(s): "
              f"{', '.join(sparse)} (<2 instances). Consider re-running.")


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
