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
  "MMLU", "NaturalQuestions", "Materials Project"). Record it under `datasets:`
  whenever you have a name for it -- regardless of whether the paper happens
  to cite the originating publication alongside.
- The `references:` list is **only** for data sources where the paper offers
  *no name and no description at all*, just a bare author-year citation
  ("we used the data from <Author> <Year>") -- nothing else identifies the
  resource. This is rare; most papers describe the data they use.
- DO NOT add an entry to `references:` for a paper that merely *introduced*
  a dataset you have already listed in `datasets:`. The dataset itself is
  enough; the citation is redundant.
- DO NOT add citations of related work, baseline methods, training-data
  origins, or prior benchmarks to `references:`. Those belong in the paper's
  bibliography, not in this extraction.
- If you have nothing that meets the above bar, omit `references:` entirely
  (or emit an empty list).
- For HuggingFace loaders, emit your best-guess repo ID even if you are not
  100% certain (e.g. "cais/mmlu", "openai/gsm8k", "Anthropic/hh-rlhf").
  Downstream validation drops invalid IDs and a fallback resolver retries
  the lookup, so a real guess beats giving up. Only fall back to
  `source: unknown` when you genuinely have no candidate. Always include a
  clear, distinctive `name:` (the published dataset name) so the resolver
  can search by it.
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
research_question: "<the provided research question, verbatim -- ALWAYS wrap in double quotes>"
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
  - "<RARE: only an author-year citation that the paper offers as a data source with no name and no description; usually omit this section entirely>"
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
research_question: "<the provided research question, verbatim -- ALWAYS wrap in double quotes>"
proposed_method:
  name: "<short name of the new architecture / method>"
  summary: "<one-paragraph description of what it is>"
  key_components:
    - "<component 1 -- the essential pieces an implementation must include>"
baselines:                       # existing methods the paper compares against
  - "<baseline name>"
reference_implementation:        # FRAMEWORK HINTS ONLY, never URLs or repos
  - "<framework / library name only, e.g. 'PyTorch', 'JAX', 'PyTorch Geometric'. NEVER a URL, NEVER a GitHub or HuggingFace repo, NEVER the source paper's own codebase. Omit the section entirely if the paper does not mention a specific framework.>"
datasets:
  - name: "<dataset name>"
    source: huggingface          # huggingface | local | synthetic | unknown
    loader: |
      <Python code snippet -- omit if source is 'unknown'>
    description: "<what the data is, not how to use it>"
references:
  - "<RARE: only an author-year citation that the paper offers as a data source with no name and no description; usually omit this section entirely>"
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
research_question: "<the provided research question, verbatim -- ALWAYS wrap in double quotes>"
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
  - "<RARE: only an author-year citation that the paper offers as a data source with no name and no description; usually omit this section entirely>"
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

    cleaned = _quote_unparseable_top_level_scalars(cleaned)

    if validate_hf_loaders:
        cleaned = _validate_hf_loaders(cleaned)
        cleaned = resolve_unknown_datasets(cleaned, model=model)
        cleaned = find_external_dataset_links(cleaned, model=model)

    return cleaned


def _quote_unparseable_top_level_scalars(yaml_text: str) -> str:
    """If the YAML fails to parse, wrap any top-level unquoted scalar value
    that contains a colon in double quotes and retry. LLMs frequently emit
    `research_question: RQ1: Do X...` which YAML reads as a nested mapping.
    """
    import yaml as _yaml
    try:
        _yaml.safe_load(yaml_text)
        return yaml_text
    except _yaml.YAMLError:
        pass

    fixed_lines = []
    for line in yaml_text.splitlines():
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*):[ \t]+(.+?)[ \t]*$', line)
        if m:
            key, value = m.group(1), m.group(2)
            if value and value[0] not in '"\'[{|>&*!' and ':' in value:
                escaped = value.replace('\\', '\\\\').replace('"', '\\"')
                line = f'{key}: "{escaped}"'
        fixed_lines.append(line)
    candidate = '\n'.join(fixed_lines)
    try:
        _yaml.safe_load(candidate)
        return candidate
    except _yaml.YAMLError:
        return yaml_text


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
# HF Hub auto-resolution for source: unknown datasets
# ---------------------------------------------------------------------------

_HF_RESOLVE_PROMPT = """\
You are a HuggingFace Hub expert. Given a list of dataset names with brief
descriptions, return the canonical HuggingFace Hub repository ID for each
dataset (or null when no widely-used HF release exists for it).

Rules:
- Use the exact published repo ID (e.g. "cais/mmlu", "openai/gsm8k",
  "trivia_qa", "Anthropic/hh-rlhf", "THUDM/AgentBench").
- If the "dataset" describes paper-internal data (a user study, a survey
  the authors collected, transcripts they annotated, etc.) return null --
  it is not on the Hub.
- If you are not confident a real Hub repo exists, return null. Do NOT
  invent IDs; downstream validation will reject them and we'd rather have
  null than a hallucination.

Output ONLY a JSON object mapping each input name to a string ID or null.
No commentary, no markdown.
"""


def _is_paper_internal_data(name: str, description: str = "") -> bool:
    """Heuristic: does the 'dataset' describe paper-internal experimental data
    (e.g. a user study) rather than a published dataset?"""
    blob = f"{name} {description}".lower()
    bad_phrases = (
        "user study", "survey response", "post-study", "human evaluation",
        "questionnaire", "interview", "annotated by the authors", "we collected",
        "participants", "our annotators", "in-house",
    )
    return any(p in blob for p in bad_phrases)


def resolve_unknown_datasets(yaml_text: str, model: str = "gpt-4o") -> str:
    """Look up a HuggingFace Hub repo ID for every dataset with
    ``source: unknown``. Promotes confirmed matches to ``source: huggingface``
    with a generated ``load_dataset(...)`` loader. No-ops if the LLM call,
    JSON parse, or HF validation fails.

    Strategy:
      1. Collect all unknown datasets in one task_config (skip clearly
         paper-internal ones via name/description heuristic).
      2. One LLM call returns a {name -> hf_id_or_null} mapping.
      3. Each suggested ID is verified with ``HfApi().dataset_info()``;
         hallucinations are dropped.
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

    unknowns: list[dict] = []
    for key in ("datasets", "data_sources"):
        items = config.get(key) or []
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            if (item.get("source") or "").lower() != "unknown":
                continue
            name = (item.get("name") or "").strip()
            desc = (item.get("description") or "").strip()
            if not name or _is_paper_internal_data(name, desc):
                continue
            unknowns.append(item)

    if not unknowns:
        return yaml_text

    user_msg = "Resolve these dataset names to HuggingFace Hub IDs:\n\n"
    for it in unknowns:
        name = it["name"]
        desc = (it.get("description") or "")[:200]
        user_msg += f"- {name}: {desc}\n"

    try:
        response = chat(
            model=model,
            messages=[
                {"role": "system", "content": _HF_RESOLVE_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=600,
            temperature=0.0,
        )
        raw = response.content.strip()
        raw = re.sub(r"^```(?:json)?\s*\n", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\n```\s*$", "", raw, flags=re.MULTILINE).strip()
        suggestions = json.loads(raw)
        if not isinstance(suggestions, dict):
            return yaml_text
    except Exception as e:
        print(f"[hf-resolve] LLM call failed: {e}")
        return yaml_text

    api = HfApi()
    changed = False
    for it in unknowns:
        name = it["name"]
        suggested = suggestions.get(name)
        if not suggested or not isinstance(suggested, str):
            continue
        try:
            api.dataset_info(suggested)
        except RepositoryNotFoundError:
            print(f"[hf-resolve] {name!r} -> {suggested!r} (not on Hub, skipped)")
            continue
        except Exception as e:
            print(f"[hf-resolve] HF check failed for {suggested!r}: {e}")
            continue
        it["source"] = "huggingface"
        it["loader"] = (
            "from datasets import load_dataset\n"
            f"ds = load_dataset({suggested!r})\n"
        )
        changed = True
        print(f"[hf-resolve] {name!r} -> {suggested}")

    if not changed:
        return yaml_text
    return _yaml.safe_dump(config, sort_keys=False)


# ---------------------------------------------------------------------------
# External link discovery (GitHub repos, project pages) for unknowns
# the HF resolver couldn't place
# ---------------------------------------------------------------------------

_EXTERNAL_LINK_PROMPT = """\
You are a dataset locator. For each dataset name (with brief description),
return the most authoritative public URL where the dataset (or instructions
to obtain it) lives.

Preferred URL kinds, in order:
- "github"  -- a GitHub repo that hosts the data or its loader scripts
- "website" -- the dataset's official project page
- "paper"   -- arxiv / ACL / OpenReview page if it is the canonical source
- null      -- if the dataset is paper-internal (a user study the authors
               collected, transcripts they annotated, an in-house corpus,
               etc.) OR you don't know a specific real URL

Hard rules:
- Do NOT invent URLs. If you are not confident a real URL exists, return null.
  We will validate every URL with an HTTP request and drop the ones that 404.
- Do NOT return generic search-engine URLs (e.g. google.com/search?q=...).
- Prefer https:// over http://.

Output ONLY a JSON object mapping each input name to an object:
  {"<name>": {"url": "<https://...>" | null, "kind": "github|website|paper" | null}}
No commentary, no markdown.
"""


def _url_reachable(url: str, timeout: float = 6.0) -> bool:
    """Best-effort HEAD probe; many sites reject HEAD so fall back to GET
    with a 1-byte read. Treat 2xx and 3xx as reachable."""
    import urllib.error
    import urllib.request
    headers = {"User-Agent": "paper2bench/1.0 (+dataset-locator)"}
    for method in ("HEAD", "GET"):
        try:
            req = urllib.request.Request(url, method=method, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return 200 <= getattr(r, "status", 200) < 400
        except urllib.error.HTTPError as e:
            if e.code in (405, 501) and method == "HEAD":
                continue  # try GET
            return False
        except Exception:
            return False
    return False


def find_external_dataset_links(yaml_text: str, model: str = "gpt-4o") -> str:
    """For each dataset still ``source: unknown``, ask an LLM for an
    authoritative public URL (GitHub / project page / arxiv) and validate
    it with an HTTP request. Confirmed URLs are written into the dataset
    entry as ``source: external`` plus a ``url:`` field, so the renderer
    can show the agent where to fetch the data.

    Skips datasets whose name/description looks paper-internal.
    """
    try:
        import yaml as _yaml
    except ImportError:
        return yaml_text

    try:
        config = _yaml.safe_load(yaml_text)
    except Exception:
        return yaml_text
    if not isinstance(config, dict):
        return yaml_text

    unknowns: list[dict] = []
    for key in ("datasets", "data_sources"):
        items = config.get(key) or []
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            if (item.get("source") or "").lower() != "unknown":
                continue
            name = (item.get("name") or "").strip()
            desc = (item.get("description") or "").strip()
            if not name or _is_paper_internal_data(name, desc):
                continue
            unknowns.append(item)

    if not unknowns:
        return yaml_text

    user_msg = "Find an authoritative URL for each of these datasets:\n\n"
    for it in unknowns:
        desc = (it.get("description") or "")[:200]
        user_msg += f"- {it['name']}: {desc}\n"

    try:
        response = chat(
            model=model,
            messages=[
                {"role": "system", "content": _EXTERNAL_LINK_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=800,
            temperature=0.0,
        )
        raw = response.content.strip()
        raw = re.sub(r"^```(?:json)?\s*\n", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\n```\s*$", "", raw, flags=re.MULTILINE).strip()
        suggestions = json.loads(raw)
        if not isinstance(suggestions, dict):
            return yaml_text
    except Exception as e:
        print(f"[external-link] LLM call failed: {e}")
        return yaml_text

    changed = False
    for it in unknowns:
        name = it["name"]
        sug = suggestions.get(name) or {}
        if not isinstance(sug, dict):
            continue
        url = sug.get("url")
        kind = sug.get("kind") or "website"
        if not url or not isinstance(url, str) or not url.startswith(("http://", "https://")):
            continue
        if "google.com/search" in url or "bing.com/search" in url:
            continue
        if not _url_reachable(url):
            print(f"[external-link] {name!r} -> {url} (unreachable, skipped)")
            continue
        it["source"] = "external"
        it["url"] = url
        it["url_kind"] = kind
        if kind == "github":
            it["loader"] = (
                f"# Clone the repo and follow its README to load the data:\n"
                f"# git clone {url}.git\n"
            )
        else:
            it["loader"] = (
                f"# Download instructions and data are at:\n# {url}\n"
            )
        changed = True
        print(f"[external-link] {name!r} -> {url}  ({kind})")

    if not changed:
        return yaml_text
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
            if not isinstance(ref, str):
                continue
            ref_clean = ref.strip()
            if not ref_clean:
                continue
            # Defense-in-depth: even though the extract prompt forbids
            # URLs/repos in reference_implementation, the LLM occasionally
            # ignores it. Drop anything that looks like a link.
            lowered = ref_clean.lower()
            if (lowered.startswith(("http://", "https://", "www.", "github.com/",
                                    "huggingface.co/"))
                    or "://" in lowered
                    or "github.com/" in lowered
                    or "huggingface.co/" in lowered):
                continue
            models_lines.append(f"- Reference hint: {ref_clean}")
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
        elif source == "external":
            url = ds.get("url", "")
            kind = ds.get("url_kind", "website")
            label = {"github": "GitHub repo", "paper": "Paper page",
                     "website": "Project page"}.get(kind, "Source")
            datasets_lines.append(f"- {name}: {desc}")
            if url:
                datasets_lines.append(f"  {label}: {url}")
            if loader:
                datasets_lines.append("  Setup:")
                for line in loader.split("\n"):
                    datasets_lines.append(f"    {line}")
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
