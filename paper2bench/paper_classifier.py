"""Classify a paper into one of three benchmark archetypes.

The classification drives which extraction prompt and which instruction
template are used downstream.  Without this step, the pipeline assumes every
paper evaluates LLMs, which produces empty or hallucinated model lists for
papers that propose novel non-LLM architectures.

Types:
  - llm_evaluation     -- paper evaluates existing models on tasks
                          (e.g. Lost in the Middle, Chain-of-Thought)
  - novel_architecture -- paper introduces a new model / method
                          (e.g. CGCNN, SchNet, ESM, Transformer)
  - empirical_study    -- observational / analytical, no new model training
                          (e.g. meta-analysis, benchmark studies)
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .llm import chat
from .pdf_utils import extract_text_from_pdf

load_dotenv()


PAPER_TYPES = ("llm_evaluation", "novel_architecture", "empirical_study")


CLASSIFIER_PROMPT = """\
You are an expert research scientist.  Read the paper excerpt below and
classify it into exactly ONE of these three benchmark archetypes:

1. "llm_evaluation"
   The paper's main contribution is an *evaluation* of existing language
   models or AI systems on a task or benchmark.  The paper does not
   introduce a new neural architecture.  Examples: Lost in the Middle,
   Chain-of-Thought Prompting, MMLU, HELM.

2. "novel_architecture"
   The paper's main contribution is a *new model, method, or architecture*
   that must be implemented and trained.  The evaluation is in service of
   demonstrating the new method works.  Examples: Transformer, ResNet,
   CGCNN, SchNet, AlphaFold, ESM-2.

3. "empirical_study"
   The paper is primarily observational, analytical, or a meta-study.
   It neither proposes a new architecture nor centrally evaluates existing
   models on a task.  Examples: dataset surveys, training-data analyses,
   theoretical analyses.

Output ONLY a JSON object:
{
  "type": "<one of: llm_evaluation | novel_architecture | empirical_study>",
  "confidence": "<high | medium | low>",
  "rationale": "<one sentence explaining the classification>"
}
"""


def classify_paper(
    pdf_path: str,
    *,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    excerpt_chars: int = 12000,
) -> dict:
    """Classify a paper PDF into one of three benchmark archetypes.

    Uses only the first ``excerpt_chars`` of the paper (enough for the title,
    abstract, intro, and usually the start of methods) to keep the call cheap.

    Returns a dict with ``type``, ``confidence``, and ``rationale``.
    """
    paper_text = extract_text_from_pdf(pdf_path)
    if not paper_text.strip():
        raise ValueError(f"No text extracted from {pdf_path}")

    excerpt = paper_text[:excerpt_chars]

    response = chat(
        model=model,
        messages=[
            {"role": "system", "content": CLASSIFIER_PROMPT},
            {"role": "user", "content": excerpt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        api_key=api_key,
    )
    parsed = _parse_json(response.content)

    # Defensive: fall back to llm_evaluation if the model emits junk.
    if parsed.get("type") not in PAPER_TYPES:
        print(f"[classify] Unrecognised type '{parsed.get('type')}', "
              "defaulting to llm_evaluation.")
        parsed = {"type": "llm_evaluation", "confidence": "low",
                  "rationale": "classifier output did not match known types"}

    return parsed


def _parse_json(text: str) -> dict:
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
    return {}
