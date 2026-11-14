"""Extract a structured 7-component paper specification from a PDF.

The specification captures the transferable scientific content of a paper,
separating it from the original experimental packaging.  This is the input
for benchmark-variant generation (see benchmark_generator.py).

Components (per Section 3.2 of the Paper2Bench design doc):
  1. scientific_question  -- phenomenon / capability the paper establishes
  2. method               -- intervention / mechanism behind the claimed result
  3. claim                -- qualitative or quantitative conclusion
  4. evaluation_protocol  -- datasets, metrics, baselines, ablations, controls
  5. assumptions          -- conditions under which the claim is valid
  6. ablation_structure   -- components varied and causal hypotheses tested
  7. future_work          -- limitations, open questions, natural extensions
"""

import json
import os
import re
from pathlib import Path
from typing import Optional


from .llm import chat
from .pdf_utils import extract_text_from_pdf


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SPEC_EXTRACTION_PROMPT = """\
You are an expert research scientist who specialises in distilling the
transferable scientific content from academic papers.

Read the full paper text below and produce a **structured paper specification**
with exactly the following seven components.  Each component should be a concise
but complete description -- concrete enough that a different researcher could
use it to design new experiments testing the same scientific logic.

Output ONLY valid JSON matching this schema (no preamble, no markdown fences):

{
  "paper": {
    "title": "",
    "authors": [],
    "venue": "",
    "year": ""
  },
  "spec": {
    "scientific_question": "<What phenomenon or capability is the paper trying to establish?>",
    "method": {
      "summary": "<One-paragraph description of the core intervention or mechanism>",
      "key_components": [
        {
          "name": "<component name>",
          "role": "<what it does and why it matters>"
        }
      ]
    },
    "claim": [
      "<Each major qualitative or quantitative conclusion, one per item>"
    ],
    "evaluation_protocol": {
      "datasets": ["<dataset name and brief description>"],
      "metrics": ["<metric name and definition>"],
      "baselines": ["<baseline name>"],
      "controls": "<any controls or statistical tests used>"
    },
    "assumptions": [
      "<Each condition that must hold for the claim to be valid>"
    ],
    "ablation_structure": [
      {
        "component_varied": "<which part of the method is changed>",
        "hypothesis": "<what causal claim does this ablation test>",
        "finding": "<what was observed>"
      }
    ],
    "future_work": [
      {
        "direction": "<concise description of the open question or extension>",
        "type": "<one of: domain_extension | robustness | scaling | efficiency | failure_mode | other>",
        "grounding": "<where in the paper this is mentioned (section, paragraph, or quote)>"
      }
    ]
  }
}

Rules:
1. Derive every field from the paper alone -- do NOT invent or hallucinate.
2. If a component is absent from the paper (e.g. no ablations), use an empty list.
3. Be concrete: include exact dataset names, metric formulas, thresholds, and
   component names when the paper provides them.
4. For "claim", separate distinct conclusions into individual list items.
5. For "future_work", capture both explicit future-work statements AND implicit
   limitations that suggest natural follow-ups.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_paper_spec(
    pdf_path: str,
    *,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    max_tokens: int = 16384,
    temperature: float = 0.0,
) -> dict:
    """Read a paper PDF and return its 7-component specification as a dict."""
    paper_text = extract_text_from_pdf(pdf_path)
    if not paper_text.strip():
        raise ValueError(f"No text extracted from {pdf_path}")

    response = chat(
        model=model,
        messages=[
            {"role": "system", "content": SPEC_EXTRACTION_PROMPT},
            {"role": "user", "content": paper_text},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        api_key=api_key,
    )
    print(
        f"Tokens  --  input: {response.input_tokens}  |  "
        f"output: {response.output_tokens}  |  "
        f"total: {response.total_tokens}  |  provider: {response.provider}"
    )
    return _parse_json(response.content)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict:
    """Best-effort JSON extraction from LLM output."""
    # Strip markdown fences if present
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: find outermost { ... }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start : end + 1])
        raise ValueError("No valid JSON found in the model response")
