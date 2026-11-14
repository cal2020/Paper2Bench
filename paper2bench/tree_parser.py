"""Parse a research paper PDF into a structured problem tree via LLM."""

import json
import re
import argparse
import os
from pathlib import Path
from typing import Optional


from .llm import chat
from .pdf_utils import extract_text_from_pdf



TREE_PARSING_PROMPT = """
You are a research-paper expert specializing in methodological analysis and problem
decomposition of scientific studies.

**GOAL**
- Fully comprehend the paper, understand its core research problems and experiments.
- Parse the paper and construct a **three-level research-problem tree**:
  * **Root node** -- the single, broadest research problem tackled by the paper.
  * **Depth-1 nodes (Research Questions)** -- **fully self-contained research
    questions**. A researcher reading ONLY the question (with no access to the
    paper, abstract, or any other context) must know exactly what to
    investigate. To meet that bar, each RQ MUST:
      - DEFINE any specialized term that is not standard ML/NLP common
        knowledge, inline as a short parenthetical or appositive clause
        (e.g. "sycophancy -- the tendency to adjust responses to match
        perceived user beliefs rather than maintaining accuracy").
      - SPECIFY the concrete dimensions, conditions, or modalities covered
        by the experiments under it: name the kinds of model classes, the
        dataset families, the environment categories, or the evaluation
        regimes the paper actually uses (you'll list these as leaves below).
      - PRESERVE the paper's "RQ1:"/"RQ2:" prefix when present.
      - Stay as a SINGLE question, one or two sentences, ~50 words max.
      - NOT reveal the paper's findings -- describe *what is investigated*,
        not *what was concluded*.

    Bad RQ:  "RQ1: Do AI assistants exhibit sycophancy in varied settings?"
    Good RQ: "RQ1: Do state-of-the-art AI assistants exhibit sycophancy --
              adjusting responses to match perceived user beliefs rather
              than maintaining accuracy -- across free-form generation
              feedback, challenged-QA, belief-conforming QA, and
              content-mimicry tasks?"

    Bad RQ:  "RQ1: How well do VLMs understand compositional relationships?"
    Good RQ: "RQ1: How well do current Vision-Language Models (CLIP, BLIP,
              FLAVA, X-VLM) understand compositional relationships,
              attribute binding, and word order in image-caption pairs,
              evaluated via the Attribution, Relation, and Order benchmarks
              built on Visual Genome, COCO, and Flickr30k?"

  * **Leaf nodes** -- fully specified experimental tasks (datasets, models, metrics,
    or protocols) that directly answer the parent research question and map to a
    figure, table, or named result section in the paper.

The tree has exactly three levels: Root -> Research Questions -> Experiments.

---

### Reading & Extraction Rules

1. **Locate the root** in the title, abstract, introduction, or discussion.
2. **Derive research questions** by examining the paper's headings, stated objectives,
   hypotheses, "RQ" labels, discussion structure, and figure/table captions.
3. **Identify leaves**: a node is a leaf *only if* it describes a concrete experiment
   and you can cite the corresponding Figure / Table / Section ID.
4. **Stay faithful** to the paper's wording for technical terms.
5. **No outside invention** -- derive every node from the paper alone.

---

**Before you output**, check every RQ node against this list:
- [ ] Does it inline-define every specialized term that isn't standard ML/NLP?
- [ ] Does it name the concrete dimensions / datasets / settings covered by
  the leaves under it?
- [ ] Is it under ~50 words and a single question?
- [ ] Does it stay neutral about findings (no spoiler of the conclusion)?

If any answer is "no", REWRITE the RQ before emitting the JSON.

Strictly output the tree in the following JSON format:

```json
{
  "paper": {
    "title": "",
    "authors": [],
    "venue": "",
    "year": ""
  },
  "problem_tree": {
    "node": "Root: broadest research problem tackled by the paper",
    "type": "root",
    "description": "Detailed description of the overarching research problem",
    "evidence": "References back to the original paper",
    "children": [
      {
        "node": "RQ1: <fully self-contained question -- inline-defines specialized terms AND names the concrete dimensions/datasets/conditions covered by the leaves below; e.g. 'RQ1: Do state-of-the-art AI assistants exhibit sycophancy -- adjusting responses to match perceived user beliefs rather than maintaining accuracy -- across feedback, challenged-QA, belief-conforming QA, and content-mimicry tasks?'>",
        "type": "research_question",
        "description": "What this question investigates",
        "evidence": "References back to the original paper",
        "children": [
          {
            "type": "leaf",
            "task": "Concrete experimental task",
            "dataset": ["..."],
            "model_or_method": ["..."],
            "metrics": ["..."],
            "protocol_or_setup": "Key settings if stated",
            "evidence": {
              "figure": "Fig. X",
              "table": "Table Y",
              "section": "Sec. Z"
            },
            "conclusion": "Explicit conclusions from this experiment",
            "status": ""
          }
        ]
      }
    ]
  }
}
```
"""


def extract_json_from_response(text: str) -> dict:
    """Pull the first JSON object out of a model response, handling markdown fences."""
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return json.loads(text[start : end + 1])
    raise ValueError("No valid JSON found in the model response")


def parse_paper_to_tree(
    pdf_path: str,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    max_tokens: int = 16384,
    temperature: float = 0.0,
) -> dict:
    """Read a paper PDF and return the research-problem tree as a dict."""
    paper_text = extract_text_from_pdf(pdf_path)
    if not paper_text.strip():
        raise ValueError(f"No text could be extracted from {pdf_path}")

    user_message = (
        "Below is the full text of the paper:\n\n"
        f"{paper_text}\n\n"
        "Now parse this paper according to the instructions."
    )

    response = chat(
        model=model,
        messages=[
            {"role": "system", "content": TREE_PARSING_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        api_key=api_key,
    )
    tree = extract_json_from_response(response.content)
    print(
        f"Tokens  --  input: {response.input_tokens}  |  "
        f"output: {response.output_tokens}  |  "
        f"total: {response.total_tokens}  |  provider: {response.provider}"
    )
    return tree


_POLISH_RQ_PROMPT = """\
You rewrite vague research questions to be self-contained.

INPUT: an original research question (often vague), the paper's description
of what it investigates, the experiments that operationalize it (datasets,
metrics, protocol), and (optionally) a list of specific issues a reviewer
flagged with the current wording.

OUTPUT: a single polished question that an AI agent or human researcher can
read on its own and understand exactly what to investigate. Concretely:

- DEFINE any specialized term that is not standard ML/NLP common knowledge
  (e.g. "sycophancy", "compositionality gap", "the reversal curse",
  "in-context learning"). One short clause inline; do not turn it into a
  separate sentence.
- SPECIFY the concrete aspects, conditions, or dimensions being studied --
  pull in the kinds of environments, model classes, dataset families, and
  evaluation regimes implied by the listed experiments.
- DO NOT reveal the paper's METHODOLOGY, intervention, or proposed approach
  -- the agent must design experiments itself. Naming the datasets and
  evaluation regimes is fine; describing how the paper *fixes* or *attacks*
  the problem is not.
- DO NOT reveal the paper's findings. Describe what is being investigated,
  not what was concluded.
- DO NOT introduce experiments not implied by the inputs.
- Keep it as ONE question. One or two sentences, ~50 words max.
- PRESERVE the original "RQ1:" / "RQ2:" prefix if present.
- If the input includes a list of issues to fix, address EVERY one of them.

Output ONLY the rewritten question text. No quotes, no markdown, no
commentary.
"""


_DETECT_RQ_PROMPT = """\
You are a quality reviewer for research-question prompts that will be sent
to an AI research agent. The agent gets only the RQ plus a list of
resources (datasets, models, constraints) and must DESIGN its own
experiments. So the RQ must give it enough context to know what to study,
without revealing how the paper studied it.

Judge the given RQ on these criteria:

(1) SELF-CONTAINED -- a reader who has not seen the paper or its abstract
    can tell exactly what is being investigated. Specialized terms have
    inline definitions. Concrete dimensions / datasets / settings / model
    classes are named when relevant.

(2) NEUTRAL ABOUT METHODOLOGY -- the RQ does NOT name the paper's specific
    technique, intervention, or proposed solution. Examples:
      Bad:  "Does <Method-X> improve <task>?"            (leaks the method)
      Bad:  "Can prompting with X mitigate Y?"           (leaks intervention)
      Good: "How does model performance change as <factor> varies?"

(3) NEUTRAL ABOUT FINDINGS -- the RQ does NOT spoil the conclusion.
    Examples:
      Bad:  "Do larger models perform worse on <task>?"  (leaks direction)
      Good: "How does <task> performance scale with model size?"

(4) SHAPE -- one question, ~50 words max, "RQ1:" / "RQ2:" prefix preserved
    when the original had one.

Output ONLY a JSON object, no markdown or commentary:

{
  "valid": true | false,
  "issues": [
    "<one short imperative phrase per issue, e.g. 'inline-define sycophancy'>",
    ...
  ]
}

If the RQ passes all four criteria, set valid=true and issues=[].
"""


def polish_research_question(
    rq_node: dict,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    issues: Optional[list[str]] = None,
) -> str:
    """Rewrite an RQ node's question text to be self-contained, using the
    leaf experiments under it as operational context. When ``issues`` is
    non-empty, the polish is targeted -- the prompt is told to fix exactly
    those flagged problems. Returns the original text unchanged if there
    are no leaves or the LLM call fails."""
    original = rq_node.get("node") or rq_node.get("description") or ""
    leaves = [c for c in (rq_node.get("children") or [])
              if isinstance(c, dict) and c.get("type") == "leaf"]
    if not leaves or not original.strip():
        return original

    user_lines = [
        f"Original research question: {original}",
        f"Paper description: {rq_node.get('description', '')}",
        "",
        "Experiments that operationalize this question:",
    ]
    for lf in leaves:
        ds = lf.get("dataset") or []
        mt = lf.get("metrics") or []
        proto = lf.get("protocol_or_setup") or ""
        user_lines.append(
            f"- {lf.get('task', '')}\n"
            f"  datasets: {ds}\n"
            f"  metrics:  {mt}\n"
            f"  protocol: {proto}"
        )
    if issues:
        user_lines.append("")
        user_lines.append("Issues a reviewer flagged with the current wording "
                          "(address EACH):")
        for it in issues:
            user_lines.append(f"- {it}")

    try:
        response = chat(
            model=model,
            messages=[
                {"role": "system", "content": _POLISH_RQ_PROMPT},
                {"role": "user", "content": "\n".join(user_lines)},
            ],
            max_tokens=400,
            temperature=0.0,
            api_key=api_key,
        )
        polished = response.content.strip().strip('"').strip()
        if not polished or len(polished) > 600:
            return original
        return polished
    except Exception as e:
        print(f"[polish-rq] failed for {original[:60]!r}: {e}")
        return original


def detect_rq_quality(
    rq_text: str,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
) -> dict:
    """LLM-judge the RQ on self-containment + methodology/finding neutrality.
    Returns ``{"valid": bool, "issues": [str, ...]}``. On any error returns
    ``{"valid": True, "issues": []}`` so callers fall through gracefully."""
    try:
        response = chat(
            model=model,
            messages=[
                {"role": "system", "content": _DETECT_RQ_PROMPT},
                {"role": "user", "content": f"RQ to judge:\n\n{rq_text}"},
            ],
            max_tokens=300,
            temperature=0.0,
            api_key=api_key,
        )
        raw = response.content.strip()
        raw = re.sub(r"^```(?:json)?\s*\n", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\n```\s*$", "", raw, flags=re.MULTILINE).strip()
        verdict = json.loads(raw)
        if not isinstance(verdict, dict):
            return {"valid": True, "issues": []}
        verdict.setdefault("valid", True)
        issues = verdict.get("issues") or []
        verdict["issues"] = issues if isinstance(issues, list) else []
        return verdict
    except Exception as e:
        print(f"[detect-rq] failed for {rq_text[:60]!r}: {e}")
        return {"valid": True, "issues": []}


def polish_until_valid(
    rq_node: dict,
    model: str = "gpt-4o",
    max_iters: int = 3,
    api_key: Optional[str] = None,
) -> dict:
    """Detect-then-polish loop. Stops as soon as the detector says valid or
    after ``max_iters`` polish attempts. Returns ``{"text": str,
    "iterations": int, "final_issues": list[str]}``."""
    current = rq_node.get("node") or rq_node.get("description") or ""
    last_issues: list[str] = []
    iterations = 0
    while True:
        verdict = detect_rq_quality(current, model=model, api_key=api_key)
        if verdict.get("valid"):
            return {"text": current, "iterations": iterations,
                    "final_issues": []}
        last_issues = verdict.get("issues") or []
        if iterations >= max_iters:
            break
        polished = polish_research_question(
            {**rq_node, "node": current},
            model=model,
            api_key=api_key,
            issues=last_issues,
        )
        iterations += 1
        if not polished or polished == current:
            break
        current = polished
    return {"text": current, "iterations": iterations,
            "final_issues": last_issues}


def extract_rqs_from_tree(
    tree: dict,
    polish: bool = True,
    polish_model: str = "gpt-4o",
    max_polish_iters: int = 3,
) -> list[dict]:
    """Return a list of (rq_text, subtree) pairs from a parsed problem tree.

    When ``polish`` is True (default), each RQ goes through a
    detect-then-polish loop: an LLM judge checks whether the question is
    self-contained AND neutral about the paper's methodology / findings,
    and if it flags issues, ``polish_research_question`` is called with the
    issue list. The loop stops as soon as the judge says valid or after
    ``max_polish_iters`` rewrites. The final text replaces the RQ node's
    ``node`` field so it propagates into both ``rq_text`` and the persisted
    subtree.

    Each subtree is a full tree dict with the same ``paper`` metadata but with
    ``problem_tree.children`` containing only the single RQ and its leaves.
    """
    problem_tree = tree.get("problem_tree", {})
    children = problem_tree.get("children", [])
    if not children:
        return []

    paper_meta = tree.get("paper", {})
    root_fields = {k: v for k, v in problem_tree.items() if k != "children"}

    rqs = []
    for i, rq_node in enumerate(children):
        if polish:
            result = polish_until_valid(
                rq_node, model=polish_model, max_iters=max_polish_iters,
            )
            polished = result["text"]
            if polished and polished != rq_node.get("node"):
                rq_node = {**rq_node, "node": polished}

        # Derive a short RQ label from the node text (e.g. "RQ1: Does X …")
        rq_text = rq_node.get("node", rq_node.get("description", f"RQ{i + 1}"))

        subtree = {
            "paper": paper_meta,
            "problem_tree": {
                **root_fields,
                "children": [rq_node],
            },
        }
        rqs.append({"index": i, "rq_text": rq_text, "subtree": subtree})

    return rqs


def main():
    parser = argparse.ArgumentParser(
        description="Parse a research paper (PDF) into a problem tree via LLM."
    )
    parser.add_argument("pdf_path", help="Path to the paper PDF")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model (default: gpt-4o)")
    parser.add_argument("-o", "--output", default=None, help="Output JSON path")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_path = args.output or str(pdf_path.with_suffix("")) + "_tree.json"

    tree = parse_paper_to_tree(
        str(pdf_path),
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)

    print(f"Tree saved to {output_path}")


if __name__ == "__main__":
    main()
