"""Parse a research paper PDF into a structured problem tree via LLM."""

import json
import re
import argparse
import os
from pathlib import Path
from typing import Optional

import openai
from dotenv import load_dotenv

from .pdf_utils import extract_text_from_pdf

load_dotenv()


TREE_PARSING_PROMPT = """
You are a research-paper expert specializing in methodological analysis and problem
decomposition of scientific studies.

**GOAL**
- Fully comprehend the paper, understand its core research problems and experiments.
- Parse the paper and construct a **three-level research-problem tree**:
  * **Root node** -- the single, broadest research problem tackled by the paper.
  * **Depth-1 nodes (Research Questions)** -- concrete, self-contained research
    questions that the authors pose or implicitly address to tackle the root problem.
    Each depth-1 node must be phrased as a clear, answerable research question.
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
        "node": "RQ1: [A concrete research question]",
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
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be set in the environment or passed directly")

    client = openai.OpenAI(api_key=api_key)
    paper_text = extract_text_from_pdf(pdf_path)
    if not paper_text.strip():
        raise ValueError(f"No text could be extracted from {pdf_path}")

    user_message = (
        "Below is the full text of the paper:\n\n"
        f"{paper_text}\n\n"
        "Now parse this paper according to the instructions."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": TREE_PARSING_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    raw_output = response.choices[0].message.content
    tree = extract_json_from_response(raw_output)

    usage = response.usage
    print(
        f"Tokens  --  input: {usage.prompt_tokens}  |  "
        f"output: {usage.completion_tokens}  |  "
        f"total: {usage.total_tokens}"
    )

    return tree


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
