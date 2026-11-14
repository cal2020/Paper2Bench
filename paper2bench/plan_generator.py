"""Generate instruction_gt.txt by enriching instruction.txt with detailed plans from the paper."""

import json
import argparse
import os
from pathlib import Path
from typing import Optional


from .llm import chat
from .pdf_utils import extract_text_from_pdf



SUPPLEMENTARY_PLAN_PROMPT = """\
You are an expert research engineer specializing in reproducing machine learning
and NLP experiments from academic papers.

A coding agent has already been given a base instruction (shown below under
"BASE INSTRUCTION") that tells it the research question, available models,
available datasets, and a few high-level hints.

Your job is to read the full paper and write a **supplementary experiment plan**
that gives the agent the additional detail it needs to run the experiments
faithfully -- **without repeating anything already stated in the base instruction**.

Specifically, generate a Markdown document with ONLY the following sections
(omit any section that has nothing new to add beyond the base instruction):

## Experimental Design
For each distinct experiment in the paper:
- Numbered task with a short name
- Precise procedure (input format, prompting strategy, number of samples,
  any shuffling / ordering of conditions, splits to use)
- What to record (raw outputs, extracted answers, intermediate statistics)
- Correspondence to a Figure / Table / Section in the paper

## Evaluation Details
- Exact metric definitions and computation (formulas, library calls, edge cases)
- How to normalize or pre-process model outputs before scoring
- Aggregation strategy (per-sample -> per-condition -> overall)

## Implementation Notes
- Key pitfalls or caveats explicitly mentioned in the paper
- Any special prompt templates, delimiters, or output-parsing logic required
- Order of operations that must be respected

---

Rules:
1. Do NOT restate the research question, model names, dataset names, or any
   content already present in the base instruction. Do NOT expose the conclusions.
2. Be concrete: use exact numbers, thresholds, prompt wording, and metric
   definitions from the paper. Mark missing values as [uncertain].
3. Be complete: cover every experiment that maps to a figure, table, or named
   result section.
4. Output ONLY the Markdown content (no preamble, no postamble).
"""


def generate_supplementary_plan(
    pdf_path: str,
    base_instruction: str,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    max_tokens: int = 16384,
    temperature: float = 0.0,
    tree_json_path: Optional[str] = None,
) -> str:
    """
    Read a paper PDF plus its existing base instruction and return only the
    supplementary detail (Markdown) that the base instruction does not cover.
    """
    paper_text = extract_text_from_pdf(pdf_path)
    if not paper_text.strip():
        raise ValueError(f"No text could be extracted from {pdf_path}")

    user_message = (
        "BASE INSTRUCTION (already given to the agent -- do NOT repeat this):\n"
        "```\n"
        f"{base_instruction.strip()}\n"
        "```\n\n"
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

    user_message += "\n\nNow write the supplementary experiment plan."

    response = chat(
        model=model,
        messages=[
            {"role": "system", "content": SUPPLEMENTARY_PLAN_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        api_key=api_key,
    )
    supplement = response.content
    print(
        f"Tokens  --  input: {response.input_tokens}  |  "
        f"output: {response.output_tokens}  |  "
        f"total: {response.total_tokens}  |  provider: {response.provider}"
    )

    return supplement


def build_instruction_gt(base_instruction: str, supplement: str) -> str:
    """Combine the base instruction with the supplementary plan."""
    return (
        base_instruction.rstrip()
        + "\n\n"
        + "---\n\n"
        + supplement.strip()
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate instruction_gt.txt by enriching instruction.txt with paper details."
    )
    parser.add_argument("--instruction", required=True, help="Path to the base instruction.txt")
    parser.add_argument("--pdf", required=True, help="Path to the paper PDF")
    parser.add_argument("--tree", default=None, help="Optional problem tree JSON")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model (default: gpt-4o)")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("-o", "--output", required=True, help="Output path for instruction_gt.txt")
    args = parser.parse_args()

    instruction_path = Path(args.instruction)
    if not instruction_path.exists():
        raise FileNotFoundError(f"Instruction not found: {instruction_path}")
    base_instruction = instruction_path.read_text(encoding="utf-8")

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    supplement = generate_supplementary_plan(
        str(pdf_path),
        base_instruction=base_instruction,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        tree_json_path=args.tree,
    )

    instruction_gt = build_instruction_gt(base_instruction, supplement)
    output_path.write_text(instruction_gt, encoding="utf-8")
    print(f"\ninstruction_gt.txt saved to {output_path}")


if __name__ == "__main__":
    main()
