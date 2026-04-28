"""Rewrite a verbose paper-level conclusion into a direct answer to the
research question being asked. Used to align ground-truth answers with
their RQ for benchmark evaluation -- a yes/no RQ should get a yes/no
answer, not a 3-paragraph summary of every finding in the paper.
"""
from typing import Optional

from .llm import chat


_POLISH_CONCLUSION_PROMPT = """\
You rewrite verbose research conclusions into direct, concise answers to a
specific research question.

INPUT: a research question (RQ), and the paper's full conclusion paragraph
addressing it (often a multi-sentence summary that mixes the answer with
methodology details and secondary findings).

OUTPUT: a focused answer that directly answers the RQ -- nothing more.

Rules:
- The FIRST sentence must directly address the question. For yes/no RQs,
  start with "Yes" or "No" (with appropriate hedge if the original is
  hedged). For "how" RQs, lead with the mechanism/cause. For "what / when
  / why" RQs, lead with the finding.
- PRESERVE the paper's actual finding -- direction, magnitude, and any
  hedge language already in the original. Do NOT soften, strengthen, or
  add speculation.
- DROP methodology details: specific dataset/model/metric names,
  implementation specifics, and step-by-step procedure -- unless they are
  essential to *what* the answer is.
- DROP secondary findings or related observations that don't directly
  answer the RQ.
- Keep it under ~80 words. One short paragraph or 1-3 sentences.

Output ONLY the polished answer text. No quotes, no markdown, no
commentary.
"""


def polish_conclusion(
    research_question: str,
    raw_conclusion: str,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
) -> str:
    """Return a focused answer to ``research_question`` derived from
    ``raw_conclusion``. Returns the original text unchanged on any error
    or when the input is empty."""
    if not raw_conclusion or not raw_conclusion.strip():
        return raw_conclusion
    if not research_question or not research_question.strip():
        return raw_conclusion
    try:
        response = chat(
            model=model,
            messages=[
                {"role": "system", "content": _POLISH_CONCLUSION_PROMPT},
                {"role": "user", "content":
                    f"RQ: {research_question}\n\n"
                    f"Conclusion: {raw_conclusion}"},
            ],
            max_tokens=400,
            temperature=0.0,
            api_key=api_key,
        )
        polished = response.content.strip().strip('"').strip()
        if not polished or len(polished) > 1500:
            return raw_conclusion
        return polished
    except Exception as e:
        print(f"[polish-conclusion] failed: {e}")
        return raw_conclusion
