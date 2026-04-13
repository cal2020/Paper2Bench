"""Verify that instruction_gt.txt is clear enough for an agent to understand.

Runs a coding agent with the ground-truth instruction, extracts its conclusion,
decomposes both the conclusion and a reference answer into atomic claims, then
computes claim-level precision / recall / F1.  The instruction passes
verification when F1 > 80.

Evaluation approach adapted from FIRE-Bench (eval/RAGChecker/eval.py).
"""

import json
import os
import re
import subprocess
import sys
import textwrap
import time
import random
import shutil
from pathlib import Path
from typing import Optional

import openai
from dotenv import load_dotenv

from .pdf_utils import extract_text_from_pdf

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
F1_THRESHOLD = 80  # percent

AGENT_SYSTEM_PROMPT = textwrap.dedent("""\
You are an autonomous research agent. You are given a research task with instructions.
You have a working directory with data files and utility scripts.
You must write and execute Python code to carry out experiments.

RULES:
- Write Python code in ```python ... ``` blocks. Each block will be executed immediately.
- You can install packages with pip if needed.
- After each code execution you will see stdout/stderr output.
- Iterate: analyze results, fix errors, run more experiments.
- When you are done, write your final conclusions in a block starting with "FINAL CONCLUSION:".
- Be concise. Focus on running experiments and producing quantitative results.
- Do NOT use web search. Run FULL experiments from the data provided.
""")

MAX_TURNS = 30
CODE_TIMEOUT = 300  # seconds per code block

# ---------------------------------------------------------------------------
# Ground-truth conclusion extraction
# ---------------------------------------------------------------------------
CONCLUSION_EXTRACTION_PROMPT = """\
You are an expert research scientist.  Read the full text of the paper below and
produce a concise summary of its **main experimental findings and conclusions**.

Rules:
1. Focus on quantitative results and qualitative trends reported in the paper.
2. Do NOT include background, motivation, or method descriptions -- only findings.
3. Be concrete: mention directions of effects, relative magnitudes, and key
   conditions under which results hold.
4. Output ONLY the summary (no preamble, no postamble).
"""


def extract_ground_truth_conclusion(
    pdf_path: str,
    *,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> str:
    """Use an LLM to extract the expected conclusion from a paper PDF."""
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be set")

    client = openai.OpenAI(api_key=api_key)
    paper_text = extract_text_from_pdf(pdf_path)
    if not paper_text.strip():
        raise ValueError(f"No text extracted from {pdf_path}")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CONCLUSION_EXTRACTION_PROMPT},
            {"role": "user", "content": paper_text},
        ],
        max_completion_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Simplified agent runner
# ---------------------------------------------------------------------------

def _extract_code_blocks(text: str):
    return re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)


def _extract_conclusion(text: str) -> Optional[str]:
    if "FINAL CONCLUSION:" in text:
        return text.split("FINAL CONCLUSION:", 1)[1].strip()
    return None


def _run_code(code: str, cwd: Path, env: dict) -> str:
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=CODE_TIMEOUT,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += "\nSTDERR:\n" + result.stderr
        if len(output) > 8000:
            output = output[:4000] + "\n... [truncated] ...\n" + output[-4000:]
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: Code execution timed out after 5 minutes."
    except Exception as e:
        return f"ERROR: {e}"


def run_agent(
    instruction_gt_path: str,
    *,
    data_dir: Optional[str] = None,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    max_turns: int = MAX_TURNS,
) -> str:
    """Run a coding agent with *instruction_gt.txt* and return its conclusion.

    Parameters
    ----------
    instruction_gt_path : str
        Path to the ground-truth instruction file.
    data_dir : str, optional
        Directory containing task data files to copy into the sandbox.
    model : str
        OpenAI model used by the agent.
    api_key : str, optional
        OpenAI API key (falls back to env var).
    max_turns : int
        Maximum conversation turns for the agent loop.

    Returns
    -------
    str
        The agent's final conclusion text.
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be set")

    client = openai.OpenAI(api_key=api_key)
    instruction_text = Path(instruction_gt_path).read_text(encoding="utf-8").strip()

    # ---- sandbox setup ----
    timestamp = time.strftime("%Y%m%d%H%M%S")
    rd = random.randint(10000, 99999)
    sandbox = Path(os.environ.get("PAPER2BENCH_SANDBOX_ROOT", "./runs")) / f"verify_{timestamp}_{rd}"

    if data_dir and Path(data_dir).exists() and any(Path(data_dir).iterdir()):
        shutil.copytree(data_dir, sandbox)
    else:
        sandbox.mkdir(parents=True, exist_ok=True)

    # write .env into sandbox
    with open(sandbox / ".env", "w") as f:
        f.write(f"OPENAI_API_KEY={api_key}\n")

    exec_env = os.environ.copy()
    exec_env["OPENAI_API_KEY"] = api_key

    # file listing for the agent
    sandbox_files = []
    for p in sorted(sandbox.rglob("*")):
        if p.is_file() and ".env" not in p.name:
            sandbox_files.append(str(p.relative_to(sandbox)))
    files_listing = "\n".join(sandbox_files[:50]) or "(empty)"

    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"## Research Task\n\n{instruction_text}\n\n"
                f"## Working Directory\n"
                f"Your working directory is: {sandbox}\n\n"
                f"## Files Available\n```\n{files_listing}\n```\n\n"
                "Begin your research. Write Python code to execute experiments."
            ),
        },
    ]

    # ---- log ----
    log_dir = sandbox / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_lines = [f"model: {model}", f"timestamp: {timestamp}", "=" * 40]

    conclusion = None
    for turn in range(max_turns):
        print(f"  [verify] Turn {turn + 1}/{max_turns} ...")

        api_kwargs = {"model": model, "messages": messages}
        if any(model.startswith(p) for p in ("o4", "o3", "o1")):
            api_kwargs["max_completion_tokens"] = 16384
        else:
            api_kwargs["max_completion_tokens"] = 4096

        try:
            resp = client.chat.completions.create(**api_kwargs)
            assistant_msg = resp.choices[0].message.content or ""
        except Exception as e:
            assistant_msg = f"API ERROR: {e}"
            log_lines.append(f"\n--- TURN {turn + 1} (API ERROR) ---\n{assistant_msg}")
            break

        messages.append({"role": "assistant", "content": assistant_msg})
        log_lines.append(f"\n--- TURN {turn + 1} (ASSISTANT) ---\n{assistant_msg}")

        conclusion = _extract_conclusion(assistant_msg)
        if conclusion:
            log_lines.append(f"\n--- FINAL CONCLUSION ---\n{conclusion}")
            break

        code_blocks = _extract_code_blocks(assistant_msg)
        if not code_blocks:
            messages.append({
                "role": "user",
                "content": "Please write Python code to run experiments, or provide your FINAL CONCLUSION: if you are done.",
            })
            continue

        all_output = []
        for i, code in enumerate(code_blocks):
            output = _run_code(code, sandbox, exec_env)
            all_output.append(f"[Code block {i + 1} output]\n{output}")
            log_lines.append(f"\n--- CODE BLOCK {i + 1} ---\n{code}\n--- OUTPUT ---\n{output}")

        messages.append({"role": "user", "content": f"Execution results:\n\n" + "\n\n".join(all_output)})

    # Force a conclusion if none was produced
    if not conclusion:
        messages.append({
            "role": "user",
            "content": "Time is up. Please provide your FINAL CONCLUSION: summarizing all findings.",
        })
        try:
            api_kwargs["messages"] = messages
            resp = client.chat.completions.create(**api_kwargs)
            final = resp.choices[0].message.content or ""
            conclusion = _extract_conclusion(final) or final
            log_lines.append(f"\n--- FORCED FINAL ---\n{final}")
        except Exception as e:
            conclusion = f"Failed to get conclusion: {e}"

    # Save log
    log_file = log_dir / "log.log"
    log_lines.append(f"\n{'=' * 40}\n" + json.dumps({"result": conclusion}))
    log_file.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"  [verify] Agent log saved to {log_file}")

    return conclusion


# ---------------------------------------------------------------------------
# Claim-based evaluation  (adapted from FIRE-Bench eval/RAGChecker/eval.py)
# ---------------------------------------------------------------------------

def _decompose_into_claims(text: str, client: openai.OpenAI, model: str) -> list[str]:
    """Break a piece of text into atomic sub-claims using an LLM."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Given a claim, break it down into a list of meaningful and reasonable subclaims. "
                    "Each subclaim should be a standalone and specific statement. "
                    "DO NOT infer any detail, context information, or background knowledge "
                    "that is not mentioned in the original text. "
                    'Return the result as a JSON object: {"subclaims": ["<first>", "<second>", ...]}'
                ),
            },
            {"role": "user", "content": f"Input statement: {text}"},
        ],
        temperature=0.0,
        max_completion_tokens=2048,
    )
    raw = response.choices[0].message.content
    parsed = _parse_json(raw)
    return parsed.get("subclaims", [])


def _check_claim(claim: str, reference: str, client: openai.OpenAI, model: str) -> bool:
    """Return True if *claim* is entailed by / overlaps with *reference*."""
    prompt = textwrap.dedent(f"""\
        You are a neutral judge asked to determine if a claim is part of the given reference.
        The claim does not need to mention every aspect of the reference.
        If it overlaps with part of the reference, return True.
        If it conflicts with or cannot be inferred from the reference, return False.

        =================  INPUT  =================
        [1]  CLAIM:
        {claim}

        [2]  REFERENCE:
        {reference}
        ===========================================

        **Output format (JSON)**
        ```json
        {{
          "rationale": "<2-3 sentence explanation>",
          "result": <true/false>
        }}
        ```
    """)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an impartial judge engine. Follow the instructions exactly and output *only* the JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_completion_tokens=1024,
    )
    parsed = _parse_json(response.choices[0].message.content)
    return bool(parsed.get("result", False))


def _extract_core_idea(text: str, client: openai.OpenAI, model: str) -> str:
    """Distil a conclusion down to its core insight (no specific numbers)."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert summarizer. "
                    "Given a conclusion, extract only the core insight or main idea, "
                    "omitting all concrete values, specific numbers, background details, methods, "
                    "file names, or references to artifacts. "
                    "Focus on general trends or main conclusions/findings, and express them in a "
                    "generalized way without referring to any precise data or background context. "
                    "DO NOT infer any detail, context information, or background knowledge "
                    "that is not mentioned in the original conclusion."
                ),
            },
            {"role": "user", "content": f"Input: {text}"},
        ],
        temperature=0.0,
        max_completion_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def _parse_json(text: str) -> dict:
    """Best-effort JSON extraction from LLM output."""
    # Try to find a JSON block in markdown fences first
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Last resort: find first { ... }
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def evaluate_conclusion(
    agent_conclusion: str,
    ground_truth: str,
    *,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
) -> dict:
    """Compute claim-level precision, recall, and F1.

    1. Extract the core idea from the agent's conclusion.
    2. Decompose the ground truth into atomic claims and check each against the
       agent conclusion  →  **precision**  (how many GT claims are covered).
    3. Decompose the agent conclusion into atomic claims and check each against
       the ground truth  →  **recall**  (how many agent claims are correct).
    4. F1 = harmonic mean.

    Returns
    -------
    dict
        Keys: ``precision``, ``recall``, ``f1`` (0-100 scale),
        ``gt_claims``, ``agent_claims``, ``pass`` (bool, f1 >= F1_THRESHOLD).
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be set")

    client = openai.OpenAI(api_key=api_key)

    # Distil the agent conclusion
    core_idea = _extract_core_idea(agent_conclusion, client, model)
    print(f"  [eval] Core idea: {core_idea[:200]}...")

    # --- Precision: GT claims checked against agent response ---
    gt_claims = _decompose_into_claims(ground_truth, client, model)
    print(f"  [eval] Ground-truth decomposed into {len(gt_claims)} claims")
    precision_hits = 0
    for claim in gt_claims:
        if _check_claim(claim, core_idea, client, model):
            precision_hits += 1
    precision = (precision_hits / len(gt_claims) * 100) if gt_claims else 0.0

    # --- Recall: agent claims checked against ground truth ---
    agent_claims = _decompose_into_claims(core_idea, client, model)
    print(f"  [eval] Agent conclusion decomposed into {len(agent_claims)} claims")
    recall_hits = 0
    for claim in agent_claims:
        if _check_claim(claim, ground_truth, client, model):
            recall_hits += 1
    recall = (recall_hits / len(agent_claims) * 100) if agent_claims else 0.0

    # --- F1 ---
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        "precision": round(precision, 1),
        "recall": round(recall, 1),
        "f1": round(f1, 1),
        "gt_claims_count": len(gt_claims),
        "agent_claims_count": len(agent_claims),
        "pass": f1 >= F1_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def verify_instruction_gt(
    instruction_gt_path: str,
    pdf_path: str,
    *,
    data_dir: Optional[str] = None,
    agent_model: str = "gpt-4o",
    eval_model: str = "gpt-4o",
    api_key: Optional[str] = None,
    max_turns: int = MAX_TURNS,
    ground_truth: Optional[str] = None,
) -> dict:
    """End-to-end verification of an instruction_gt.txt.

    Steps:
        1. Extract expected conclusion from the paper (or use provided gt).
        2. Run a coding agent with the instruction_gt.
        3. Evaluate the agent's conclusion against the expected conclusion.
        4. Return metrics and pass/fail.
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY")

    # Step 1 -- ground-truth conclusion
    if ground_truth:
        print("[verify] Using provided ground-truth conclusion.")
        gt_conclusion = ground_truth
    else:
        print("[verify] Step 1: Extracting expected conclusion from the paper ...")
        gt_conclusion = extract_ground_truth_conclusion(
            pdf_path, model=eval_model, api_key=api_key,
        )
    print(f"[verify] Ground truth ({len(gt_conclusion)} chars): {gt_conclusion[:200]}...")

    # Step 2 -- run agent
    print("\n[verify] Step 2: Running agent with instruction_gt.txt ...")
    agent_conclusion = run_agent(
        instruction_gt_path,
        data_dir=data_dir,
        model=agent_model,
        api_key=api_key,
        max_turns=max_turns,
    )
    print(f"[verify] Agent conclusion ({len(agent_conclusion)} chars): {agent_conclusion[:200]}...")

    # Step 3 -- evaluate
    print("\n[verify] Step 3: Evaluating agent conclusion ...")
    metrics = evaluate_conclusion(
        agent_conclusion, gt_conclusion, model=eval_model, api_key=api_key,
    )

    # Summary
    status = "PASS" if metrics["pass"] else "FAIL"
    print(f"\n{'=' * 50}")
    print(f"[verify] Precision : {metrics['precision']:.1f}%")
    print(f"[verify] Recall    : {metrics['recall']:.1f}%")
    print(f"[verify] F1        : {metrics['f1']:.1f}%")
    print(f"[verify] Threshold : {F1_THRESHOLD}%")
    print(f"[verify] Result    : {status}")
    print(f"{'=' * 50}")

    return metrics
