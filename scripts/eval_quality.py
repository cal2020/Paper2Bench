"""Diagnostic quality report for Paper2Bench outputs.

Walks an output directory produced by `paper2bench run` and emits a
markdown report with three signals per task:

  - **Structural** -- expected files exist; YAML/JSON parse; tree has
    3-level shape.

  - **Content quality** (LLM-judged) -- 0-100 score for
    ``instruction.txt`` (self-contained, concrete, executable, no
    leakage) and ``instruction_gt.txt`` (contains the base instruction,
    plan is faithful, conclusion is sharp, no numeric / URL leakage).

  - **HuggingFace validity** -- dataset IDs in ``task_config.yaml`` that
    look like HF paths actually resolve via the HF Hub.

This is a diagnostic tool, not a gate. It does not pass/fail anything
and always exits 0. The real solvability check lives in
``paper2bench/verify.py`` (closed-loop F1 against the paper's
conclusion). Use this script when you want a quick quality snapshot
after changing prompts.

Usage:

    python scripts/eval_quality.py <output-dir>
    python scripts/eval_quality.py output_beta --no-llm   # structural only
    python scripts/eval_quality.py output_beta --task-id mamba_rq0
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Repo-internal imports (script is meant to run from a repo checkout).
from paper2bench.llm import chat


# --------------------------------------------------------------------------- #
# Required keys / thresholds
# --------------------------------------------------------------------------- #

REQUIRED_TASK_CONFIG_KEYS = ("task_id", "research_question")


# --------------------------------------------------------------------------- #
# LLM-judge prompts (kept narrow; one prompt per artifact)
# --------------------------------------------------------------------------- #

_INSTRUCTION_JUDGE_PROMPT = """\
You are a quality reviewer for benchmark task instructions written for an
autonomous ML agent. The agent gets ONLY the instruction.txt -- no paper,
no abstract, no extra context -- and must design and run experiments to
answer the research question stated in the instruction.

Score the instruction on a 0-100 rubric:

(a) SELF-CONTAINED (0-30)  -- a competent ML practitioner who has never
    seen the source paper can understand exactly what to do. All
    specialized terms are defined inline; the RQ is unambiguous.

(b) CONCRETE  (0-25)  -- specific models, datasets (with loader IDs when
    relevant), metrics, and a compute/budget envelope are named. Not
    vague ("use a transformer") but pinned ("evaluate Llama-3-8B on
    MMLU with 5-shot accuracy, max 50 GPU-hours").

(c) EXECUTABLE  (0-25)  -- the agent could plausibly run this on a
    workstation. Inputs are obtainable, no hidden dependencies, the
    research question is decidable from quantitative results.

(d) NO LEAKAGE  (0-20)  -- the instruction must NOT contain any of:
       - URLs to the source paper's code repository, project page,
         author GitHub/HuggingFace, or "official implementation" links;
       - the paper's conclusion (the answer to the research question);
       - the paper's specific numerical results / effect sizes;
       - a step-by-step recipe that describes the paper's exact method
         (e.g. "use selective scan with input-dependent A, B, C and
         hardware-aware parallel kernel") -- the agent must design and
         implement the method itself.
    IMPORTANT: naming the paper's method (e.g. "Mamba", "Self-RAG",
    "Chain-of-Verification") is FINE -- it serves as a label for what
    is being studied, not as the answer. Do not deduct for method name
    mentions. Only deduct for the four leakage classes above.

Output ONLY this JSON, no markdown or commentary:

{
  "score": <0-100 integer>,
  "components": {"self_contained": <int>, "concrete": <int>,
                  "executable": <int>, "no_leakage": <int>},
  "issues": ["<one short phrase per concrete problem>", ...]
}
"""


_INSTRUCTION_GT_JUDGE_PROMPT = """\
You are a quality reviewer for "instruction_gt.txt" files. An
instruction_gt is the base instruction PLUS a supplementary plan PLUS a
ground-truth conclusion. It is used to grade an autonomous agent's
attempt at the task: the agent does NOT see the GT, but the evaluator
uses it to check whether the agent's results match what the source
paper reported.

You will receive the base instruction and the full instruction_gt
(which should include the base instruction's content plus the new
sections). Judge the supplementary plan and ground-truth conclusion.

Score on 0-100:

(a) CONTAINS BASE  (0-15)  -- the instruction_gt actually includes the
    base instruction's content; nothing was dropped.

(b) PLAN FAITHFUL  (0-35)  -- the supplementary plan describes the
    experimental steps a researcher reproducing the source paper would
    follow. It is concrete (datasets, models, metrics named) and
    consistent with the base instruction.

(c) CONCLUSION SHARP  (0-30)  -- the ground-truth conclusion is a
    direct answer to the research question (a substantive claim, not
    a yes/no prefix or vague paraphrase). It is the kind of statement
    a paper's abstract would make.

(d) NO LEAKAGE  (0-20)  -- the plan must NOT contain:
       - URLs to the source paper's code/project page;
       - specific numerical results / effect sizes from the source paper
         (general direction OK; "we observed X% improvement" not OK).
    Naming the paper's method by name is FINE -- the ground truth
    conclusion will reference it. Do not deduct for method name
    mentions; only for the two leakage classes above.

Output ONLY this JSON:

{
  "score": <0-100 integer>,
  "components": {"contains_base": <int>, "plan_faithful": <int>,
                  "conclusion_sharp": <int>, "no_leakage": <int>},
  "issues": ["<one short phrase per problem>", ...]
}
"""


def _parse_json_blob(text: str) -> dict | None:
    """Strip optional code fences and parse the first JSON object."""
    import re

    raw = text.strip()
    raw = re.sub(r"^```(?:json)?\s*\n", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\n```\s*$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None


def judge_instruction(instruction_text: str, model: str = "gpt-4o") -> dict:
    """LLM-judge an instruction.txt. Returns a dict with score/components/issues."""
    response = chat(
        model=model,
        messages=[
            {"role": "system", "content": _INSTRUCTION_JUDGE_PROMPT},
            {"role": "user", "content": instruction_text[:12000]},
        ],
        max_tokens=600,
        temperature=0.0,
    )
    verdict = _parse_json_blob(response.content) or {}
    verdict.setdefault("score", 0)
    verdict.setdefault("issues", [])
    return verdict


def judge_instruction_gt(
    instruction_text: str, instruction_gt_text: str, model: str = "gpt-4o"
) -> dict:
    user = (
        f"--- BASE INSTRUCTION ---\n{instruction_text[:6000]}\n\n"
        f"--- INSTRUCTION_GT (full) ---\n{instruction_gt_text[:12000]}"
    )
    response = chat(
        model=model,
        messages=[
            {"role": "system", "content": _INSTRUCTION_GT_JUDGE_PROMPT},
            {"role": "user", "content": user},
        ],
        max_tokens=600,
        temperature=0.0,
    )
    verdict = _parse_json_blob(response.content) or {}
    verdict.setdefault("score", 0)
    verdict.setdefault("issues", [])
    return verdict


# --------------------------------------------------------------------------- #
# Per-task evaluation
# --------------------------------------------------------------------------- #


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class TaskReport:
    task_id: str
    tree_path: Path | None
    task_dir: Path
    checks: list[CheckResult] = field(default_factory=list)
    rq_quality: list[dict] = field(default_factory=list)
    instruction_score: int | None = None
    instruction_issues: list[str] = field(default_factory=list)
    gt_score: int | None = None
    gt_issues: list[str] = field(default_factory=list)
    hf_invalid: list[str] = field(default_factory=list)

    @property
    def structural_ok(self) -> bool:
        return all(c.passed for c in self.checks) and not self.hf_invalid


def _load_yaml(path: Path) -> tuple[dict | None, str]:
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return None, f"top-level YAML is {type(data).__name__}, expected dict"
        return data, ""
    except Exception as e:
        return None, f"YAML parse error: {e}"


def _load_json(path: Path) -> tuple[Any, str]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f), ""
    except Exception as e:
        return None, f"JSON parse error: {e}"


def _validate_tree_shape(tree: Any) -> tuple[bool, str]:
    if not isinstance(tree, dict):
        return False, "tree is not a dict"
    pt = tree.get("problem_tree")
    if not isinstance(pt, dict):
        return False, "problem_tree missing or not a dict"
    children = pt.get("children")
    if not isinstance(children, list) or not children:
        return False, "problem_tree.children empty or missing"
    for i, rq in enumerate(children):
        if not isinstance(rq, dict):
            return False, f"RQ[{i}] is not a dict"
        if not (rq.get("node") or rq.get("description")):
            return False, f"RQ[{i}] has no 'node' or 'description'"
    return True, ""


_HF_PATH_RE = __import__("re").compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


def _check_hf_datasets(cfg: dict) -> list[str]:
    """Return list of dataset names whose HF loader could not be resolved.

    Only validates loaders that look like a proper HF path (``org/name``).
    Skips Python snippets, URLs, comments, file paths — those are
    legitimate non-HF dataset sources (synthetic generators, external
    downloads) and aren't this checker's job."""
    invalid: list[str] = []
    try:
        from huggingface_hub import dataset_info
    except ImportError:
        return invalid

    sources = cfg.get("data_sources") or cfg.get("datasets") or []
    if not isinstance(sources, list):
        return invalid

    for src in sources:
        if not isinstance(src, dict):
            continue
        loader = src.get("hf_id") or src.get("loader") or src.get("path")
        name = src.get("name") or loader or "<unnamed>"
        if not loader or not isinstance(loader, str):
            continue
        loader = loader.strip()
        # Real HF paths are one slash, no whitespace, no leading sigils.
        if not _HF_PATH_RE.match(loader):
            continue
        try:
            dataset_info(loader)
        except Exception:
            invalid.append(f"{name} (loader={loader})")
    return invalid


def _find_rq_texts(tree: dict) -> list[str]:
    out: list[str] = []
    for rq in tree.get("problem_tree", {}).get("children", []):
        text = rq.get("node") or rq.get("description")
        if text:
            out.append(text)
    return out


def evaluate_task(
    task_dir: Path,
    tree_path: Path | None,
    *,
    base_task_id: str,
    enable_llm: bool,
    model: str,
) -> TaskReport:
    task_id = task_dir.name
    report = TaskReport(task_id=task_id, tree_path=tree_path, task_dir=task_dir)

    # 1. Required files exist
    cfg_path = task_dir / "task_config.yaml"
    inst_path = task_dir / "instruction.txt"
    inst_gt_path = task_dir / "instruction_gt.txt"
    for p in (cfg_path, inst_path, inst_gt_path):
        report.checks.append(
            CheckResult(f"exists:{p.name}", p.exists(),
                        "" if p.exists() else f"missing {p}")
        )

    # 2. task_config.yaml parses + required keys
    cfg, err = _load_yaml(cfg_path) if cfg_path.exists() else (None, "missing")
    report.checks.append(
        CheckResult("task_config:parses", cfg is not None, err)
    )
    if cfg is not None:
        missing = [k for k in REQUIRED_TASK_CONFIG_KEYS if not cfg.get(k)]
        report.checks.append(
            CheckResult("task_config:required_keys", not missing,
                        f"missing keys: {missing}" if missing else "")
        )

    # 3. instruction.txt non-empty
    inst_text = ""
    if inst_path.exists():
        inst_text = inst_path.read_text(encoding="utf-8").strip()
    report.checks.append(
        CheckResult("instruction:non_empty", bool(inst_text),
                    "" if inst_text else "instruction.txt is empty")
    )

    # 4. instruction_gt.txt non-empty and contains instruction
    inst_gt_text = ""
    if inst_gt_path.exists():
        inst_gt_text = inst_gt_path.read_text(encoding="utf-8").strip()
    report.checks.append(
        CheckResult("instruction_gt:non_empty", bool(inst_gt_text),
                    "" if inst_gt_text else "instruction_gt.txt is empty")
    )

    # 5. tree shape (only for the base paper)
    if tree_path and tree_path.exists():
        tree, err = _load_json(tree_path)
        if tree is None:
            report.checks.append(CheckResult("tree:parses", False, err))
        else:
            report.checks.append(CheckResult("tree:parses", True))
            ok, why = _validate_tree_shape(tree)
            report.checks.append(CheckResult("tree:shape", ok, why))

    # 6. HF dataset validity
    if cfg is not None:
        invalid = _check_hf_datasets(cfg)
        report.hf_invalid = invalid
        report.checks.append(
            CheckResult("hf:datasets_resolve", not invalid,
                        f"unresolved: {invalid}" if invalid else "")
        )

    if not enable_llm:
        return report

    # 8. LLM-judged: instruction.txt
    # (RQ-only quality check intentionally dropped: the instruction.txt
    # judge already evaluates the rendered RQ, and using
    # detect_rq_quality here flagged method-name mentions as leaks even
    # though method names serve as labels, not answers.)
    if inst_text:
        try:
            v = judge_instruction(inst_text, model=model)
            report.instruction_score = int(v.get("score") or 0)
            report.instruction_issues = list(v.get("issues") or [])
        except Exception as e:
            report.instruction_score = 0
            report.instruction_issues = [f"judge crashed: {e}"]

    # 9. LLM-judged: instruction_gt.txt
    if inst_text and inst_gt_text:
        try:
            v = judge_instruction_gt(inst_text, inst_gt_text, model=model)
            report.gt_score = int(v.get("score") or 0)
            report.gt_issues = list(v.get("issues") or [])
        except Exception as e:
            report.gt_score = 0
            report.gt_issues = [f"judge crashed: {e}"]

    return report


# --------------------------------------------------------------------------- #
# Walk + report
# --------------------------------------------------------------------------- #


def find_tasks(output_dir: Path) -> list[tuple[Path, Path | None]]:
    """Return [(task_dir, tree_path_for_base_paper)] for every task subdir.

    A task_id is split-rqs style (e.g. ``mamba_rq0``) or combined style
    (e.g. ``mamba``). For split-rqs we point all sub-tasks at the same
    base paper's tree (``trees/mamba_tree.json``).
    """
    tasks_root = output_dir / "tasks"
    trees_root = output_dir / "trees"
    if not tasks_root.exists():
        return []
    result = []
    for task_dir in sorted(tasks_root.iterdir()):
        if not task_dir.is_dir():
            continue
        task_id = task_dir.name
        # Strip _rq<N> suffix if present
        base = task_id
        if "_rq" in task_id:
            base = task_id.rsplit("_rq", 1)[0]
        tree_path = trees_root / f"{base}_tree.json"
        if not tree_path.exists():
            tree_path = None
        result.append((task_dir, tree_path))
    return result


def _median(xs: list[int]) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def render_report(reports: list[TaskReport]) -> str:
    lines = ["# Paper2Bench Quality Report", "",
             "*Diagnostic snapshot — no pass/fail. Higher is better.*", ""]

    n = len(reports)
    struct_ok = sum(1 for r in reports if r.structural_ok)
    inst_scores = [r.instruction_score for r in reports if r.instruction_score is not None]
    gt_scores = [r.gt_score for r in reports if r.gt_score is not None]

    lines.append("## Headline")
    lines.append(f"- Structural OK: **{struct_ok}/{n}**")
    if inst_scores:
        lines.append(f"- `instruction.txt` score: **median {_median(inst_scores):.0f}/100**, "
                     f"range {min(inst_scores)}-{max(inst_scores)}")
    if gt_scores:
        lines.append(f"- `instruction_gt.txt` score: **median {_median(gt_scores):.0f}/100**, "
                     f"range {min(gt_scores)}-{max(gt_scores)}")
    lines.append("")

    # Summary table
    lines.append("| Task | Structural | instruction | instruction_gt | HF |")
    lines.append("|---|---|---|---|---|")
    for r in reports:
        struct = "OK" if r.structural_ok else "issues"
        inst = "—" if r.instruction_score is None else f"{r.instruction_score}/100"
        gt = "—" if r.gt_score is None else f"{r.gt_score}/100"
        hf = "OK" if not r.hf_invalid else f"{len(r.hf_invalid)} bad"
        lines.append(f"| `{r.task_id}` | {struct} | {inst} | {gt} | {hf} |")
    lines.append("")

    # Per-task notes: structural issues, judge issue lists
    has_notes = any(
        (not r.structural_ok) or r.instruction_issues or r.gt_issues or r.hf_invalid
        for r in reports
    )
    if has_notes:
        lines.append("## Notes")
        for r in reports:
            task_notes: list[str] = []
            for c in r.checks:
                if not c.passed:
                    task_notes.append(f"- structural: **{c.name}** — {c.detail}")
            if r.hf_invalid:
                task_notes.append(f"- HF datasets unresolved: {r.hf_invalid}")
            if r.instruction_issues:
                task_notes.append(f"- instruction.txt judge ({r.instruction_score}/100):")
                for issue in r.instruction_issues:
                    task_notes.append(f"    - {issue}")
            if r.gt_issues:
                task_notes.append(f"- instruction_gt.txt judge ({r.gt_score}/100):")
                for issue in r.gt_issues:
                    task_notes.append(f"    - {issue}")
            if task_notes:
                lines.append(f"\n### `{r.task_id}`")
                lines.extend(task_notes)
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", help="Directory produced by paper2bench run")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM-judged checks (structural + HF only)")
    parser.add_argument("--task-id", action="append",
                        help="Limit to specific task_id(s); may repeat")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--report", default=None,
                        help="Path for the markdown report "
                             "(default: <output_dir>/QUALITY_REPORT.md)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        sys.exit(f"output dir does not exist: {output_dir}")

    pairs = find_tasks(output_dir)
    if args.task_id:
        wanted = set(args.task_id)
        pairs = [(d, t) for (d, t) in pairs if d.name in wanted]
    if not pairs:
        sys.exit("no task directories found")

    reports: list[TaskReport] = []
    for task_dir, tree_path in pairs:
        print(f"[eval] {task_dir.name} ...", flush=True)
        try:
            r = evaluate_task(
                task_dir, tree_path,
                base_task_id=task_dir.name,
                enable_llm=not args.no_llm,
                model=args.model,
            )
        except Exception:
            traceback.print_exc()
            r = TaskReport(task_id=task_dir.name, tree_path=tree_path, task_dir=task_dir)
            r.checks.append(CheckResult("evaluator", False, "uncaught exception"))
        reports.append(r)

    report_text = render_report(reports)
    report_path = Path(args.report) if args.report else output_dir / "QUALITY_REPORT.md"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\nReport written to {report_path}")
    # Echo the headline lines so the run also prints a quick summary.
    for line in report_text.splitlines()[2:9]:
        if line:
            print(line)

    return 0


if __name__ == "__main__":
    sys.exit(main())
