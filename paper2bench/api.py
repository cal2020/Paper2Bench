"""Public Python SDK entrypoint for Paper2Bench.

Example::

    import paper2bench
    result = paper2bench.run(
        pdf="paper.pdf",
        research_question="Does X improve Y?",
        api_key="sk-...",          # OpenAI or Anthropic key (router decides by model name)
        output_dir="./out",
    )
    print(result.instruction_path)
    print(result.paper_type)        # auto-classified unless paper_type= was passed

The SDK orchestrates the same five core stages as ``paper2bench run``:
parse tree -> (auto-)classify -> extract YAML -> render instruction.txt ->
build instruction_gt.txt. Optionally verifies (claim-F1 or LLM-judge).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from .instruction_generator import extract_yaml_from_pdf, render_instruction
from .paper_classifier import classify_paper
from .plan_generator import build_instruction_gt, generate_supplementary_plan
from .tree_parser import parse_paper_to_tree


@dataclass
class BenchmarkResult:
    """Output bundle from a successful pipeline run."""

    task_id: str
    paper_type: str
    pdf_path: Path
    tree_path: Path
    config_path: Path
    instruction_path: Path
    instruction_gt_path: Path
    config: dict
    instruction: str
    instruction_gt: str
    classification: Optional[dict] = None
    verify_metrics: Optional[dict] = None
    extras: dict = field(default_factory=dict)


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return s or "task"


def run(
    *,
    pdf: Union[str, Path],
    research_question: str,
    paper_type: Optional[str] = None,
    api_key: Optional[str] = None,
    output_dir: Union[str, Path] = "./output",
    task_id: Optional[str] = None,
    model: str = "gpt-4o",
    template: Union[str, Path, None] = None,
    validate_hf_loaders: bool = True,
    verify: bool = False,
    eval_mode: str = "claim_f1",
) -> BenchmarkResult:
    """Run the full Paper2Bench pipeline on a local PDF.

    BYOK: pass ``api_key`` (OpenAI or Anthropic, depending on ``model``) or
    let the underlying ``paper2bench.llm.chat`` read ``OPENAI_API_KEY`` /
    ``ANTHROPIC_API_KEY`` from the environment.

    Parameters
    ----------
    pdf : str | Path
        Local path to the paper PDF. The SDK does not download papers; for
        arXiv search use the CLI's ``download`` subcommand or call
        ``paper2bench.downloader.find_and_download`` directly.
    research_question : str
        The question the agent will be asked to investigate.
    paper_type : Optional[str]
        Override the classifier; one of ``llm_evaluation`` /
        ``novel_architecture`` / ``empirical_study``. If ``None``, the
        classifier picks one and its result is stored in
        ``BenchmarkResult.classification``.
    api_key : Optional[str]
        Provider key. Routed by ``model`` name (see ``paper2bench.llm``).
    output_dir : str | Path
        Base directory for ``trees/`` and ``tasks/<task_id>/``.
    task_id : Optional[str]
        Stem of generated filenames. If ``None``, derived from the PDF name.
    model : str
        LLM model ID. Defaults to ``"gpt-4o"`` to match the CLI; override
        with ``"gpt-4o-mini"`` for tier-1-friendly runs.
    template : Optional[str | Path]
        Custom Jinja-style template path for ``render_instruction``.
    validate_hf_loaders : bool
        When ``True``, ``extract_yaml_from_pdf`` validates HuggingFace
        loader IDs against the Hub and demotes hallucinated paths.
    verify : bool
        After producing ``instruction_gt.txt``, run an agent against it and
        score with claim-F1 (or LLM-judge if ``eval_mode="llm_judge"``).
        Adds ``verify_metrics`` to the result.
    eval_mode : str
        ``"claim_f1"`` (default) or ``"llm_judge"`` -- only used when
        ``verify=True``.
    """
    pdf_path = Path(pdf).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    out = Path(output_dir)
    if task_id is None:
        task_id = _slugify(pdf_path.stem)

    trees_dir = out / "trees"
    tasks_dir = out / "tasks" / task_id
    trees_dir.mkdir(parents=True, exist_ok=True)
    tasks_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: parse the problem tree.
    tree = parse_paper_to_tree(str(pdf_path), model=model, api_key=api_key)
    tree_path = trees_dir / f"{task_id}_tree.json"
    tree_path.write_text(
        json.dumps(tree, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Stage 1b: classify (or use override).
    classification: Optional[dict] = None
    if paper_type is None:
        classification = classify_paper(str(pdf_path), model=model, api_key=api_key)
        paper_type = classification.get("type", "llm_evaluation")

    # Stage 2: extract task config YAML (validator runs inside on HF loaders).
    yaml_content = extract_yaml_from_pdf(
        str(pdf_path),
        research_question=research_question,
        task_id=task_id,
        model=model,
        api_key=api_key,
        tree_json_path=str(tree_path),
        paper_type=paper_type,
        validate_hf_loaders=validate_hf_loaders,
    )
    config_path = tasks_dir / "task_config.yaml"
    config_path.write_text(yaml_content, encoding="utf-8")

    config = yaml.safe_load(yaml_content)
    if not isinstance(config, dict):
        raise ValueError(
            f"Extracted YAML did not parse to a dict; saved raw output to {config_path}"
        )

    # Stage 3: render instruction.txt.
    instruction = render_instruction(
        config, template_path=str(template) if template else None
    )
    instruction_path = tasks_dir / "instruction.txt"
    instruction_path.write_text(instruction, encoding="utf-8")

    # Stage 4: supplementary plan and combine into instruction_gt.
    supplement = generate_supplementary_plan(
        str(pdf_path),
        base_instruction=instruction,
        model=model,
        api_key=api_key,
        tree_json_path=str(tree_path),
    )
    instruction_gt = build_instruction_gt(instruction, supplement)
    instruction_gt_path = tasks_dir / "instruction_gt.txt"
    instruction_gt_path.write_text(instruction_gt, encoding="utf-8")

    # Stage 5 (optional): verify by running an agent against instruction_gt.
    verify_metrics: Optional[dict[str, Any]] = None
    if verify:
        from .verify import verify_instruction_gt
        verify_metrics = verify_instruction_gt(
            str(instruction_gt_path),
            str(pdf_path),
            data_dir=str(tasks_dir / "data") if (tasks_dir / "data").exists() else None,
            agent_model=model,
            eval_model=model,
            research_question=research_question,
            eval_mode=eval_mode,
        )
        (tasks_dir / "verify_results.json").write_text(
            json.dumps(verify_metrics, indent=2), encoding="utf-8",
        )

    return BenchmarkResult(
        task_id=task_id,
        paper_type=paper_type,
        pdf_path=pdf_path,
        tree_path=tree_path,
        config_path=config_path,
        instruction_path=instruction_path,
        instruction_gt_path=instruction_gt_path,
        config=config,
        instruction=instruction,
        instruction_gt=instruction_gt,
        classification=classification,
        verify_metrics=verify_metrics,
    )
