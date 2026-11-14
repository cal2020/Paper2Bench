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
from contextlib import nullcontext

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


def _run(
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
    agent_model: Optional[str] = None,
    eval_model: Optional[str] = None,
    executor=None,
    execution_options=None,
    checkpoint=None,
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
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", task_id):
        raise ValueError("task_id must be a normalized identifier, not a path")

    trees_dir = out / "trees"
    tasks_dir = out / "tasks" / task_id
    trees_dir.mkdir(parents=True, exist_ok=True)
    tasks_dir.mkdir(parents=True, exist_ok=True)
    stage = checkpoint.run_stage if checkpoint is not None else lambda name, operation: operation()

    # Stage 1: parse the problem tree.
    tree = stage("parse", lambda: parse_paper_to_tree(str(pdf_path), model=model, api_key=api_key))
    tree_path = trees_dir / f"{task_id}_tree.json"
    tree_path.write_text(
        json.dumps(tree, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Stage 1b: classify (or use override).
    classification: Optional[dict] = None
    if paper_type is None:
        classification = stage("classify", lambda: classify_paper(str(pdf_path), model=model, api_key=api_key))
        paper_type = classification.get("type", "llm_evaluation")

    # Stage 2: extract task config YAML (validator runs inside on HF loaders).
    def validated_extraction():
        value = extract_yaml_from_pdf(str(pdf_path), research_question=research_question,
            task_id=task_id, model=model, api_key=api_key, tree_json_path=str(tree_path),
            paper_type=paper_type, validate_hf_loaders=validate_hf_loaders)
        if not isinstance(yaml.safe_load(value), dict):
            raise ValueError("Extracted YAML requires a mapping before its stage can complete")
        return value

    yaml_content = stage("extract", validated_extraction)
    config_path = tasks_dir / "task_config.yaml"
    config_path.write_text(yaml_content, encoding="utf-8")

    config = yaml.safe_load(yaml_content)
    if not isinstance(config, dict):
        raise ValueError(
            f"Extracted YAML did not parse to a dict; saved raw output to {config_path}"
        )

    # Stage 3: render instruction.txt.
    instruction = stage("render", lambda: render_instruction(
        config, template_path=str(template) if template else None
    ))
    instruction_path = tasks_dir / "instruction.txt"
    instruction_path.write_text(instruction, encoding="utf-8")

    # Stage 4: supplementary plan and combine into instruction_gt.
    supplement = stage("plan", lambda: generate_supplementary_plan(
        str(pdf_path),
        base_instruction=instruction,
        model=model,
        api_key=api_key,
        tree_json_path=str(tree_path),
    ))
    instruction_gt = build_instruction_gt(instruction, supplement)
    instruction_gt_path = tasks_dir / "instruction_gt.txt"
    instruction_gt_path.write_text(instruction_gt, encoding="utf-8")

    # Stage 5 (optional): verify by running an agent against instruction_gt.
    verify_metrics: Optional[dict[str, Any]] = None
    if verify:
        from .verify import verify_instruction_gt
        verify_metrics = stage("verify", lambda: verify_instruction_gt(
            str(instruction_gt_path),
            str(pdf_path),
            data_dir=str(tasks_dir / "data") if (tasks_dir / "data").exists() else None,
            agent_model=agent_model or model,
            eval_model=eval_model or model,
            api_key=api_key,
            executor=executor,
            execution_options=execution_options,
            research_question=research_question,
            eval_mode=eval_mode,
        ))
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


def run(
    *, pdf: Union[str, Path], research_question: str, paper_type: Optional[str] = None,
    api_key: Optional[str] = None, output_dir: Union[str, Path] = "./output",
    task_id: Optional[str] = None, model: str = "gpt-4o", template=None,
    validate_hf_loaders: bool = True, verify: bool = False, eval_mode: str = "claim_f1",
    runtime_config=None, agent_model=None, eval_model=None, executor=None, execution_options=None,
    model_runs_dir="./runs",
    generation_run_id=None,
) -> BenchmarkResult:
    """Generate compatible artifacts using explicit optional runtime role bindings.

    Verification preflights isolation before paid generation. Scientific readiness
    remains needs_review until a curator installs an appropriate protocol.
    """
    from .runtime_config import get_context, load_runtime_config
    from .bundles import digest, write_draft_manifest, atomic_bytes
    from .generation import GenerationCheckpoint, pipeline_identity, task_output_lock
    if runtime_config is not None and (agent_model is not None or eval_model is not None):
        raise ValueError("With a runtime profile, configure agent/judge role models in that profile")
    from .model_runs import durable_runtime_scope
    pdf_path = Path(pdf).resolve()
    if not pdf_path.is_file() or pdf_path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("PDF must be an existing file no larger than 32 MiB")
    with pdf_path.open("rb") as stream:
        source_bytes = stream.read(32 * 1024 * 1024 + 1)
    if len(source_bytes) > 32 * 1024 * 1024:
        raise ValueError("PDF changed beyond its input bound")
    effective_task_id = task_id or _slugify(pdf_path.stem)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", effective_task_id):
        raise ValueError("task_id must be a normalized identifier, not a path")
    resolved_config = load_runtime_config(runtime_config) if runtime_config is not None else None
    plan = {"pdf": str(pdf_path), "paper_sha256": digest(source_bytes), "research_question": research_question,
            "task_id": effective_task_id, "paper_type": paper_type, "model": model,
            "template": str(Path(template).resolve()) if template else None,
            "template_sha256": digest(Path(template).read_bytes()) if template else None,
            "output_dir": str(Path(output_dir).resolve()), "validate_hf_loaders": validate_hf_loaders,
            "verify": verify, "eval_mode": eval_mode, "execution_options": execution_options,
            "pipeline_sha256": pipeline_identity(), "runtime": resolved_config.to_dict() if resolved_config else None}
    if generation_run_id and resolved_config is None:
        raise ValueError("Generation resume requires the original explicit runtime profile")
    scope = durable_runtime_scope(resolved_config, api_key=api_key, runs_dir=model_runs_dir,
                                  run_id=generation_run_id, metadata=plan) if resolved_config is not None else nullcontext(get_context())
    with task_output_lock(Path(output_dir) / "tasks" / effective_task_id), scope as context:
        if context:
            for role in context.config.roles:
                context.resolve(role)
        if verify:
            from .executor_factory import build_executor
            image = context.config.execution.get("image", "python:3.11-slim") if context else "python:3.11-slim"
            executor = executor or build_executor(image=image)
            executor.preflight()
        checkpoint = (GenerationCheckpoint(Path(model_runs_dir) / "generation", context.budget_run_id, plan,
                                           resume=generation_run_id is not None,
                                           max_attempts=context.config.retry.get("max_attempts_per_step", 3))
                      if resolved_config is not None else None)
        snapshot = Path(output_dir).resolve() / ".paper2bench" / "sources" / (plan["paper_sha256"] + ".pdf")
        if snapshot.exists():
            if snapshot.is_symlink() or snapshot.stat().st_size > 32 * 1024 * 1024 or digest(snapshot.read_bytes()) != plan["paper_sha256"]:
                raise ValueError("Immutable source snapshot identity changed")
        else:
            atomic_bytes(snapshot, source_bytes)
        result = _run(pdf=snapshot, research_question=research_question, paper_type=paper_type,
                      api_key=api_key, output_dir=output_dir, task_id=effective_task_id, model=model,
                      template=template, validate_hf_loaders=validate_hf_loaders, verify=verify,
                      eval_mode=eval_mode, agent_model=agent_model, eval_model=eval_model,
                      executor=executor, execution_options=execution_options, checkpoint=checkpoint)
        result.pdf_path = pdf_path
        result.extras["source_snapshot"] = str(snapshot)
        candidate = write_draft_manifest(result.config_path.parent,
                        dict(result.config, task_id=result.task_id, research_question=research_question),
                        source={"paper_sha256": plan["paper_sha256"]})
        result.extras["task_manifest_path"] = str(candidate)
        result.extras["readiness"] = "needs_review"
        if context:
            result.extras["runtime"] = context.config.to_dict()
            result.extras["model_budget"] = context.ledger.snapshot()
            result.extras["model_run_id"] = context.budget_run_id
        if checkpoint:
            result.extras.update(generation_checkpoint=str(checkpoint.path),
                                resumed_stages=list(checkpoint.cache_hits), cache_semantics="historical_artifact_reuse")
            from dataclasses import asdict
            serializable = asdict(result)
            for key in ("pdf_path", "tree_path", "config_path", "instruction_path", "instruction_gt_path"):
                serializable[key] = str(serializable[key])
            checkpoint.complete(serializable)
        return result


def resume_generation(run_id, *, model_runs_dir="./runs", runtime_config=None, api_key=None):
    from .journal import RunJournal, JournalError
    from .generation import GenerationCheckpoint, pipeline_identity, task_output_lock
    from .bundles import digest
    from .runtime_config import load_runtime_config
    journal = RunJournal(model_runs_dir)
    run_record = journal.get(run_id)
    plan = run_record["spec"].get("generation")
    if run_record["spec"].get("kind") != "authoring" or not isinstance(plan, dict):
        raise JournalError("Run has no generation checkpoint")
    pdf_path = Path(plan["pdf"])
    if not pdf_path.is_file() or pdf_path.stat().st_size > 32 * 1024 * 1024 or digest(pdf_path.read_bytes()) != plan["paper_sha256"]:
        raise JournalError("Source paper changed since authoring")
    if pipeline_identity() != plan["pipeline_sha256"]:
        raise JournalError("Authoring software or prompts changed; create a new run")
    if plan.get("template") and digest(Path(plan["template"]).read_bytes()) != plan["template_sha256"]:
        raise JournalError("Authoring template changed; create a new run")
    if load_runtime_config(runtime_config or plan["runtime"]).to_dict() != plan["runtime"]:
        raise JournalError("Authoring runtime configuration changed; create a new run")
    with task_output_lock(Path(plan["output_dir"]) / "tasks" / plan["task_id"]):
        # The output lock also covers live controllers between model calls. A
        # settled budget alone cannot distinguish that interval from a crash.
        run_record = journal.get(run_id)
        if run_record["state"] in {"cancelled", "cancellation_requested"}:
            raise JournalError("Cancelled authoring cannot resume")
        checkpoint = GenerationCheckpoint(Path(model_runs_dir) / "generation", run_id, plan, resume=True)
        checkpoint.ensure_resumable()
        completed = checkpoint.completed_result()
        if run_record["state"] in {"running", "reconciling"}:
            journal.recover_authoring(run_id)
            run_record = journal.get(run_id)
        if completed is not None:
            if run_record["state"] != "succeeded":
                attempt_id = journal.begin_attempt(run_id)
                journal.finish(run_id, attempt_id, "succeeded", {
                    "recovered_generation_checkpoint": True, "budget": run_record["budget"]})
            for key in ("pdf_path", "tree_path", "config_path", "instruction_path", "instruction_gt_path"):
                completed[key] = Path(completed[key])
            completed["extras"]["cache_semantics"] = "historical_artifact_reuse"
            return BenchmarkResult(**completed)
    return globals()["run"](pdf=plan["pdf"], research_question=plan["research_question"],
        paper_type=plan["paper_type"], model=plan["model"], task_id=plan["task_id"],
        output_dir=plan["output_dir"], template=plan["template"], validate_hf_loaders=plan["validate_hf_loaders"],
        verify=plan["verify"], eval_mode=plan["eval_mode"], execution_options=plan["execution_options"],
        runtime_config=runtime_config or plan["runtime"], api_key=api_key,
        model_runs_dir=model_runs_dir, generation_run_id=run_id)
