"""Unified CLI for Paper2Bench."""

import argparse
import os
import sys
from pathlib import Path


def cmd_download(args):
    from .downloader import find_and_download
    result = find_and_download(
        query=args.query,
        task_id=args.task_id,
        output_dir=args.output_dir,
        auto=args.auto,
        max_results=args.max_results,
    )
    if result:
        print(f"\nPDF saved to: {result}")
    else:
        print("\nNo PDF downloaded.")
        sys.exit(1)


def cmd_parse(args):
    from .tree_parser import parse_paper_to_tree
    import json
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        sys.exit(f"PDF not found: {pdf_path}")
    output_path = args.output or str(pdf_path.with_suffix("")) + "_tree.json"
    tree = parse_paper_to_tree(
        str(pdf_path), model=args.model,
        max_tokens=args.max_tokens, temperature=args.temperature,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)
    print(f"Tree saved to {output_path}")


def cmd_classify(args):
    from .paper_classifier import classify_paper
    import json
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        sys.exit(f"PDF not found: {pdf_path}")
    result = classify_paper(str(pdf_path), model=args.model)
    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Classification saved to {args.output}")


def cmd_extract(args):
    from .instruction_generator import extract_yaml_from_pdf
    import yaml
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        sys.exit(f"PDF not found: {pdf_path}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_content = extract_yaml_from_pdf(
        str(pdf_path),
        research_question=args.research_question,
        task_id=args.task_id,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        tree_json_path=args.tree,
        paper_type=args.paper_type,
        validate_hf_loaders=not args.skip_hf_validation,
    )
    try:
        parsed = yaml.safe_load(yaml_content)
        if not isinstance(parsed, dict):
            raise ValueError("YAML did not parse to a dict")
    except Exception as e:
        print(f"[warn] LLM output may not be valid YAML: {e}")
    output_path.write_text(yaml_content, encoding="utf-8")
    print(f"\nConfig saved to {output_path}")


def cmd_render(args):
    from .instruction_generator import render_instruction
    import yaml
    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    instruction = render_instruction(config, template_path=args.template)
    output_path.write_text(instruction, encoding="utf-8")
    print(f"instruction.txt saved to {output_path}")


def cmd_plan(args):
    from .plan_generator import generate_supplementary_plan, build_instruction_gt
    instruction_path = Path(args.instruction)
    if not instruction_path.exists():
        sys.exit(f"Instruction not found: {instruction_path}")
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        sys.exit(f"PDF not found: {pdf_path}")
    base_instruction = instruction_path.read_text(encoding="utf-8")
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


def cmd_specextract(args):
    from .paper_spec import extract_paper_spec
    import json

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        sys.exit(f"PDF not found: {pdf_path}")

    spec = extract_paper_spec(
        str(pdf_path),
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"\nPaper spec saved to {output_path}")


def cmd_generate(args):
    from .benchmark_generator import generate_variants
    import json

    spec_path = Path(args.spec)
    if not spec_path.exists():
        sys.exit(f"Spec not found: {spec_path}")
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        sys.exit(f"PDF not found: {pdf_path}")

    instances = generate_variants(
        str(spec_path),
        str(pdf_path),
        paper_type=args.paper_type,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write the full manifest
    manifest_path = output_dir / "instances.json"
    manifest_path.write_text(
        json.dumps(instances, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"\nManifest saved to {manifest_path}")

    # Write each instance's instruction as a separate file
    clean_count = 0
    leaked_count = 0
    unknown_count = 0
    for inst in instances:
        inst_id = inst.get("id", "unknown")
        import re
        if not isinstance(inst_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", inst_id):
            raise ValueError("Variant ID must be a normalized identifier")
        inst_dir = output_dir / inst_id
        inst_dir.mkdir(parents=True, exist_ok=True)

        # instruction.txt (rendered from task_config by the generator)
        (inst_dir / "instruction.txt").write_text(
            inst.get("instruction", ""), encoding="utf-8",
        )
        # task_config.yaml -- the resource spec used to render the instruction
        cfg = inst.get("task_config")
        if cfg:
            try:
                import yaml as _yaml
                (inst_dir / "task_config.yaml").write_text(
                    _yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8",
                )
                from .bundles import write_draft_manifest
                write_draft_manifest(inst_dir, dict(cfg, task_id=inst_id))
            except Exception as e:
                print(f"  [{inst_id}] failed to write task_config.yaml: {e}")
        # metadata.json (everything except the rendered instruction text)
        meta = {k: v for k, v in inst.items() if k != "instruction"}
        (inst_dir / "metadata.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8",
        )

        leaked = inst.get("leakage_check", {}).get("leaked")
        if leaked is True:
            leaked_count += 1
        elif leaked is False:
            clean_count += 1
        else:
            unknown_count += 1

    print(f"\n{len(instances)} instance(s) written to {output_dir}/")
    print(f"  No leak detected (advisory): {clean_count}  |  Leaked: {leaked_count}  |  Ungraded: {unknown_count}")
    if leaked_count or unknown_count:
        print("  [warn] Leaked or ungraded instances require review or regeneration.")


def cmd_verify(args):
    from .verify import verify_instruction_gt
    import json

    instruction_gt_path = Path(args.instruction_gt)
    if not instruction_gt_path.exists():
        sys.exit(f"instruction_gt not found: {instruction_gt_path}")
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        sys.exit(f"PDF not found: {pdf_path}")

    gt_text = None
    if args.ground_truth:
        gt_path = Path(args.ground_truth)
        if gt_path.exists():
            gt_text = gt_path.read_text(encoding="utf-8").strip()
        else:
            # treat as inline text
            gt_text = args.ground_truth

    metrics = verify_instruction_gt(
        str(instruction_gt_path),
        str(pdf_path),
        data_dir=args.data_dir,
        agent_model=args.agent_model,
        eval_model=args.eval_model,
        max_turns=args.max_turns,
        ground_truth=gt_text,
        research_question=args.research_question or "",
        eval_mode=args.eval_mode,
        data_files=args.data_file,
        executor=_executor(args),
        execution_options={"timeout_seconds": args.code_timeout},
    )

    # Persist results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nVerification results saved to {output_path}")

    if not metrics["pass"]:
        sys.exit(1)


def _run_pipeline_for_rq(
    pdf_path, task_id, rq_text, tree_json_path, tasks_dir, args, subtree=None,
    paper_type="llm_evaluation",
):
    """Run steps 3-6 (extract → render → plan → verify) for a single research question.

    When *subtree* is provided the per-RQ subtree JSON is written next to the
    main tree so that downstream LLM calls only see the relevant branch.
    """
    from .instruction_generator import extract_yaml_from_pdf, render_instruction
    from .plan_generator import generate_supplementary_plan, build_instruction_gt
    import json
    import yaml

    tasks_dir = Path(tasks_dir)
    tasks_dir.mkdir(parents=True, exist_ok=True)

    # If a subtree was given, persist it so extract/plan prompts see only this RQ
    if subtree is not None:
        rq_tree_path = str(tasks_dir / "subtree.json")
        Path(rq_tree_path).write_text(
            json.dumps(subtree, indent=2, ensure_ascii=False), encoding="utf-8",
        )
    else:
        rq_tree_path = tree_json_path

    # Step 3: Extract YAML
    print(f"\n{'='*50}\nStep 3: Extracting task config YAML (paper_type={paper_type})\n{'='*50}")
    yaml_path = str(tasks_dir / "task_config.yaml")
    yaml_content = extract_yaml_from_pdf(
        pdf_path,
        research_question=rq_text,
        task_id=task_id,
        model=args.model,
        tree_json_path=rq_tree_path,
        paper_type=paper_type,
        validate_hf_loaders=not getattr(args, "skip_hf_validation", False),
    )
    Path(yaml_path).write_text(yaml_content, encoding="utf-8")
    print(f"Config saved to {yaml_path}")

    # Validate YAML
    try:
        config = yaml.safe_load(yaml_content)
        if not isinstance(config, dict):
            raise ValueError("YAML did not parse to a dict")
    except Exception as e:
        print(f"[warn] YAML may need manual editing: {e}")
        print(f"Edit {yaml_path} then run: paper2bench render --config {yaml_path} -o {tasks_dir}/instruction.txt")
        return None

    # Step 4: Render instruction.txt
    print(f"\n{'='*50}\nStep 4: Rendering instruction.txt\n{'='*50}")
    instruction_path = tasks_dir / "instruction.txt"
    instruction = render_instruction(config, template_path=args.template)
    instruction_path.write_text(instruction, encoding="utf-8")
    print(f"instruction.txt saved to {instruction_path}")

    # Step 5: Generate instruction_gt.txt
    print(f"\n{'='*50}\nStep 5: Generating instruction_gt.txt\n{'='*50}")
    supplement = generate_supplementary_plan(
        pdf_path,
        base_instruction=instruction,
        model=args.model,
        tree_json_path=rq_tree_path,
    )
    instruction_gt = build_instruction_gt(instruction, supplement)
    gt_path = tasks_dir / "instruction_gt.txt"
    gt_path.write_text(instruction_gt, encoding="utf-8")
    print(f"instruction_gt.txt saved to {gt_path}")

    from .bundles import write_draft_manifest, digest
    write_draft_manifest(tasks_dir, dict(config, task_id=task_id, research_question=rq_text),
                         source={"paper_sha256": digest(Path(pdf_path).read_bytes())})

    # Step 6: (optional) Verify instruction_gt.txt
    metrics = None
    if args.verify:
        from .verify import verify_instruction_gt
        print(f"\n{'='*50}\nStep 6: Verifying instruction_gt.txt\n{'='*50}")
        verify_output = tasks_dir / "verify_results.json"
        metrics = verify_instruction_gt(
            str(gt_path),
            pdf_path,
            data_dir=str(tasks_dir / "data") if (tasks_dir / "data").exists() else None,
            agent_model=args.model,
            eval_model=args.model,
            research_question=rq_text,
            eval_mode=getattr(args, "eval_mode", "claim_f1"),
            executor=_executor(args),
            execution_options={"timeout_seconds": args.code_timeout},
        )
        verify_output.write_text(
            json.dumps(metrics, indent=2), encoding="utf-8",
        )
        print(f"Verification results saved to {verify_output}")

    return {
        "task_id": task_id,
        "rq": rq_text,
        "yaml": yaml_path,
        "instruction": str(instruction_path),
        "instruction_gt": str(gt_path),
        "verify": metrics,
    }


def cmd_run(args):
    """Full pipeline: download/locate PDF -> parse -> extract -> render -> plan."""
    from .downloader import find_and_download
    from .tree_parser import parse_paper_to_tree, extract_rqs_from_tree
    import json

    out = Path(args.output_dir)
    task_id = args.task_id
    papers_dir = out / "papers"
    trees_dir = out / "trees"

    # Step 1: Get PDF (either from local path or arXiv search)
    if args.pdf:
        pdf_path = str(Path(args.pdf).resolve())
        if not Path(pdf_path).exists():
            sys.exit(f"PDF not found: {pdf_path}")
        print(f"\n{'='*50}\nStep 1: Using local PDF\n{'='*50}")
        print(f"  {pdf_path}")
    elif args.query:
        print(f"\n{'='*50}\nStep 1: Downloading paper from arXiv\n{'='*50}")
        pdf_path = find_and_download(
            query=args.query,
            task_id=task_id,
            output_dir=str(papers_dir),
            auto=args.auto,
        )
        if not pdf_path:
            sys.exit("Failed to download paper.")
    else:
        sys.exit("Provide either --pdf (local file) or --query (arXiv search).")

    if not args.split_rqs:
        if not args.research_question:
            raise ValueError("--research-question is required unless --split-rqs is used")
        from .api import run as generate
        from .runtime_config import get_context
        context = get_context()
        result = generate(pdf=pdf_path, research_question=args.research_question,
                          paper_type=args.paper_type, output_dir=out, task_id=task_id,
                          model=args.model, template=args.template,
                          validate_hf_loaders=not args.skip_hf_validation, verify=args.verify,
                          eval_mode=args.eval_mode, runtime_config=context.config if context else None,
                          executor=_executor(args) if args.verify else None,
                          execution_options={"timeout_seconds": args.code_timeout},
                          model_runs_dir=args.model_runs_dir)
        print(json.dumps({"task_id": result.task_id, "instruction": str(result.instruction_path),
                          "reference_plan": str(result.instruction_gt_path), "extras": result.extras}, indent=2))
        return

    # Step 2: Parse tree
    print(f"\n{'='*50}\nStep 2: Parsing problem tree\n{'='*50}")
    trees_dir.mkdir(parents=True, exist_ok=True)
    tree_path = str(trees_dir / f"{task_id}_tree.json")
    tree = parse_paper_to_tree(pdf_path, model=args.model)
    with open(tree_path, "w", encoding="utf-8") as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)
    print(f"Tree saved to {tree_path}")

    # Step 2b: Classify paper type (drives extraction prompt + template)
    if args.paper_type:
        paper_type = args.paper_type
        print(f"\n[classify] Using user-specified paper_type: {paper_type}")
    else:
        from .paper_classifier import classify_paper
        print(f"\n{'='*50}\nStep 2b: Classifying paper type\n{'='*50}")
        classification = classify_paper(pdf_path, model=args.model)
        paper_type = classification["type"]
        print(f"[classify] type={paper_type}  confidence={classification.get('confidence')}")
        print(f"[classify] rationale: {classification.get('rationale')}")

    # ----- combined (default) vs split-rqs mode -----
    if not args.split_rqs and not args.research_question:
        sys.exit("--research-question is required unless --split-rqs is used.")

    if args.split_rqs:
        rqs = extract_rqs_from_tree(tree)
        if not rqs:
            sys.exit("No research questions found in the problem tree.")
        print(f"\n[split-rqs] Found {len(rqs)} research question(s). "
              "Generating separate tasks per RQ.\n")

        results = []
        for rq in rqs:
            rq_label = f"rq{rq['index']}"
            sub_task_id = f"{task_id}_{rq_label}"
            sub_tasks_dir = out / "tasks" / sub_task_id
            print(f"\n{'#'*50}")
            print(f"# {sub_task_id}: {rq['rq_text'][:70]}")
            print(f"{'#'*50}")
            result = _run_pipeline_for_rq(
                pdf_path, sub_task_id, rq["rq_text"],
                tree_path, sub_tasks_dir, args,
                subtree=rq["subtree"],
                paper_type=paper_type,
            )
            if result:
                results.append(result)

        # Summary
        print(f"\n{'='*50}\nPipeline complete for: {task_id} "
              f"({len(results)} RQ task(s))\n{'='*50}")
        print(f"  PDF:   {pdf_path}")
        print(f"  Tree:  {tree_path}")
        for r in results:
            v = ""
            if r["verify"] is not None:
                status = "PASS" if r["verify"]["pass"] else "FAIL"
                if r['verify'].get('eval_mode') == 'llm_judge':
                    v = f"  score={r['verify']['score']}/100 ({status})"
                else:
                    v = f"  F1={r['verify']['f1']:.1f}% ({status})"
            print(f"  [{r['task_id']}] {r['instruction_gt']}{v}")
    else:
        # Combined mode -- all RQs in one prompt (original behaviour)
        tasks_dir = out / "tasks" / task_id
        result = _run_pipeline_for_rq(
            pdf_path, task_id, args.research_question,
            tree_path, tasks_dir, args,
            paper_type=paper_type,
        )

        print(f"\n{'='*50}\nPipeline complete for: {task_id}\n{'='*50}")
        print(f"  PDF:              {pdf_path}")
        print(f"  Tree:             {tree_path}")
        if result:
            print(f"  Config:           {result['yaml']}")
            print(f"  instruction.txt:  {result['instruction']}")
            print(f"  instruction_gt:   {result['instruction_gt']}")
            if result["verify"] is not None:
                status = "PASS" if result["verify"]["pass"] else "FAIL"
                if result['verify'].get('eval_mode') == 'llm_judge':
                    print(f"  Verify score:     {result['verify']['score']}/100 ({status})")
                else:
                    print(f"  Verify F1:        {result['verify']['f1']:.1f}% ({status})")


def _executor(args):
    from .executor_factory import build_executor
    return build_executor(image=getattr(args, "execution_image", None))


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="paper2bench",
        description="Paper2Bench: Convert academic papers into benchmark tasks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- download ---
    dl = subparsers.add_parser("download", help="Search arXiv and download a paper PDF")
    dl.add_argument("query", help="Search query (title and/or authors)")
    dl.add_argument("--task-id", required=True, help="Task ID for naming the PDF")
    dl.add_argument("-o", "--output-dir", default="./papers", help="Output directory")
    dl.add_argument("--auto", action="store_true", help="Auto-select first result")
    dl.add_argument("--max-results", type=int, default=5)
    dl.set_defaults(func=cmd_download)

    # --- parse ---
    ps = subparsers.add_parser("parse", help="Parse a paper PDF into a problem tree")
    ps.add_argument("pdf_path", help="Path to the paper PDF")
    ps.add_argument("-o", "--output", default=None, help="Output JSON path")
    ps.add_argument("--model", default="gpt-4o")
    ps.add_argument("--max-tokens", type=int, default=16384)
    ps.add_argument("--temperature", type=float, default=0.0)
    ps.set_defaults(func=cmd_parse)

    # --- classify ---
    cl = subparsers.add_parser("classify", help="Classify a paper into a benchmark archetype")
    cl.add_argument("--pdf", required=True)
    cl.add_argument("--model", default="gpt-4o")
    cl.add_argument("-o", "--output", default=None, help="Optional JSON output path")
    cl.set_defaults(func=cmd_classify)

    # --- extract ---
    ex = subparsers.add_parser("extract", help="Extract task config YAML from a paper PDF")
    ex.add_argument("--pdf", required=True)
    ex.add_argument("--research-question", required=True)
    ex.add_argument("--task-id", required=True)
    ex.add_argument("--tree", default=None, help="Optional problem tree JSON")
    ex.add_argument("--model", default="gpt-4o")
    ex.add_argument("--max-tokens", type=int, default=4096)
    ex.add_argument("--temperature", type=float, default=0.0)
    ex.add_argument("--paper-type", default="llm_evaluation",
                     choices=["llm_evaluation", "novel_architecture", "empirical_study"],
                     help="Benchmark archetype (defaults to llm_evaluation)")
    ex.add_argument("--skip-hf-validation", action="store_true",
                     help="Skip HuggingFace loader verification")
    ex.add_argument("-o", "--output", required=True)
    ex.set_defaults(func=cmd_extract)

    # --- render ---
    rn = subparsers.add_parser("render", help="Render YAML config into instruction.txt")
    rn.add_argument("--config", required=True)
    rn.add_argument("--template", default=None, help="Custom template file")
    rn.add_argument("-o", "--output", required=True)
    rn.set_defaults(func=cmd_render)

    # --- plan ---
    pl = subparsers.add_parser("plan", help="Generate instruction_gt.txt from instruction.txt + PDF")
    pl.add_argument("--instruction", required=True, help="Path to instruction.txt")
    pl.add_argument("--pdf", required=True)
    pl.add_argument("--tree", default=None)
    pl.add_argument("--model", default="gpt-4o")
    pl.add_argument("--max-tokens", type=int, default=16384)
    pl.add_argument("--temperature", type=float, default=0.0)
    pl.add_argument("-o", "--output", required=True)
    pl.set_defaults(func=cmd_plan)

    # --- verify ---
    vf = subparsers.add_parser("verify", help="Verify instruction_gt.txt by running an agent and checking F1 >= 80")
    vf.add_argument("--instruction-gt", required=True, help="Path to instruction_gt.txt")
    vf.add_argument("--pdf", required=True, help="Path to the paper PDF")
    vf.add_argument("--data-dir", default=None, help="Directory with task data files")
    vf.add_argument("--ground-truth", default=None, help="Ground-truth conclusion text or file path (auto-extracted from PDF if omitted)")
    vf.add_argument("--agent-model", default="gpt-4o", help="Model for the agent")
    vf.add_argument("--eval-model", default="gpt-4o", help="Model for evaluation")
    vf.add_argument("--max-turns", type=int, default=30, help="Max agent turns")
    vf.add_argument("--eval-mode", default="claim_f1",
                     choices=["claim_f1", "llm_judge"],
                     help="Evaluation method: claim_f1 (precision/recall/F1 via claim "
                          "decomposition, default) or llm_judge (single-call rubric scoring)")
    vf.add_argument("--research-question", default=None,
                     help="Research question (used by llm_judge mode for context)")
    vf.add_argument("-o", "--output", default="./verify_results.json", help="Output JSON path")
    vf.set_defaults(func=cmd_verify)

    # --- specextract ---
    se = subparsers.add_parser("specextract",
                               help="Extract 7-component paper specification from a PDF")
    se.add_argument("--pdf", required=True, help="Path to the paper PDF")
    se.add_argument("--model", default="gpt-4o")
    se.add_argument("--max-tokens", type=int, default=16384)
    se.add_argument("--temperature", type=float, default=0.0)
    se.add_argument("-o", "--output", required=True, help="Output JSON path for paper spec")
    se.set_defaults(func=cmd_specextract)

    # --- generate ---
    gn = subparsers.add_parser("generate",
                               help="Generate benchmark variant instances from a paper spec")
    gn.add_argument("--spec", required=True, help="Path to paper spec JSON (from specextract)")
    gn.add_argument("--pdf", required=True, help="Path to the paper PDF")
    gn.add_argument("--paper-type", default=None,
                     choices=["llm_evaluation", "novel_architecture", "empirical_study"],
                     help="Skip auto-classification and force a benchmark archetype")
    gn.add_argument("--model", default="gpt-4o")
    gn.add_argument("--max-tokens", type=int, default=16384)
    gn.add_argument("--temperature", type=float, default=0.3)
    gn.add_argument("-o", "--output-dir", default="./variants", help="Output directory for instances")
    gn.set_defaults(func=cmd_generate)

    # --- run (full pipeline) ---
    rp = subparsers.add_parser("run", help="Run full pipeline: PDF -> parse -> extract -> render -> plan")
    rp.add_argument("--task-id", required=True)
    rp.add_argument("--pdf", default=None, help="Path to a local PDF (skips download)")
    rp.add_argument("--query", default=None, help="arXiv search query (downloads PDF)")
    rp.add_argument("--research-question", default=None,
                     help="Research question (required unless --split-rqs is used)")
    rp.add_argument("--output-dir", default="./output", help="Base output directory")
    rp.add_argument("--model", default="gpt-4o")
    rp.add_argument("--template", default=None, help="Custom instruction template")
    rp.add_argument("--auto", action="store_true", help="Auto-select first arXiv result")
    rp.add_argument("--paper-type", default=None,
                     choices=["llm_evaluation", "novel_architecture", "empirical_study"],
                     help="Skip auto-classification and force a benchmark archetype")
    rp.add_argument("--skip-hf-validation", action="store_true",
                     help="Skip HuggingFace loader verification in the extract step")
    rp.add_argument("--split-rqs", action="store_true",
                     help="Generate a separate task per research question instead of one combined task")
    rp.add_argument("--verify", action="store_true", help="Run verification step (requires computer-executable experiments)")
    rp.add_argument("--eval-mode", default="claim_f1",
                     choices=["claim_f1", "llm_judge"],
                     help="Verification evaluator: claim_f1 (default) or llm_judge")
    rp.set_defaults(func=cmd_run)

    from .cli_runtime import dispatch, register
    register(subparsers)
    args = parser.parse_args(argv)
    try:
        raw_arguments = argv if argv is not None else sys.argv[1:]
        if (args.runtime_config or os.environ.get("PAPER2BENCH_RUNTIME_CONFIG")) and any(
                a.split("=", 1)[0] in {"--model", "--agent-model", "--eval-model"} for a in raw_arguments):
            raise ValueError("With a runtime profile, set per-role models in that profile rather than legacy model flags")
        # Validate path-like task IDs before legacy output directories are created.
        if hasattr(args, "task_id"):
            import re
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.task_id):
                raise ValueError("task-id must be an identifier, not a path")
        result = dispatch(args)
        if isinstance(result, int) and result:
            raise SystemExit(result)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"paper2bench: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
