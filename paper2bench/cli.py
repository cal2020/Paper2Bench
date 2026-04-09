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


def cmd_run(args):
    """Full pipeline: download/locate PDF -> parse -> extract -> render -> plan."""
    from .downloader import find_and_download
    from .tree_parser import parse_paper_to_tree
    from .instruction_generator import extract_yaml_from_pdf, render_instruction
    from .plan_generator import generate_supplementary_plan, build_instruction_gt
    import json
    import yaml

    out = Path(args.output_dir)
    task_id = args.task_id
    papers_dir = out / "papers"
    trees_dir = out / "trees"
    tasks_dir = out / "tasks" / task_id

    # Step 1: Get PDF (either from local path or arXiv search)
    if args.pdf:
        # Mode: local PDF upload
        pdf_path = str(Path(args.pdf).resolve())
        if not Path(pdf_path).exists():
            sys.exit(f"PDF not found: {pdf_path}")
        print(f"\n{'='*50}\nStep 1: Using local PDF\n{'='*50}")
        print(f"  {pdf_path}")
    elif args.query:
        # Mode: arXiv search
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

    # Step 2: Parse tree
    print(f"\n{'='*50}\nStep 2: Parsing problem tree\n{'='*50}")
    trees_dir.mkdir(parents=True, exist_ok=True)
    tree_path = str(trees_dir / f"{task_id}_tree.json")
    tree = parse_paper_to_tree(pdf_path, model=args.model)
    with open(tree_path, "w", encoding="utf-8") as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)
    print(f"Tree saved to {tree_path}")

    # Step 3: Extract YAML
    print(f"\n{'='*50}\nStep 3: Extracting task config YAML\n{'='*50}")
    tasks_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = str(tasks_dir / "task_config.yaml")
    yaml_content = extract_yaml_from_pdf(
        pdf_path,
        research_question=args.research_question,
        task_id=task_id,
        model=args.model,
        tree_json_path=tree_path,
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
        return

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
        tree_json_path=tree_path,
    )
    instruction_gt = build_instruction_gt(instruction, supplement)
    gt_path = tasks_dir / "instruction_gt.txt"
    gt_path.write_text(instruction_gt, encoding="utf-8")
    print(f"instruction_gt.txt saved to {gt_path}")

    # Summary
    print(f"\n{'='*50}\nPipeline complete for: {task_id}\n{'='*50}")
    print(f"  PDF:              {pdf_path}")
    print(f"  Tree:             {tree_path}")
    print(f"  Config:           {yaml_path}")
    print(f"  instruction.txt:  {instruction_path}")
    print(f"  instruction_gt:   {gt_path}")


def main():
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

    # --- extract ---
    ex = subparsers.add_parser("extract", help="Extract task config YAML from a paper PDF")
    ex.add_argument("--pdf", required=True)
    ex.add_argument("--research-question", required=True)
    ex.add_argument("--task-id", required=True)
    ex.add_argument("--tree", default=None, help="Optional problem tree JSON")
    ex.add_argument("--model", default="gpt-4o")
    ex.add_argument("--max-tokens", type=int, default=4096)
    ex.add_argument("--temperature", type=float, default=0.0)
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

    # --- run (full pipeline) ---
    rp = subparsers.add_parser("run", help="Run full pipeline: PDF -> parse -> extract -> render -> plan")
    rp.add_argument("--task-id", required=True)
    rp.add_argument("--pdf", default=None, help="Path to a local PDF (skips download)")
    rp.add_argument("--query", default=None, help="arXiv search query (downloads PDF)")
    rp.add_argument("--research-question", required=True)
    rp.add_argument("--output-dir", default="./output", help="Base output directory")
    rp.add_argument("--model", default="gpt-4o")
    rp.add_argument("--template", default=None, help="Custom instruction template")
    rp.add_argument("--auto", action="store_true", help="Auto-select first arXiv result")
    rp.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
