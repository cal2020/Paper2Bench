"""Generate static example detail pages from output artifacts.

Reads docs/examples/<task_id>/{tree.json, instruction.txt, instruction_gt.txt,
task_config.yaml} and emits docs/examples/<task_id>/index.html.

Run from the repo root: ``python3 docs/build_examples.py``.
"""

import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).parent
EXAMPLES_DIR = ROOT / "examples"

# Hand-curated metadata for the featured examples (paper-type + domain
# come from our validation runs; titles/authors are pulled from the tree
# JSON when available).
META = {
    "cgcnn":              {"domain": "materials", "paper_type": "novel_architecture"},
    "dnabert2":           {"domain": "biology",   "paper_type": "novel_architecture"},
    "cot_prompting":      {"domain": "ml_nlp",    "paper_type": "llm_evaluation"},
    "lost_in_the_middle": {"domain": "ml_nlp",    "paper_type": "llm_evaluation"},
    "mace_potential":     {"domain": "materials", "paper_type": "novel_architecture"},
}


def render_tree(tree_data: dict) -> str:
    """Render the parsed tree.json into HTML ASCII-art tree lines."""
    lines = []
    pt = tree_data.get("problem_tree", {})
    root_text = pt.get("node", "Root")
    lines.append(f'<div class="tree-line"><strong>📄 {escape(root_text)}</strong></div>')

    rqs = pt.get("children") or []
    for i, rq in enumerate(rqs):
        is_last = i == len(rqs) - 1
        connector = "└─" if is_last else "├─"
        rq_text = rq.get("node", f"RQ{i+1}")
        lines.append(
            f'<div class="tree-line indent-1">{connector} <strong>{escape(rq_text)}</strong></div>'
        )
        leaves = rq.get("children") or []
        for j, leaf in enumerate(leaves):
            leaf_last = j == len(leaves) - 1
            leaf_connector = "└─" if leaf_last else "├─"
            task = leaf.get("task") or leaf.get("node") or "experiment"
            ev = leaf.get("evidence") or {}
            cite_parts = [v for k, v in ev.items() if k in ("figure", "table", "section") and v]
            cite = " · ".join(cite_parts)
            line = f"{leaf_connector} {task}"
            if cite:
                line += f"  ({cite})"
            indent_pad = "│   " if not is_last else "    "
            lines.append(
                f'<div class="tree-line indent-2 muted">{indent_pad}{escape(line)}</div>'
            )
    return "\n".join(lines)


def derive_research_question(yaml_text: str, instruction_text: str) -> str:
    """Pull the research question out of the artifacts."""
    # Prefer the YAML's research_question line; fall back to the
    # instruction.txt's first quoted question.
    for line in yaml_text.splitlines():
        if line.startswith("research_question:"):
            value = line.split(":", 1)[1].strip()
            return value.strip('"').strip("'")
    if 'question: "' in instruction_text:
        head = instruction_text.split('question: "', 1)[1]
        return head.split('"', 1)[0]
    return ""


PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} — Paper2Bench example</title>
  <link rel="stylesheet" href="../../styles.css">
</head>
<body>

<nav class="nav">
  <div class="container nav-inner">
    <a class="nav-brand" href="../../"><span class="dot">●</span> paper2bench</a>
    <div class="nav-links">
      <a href="../../#examples">examples</a>
      <a href="../../#install">install</a>
      <a href="https://github.com/AbhayAnandUCSD/Paper2Bench">github</a>
    </div>
  </div>
</nav>

<header class="example-header">
  <div class="container">
    <div class="crumb"><a href="../../#examples">← back to examples</a></div>
    <h1>{title}</h1>
    <div class="authors">{authors} · {domain} · <span style="opacity:0.85">{paper_type}</span></div>
    <div class="rq-callout"><strong>Research question:</strong> {research_question}</div>
  </div>
</header>

<div class="container">
  <div class="example-body">

    <aside class="tree-rail">
      <div class="tree-label">Problem tree</div>
      {tree_html}
    </aside>

    <div>
      <div class="tabs">
        <button class="tab active" data-tab="instruction">instruction.txt</button>
        <button class="tab" data-tab="gt">instruction_gt.txt</button>
        <button class="tab" data-tab="config">task_config.yaml</button>
      </div>

      <div class="tab-panel active" data-tab="instruction">
        <pre><code>{instruction_text}</code></pre>
      </div>
      <div class="tab-panel" data-tab="gt">
        <pre><code>{instruction_gt_text}</code></pre>
      </div>
      <div class="tab-panel" data-tab="config">
        <pre><code>{task_config_text}</code></pre>
      </div>

      <div class="actions">
        <a class="btn btn-primary" href="bundle.zip" download>↓ Download bundle (.zip)</a>
        <a class="btn" href="instruction.txt">View raw instruction.txt</a>
        <a class="btn" href="instruction_gt.txt">View raw instruction_gt.txt</a>
        <a class="btn" href="task_config.yaml">View raw task_config.yaml</a>
      </div>
    </div>

  </div>
</div>

<footer class="footer">
  <div class="container footer-inner">
    <span>Paper2Bench · v0.3.0 · MIT</span>
    <span>
      <a href="https://github.com/AbhayAnandUCSD/Paper2Bench">GitHub</a> ·
      <a href="../../#install">Install</a>
    </span>
  </div>
</footer>

<script src="../../scripts.js"></script>
</body>
</html>
"""


def build(task_id: str) -> None:
    d = EXAMPLES_DIR / task_id
    tree = json.loads((d / "tree.json").read_text())
    instruction = (d / "instruction.txt").read_text()
    instruction_gt = (d / "instruction_gt.txt").read_text()
    task_config = (d / "task_config.yaml").read_text()

    paper = tree.get("paper", {})
    title = paper.get("title") or task_id
    authors = paper.get("authors") or []
    year = paper.get("year") or ""
    if isinstance(authors, list) and authors:
        first_author = authors[0]
        if len(authors) > 1:
            author_str = f"{first_author} et al."
        else:
            author_str = first_author
        if year:
            author_str = f"{author_str}, {year}"
    else:
        author_str = year or "—"

    meta = META.get(task_id, {})
    rq = derive_research_question(task_config, instruction)

    html = PAGE_TEMPLATE.format(
        title=escape(title),
        authors=escape(author_str),
        domain=escape(meta.get("domain", "")),
        paper_type=escape(meta.get("paper_type", "")),
        research_question=escape(rq),
        tree_html=render_tree(tree),
        instruction_text=escape(instruction),
        instruction_gt_text=escape(instruction_gt),
        task_config_text=escape(task_config),
    )
    (d / "index.html").write_text(html)
    print(f"  wrote {d.relative_to(ROOT.parent)}/index.html")


def main() -> None:
    for task_id in sorted(META):
        if not (EXAMPLES_DIR / task_id).exists():
            print(f"  ! missing {task_id}, skipping")
            continue
        build(task_id)


if __name__ == "__main__":
    main()
