"""Export selected, recorded Adam-paper evidence; never execute or invent a run.

Private PDF, extracted text, model responses, absolute paths, environment and
Docker daemon identifiers are deliberately absent from this public projection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from paper2bench.adam_benchmark import PAPER_SHA256, trajectory_statistics
from paper2bench.bundles import canonical_json, digest, read_json, validate_task
from paper2bench.journal import RunJournal
from paper2bench.generation import GenerationCheckpoint
from paper2bench.scientific_runs import _checked_report, resume_scientific

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/papers/adam-counterexample"


def export(private_root: Path):
    author = read_json(private_root / "authoring-result.json")
    extras = author["extras"]
    author_journal = RunJournal(private_root / "model-runs")
    author_record = author_journal.get(extras["model_run_id"])
    if author_record["state"] != "succeeded" or extras["readiness"] != "needs_review":
        raise ValueError("A completed, explicitly draft authoring run is required")
    verified = GenerationCheckpoint(private_root / "model-runs/generation", author_record["id"],
                                    author_record["spec"]["generation"], resume=True)
    verified.ensure_resumable()
    if verified.state["state"] != "succeeded" or verified.completed_result() != author:
        raise ValueError("Authoring result differs from its completed checkpoint")
    generation = private_root / "model-runs/generation" / author_record["id"]
    checkpoint = read_json(generation / "generation.json")
    stages = checkpoint["stages"]
    for name in ("parse", "classify", "extract", "render", "plan"):
        stage = stages[name]
        if stage["state"] != "succeeded" or digest((generation / (name + ".json")).read_bytes()) != stage["sha256"]:
            raise ValueError("Authoring stage receipt integrity differs")
    if checkpoint["plan"]["paper_sha256"] != PAPER_SHA256:
        raise ValueError("Authoring and experiment source papers differ")
    original_report = read_json(private_root / "execution-report.json")
    runs = private_root / "execution"
    journal = RunJournal(runs)
    parent = journal.get(original_report["run_id"])
    if parent["state"] != "succeeded":
        raise ValueError("Exporter never dispatches incomplete scientific runs")
    report = resume_scientific(original_report["run_id"], runs_dir=runs)
    if report != original_report or not report["pass"]:
        raise ValueError("Export requires the exact completed scientific report")
    task = validate_task(EXAMPLE / "task_manifest.json", require_executable=True)
    protocol = read_json(EXAMPLE / "protocol.json")
    if digest(canonical_json(task)) != parent["spec"]["task_sha256"] or digest(canonical_json(protocol)) != report["protocol_sha256"]:
        raise ValueError("Public task or protocol differs from the recorded condition")
    curation = read_json(EXAMPLE / "curation.json")
    if (digest(canonical_json(curation)) != task["source"]["curation_sha256"]
            or curation["authoring_run_id"] != author_record["id"]
            or curation["candidate_sha256"] != task["files"]["solution.py"]
            or curation["authoring_stage_receipts"] != {name: stage["sha256"] for name, stage in stages.items()}):
        raise ValueError("Authoring, curation and executable task must form one bound evidence chain")
    seeds, children, metrics, predictions = [], [], [], []
    for child_id in report["child_run_ids"]:
        record = journal.get(child_id)
        child = _checked_report(record)
        attempt = record["attempts"][-1]["id"]
        path = runs / child_id / "attempts" / attempt
        execution = read_json(path / "execution_result.json")
        raw = (path / "workspace/predictions.json").read_bytes()
        if digest(raw) != child["predictions_sha256"] or not child["pass"] or execution["evidence"].get("cleanup_confirmed") is not True:
            raise ValueError("Child evidence must be complete, unaltered and cleaned up")
        if (record["attempts"][-1]["result"].get("cleanup_confirmed") is not True
                or execution["evidence"]["image_id"] != parent["spec"]["image"]
                or execution["backend"] != "docker" or execution["isolation"] != "single_user_linux_container"
                or execution["evidence"]["network"] != "none"):
            raise ValueError("Execution isolation claims differ from the recorded condition")
        seed, metric = child["seed"], child["metric"]
        seeds.append(seed)
        metrics.append(metric)
        predictions.append(json.loads(raw))
        children.append({"run_id": child_id, "seed": seed, "status": record["state"],
                         "report_sha256": digest(canonical_json(child)),
                         "predictions_sha256": child["predictions_sha256"], "metric": metric,
                         "duration_seconds": execution["duration_seconds"],
                         "isolation": execution["isolation"], "network": execution["evidence"]["network"],
                         "cleanup_confirmed": True, "image_id": execution["evidence"]["image_id"]})
    if seeds != protocol["plan"]["seed_schedule"] or any(p != predictions[0] for p in predictions):
        raise ValueError("Every prescribed deterministic seed must be present and identical")
    first = metrics[0]
    points = [{"step": p["horizon"], "adam_position": predictions[0]["adam"]["positions"][p["horizon"] - 1],
               "amsgrad_position": predictions[0]["amsgrad"]["positions"][p["horizon"] - 1],
               "adam_average_regret": p["adam_average_regret"], "amsgrad_average_regret": p["amsgrad_average_regret"]}
              for p in trajectory_statistics(predictions[0])]
    model = extras["runtime"]["models"]["local"]
    usage = extras["model_budget"]["used"]["authoring"]
    if (author_record["spec"]["runtime_sha256"] != digest(canonical_json(extras["runtime"]))
            or author_record["budget"]["reserved"]
            or any(author_record["budget"]["used"]["authoring." + key] != usage[key]
                   for key in ("requests", "input_tokens", "output_tokens"))):
        raise ValueError("Model identity or token usage differs from the durable journal")
    review = ("The local model completed parse, classify, extract, render and plan. Its draft needed corrections: "
              "the generated loader omitted the decaying step size and projection, and used squared distance instead of pre-update linear regret. "
              "The executed task uses a separately source-reviewed curated implementation with the prescribed schedule, projection and regret metric. "
              "The executed candidate was not generated by this model run. This is an assisted end-to-end workflow.")
    result = {
        "schema_version": 1,
        "paper": {"title": "On the Convergence of Adam and Beyond", "authors": ["Sashank J. Reddi", "Satyen Kale", "Sanjiv Kumar"],
                  "venue": "ICLR", "year": 2018, "url": "https://arxiv.org/abs/1904.09237v1",
                  "pdf_url": "https://arxiv.org/pdf/1904.09237v1", "pdf_sha256": PAPER_SHA256},
        "recorded_at": datetime.fromtimestamp(max(st["finished_at"] for st in stages.values()), timezone.utc).isoformat(),
        "question": "On the Appendix A cyclic convex counterexample, how do Adam and AMSGrad compare after 30,000 steps?",
        "scope": ("A finite numerical check of the prescribed one-dimensional sequence. Adam average regret is "
                  f"{first['adam_average_regret']:.6f}; AMSGrad is {first['amsgrad_average_regret']:.6f} at the fixed horizon."),
        "limitations": ["Appendix A only; Figure 1 and neural benchmarks were not reproduced.",
                        "Three deterministic repeats check fresh isolation; they do not estimate population uncertainty.",
                        "Source and implementation review were automated; no external human expert certified the result.",
                        "Local inference locality is operator-trusted. This static page displays saved evidence and makes no model or compute calls."],
        "authoring": {"status": "Completed · draft requires review", "model": model["model"], "provider": model["provider"],
                      "model_digest": model["revision"], "context_tokens": model["context_tokens"],
                      "model_calls": usage["requests"], "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
                      "run_id": author_record["id"], "source_commit": "7179845fee5318bc95d3ce84118f1aa4028d470c",
                      "pipeline_sha256": checkpoint["plan"]["pipeline_sha256"], "review": review,
                      "curation_record_sha256": task["source"]["curation_sha256"],
                      "curation": {"required": True, "executed_model_draft": False,
                                   "corrections": ["Use alpha/sqrt(t), not a constant learning rate.",
                                                   "Project every update to [-1,1].",
                                                   "Evaluate g_t*(x_t+1) at the pre-update position; squared distance is a different metric.",
                                                   "Classify as a deterministic optimization check; keep broader paper claims out of scope."]},
                      "stages": [{"label": name.capitalize(), "status": stages[name]["state"], "receipt_sha256": stages[name]["sha256"]}
                                 for name in ("parse", "classify", "extract", "render", "plan")]},
        "execution": {"status": "3 / 3 isolated runs passed", "run_id": report["run_id"], "steps": 30000,
                      "seed_schedule": seeds, "image_id": children[0]["image_id"], "cleanup_confirmed": True,
                      "protocol_sha256": report["protocol_sha256"], "task_sha256": parent["spec"]["task_sha256"],
                      "submission_sha256": report["submission_sha256"], "scorer_protocol": report["scorer_protocol"],
                      "parameters": read_json(EXAMPLE / "input.json"), "children": children,
                      "scientific_report_sha256": digest(canonical_json(report)),
                      "agreement": report["criteria"][-1]["comparison"], "network": "none"},
        "metrics": [{"label": "Adam average regret", "value": first["adam_average_regret"], "unit": "loss / round", "precision": 6},
                    {"label": "AMSGrad average regret", "value": first["amsgrad_average_regret"], "unit": "loss / round", "precision": 6},
                    {"label": "Adam − AMSGrad", "value": first["value"], "unit": "loss / round", "precision": 6},
                    {"label": "Clean isolated runs", "value": len(children), "unit": "of 3 prescribed seeds", "precision": 0}],
        "checkpoints": points,
        "reproduction": {"command": "paper2bench execute --task examples/papers/adam-counterexample/task_manifest.json --evaluation-bundle examples/papers/adam-counterexample/curator_manifest.json --evaluation-profile reproduction/v1 --runs-dir runs/adam-reproduction",
                         "source_path": "examples/papers/adam-counterexample", "receipt_path": "results/adam-counterexample.json",
                         "raw_predictions_path": "results/adam-predictions.json"},
        "assurance": {"method_fidelity": "The source review binds the exact code and PDF digests. An installed geometric-sum oracle checks all 60,002 positions, including final states, against tolerance 1e-10.",
                      "scientific_scope": "Passing means agreement with this frozen finite condition; it is not an automatically proven scientific conclusion."}}
    preliminary = read_json(private_root / "implementation-check-report.json")
    result["prior_implementation_check"] = {
        "run_id": preliminary["run_id"], "purpose": "Isolated implementation check before completed model-authoring provenance was bound to the final task.",
        "report_sha256": digest(canonical_json(preliminary)), "pass": preliminary["pass"],
        "seed_schedule": preliminary["criteria"][-1]["expected_seeds"],
        "child_run_ids": preliminary["child_run_ids"],
        "values": preliminary["criteria"][-1]["summary"]["values"],
        "selection": "All implementation checks and all final prescribed seeds passed; no failed outcomes were dropped."}
    raw = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    destination = ROOT / "docs/results/adam-counterexample.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(raw, encoding="utf-8")
    (ROOT / "docs/results/adam-predictions.json").write_bytes(
        (runs / report["child_run_ids"][0] / "attempts" / journal.get(report["child_run_ids"][0])["attempts"][-1]["id"] / "workspace/predictions.json").read_bytes())
    (ROOT / "docs/paper-result.js").write_text("// Recorded evidence; regenerated from durable controller receipts.\nwindow.P2B_REAL_PAPER = " + raw.rstrip() + ";\n", encoding="utf-8")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("private_root", type=Path)
    print(export(parser.parse_args().private_root))
