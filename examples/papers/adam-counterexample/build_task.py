"""Freeze the candidate files and independent numerical reference before a run.

Usage: python -m examples.papers.adam-counterexample.build_task OUTPUT_DIRECTORY
Or import this file's build() using importlib. The destination must be new.
This builder does not manufacture a passing scientific fidelity review. Attach a
source- and submission-bound reviewed protocol before using reproduction/v1.
"""
from pathlib import Path
import argparse
import shutil

from paper2bench.adam_benchmark import PAPER_SHA256, SCORER, build_reference, validate_inputs
from paper2bench.bundles import atomic_json, canonical_json, digest, read_json, validate_task

HERE = Path(__file__).resolve().parent


def manifests(directory, *, paper_sha256=PAPER_SHA256):
    """Build without executing candidate code; expected results come from oracle."""
    directory = Path(directory)
    if paper_sha256 != PAPER_SHA256:
        raise ValueError("Paper identity differs from the reviewed arXiv v1 source")
    parameters = read_json(directory / "input.json")
    validate_inputs(parameters)
    files = {name: digest((directory / name).read_bytes()) for name in ("solution.py", "input.json")}
    task = {
        "schema_version": 1, "task_id": "adam-appendix-a-counterexample", "revision": 1,
        "research_question": "At the prescribed finite horizon, does the Appendix A cyclic convex sequence distinguish Adam from AMSGrad?",
        "paper_type": "theoretical_study", "method_kind": "optimization", "task_mode": "adaptation", "readiness": "ready",
        "source": {"paper_sha256": paper_sha256, "url": "https://arxiv.org/abs/1904.09237",
                   "scope": "Appendix A finite numerical check; not Figure 1, neural experiments, or an asymptotic proof"},
        "files": files, "entrypoint": "solution.py", "inputs": "input.json", "predictions": "predictions.json",
        "limits": {"timeout_seconds": 30, "memory_bytes": 268435456, "cpus": 1, "pids_limit": 64,
                   "workspace_bytes": 16777216, "output_limit_bytes": 16384}}
    curator = {"schema_version": 1, "task_manifest_sha256": digest(canonical_json(task)),
               "scorer": SCORER, "reference": build_reference(parameters)}
    return task, curator


def build(output, *, solution=None):
    output = Path(output)
    if output.exists():
        raise ValueError("Task destination must not exist")
    output.mkdir(parents=True)
    shutil.copyfile(HERE / "input.json", output / "input.json")
    shutil.copyfile(Path(solution) if solution is not None else HERE / "solution.py", output / "solution.py")
    task, curator = manifests(output)
    atomic_json(output / "task_manifest.json", task)
    atomic_json(output / "curator_manifest.json", curator)
    validate_task(output / "task_manifest.json", require_executable=True)
    return output / "task_manifest.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--solution", type=Path, help="Candidate entrypoint to freeze instead of the reviewed example")
    args = parser.parse_args()
    print(build(args.output, solution=args.solution))
