"""Build an executable synthetic protocol fixture, never a published-paper claim.

Usage: python examples/protocols/build_contract_fixture.py OUTPUT_DIRECTORY
The test-only curator review is explicitly declared by this fixture author.
Real research needs an independently reviewed source and implementation.
"""
from pathlib import Path
import shutil
import sys

from paper2bench.bundles import atomic_json, canonical_json, digest, read_json


def build(output):
    output = Path(output)
    if output.exists():
        raise ValueError("Fixture destination must not exist")
    original = Path(__file__).resolve().parents[1] / "tasks/linear-fit"
    shutil.copytree(original, output)
    source = ("Synthetic contract fixture, not a published paper.\n"
              "Fit ordinary least squares to the supplied exact line y=2*x+1.\n"
              "For the three held-out inputs the declared target mean squared error is zero.\n")
    (output / "source_fixture.txt").write_text(source, encoding="utf-8")
    paper_hash = digest(source.encode())
    task = read_json(output / "task_manifest.json")
    task["task_id"] = "synthetic-scientific-contract"
    task["source"] = {"paper_sha256": paper_hash, "fixture": "synthetic contract; not a published-paper reproduction"}
    atomic_json(output / "task_manifest.json", task)
    task_hash = digest(canonical_json(task))
    code_hash = task["files"][task["entrypoint"]]
    protocol = {
        "schema_version": 1, "profile": "reproduction/v1", "protocol_id": "synthetic-contract-fixture",
        "task_manifest_sha256": task_hash, "submission_sha256": code_hash,
        "source": {"paper_sha256": paper_hash, "citations": [
            {"id": "fixture-method", "locator": "source_fixture.txt lines 1 to 3",
             "claim": "Synthetic ordinary-least-squares contract; no published-paper claim"}]},
        "datasets": {task["inputs"]: task["files"][task["inputs"]]},
        "plan": {"seed_schedule": [0, 1, 2], "stopping_rule": "fixed_schedule",
                 "variability_source": "deterministic implementation; seeds test isolation, not population uncertainty",
                 "independence_assumption": False},
        "criteria": [
            {"id": "execution", "kind": "execution", "dimension": "execution", "critical": True, "depends_on": []},
            {"id": "fidelity", "kind": "review", "dimension": "implementation_fidelity", "critical": True,
             "depends_on": ["execution"], "review_id": "fixture-review"},
            {"id": "agreement", "kind": "metric", "dimension": "empirical_agreement", "critical": True,
             "depends_on": ["execution", "fidelity"], "metric_id": "mean_squared_error",
             "units": "target_units_squared", "rule": "absolute_tolerance", "reference": 0,
             "tolerance": 1e-12, "min_samples_per_run": 3}],
        "reviews": [{"id": "fixture-review", "reviewer": "synthetic fixture author; not an independent scientific reviewer",
                     "status": "pass", "submission_sha256": code_hash, "paper_sha256": paper_hash,
                     "source_citations": ["fixture-method"], "evidence_refs": ["source_fixture.txt"]}]}
    curator = read_json(output / "curator_manifest.json")
    curator.update(task_manifest_sha256=task_hash, protocol=protocol)
    atomic_json(output / "curator_manifest.json", curator)
    return output / "task_manifest.json"


if __name__ == "__main__":
    print(build(sys.argv[1]))
