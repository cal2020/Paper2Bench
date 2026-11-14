import json
import shutil
from pathlib import Path

import pytest

from paper2bench.bundles import (BundleError, atomic_json, canonical_json, digest, export_bundle, public_config,
                                 read_json, safe_file, validate_task, write_draft_manifest)

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/tasks/linear-fit"


def test_candidate_export_excludes_curator_and_unknown_fields(tmp_path):
    source = tmp_path / "source"
    shutil.copytree(EXAMPLE, source)
    manifest = read_json(source / "task_manifest.json")
    manifest.update(secret_answer="gold", source={"paper_id": "demo", "hidden_answer": "gold"})
    atomic_json(source / "task_manifest.json", manifest)
    output = export_bundle(source, tmp_path / "candidate")
    exported = read_json(output)
    assert "secret_answer" not in exported
    assert exported["source"] == {"paper_id": "demo"}
    assert not (output.parent / "curator_manifest.json").exists()
    assert set(p.name for p in output.parent.iterdir()) == {"task_manifest.json", "solution.py", "input.json"}


@pytest.mark.parametrize("name", ["../private.json", "/private.json", ".env", "data/.secret", "data\\escape"])
def test_escape_or_hidden_file_paths_rejected(tmp_path, name):
    with pytest.raises(BundleError):
        safe_file(tmp_path, name)


def test_symlink_and_hash_changes_rejected(tmp_path):
    shutil.copytree(EXAMPLE, tmp_path / "task")
    task = tmp_path / "task/task_manifest.json"
    (task.parent / "input.json").write_text("{}")
    with pytest.raises(BundleError, match="digest"):
        validate_task(task)
    (tmp_path / "alias").symlink_to(task.parent)
    with pytest.raises(BundleError):
        safe_file(tmp_path, "alias/solution.py")


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', '{"x":NaN}', '{"x":1e999}', '[] trailing'])
def test_strict_json_rejects_ambiguous_or_nonfinite_data(tmp_path, raw):
    path = tmp_path / "data.json"
    path.write_text(raw)
    with pytest.raises(BundleError):
        read_json(path)


def test_legacy_config_projection_is_draft_and_drops_loader_and_reference(tmp_path):
    config = {"task_id": "demo", "research_question": "Question", "expected_outcome": "gold",
              "reference_implementation": "secret", "datasets": [{"name": "Data", "loader": "dangerous()"}]}
    manifest = write_draft_manifest(tmp_path, config)
    data = validate_task(manifest)
    assert data["readiness"] == "needs_review"
    projected = public_config(config)
    assert "reference_implementation" not in projected
    assert "loader" not in projected["datasets"][0]
    with pytest.raises(BundleError):
        validate_task(manifest, require_executable=True)


def scientific_bundle(tmp_path):
    """Synthetic reviewed contract: tests identity semantics, not paper reproduction."""
    root = tmp_path / "scientific"
    shutil.copytree(EXAMPLE, root)
    manifest = read_json(root / "task_manifest.json")
    paper_hash = digest(b"synthetic export contract, not a published paper")
    manifest["source"] = {"paper_sha256": paper_hash}
    task_hash = digest(canonical_json(manifest))
    submission_hash = manifest["files"][manifest["entrypoint"]]
    protocol = {
        "schema_version": 1, "profile": "reproduction/v1", "protocol_id": "export-contract",
        "task_manifest_sha256": task_hash, "submission_sha256": submission_hash,
        "source": {"paper_sha256": paper_hash, "citations": [
            {"id": "method", "locator": "synthetic definition", "claim": "Export contract fixture"}]},
        "datasets": {manifest["inputs"]: manifest["files"][manifest["inputs"]]},
        "plan": {"seed_schedule": [0], "stopping_rule": "fixed_schedule",
                 "variability_source": "controlled fixture", "independence_assumption": False},
        "criteria": [
            {"id": "execution", "kind": "execution", "dimension": "execution", "critical": True, "depends_on": []},
            {"id": "fidelity", "kind": "review", "dimension": "implementation_fidelity", "critical": True,
             "depends_on": ["execution"], "review_id": "review"},
            {"id": "metric", "kind": "metric", "dimension": "empirical_agreement", "critical": True,
             "depends_on": ["execution", "fidelity"], "metric_id": "mean_squared_error",
             "units": "target_units_squared", "rule": "absolute_tolerance", "reference": 0, "tolerance": 1e-9}],
        "reviews": [{"id": "review", "reviewer": "synthetic test curator", "status": "pass",
                     "submission_sha256": submission_hash, "paper_sha256": paper_hash,
                     "source_citations": ["method"], "evidence_refs": ["curator/synthetic-review.json"]}],
    }
    curator = read_json(root / "curator_manifest.json")
    curator.update(task_manifest_sha256=task_hash, protocol=protocol)
    atomic_json(root / "task_manifest.json", manifest)
    atomic_json(root / "curator_manifest.json", curator)
    return root, manifest, curator


@pytest.mark.parametrize("change", ["private_manifest", "private_source", "yaml_bytes"])
def test_scientific_export_refuses_identity_changing_projection(tmp_path, change):
    root, manifest, curator = scientific_bundle(tmp_path)
    if change == "private_manifest":
        manifest["private_annotations"] = {"expected_result": "not public"}
    elif change == "private_source":
        manifest["source"]["private_note"] = "not public"
    else:
        config = root / "task_config.yaml"
        # Semantically equivalent YAML still changes pinned bytes when exported.
        config.write_text("research_question: 'Question'\n")
        manifest["files"]["task_config.yaml"] = digest(config.read_bytes())
    source_identity = digest(canonical_json(manifest))
    curator["task_manifest_sha256"] = source_identity
    curator["protocol"]["task_manifest_sha256"] = source_identity
    atomic_json(root / "task_manifest.json", manifest)
    atomic_json(root / "curator_manifest.json", curator)
    original_protocol = canonical_json(curator["protocol"])
    destination = tmp_path / "protected"
    with pytest.raises(BundleError, match="reviewed scientific task identity"):
        export_bundle(root, destination, audience="curator")
    assert not destination.exists()
    assert canonical_json(read_json(root / "curator_manifest.json")["protocol"]) == original_protocol


def test_identity_preserving_scientific_export_keeps_reviewed_protocol_unchanged(tmp_path):
    root, manifest, curator = scientific_bundle(tmp_path)
    public = read_json(export_bundle(root, tmp_path / "candidate"))
    protected = read_json(export_bundle(root, tmp_path / "protected", audience="curator"))
    assert digest(canonical_json(public)) == digest(canonical_json(manifest))
    assert protected["task_manifest_sha256"] == protected["protocol"]["task_manifest_sha256"]
    assert canonical_json(protected["protocol"]) == canonical_json(curator["protocol"])
    assert read_json(root / "curator_manifest.json") == curator


@pytest.mark.parametrize("corrupt", ["task_binding", "review_binding", "malformed_protocol"])
def test_identity_preserving_export_rejects_invalid_scientific_protocol(tmp_path, corrupt):
    root, manifest, curator = scientific_bundle(tmp_path)
    if corrupt == "task_binding":
        curator["protocol"]["task_manifest_sha256"] = "0" * 64
    elif corrupt == "review_binding":
        curator["protocol"]["reviews"][0]["submission_sha256"] = "0" * 64
    else:
        curator["protocol"] = {"profile": "reproduction/v1"}
    atomic_json(root / "curator_manifest.json", curator)
    destination = tmp_path / "protected"
    with pytest.raises(BundleError, match="Scientific protocol is invalid"):
        export_bundle(root, destination, audience="curator")
    assert not destination.exists()
