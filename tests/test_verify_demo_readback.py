"""Exercise the actual readback comparison with bounded authored artifacts."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

from test_release_artifact import artifact, packager, reseal, write_json  # noqa: F401

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_demo_readback.py"
SPEC = importlib.util.spec_from_file_location("verify_demo_readback", SCRIPT)


@pytest.fixture
def verifier(monkeypatch):
    monkeypatch.setitem(sys.modules, "package_demo", packager)
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    return module


def set_node(root, node):
    path = root / "dist/build-provenance.json"
    payload = json.loads(path.read_bytes())
    payload["build"]["node"] = node
    write_json(path, payload)
    reseal(root)


def test_exact_capture_is_not_a_publication_attestation(artifact, verifier, tmp_path):
    observed, commit = artifact
    reference = tmp_path / "ref"
    shutil.copytree(observed, reference)
    result = verifier.compare(observed, reference, commit)
    assert result["public_dist_files_compared"] == result["public_dist_files_byte_equal"] == 5
    assert result["publication_attested_by_this_command"] is False


@pytest.mark.parametrize("mode", ["changed_asset", "missing_asset", "prev_commit", "different_minor",
                                  "non_node_provenance", "same_minor_exact_required"])
def test_rejects_bad_readback_even_after_manifest_resealed(artifact, verifier, tmp_path, mode):
    observed, commit = artifact
    reference = tmp_path / "ref"
    set_node(observed, "v24.22.1")
    shutil.copytree(observed, reference)
    if mode == "changed_asset":
        (observed / "dist/assets/main.js").write_text("updated, but not the reviewed content")
        reseal(observed)
    elif mode == "missing_asset":
        (observed / "dist/assets/main.js").unlink()
    elif mode == "prev_commit":
        commit = "0" * 40
    elif mode == "different_minor":
        set_node(reference, "v24.23.1")
    elif mode == "non_node_provenance":
        path = observed / "dist/build-provenance.json"
        payload = json.loads(path.read_bytes())
        payload["build"]["esbuild"] = "unexpected"
        write_json(path, payload)
        reseal(observed)
    else:
        set_node(reference, "v24.22.3")
    with pytest.raises(packager.ArtifactError):
        verifier.compare(observed, reference, commit, local_rebuild=mode != "same_minor_exact_required")


def test_local_rebuild_permits_only_node_patch_and_its_manifest_hash(artifact, verifier, tmp_path):
    observed, commit = artifact
    reference = tmp_path / "ref"
    set_node(observed, "v24.22.1")
    shutil.copytree(observed, reference)
    set_node(reference, "v24.22.3")
    result = verifier.compare(observed, reference, commit, local_rebuild=True)
    assert result["public_dist_files_compared"] == 5
    assert result["public_dist_files_byte_equal"] == 3
    assert result["node_patch_only_provenance_difference"] == {
        "observed_node": "v24.22.1", "rebuild_node": "v24.22.3"
    }
