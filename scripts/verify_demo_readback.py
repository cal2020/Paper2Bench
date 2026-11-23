"""Compare a captured Site dist to a separately built or staged Paper2Bench release."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

from package_demo import ArtifactError, validate

GENERATED = {"dist/build-provenance.json", "dist/asset-manifest.json"}
NODE = re.compile(r"v(\d+\.\d+)\.\d+")


def compare(observed, reference, expected_commit, *, local_rebuild=False):
    """Validate exact committed inputs on both sides before comparing file bytes."""
    actual, receipt = validate(observed, expected_commit)
    approved, ref_receipt = validate(reference, expected_commit)
    if actual.keys() != approved.keys() or receipt["project_id"] != ref_receipt["project_id"]:
        raise ArtifactError("Site and reference disagree about the project or files")
    differing = {p for p in actual if actual[p] != approved[p]}
    details = None
    if differing:
        if not local_rebuild or differing != GENERATED:
            raise ArtifactError("Site assets do not match the verified reference")
        old = json.loads(actual["dist/build-provenance.json"])
        new = json.loads(approved["dist/build-provenance.json"])
        old_node = old["build"].pop("node", None)
        new_node = new["build"].pop("node", None)
        if (
            old != new
            or not isinstance(old_node, str)
            or not isinstance(new_node, str)
            or not (old_version := NODE.fullmatch(old_node))
            or not (new_version := NODE.fullmatch(new_node))
            or old_version.group(1) != new_version.group(1)
        ):
            raise ArtifactError("Build provenance differs beyond the Node patch version")
        # The manifest keys and every declared hash except provenance still match.
        old_manifest = json.loads(actual["dist/asset-manifest.json"])
        new_manifest = json.loads(approved["dist/asset-manifest.json"])
        for manifest in (old_manifest, new_manifest):
            del manifest["files"]["build-provenance.json"]
        if old_manifest != new_manifest:
            raise ArtifactError("Asset manifest drift exceeds build provenance")
        details = {"observed_node": old_node, "rebuild_node": new_node}
    # The metadata directory comes from the source repo; it is not a public URL.
    asset_paths = {p for p in actual if p.startswith("dist/")}
    return {
        "source_commit": receipt["source_commit"],
        "source_tree": receipt["source_tree"],
        "project_id": receipt["project_id"],
        "reference_kind": "local_rebuild" if local_rebuild else "caller_supplied_reference",
        "public_dist_files_compared": len(asset_paths),
        "public_dist_files_byte_equal": len(asset_paths - differing),
        "node_patch_only_provenance_difference": details,
        "publication_attested_by_this_command": False,
    }


def main():
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("--observed", required=True, type=Path,
                     help="Trusted, separately captured public dist and committed .openai metadata")
    arg.add_argument("--reference", required=True, type=Path,
                     help="Separately verified read-only reference with dist and .openai metadata")
    arg.add_argument("--expected-commit", required=True)
    arg.add_argument("--local-rebuild", action="store_true",
                     help="Permit only Node patch-version difference in local rebuild metadata")
    args = arg.parse_args()
    try:
        receipt = compare(args.observed, args.reference, args.expected_commit,
                          local_rebuild=args.local_rebuild)
    except (ArtifactError, OSError, ValueError) as error:
        print(f"Demo readback rejected: {error}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
