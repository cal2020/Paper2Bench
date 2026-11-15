"""Model-free verification for an enrolled Control Center repository snapshot.

Run from the repository root with PYTHONPATH=. and the repro dependencies already
installed. All state lives in a temporary directory; no generated code, model,
package installation, external network, or nested Docker execution is needed.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch

from paper2bench.repro.bundles import export_bundle, replay_bundle
from paper2bench.repro.cli import main
from paper2bench.repro.demo import create_demo, create_optimization_demo
from paper2bench.repro.engine import Engine
from paper2bench.repro.storage import IdempotencyConflict


def verify() -> dict:
    checks = []

    def require(condition: bool, name: str):
        if not condition:
            raise AssertionError(name)
        checks.append({"name": name, "passed": True})

    def forbidden(*_args, **_kwargs):
        raise AssertionError("This verifier must not call a model or execute generated code")

    with (
        tempfile.TemporaryDirectory(prefix="paper2bench-control-center-") as temporary,
        patch("paper2bench.repro.providers.ProviderClient.generate", forbidden),
        patch("paper2bench.repro.runtime.DockerRunner.run_input", forbidden),
    ):
        root = Path(temporary)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["--state", str(root / "unused"), "schema"])
        schema = json.loads(output.getvalue())
        require(code == 0 and schema["title"] == "ExperimentSpec", "optional CLI schema entrypoint")
        require(not (root / "unused").exists(), "schema inspection creates no execution state")

        engine = Engine(root / "state")
        for task, factory, metric in (
            ("classification", create_demo, "accuracy"),
            ("knapsack", create_optimization_demo, "optimality_ratio"),
        ):
            spec_hash = factory(engine)
            spec = engine.load_spec(spec_hash)
            require(bool(spec.evidence) and not spec.unresolved, task + " source-linked frozen spec")
            run = engine.run(spec_hash, "control-center-" + task)
            require(run["state"] == "completed", task + " trusted builtin execution")
            result = engine.result(run["run_id"])["result"]
            assessment = result["assessment"]
            actual = assessment["accuracy"] if task == "classification" else assessment["mean_score"]
            require(assessment["metric"] == metric and actual == 1.0, task + " trusted numerical score")
            require(result["provider_calls"] == [], task + " zero provider calls")
            repeated = engine.run(spec_hash, "control-center-" + task)
            require(
                repeated["run_id"] == run["run_id"] and repeated["attempt"] == run["attempt"],
                task + " idempotent run identity without another execution",
            )
            archive = export_bundle(engine, run["run_id"], root / (task + ".zip"))
            replay = replay_bundle(archive)
            require(
                replay["integrity"] == "verified"
                and replay["authentication"] == "not_verified"
                and replay["scores"] == "recomputed"
                and replay["execution_rerun"] is False
                and replay["assessment"] == assessment,
                task + " portable integrity and score recomputation without code execution",
            )

        changed = engine.load_spec(spec_hash).model_copy(update={"seed": spec.seed + 1})
        changed_hash = engine.freeze(changed)
        require(changed_hash != spec_hash, "changed experiment setting changes frozen identity")
        try:
            engine.submit(changed_hash, "control-center-knapsack")
        except IdempotencyConflict:
            checks.append({"name": "idempotency key rejects a different frozen specification", "passed": True})
        else:
            raise AssertionError("idempotency conflict was not rejected")

        queued = engine.submit(spec_hash, "control-center-cancellation")
        lease = engine.runs.claim(queued["run_id"], "control-center-verifier")
        engine.runs.cancel(queued["run_id"])
        require(
            lease is not None
            and not engine.runs.finish(queued["run_id"], lease["token"], "failed", error="fixture")
            and engine.runs.get_run(queued["run_id"])["state"] == "cancelled",
            "cancelled lease cannot publish a stale result",
        )

        with zipfile.ZipFile(archive) as handle:
            members = {name: handle.read(name) for name in handle.namelist()}
        members["artifacts/" + spec_hash] += b" "
        corrupt = root / "corrupt.zip"
        with zipfile.ZipFile(corrupt, "w") as handle:
            for name, data in members.items():
                handle.writestr(name, data)
        try:
            replay_bundle(corrupt)
        except ValueError:
            checks.append({"name": "corrupted hashed artifact is rejected", "passed": True})
        else:
            raise AssertionError("corrupted artifact was accepted")

    return {
        "schema_version": 1,
        "status": "passed",
        "check_count": len(checks),
        "checks": checks,
        "model_requests": 0,
        "generated_code_execution": "not_tested",
        "isolation_attestation": "inspect_outer_control_center_receipt",
        "scientific_scope": "synthetic_software_checks_not_published_paper_reproduction",
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, allow_nan=False))
