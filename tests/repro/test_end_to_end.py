"""Actual HTTP -> draft -> frozen code -> Docker -> trusted score -> replay.

The HTTP server is explicitly a protocol fixture, not a foundation model.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from paper2bench.repro.authoring import draft_experiment
from paper2bench.repro.bundles import export_bundle, replay_bundle
from paper2bench.repro.demo import CODE, PAPER, create_demo
from paper2bench.repro.engine import Engine
from paper2bench.repro.providers import ProviderClient, ProviderProfile


@pytest.mark.docker
def test_http_authoring_to_isolated_execution_and_replay(tmp_path):
    image = os.environ.get("PAPER2BENCH_DOCKER_IMAGE")
    if not image:
        pytest.skip("set PAPER2BENCH_DOCKER_IMAGE to a staged immutable Python image ID")
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            draft = {
                "title": "Synthetic protocol-fixture authored implementation",
                "evidence": [{"start": 0, "end": len(PAPER), "quote": PAPER}],
                "assumptions": ["Synthetic test; author endpoint is a protocol fixture."],
                "unresolved": [],
                "claim": {"target": 1.0},
                "implementation": CODE,
            }
            response = json.dumps(
                {
                    "id": "fixture-generation",
                    "model": "protocol-fixture",
                    "choices": [
                        {"message": {"content": json.dumps(draft)}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 100},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *_):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        engine = Engine(tmp_path / "state")
        reference = engine.load_spec(create_demo(engine))
        profile = ProviderProfile(
            name="protocol-fixture",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            location="local",
        )
        draft = draft_experiment(
            reference.source_sha256,
            reference.dataset_sha256,
            engine.artifacts,
            ProviderClient(profile),
            "protocol-fixture",
            True,
            image,
        )
        assert len(requests) == 1
        context = json.loads(requests[0]["messages"][1]["content"])
        # The exact key set may evolve, but no held-out example reaches authoring.
        assert "test-000" not in json.dumps(context)
        assert draft.authoring_receipt_sha256
        run = engine.run(engine.freeze(draft), "full-protocol-fixture")
        assert run["state"] == "completed"
        result = engine.result(run["run_id"])["result"]
        assert result["assessment"]["accuracy"] == 1.0
        assert result["measurements"]["cleanup_confirmed"] is True
        replay = replay_bundle(export_bundle(engine, run["run_id"], tmp_path / "full.zip"))
        assert replay["scores"] == "recomputed"
        assert replay["assessment"] == result["assessment"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
