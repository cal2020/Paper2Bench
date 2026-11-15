"""Optional authenticated HTTP control plane for one trusted operator.

Requests may register artifacts and queue work, but never execute candidates or
invoke providers. Run a separate CLI worker against the same state directory.
This module intentionally offers no filesystem-path, shell-command, or raw
artifact-reading endpoints.
"""

from __future__ import annotations

import hmac
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .budgets import BudgetConflict, BudgetNotFound
from .engine import Engine
from .models import Dataset, ExperimentSpec, KnapsackDataset
from .providers import ProviderConfigurationError, ProviderProfile
from .remote import RemoteConflict, RemoteCoordinator
from .storage import ArtifactNotFound, IdempotencyConflict, RunNotFound, StorageError

MAX_REQUEST_BYTES = 4 * 1024 * 1024


class _Guard:
    """Authenticate before parsing data and cap streamed request bodies."""

    def __init__(self, app, *, token: str, worker_tokens: dict[str, str] | None = None):
        self.app = app
        self._token = token.encode("utf-8")
        self._workers = {key: value.encode("utf-8") for key, value in (worker_tokens or {}).items()}

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        public_health = scope["path"] == "/health" and scope["method"] == "GET"
        if not public_health:
            values = [
                value for key, value in scope.get("headers", []) if key.lower() == b"authorization"
            ]
            supplied = values[0] if len(values) == 1 else b""
            scheme, separator, credential = supplied.partition(b" ")
            worker_route = scope["path"].startswith("/worker/")
            if worker_route:
                matches = [
                    key
                    for key, value in self._workers.items()
                    if hmac.compare_digest(credential, value)
                ]
                token_matches = len(matches) == 1
                if token_matches:
                    scope.setdefault("state", {})["worker_id"] = matches[0]
            else:
                token_matches = hmac.compare_digest(credential, self._token)
            if not separator or scheme.lower() != b"bearer" or not token_matches:
                await JSONResponse(
                    {"detail": "unauthorized"},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )(scope, receive, send)
                return

        chunks = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            chunk = message.get("body", b"")
            size += len(chunk)
            limit = 20 * 1024 * 1024 if scope["path"].startswith("/worker/") else MAX_REQUEST_BYTES
            if size > limit:
                await JSONResponse({"detail": "request_too_large"}, status_code=413)(
                    scope, receive, send
                )
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break

        body = b"".join(chunks)
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


class _SourceInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    text: Annotated[StrictStr, Field(min_length=1, max_length=400_000)]


class _RunInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    spec_hash: Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
    idempotency_key: Annotated[StrictStr, Field(min_length=1, max_length=256)]


class _WorkerClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    worker_id: Annotated[StrictStr, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
    images: Annotated[tuple[StrictStr, ...], Field(min_length=1, max_length=100)]
    max_memory_mb: Annotated[StrictInt, Field(ge=16, le=65536)] = 256


class _WorkerLease(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    worker_id: Annotated[StrictStr, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
    run_id: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{32}$")]
    token: Annotated[StrictInt, Field(gt=0)]


class _WorkerComplete(_WorkerLease):
    completion: dict


class _BudgetInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    budget_id: Annotated[StrictStr, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")]
    max_calls: Annotated[StrictInt, Field(ge=0, le=10**9)]
    max_output_tokens: Annotated[StrictInt, Field(ge=0, le=10**9)]


def create_app(
    state: Path, token: str | None = None, *, worker_tokens: dict[str, str] | None = None
) -> FastAPI:
    """Build a queue-only service; fail closed if authentication is unconfigured."""
    configured = os.environ.get("PAPER2BENCH_SERVER_TOKEN") if token is None else token
    if (
        not isinstance(configured, str)
        or not configured
        or configured != configured.strip()
        or len(configured) > 4096
        or any(not 33 <= ord(char) <= 126 for char in configured)
    ):
        raise RuntimeError("A nonempty server authentication token is required")

    if worker_tokens is None:
        references = json.loads(os.environ.get("PAPER2BENCH_WORKERS", "{}"))
        if not isinstance(references, dict):
            raise RuntimeError(
                "Worker configuration must map worker IDs to credential environment names"
            )
        worker_tokens = {}
        for worker_id, env_name in references.items():
            if not isinstance(env_name, str) or not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*", env_name
            ):
                raise RuntimeError("Invalid worker credential reference")
            worker_tokens[worker_id] = os.environ.get(env_name, "")
    if len(worker_tokens) > 100:
        raise RuntimeError("Too many configured workers")
    for worker_id, worker_secret in worker_tokens.items():
        if (
            not isinstance(worker_id, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", worker_id)
            or not isinstance(worker_secret, str)
            or not worker_secret
            or len(worker_secret) > 4096
            or any(not 33 <= ord(c) <= 126 for c in worker_secret)
        ):
            raise RuntimeError("Invalid worker identity or credential")
    if (
        len(set(worker_tokens.values())) != len(worker_tokens)
        or configured in worker_tokens.values()
    ):
        raise RuntimeError("Administrator and worker credentials must be distinct")

    app = FastAPI(
        title="Paper2Bench",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    engine = Engine(state)
    app.state.engine = engine
    coordinator = RemoteCoordinator(engine)
    app.add_middleware(_Guard, token=configured, worker_tokens=worker_tokens)

    async def invalid_request(_request: Request, _error: Exception):
        # Pydantic/FastAPI error details can contain complete submitted values,
        # including credentials accidentally placed in a URL or extra field.
        return JSONResponse({"detail": "invalid_request"}, status_code=422)

    async def missing_resource(_request: Request, _error: Exception):
        return JSONResponse({"detail": "resource_not_found"}, status_code=404)

    async def conflicting_request(_request: Request, _error: Exception):
        return JSONResponse({"detail": "idempotency_conflict"}, status_code=409)

    async def invalid_operation(_request: Request, _error: Exception):
        return JSONResponse({"detail": "invalid_operation"}, status_code=400)

    async def internal_failure(_request: Request, _error: Exception):
        return JSONResponse({"detail": "internal_error"}, status_code=500)

    async def route_error(_request: Request, error: StarletteHTTPException):
        detail = {404: "not_found", 405: "method_not_allowed", 403: "forbidden"}.get(
            error.status_code, "request_failed"
        )
        return JSONResponse({"detail": detail}, status_code=error.status_code)

    app.add_exception_handler(RequestValidationError, invalid_request)
    app.add_exception_handler(StarletteHTTPException, route_error)
    app.add_exception_handler(IdempotencyConflict, conflicting_request)
    app.add_exception_handler(RemoteConflict, conflicting_request)
    app.add_exception_handler(BudgetConflict, conflicting_request)
    app.add_exception_handler(BudgetNotFound, missing_resource)
    app.add_exception_handler(RunNotFound, missing_resource)
    app.add_exception_handler(ArtifactNotFound, missing_resource)
    app.add_exception_handler(FileNotFoundError, missing_resource)
    app.add_exception_handler(ProviderConfigurationError, invalid_operation)
    app.add_exception_handler(ValueError, invalid_operation)
    app.add_exception_handler(StorageError, internal_failure)
    app.add_exception_handler(Exception, internal_failure)

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__}

    @app.post("/sources", status_code=201)
    def sources(value: _SourceInput):
        text = unicodedata.normalize(
            "NFC", value.text.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
        )
        if not text.strip() or "\x00" in text or len(text) > 400_000:
            raise ValueError("invalid source text")
        receipt = {
            "source_sha256": engine.artifacts.put(text.encode("utf-8")),
            "original_sha256": engine.artifacts.put(value.text.encode("utf-8")),
            "media_type": "text/plain",
            "text_length": len(text),
            "page_count": 1,
            "warnings": [],
        }
        receipt["source_manifest_sha256"] = engine.artifacts.put_json(receipt)
        return receipt

    @app.post("/datasets", status_code=201)
    def datasets(value: Dataset | KnapsackDataset):
        return {"dataset_sha256": engine.register_dataset(value)}

    @app.post("/profiles", status_code=201)
    def profiles(value: ProviderProfile):
        return {"provider_profile_sha256": engine.register_profile(value)}

    @app.post("/budgets", status_code=201)
    def create_budget(value: _BudgetInput):
        return engine.budgets.create(value.budget_id, value.max_calls, value.max_output_tokens)

    @app.get("/budgets/{budget_id}")
    def get_budget(budget_id: str):
        return engine.budgets.get(budget_id)

    @app.post("/specs", status_code=201)
    def specs(value: ExperimentSpec):
        return {"spec_hash": engine.freeze(value)}

    @app.post("/runs", status_code=202)
    def submit(value: _RunInput):
        return engine.submit(value.spec_hash, value.idempotency_key)

    @app.get("/runs")
    def runs():
        return {"runs": engine.runs.list_runs()}

    @app.get("/runs/{run_id}")
    def run(run_id: str):
        return engine.runs.get_run(run_id)

    @app.get("/runs/{run_id}/report")
    def report(run_id: str):
        return {
            "run_id": run_id,
            "text": engine.report(run_id),
            "result": engine.result(run_id)["result"],
        }

    @app.post("/runs/{run_id}/cancel")
    def cancel(run_id: str):
        return engine.runs.cancel(run_id)

    def worker_matches(request: Request, worker_id: str) -> None:
        if getattr(request.state, "worker_id", None) != worker_id:
            raise StarletteHTTPException(status_code=403)

    @app.post("/worker/claim")
    def worker_claim(value: _WorkerClaim, request: Request):
        worker_matches(request, value.worker_id)
        return {"assignment": coordinator.claim(value.worker_id, value.images, value.max_memory_mb)}

    @app.post("/worker/heartbeat")
    def worker_heartbeat(value: _WorkerLease, request: Request):
        worker_matches(request, value.worker_id)
        return {"active": coordinator.heartbeat(value.worker_id, value.run_id, value.token)}

    @app.post("/worker/complete")
    def worker_complete(value: _WorkerComplete, request: Request):
        worker_matches(request, value.worker_id)
        return coordinator.complete(value.worker_id, value.run_id, value.token, value.completion)

    return app
